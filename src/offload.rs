//! 100% OOM-safe `F_NOCACHE` Disk Expert Offloader for Inkling 2 Mini (`inkling2_loop`).
//!
//! Supports four complementary disk-offloading execution modes with ZERO changes to existing Metal MoE kernels:
//! - **Mode A (`OffloadMode::PerStep`)**: Per-step double-buffered async streaming (`B >= 32`) + sparse
//!   active-expert `pread` streaming (`B <= 16`, up to 16x less I/O at `B = 1`).
//! - **Mode B (`OffloadMode::HotCold`)**: Zipf hot/cold expert RAM cache pinning the top `hot_per_layer`
//!   experts per offloaded layer and `F_NOCACHE pread`ing only cold misses (overlapped with RAM copies).
//! - **Mode C (`OffloadMode::MacroBatch`)**: Layer-pipelined macro-batching (`B_macro = 512..4096`,
//!   `B_micro = 128..256`) keeping only compact FP16 inter-layer residual states (`31.5 KiB/seq`) across
//!   all `B_macro` sequences while streaming each offloaded layer ONCE per macro-step, overlapped with GPU.
//!
//! Zero-page-cache guarantee:
//! - Dedicated read-only `File` handle with `fcntl(fd, F_NOCACHE, 1)` and `fcntl(fd, F_RDAHEAD, 0)`.
//! - Explicit `madvise(..., MADV_DONTNEED)` on strictly interior page-aligned ranges of offloaded mmap tensors.
//! - Live `proc_pid_rusage` (`ri_phys_footprint`) + `mincore()` residency verification printed on every run.

use crate::batch::{BatchEngine, PagedKvPool, PrefixCache, SampleCfg, SeqSlot, KV_BLK, NTC_B, SWA_RING_BLKS};
use crate::engine::{
    slot_of, EmbedArgs, Engine, Layer, MoeGemmHArgs, TopkBArgs, VecAlloc, BLOCK, B_NVEC, C_ARGMAX, C_ATTN, C_COMB,
    C_COMB2, C_DOWN, C_EMBED, C_GATHER, C_GU, C_KVPREP, C_LMHEAD, C_MIXA, C_MIXF, C_OMIX, C_QKV, C_ROUTER, C_TOPK,
    C_WO, E, EXT, FF, HD, LOOP_END, LOOP_START, NCH, NEXP, NH, NKV, NREL, N_PASS, N_PHYS, N_SLOT, TOPK,
};
use crate::metal::{Buf, CmdBuf, Enc, Gpu, Pso};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCompileOptions, MTLDevice, MTLLanguageVersion, MTLLibrary, MTLMathMode,
};
use std::ffi::c_void;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::{AsRawFd, RawFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// Architectural constants mirroring batch.rs
const LM_VCHUNK: usize = 16 * 64; // 1024 vocab rows per TG in k_lmhead_argmax_sg
const LM_VSLICE: usize = 12800;   // f32 vocab slice size for full logits extraction
const N_SWA: usize = 40;
const N_FULL: usize = 8;
const CS_SEQ_ELEMS: usize = N_SLOT * NCH * 3; // 48 * 6656 * 3 = 958,464 f32 (~3.66 MiB/seq)

/// Chunk size for parallel `F_NOCACHE pread` tasks (4 MiB hits peak Apple Silicon NVMe bandwidth).
const READ_CHUNK_BYTES: usize = 4 << 20;

/// Maximum simultaneous live inter-layer residual vectors in `inkling2_loop`:
/// blocks (up to 4) + cur (1) + loop bank (2) -> 7 FP16 slots = 7 * 2304 * 2 B = 31.5 KiB/seq!
pub const B_NVEC_LIVE: usize = 7;

/// Fixed intra-layer `mid` vector slot inside `DecodeArena` (B_NVEC = 10; slots 0..7 are inter-layer live).
const MACRO_MID_SLOT: usize = 7;

// ----------------------------------------------------------------------------- Local POD kernel arg structs (mirror batch.metal)

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Route4Args {
    sel_off: u32,
    t: u32,
    perm_off: u32,
    inv_off: u32,
    aw_off: u32,
    ind_off: u32,
    gu_y: [u32; NTC_B],
    dn_y: [u32; NTC_B],
    tiles_off: [u32; NTC_B],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct WoPostArgs {
    w_lo: u32,
    w_hi: u32,
    row_bytes: u32,
    wtype: u32,
    nrows: u32,
    k: u32,
    xh_off: u32,
    out_off: u32,
    res_off: u32,
    has_res: u32,
    layer_slot: u32,
    max_batch: u32,
    convw_t_off: u32,
    b: u32,
    seq_cs_off: u32,
    seq_pos_off: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MoeCombPostArgs {
    yd_off: u32,
    inv_off: u32,
    res_off: u32,
    out_off: u32,
    layer_slot: u32,
    max_batch: u32,
    convw_t_off: u32,
    seq_cs_off: u32,
    seq_pos_off: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PagedAttnArgs {
    b: u32,
    pool_cap: u32,
    slot_idx: u32,
    is_swa: u32,
    pt_stride: u32,
    pt_mod: u32,
    seq_pos_off: u32,
    seq_suf_off: u32,
    seq_cs_off: u32,
    pt_off: u32,
    layer_slot: u32,
    max_batch: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    r_off: u32,
    sink_off: u32,
    kdyn_off: i32,
    vdyn_off: i32,
    kconv_t_w: u32,
    vconv_t_w: u32,
    qclip_w: u32,
    relclip_w: u32,
    relbias_w: u32,
    log_alpha: f32,
    log_floor: f32,
    gate_off: u32,
    out_off: u32,
    part_off: u32,
    nsplit: u32,
    prefix_len: u32,
    qp_off: u32,
    rp_off: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CombineBArgs {
    part_off: u32,
    nsplit: u32,
    gate_off: u32,
    out_off: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct PrefixAttnArgs {
    b: u32,
    pre_n: u32,
    pre_p0: u32,
    pre_cap: u32,
    k_base: u32,
    v_base: u32,
    window: u32,
    qp_off: u32,
    rp_off: u32,
    sink_off: u32,
    mlo_off: u32,
    relbias_w: u32,
    seq_pos_off: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct LmHeadArgs {
    w_lo: u32,
    w_hi: u32,
    row_bytes: u32,
    wtype: u32,
    n_vocab: u32,
    k: u32,
    xh_off: u32,
    part_off: u32,
    b: u32,
    nvch: u32,
    vstart: u32,
    vcount: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct ArgmaxBatchArgs {
    part_off: u32,
    nvch: u32,
    out_base: u32,
    pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct TopkMergeArgs {
    slice_off: u32,
    vstart: u32,
    vcount: u32,
    topk_off: u32,
    top_k: u32,
    first_slice: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SampleBatchArgs {
    topk_off: u32,
    top_k: u32,
    out_base: u32,
    rng_off: u32,
    temp: f32,
    top_p: f32,
    softcap: f32,
    pad: f32,
}

/// Args for standalone FP32 <-> FP16 macro-state pack/unpack kernels (compiled on `eng.gpu.dev`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VecCopyArgs {
    src_off: u64,
    dst_off: u64,
    count: u32,
    pad: u32,
}

const MACRO_PACK_METAL_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VecCopyArgs {
    ulong src_off;
    ulong dst_off;
    uint count;
    uint pad;
};

// Pack f32 [B][E] in A (buffer 2) -> half [B][E] in M (buffer 9)
kernel void k_pack_f32_to_f16(
    device const float * A [[buffer(2)]],
    device half * M [[buffer(9)]],
    constant VecCopyArgs & a [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const ulong i = (ulong) gid * 4ul;
    if (i + 3ul < (ulong) a.count) {
        *(device half4 *) (M + a.dst_off + i) = half4(*(device const float4 *) (A + a.src_off + i));
    }
}

// Unpack half [B][E] in M (buffer 9) -> f32 [B][E] in A (buffer 2)
kernel void k_unpack_f16_to_f32(
    device float * A [[buffer(2)]],
    device const half * M [[buffer(9)]],
    constant VecCopyArgs & a [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const ulong i = (ulong) gid * 4ul;
    if (i + 3ul < (ulong) a.count) {
        *(device float4 *) (A + a.dst_off + i) = float4(*(device const half4 *) (M + a.src_off + i));
    }
}
"#;

// ----------------------------------------------------------------------------- Offload Policies & Modes

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OffloadPolicy {
    /// Offload all 12 non-looped layers (0..6 prelude + 18..24 coda = ~7.6 GiB);
    /// keep 12 looped layers (6..18, executed 3x/step) + dense weights pinned in RAM (~8.7 GiB).
    PreludeCoda,
    /// Offload 6 non-looped prelude layers (0..6 = ~3.8 GiB).
    HalfNonLoop,
    /// Offload ALL 24 physical layers (~15.25 GiB); keep only ~1.05 GiB dense weights pinned in RAM.
    All24,
}

impl OffloadPolicy {
    pub fn parse(s: &str) -> Result<OffloadPolicy, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "prelude_coda" | "prelude-coda" | "nonloop" | "12" => Ok(OffloadPolicy::PreludeCoda),
            "half_nonloop" | "half-nonloop" | "prelude" | "6" => Ok(OffloadPolicy::HalfNonLoop),
            "all24" | "all_24" | "all-24" | "24" => Ok(OffloadPolicy::All24),
            other => Err(format!(
                "unknown offload policy '{other}' (expected: prelude_coda, half_nonloop, all24, or sweep)"
            )),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            OffloadPolicy::PreludeCoda => "prelude_coda",
            OffloadPolicy::HalfNonLoop => "half_nonloop",
            OffloadPolicy::All24 => "all24",
        }
    }

    pub fn mask(&self) -> [bool; N_PHYS] {
        let mut m = [false; N_PHYS];
        match self {
            OffloadPolicy::PreludeCoda => {
                for ip in 0..LOOP_START {
                    m[ip] = true;
                }
                for ip in LOOP_END..N_PHYS {
                    m[ip] = true;
                }
            }
            OffloadPolicy::HalfNonLoop => {
                for ip in 0..LOOP_START {
                    m[ip] = true;
                }
            }
            OffloadPolicy::All24 => {
                m = [true; N_PHYS];
            }
        }
        m
    }

    pub fn count(&self) -> usize {
        self.mask().iter().filter(|&&b| b).count()
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum OffloadMode {
    PerStep { policy: OffloadPolicy, sparse_io: bool },
    HotCold { policy: OffloadPolicy, hot_per_layer: usize },
    MacroBatch { policy: OffloadPolicy, b_macro: usize, b_micro: usize },
}

// ----------------------------------------------------------------------------- RAM & Zero-Page-Cache Verification

#[derive(Clone, Copy, Debug, Default)]
pub struct RamStats {
    pub phys_footprint_mib: f64,
    pub mmap_resident_total_mib: f64,
    pub mmap_resident_pinned_mib: f64,
    pub mmap_resident_offloaded_mib: f64,
    pub offloaded_logical_mib: f64,
}

fn os_page_size() -> usize {
    let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if p > 0 { p as usize } else { 16384 }
}

/// Strictly interior page-aligned byte range `[start, end)` inside a tensor `[off, off + len)`.
/// Never touches boundary pages that might be shared with adjacent dense weights.
fn interior_page_range(off: u64, len: u64, page: usize, map_len: usize) -> Option<(usize, usize)> {
    let start = (off as usize).div_ceil(page) * page;
    let end = (((off + len) as usize).min(map_len) / page) * page;
    if end > start { Some((start, end)) } else { None }
}

/// Call `madvise(..., MADV_DONTNEED)` on the mmapped GGUF ranges of all offloaded expert tensors (`gu` & `dn`).
/// Returns the total byte span advised.
pub fn evict_offloaded_mmap(eng: &Engine, offload_mask: &[bool; N_PHYS]) -> u64 {
    let page = os_page_size();
    let mut advised = 0u64;
    for (ip, &off) in offload_mask.iter().enumerate() {
        if !off {
            continue;
        }
        let l = &eng.layers[ip];
        for (foff, nbytes) in [
            (l.gu.off, l.gu.expert_bytes * NEXP as u64),
            (l.dn.off, l.dn.expert_bytes * NEXP as u64),
        ] {
            if let Some((start, end)) = interior_page_range(foff, nbytes, page, eng.g.map_len) {
                let len = end - start;
                unsafe {
                    libc::madvise(eng.g.map.add(start) as *mut c_void, len, libc::MADV_DONTNEED);
                }
                advised += len as u64;
            }
        }
    }
    advised
}

/// Measure process physical footprint (`proc_pid_rusage`) AND exact mmap residency (`mincore`) split into
/// pinned (dense + resident layers) vs offloaded expert tensors.
pub fn measure_ram(eng: &Engine, offload_mask: &[bool; N_PHYS]) -> RamStats {
    let phys_footprint_mib = unsafe {
        let mut ri: libc::rusage_info_v4 = std::mem::zeroed();
        let rc = libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            &mut ri as *mut _ as *mut libc::rusage_info_t,
        );
        if rc == 0 {
            ri.ri_phys_footprint as f64 / 1048576.0
        } else {
            0.0
        }
    };

    let page = os_page_size();
    let n_pages = eng.g.map_len / page;
    let mut vec: Vec<libc::c_char> = vec![0; n_pages];
    let rc = unsafe { libc::mincore(eng.g.map as *mut c_void, n_pages * page, vec.as_mut_ptr()) };
    if rc != 0 {
        return RamStats { phys_footprint_mib, ..Default::default() };
    }

    let mut is_offloaded = vec![false; n_pages];
    let mut offloaded_logical_bytes = 0u64;
    for (ip, &off) in offload_mask.iter().enumerate() {
        if !off {
            continue;
        }
        let l = &eng.layers[ip];
        for (foff, nbytes) in [
            (l.gu.off, l.gu.expert_bytes * NEXP as u64),
            (l.dn.off, l.dn.expert_bytes * NEXP as u64),
        ] {
            offloaded_logical_bytes += nbytes;
            if let Some((start, end)) = interior_page_range(foff, nbytes, page, n_pages * page) {
                for p in (start / page)..(end / page) {
                    is_offloaded[p] = true;
                }
            }
        }
    }

    let mut res_total = 0usize;
    let mut res_offloaded = 0usize;
    for p in 0..n_pages {
        if (vec[p] & 1) != 0 {
            res_total += 1;
            if is_offloaded[p] {
                res_offloaded += 1;
            }
        }
    }
    let page_mib = page as f64 / 1048576.0;
    let mmap_resident_total_mib = res_total as f64 * page_mib;
    let mmap_resident_offloaded_mib = res_offloaded as f64 * page_mib;
    let mmap_resident_pinned_mib = (res_total - res_offloaded) as f64 * page_mib;
    RamStats {
        phys_footprint_mib,
        mmap_resident_total_mib,
        mmap_resident_pinned_mib,
        mmap_resident_offloaded_mib,
        offloaded_logical_mib: offloaded_logical_bytes as f64 / 1048576.0,
    }
}

/// Convenience tuple helper returning `(phys_footprint_mib, mmap_resident_total_mib, mmap_resident_offloaded_mib)`.
#[allow(dead_code)]
pub fn measure_ram_tuple(eng: &Engine, offload_mask: &[bool; N_PHYS]) -> (f64, f64, f64) {
    let r = measure_ram(eng, offload_mask);
    (r.phys_footprint_mib, r.mmap_resident_total_mib, r.mmap_resident_offloaded_mib)
}

// ----------------------------------------------------------------------------- Parallel F_NOCACHE Worker Pool & Async DiskExpertLoader

#[cfg(target_os = "macos")]
fn path_from_fd(fd: RawFd) -> std::io::Result<PathBuf> {
    use std::os::unix::ffi::OsStrExt;
    let mut buf = [0u8; libc::PATH_MAX as usize];
    let rc = unsafe { libc::fcntl(fd, libc::F_GETPATH, buf.as_mut_ptr() as *mut libc::c_char) };
    if rc != 0 {
        return Err(std::io::Error::last_os_error());
    }
    let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(PathBuf::from(std::ffi::OsStr::from_bytes(&buf[..len])))
}

#[cfg(not(target_os = "macos"))]
fn path_from_fd(fd: RawFd) -> std::io::Result<PathBuf> {
    std::fs::read_link(format!("/proc/self/fd/{fd}"))
}

fn open_nocache_file(path: &Path) -> std::io::Result<File> {
    let file = File::open(path)?;
    #[cfg(target_os = "macos")]
    unsafe {
        let fd = file.as_raw_fd();
        if libc::fcntl(fd, libc::F_NOCACHE, 1) != 0 {
            eprintln!("[offload] warning: fcntl(F_NOCACHE, 1) failed: {}", std::io::Error::last_os_error());
        }
        if libc::fcntl(fd, libc::F_RDAHEAD, 0) != 0 {
            eprintln!("[offload] warning: fcntl(F_RDAHEAD, 0) failed: {}", std::io::Error::last_os_error());
        }
    }
    Ok(file)
}

#[derive(Clone, Copy, Debug)]
pub struct LayerDiskSpec {
    pub gu_file_off: u64,
    pub gu_expert_bytes: u64,
    pub gu_total_bytes: u64,
    pub dn_file_off: u64,
    pub dn_expert_bytes: u64,
    pub dn_total_bytes: u64,
    /// Offset of `dn` inside `stg[buf_idx]` (4096-byte aligned after `gu`).
    pub dn_stg_off: u64,
    /// Total staging bytes needed for this layer (`dn_stg_off + dn_total_bytes`).
    pub stg_bytes: u64,
}

#[derive(Clone, Copy)]
struct ReadTask {
    file_off: u64,
    dst_ptr: usize,
    len: usize,
}

struct TaskBatch {
    tasks: Vec<ReadTask>,
    cursor: AtomicUsize,
    remaining: AtomicUsize,
}

struct PoolState {
    gen: u64,
    batch: Option<Arc<TaskBatch>>,
    shutdown: bool,
    err: Option<String>,
}

/// Persistent worker pool executing parallel `F_NOCACHE pread` tasks with generation-scoped `Arc<TaskBatch>`.
struct WorkerPool {
    file: Arc<File>,
    run_lock: Mutex<()>,
    shared: Arc<(Mutex<PoolState>, Condvar, Condvar)>,
    workers: Vec<JoinHandle<()>>,
    n_threads: usize,
}

impl WorkerPool {
    fn new(file: Arc<File>, n_threads: usize) -> Self {
        let n_threads = n_threads.clamp(1, 32);
        let shared = Arc::new((
            Mutex::new(PoolState { gen: 0, batch: None, shutdown: false, err: None }),
            Condvar::new(),
            Condvar::new(),
        ));
        let mut workers = Vec::with_capacity(n_threads);
        for wi in 0..n_threads {
            let file_c = Arc::clone(&file);
            let shared_c = Arc::clone(&shared);
            let handle = thread::Builder::new()
                .name(format!("offload-io-{wi}"))
                .spawn(move || {
                    let mut my_gen = 0u64;
                    let (ref lock, ref wake_workers, ref wake_caller) = *shared_c;
                    loop {
                        let batch = {
                            let mut st = lock.lock().unwrap();
                            while st.gen == my_gen && !st.shutdown {
                                st = wake_workers.wait(st).unwrap();
                            }
                            if st.shutdown {
                                break;
                            }
                            my_gen = st.gen;
                            st.batch.clone()
                        };
                        let Some(batch) = batch else { continue };
                        let n = batch.tasks.len();
                        loop {
                            let idx = batch.cursor.fetch_add(1, Ordering::Relaxed);
                            if idx >= n {
                                break;
                            }
                            let task = batch.tasks[idx];
                            let slice = unsafe { std::slice::from_raw_parts_mut(task.dst_ptr as *mut u8, task.len) };
                            if let Err(e) = file_c.read_exact_at(slice, task.file_off) {
                                let mut st = lock.lock().unwrap();
                                if st.err.is_none() {
                                    st.err = Some(format!(
                                        "pread failed at off={} len={}: {e}",
                                        task.file_off, task.len
                                    ));
                                }
                            }
                            if batch.remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                                let _st = lock.lock().unwrap();
                                wake_caller.notify_one();
                            }
                        }
                    }
                })
                .expect("spawn offload IO worker");
            workers.push(handle);
        }
        WorkerPool { file, run_lock: Mutex::new(()), shared, workers, n_threads }
    }

    fn execute(&self, tasks: Vec<ReadTask>) -> Result<u64, String> {
        if tasks.is_empty() {
            return Ok(0);
        }
        let total_bytes: u64 = tasks.iter().map(|t| t.len as u64).sum();
        if tasks.len() == 1 || self.n_threads <= 1 {
            for t in &tasks {
                let slice = unsafe { std::slice::from_raw_parts_mut(t.dst_ptr as *mut u8, t.len) };
                self.file
                    .read_exact_at(slice, t.file_off)
                    .map_err(|e| format!("pread failed at off={} len={}: {e}", t.file_off, t.len))?;
            }
            return Ok(total_bytes);
        }
        let _guard = self.run_lock.lock().unwrap();
        let n = tasks.len();
        let batch = Arc::new(TaskBatch {
            tasks,
            cursor: AtomicUsize::new(0),
            remaining: AtomicUsize::new(n),
        });
        let (ref lock, ref wake_workers, ref wake_caller) = *self.shared;
        {
            let mut st = lock.lock().unwrap();
            st.err = None;
            st.batch = Some(Arc::clone(&batch));
            st.gen += 1;
            wake_workers.notify_all();
            while batch.remaining.load(Ordering::Acquire) > 0 {
                st = wake_caller.wait(st).unwrap();
            }
            st.batch = None;
            if let Some(err) = st.err.take() {
                return Err(err);
            }
        }
        Ok(total_bytes)
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        {
            let (ref lock, ref wake_workers, _) = *self.shared;
            let mut st = lock.lock().unwrap();
            st.shutdown = true;
            st.batch = None;
            wake_workers.notify_all();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

enum CoordReq {
    FullLayer { ip: usize, buf_idx: usize },
    Sparse { ip: usize, buf_idx: usize, experts: Vec<usize> },
    Shutdown,
}

fn push_chunked_tasks(tasks: &mut Vec<ReadTask>, file_off: u64, dst_ptr: usize, total_len: usize) {
    let mut off = 0usize;
    while off < total_len {
        let len = (total_len - off).min(READ_CHUNK_BYTES);
        tasks.push(ReadTask { file_off: file_off + off as u64, dst_ptr: dst_ptr + off, len });
        off += len;
    }
}

fn build_full_layer_tasks(spec: &LayerDiskSpec, stg_base: usize) -> Vec<ReadTask> {
    let mut tasks = Vec::with_capacity(200);
    push_chunked_tasks(&mut tasks, spec.gu_file_off, stg_base, spec.gu_total_bytes as usize);
    push_chunked_tasks(
        &mut tasks,
        spec.dn_file_off,
        stg_base + spec.dn_stg_off as usize,
        spec.dn_total_bytes as usize,
    );
    tasks
}

/// Coalesce sorted/deduplicated expert IDs into contiguous half-open runs `[e0, e1)` for fewer syscalls.
fn coalesce_expert_runs(experts: &[usize]) -> Vec<(usize, usize)> {
    if experts.is_empty() {
        return Vec::new();
    }
    let mut sorted: Vec<usize> = experts.iter().copied().filter(|&e| e < NEXP).collect();
    sorted.sort_unstable();
    sorted.dedup();
    let mut runs = Vec::new();
    let mut i = 0;
    while i < sorted.len() {
        let e0 = sorted[i];
        let mut e1 = e0 + 1;
        while i + 1 < sorted.len() && sorted[i + 1] == e1 {
            i += 1;
            e1 += 1;
        }
        runs.push((e0, e1));
        i += 1;
    }
    runs
}

fn build_sparse_tasks(spec: &LayerDiskSpec, stg_base: usize, experts: &[usize]) -> Vec<ReadTask> {
    let runs = coalesce_expert_runs(experts);
    let mut tasks = Vec::with_capacity(runs.len() * 2);
    for (e0, e1) in runs {
        let cnt = (e1 - e0) as u64;
        let gu_off = e0 as u64 * spec.gu_expert_bytes;
        let gu_len = (cnt * spec.gu_expert_bytes) as usize;
        push_chunked_tasks(&mut tasks, spec.gu_file_off + gu_off, stg_base + gu_off as usize, gu_len);

        let dn_off = e0 as u64 * spec.dn_expert_bytes;
        let dn_len = (cnt * spec.dn_expert_bytes) as usize;
        push_chunked_tasks(
            &mut tasks,
            spec.dn_file_off + dn_off,
            stg_base + spec.dn_stg_off as usize + dn_off as usize,
            dn_len,
        );
    }
    tasks
}

/// Double-buffered parallel `F_NOCACHE` disk expert loader with fixed `~1.3–1.4 GiB` Metal staging RAM.
pub struct DiskExpertLoader {
    pub stg: [Buf; 2],
    stg_ptrs: [usize; 2],
    specs: [LayerDiskSpec; N_PHYS],
    pool: Arc<WorkerPool>,
    coord_tx: Option<mpsc::Sender<CoordReq>>,
    coord_thread: Option<JoinHandle<()>>,
    done_slots: Arc<(Mutex<[Option<(u64, Duration)>; 2]>, Condvar)>,
    in_flight: [bool; 2],
    pub n_threads: usize,
    pub path: PathBuf,
}

impl DiskExpertLoader {
    pub fn compute_specs(eng: &Engine) -> ([LayerDiskSpec; N_PHYS], usize) {
        let mut specs = [LayerDiskSpec {
            gu_file_off: 0,
            gu_expert_bytes: 0,
            gu_total_bytes: 0,
            dn_file_off: 0,
            dn_expert_bytes: 0,
            dn_total_bytes: 0,
            dn_stg_off: 0,
            stg_bytes: 0,
        }; N_PHYS];
        let mut max_stg = 0u64;
        for ip in 0..N_PHYS {
            let l = &eng.layers[ip];
            let gu_total = l.gu.expert_bytes * NEXP as u64;
            let dn_total = l.dn.expert_bytes * NEXP as u64;
            let dn_stg_off = gu_total.div_ceil(4096) * 4096;
            let stg_bytes = dn_stg_off + dn_total;
            assert!(
                l.gu.off + gu_total <= eng.g.file_len as u64 && l.dn.off + dn_total <= eng.g.file_len as u64,
                "layer {ip} expert tensor exceeds GGUF file length"
            );
            specs[ip] = LayerDiskSpec {
                gu_file_off: l.gu.off,
                gu_expert_bytes: l.gu.expert_bytes,
                gu_total_bytes: gu_total,
                dn_file_off: l.dn.off,
                dn_expert_bytes: l.dn.expert_bytes,
                dn_total_bytes: dn_total,
                dn_stg_off,
                stg_bytes,
            };
            max_stg = max_stg.max(stg_bytes);
        }
        let page = os_page_size() as u64;
        let alloc_bytes = (max_stg.div_ceil(page) * page) as usize;
        (specs, alloc_bytes)
    }

    pub fn new(eng: &Engine, n_threads: usize) -> std::io::Result<Self> {
        let path = path_from_fd(eng.g.file.as_raw_fd())?;
        Self::new_with_path(eng, &path, n_threads)
    }

    pub fn new_with_path(eng: &Engine, path: &Path, n_threads: usize) -> std::io::Result<Self> {
        let n_threads = n_threads.clamp(1, 16);
        let file = Arc::new(open_nocache_file(path)?);
        let (specs, alloc_bytes) = Self::compute_specs(eng);

        let stg = [eng.gpu.buffer(alloc_bytes), eng.gpu.buffer(alloc_bytes)];
        let stg_ptrs = [stg[0].contents().as_ptr() as usize, stg[1].contents().as_ptr() as usize];

        let pool = Arc::new(WorkerPool::new(file, n_threads));
        let done_slots = Arc::new((Mutex::new([None, None]), Condvar::new()));
        let (tx, rx) = mpsc::channel::<CoordReq>();

        let pool_c = Arc::clone(&pool);
        let done_c = Arc::clone(&done_slots);
        let specs_c = specs;
        let ptrs_c = stg_ptrs;
        let coord_thread = thread::Builder::new()
            .name("offload-coord".into())
            .spawn(move || {
                while let Ok(req) = rx.recv() {
                    match req {
                        CoordReq::FullLayer { ip, buf_idx } => {
                            let t0 = Instant::now();
                            let tasks = build_full_layer_tasks(&specs_c[ip], ptrs_c[buf_idx]);
                            let bytes = pool_c.execute(tasks).expect("async full-layer pread");
                            let elapsed = t0.elapsed();
                            let (ref lock, ref cvar) = *done_c;
                            let mut slots = lock.lock().unwrap();
                            slots[buf_idx] = Some((bytes, elapsed));
                            cvar.notify_all();
                        }
                        CoordReq::Sparse { ip, buf_idx, experts } => {
                            let t0 = Instant::now();
                            let tasks = build_sparse_tasks(&specs_c[ip], ptrs_c[buf_idx], &experts);
                            let bytes = pool_c.execute(tasks).expect("async sparse pread");
                            let elapsed = t0.elapsed();
                            let (ref lock, ref cvar) = *done_c;
                            let mut slots = lock.lock().unwrap();
                            slots[buf_idx] = Some((bytes, elapsed));
                            cvar.notify_all();
                        }
                        CoordReq::Shutdown => break,
                    }
                }
            })
            .expect("spawn offload coordinator");

        Ok(DiskExpertLoader {
            stg,
            stg_ptrs,
            specs,
            pool,
            coord_tx: Some(tx),
            coord_thread: Some(coord_thread),
            done_slots,
            in_flight: [false, false],
            n_threads,
            path: path.to_path_buf(),
        })
    }

    #[inline]
    pub fn staging_bytes(&self) -> usize {
        self.stg[0].length() + self.stg[1].length()
    }

    #[inline]
    pub fn layer_spec(&self, ip: usize) -> LayerDiskSpec {
        self.specs[ip]
    }

    /// Asynchronously prefetch full layer `ip` (`gu` + `dn`) into `stg[buf_idx]` on background threads.
    /// Caller MUST guarantee the GPU is no longer reading `stg[buf_idx]`.
    pub fn prefetch_layer(&mut self, ip: usize, buf_idx: usize) {
        assert!(buf_idx < 2 && ip < N_PHYS);
        assert!(!self.in_flight[buf_idx], "staging slot {buf_idx} already has an in-flight prefetch");
        {
            let (ref lock, _) = *self.done_slots;
            let mut slots = lock.lock().unwrap();
            slots[buf_idx] = None;
        }
        self.in_flight[buf_idx] = true;
        self.coord_tx
            .as_ref()
            .unwrap()
            .send(CoordReq::FullLayer { ip, buf_idx })
            .expect("offload coordinator channel closed");
    }

    /// Asynchronously prefetch only `active_experts` of layer `ip` into their exact offsets in `stg[buf_idx]`.
    pub fn prefetch_sparse_experts(&mut self, ip: usize, buf_idx: usize, active_experts: &[usize]) {
        assert!(buf_idx < 2 && ip < N_PHYS);
        assert!(!self.in_flight[buf_idx], "staging slot {buf_idx} already has an in-flight prefetch");
        {
            let (ref lock, _) = *self.done_slots;
            let mut slots = lock.lock().unwrap();
            slots[buf_idx] = None;
        }
        self.in_flight[buf_idx] = true;
        self.coord_tx
            .as_ref()
            .unwrap()
            .send(CoordReq::Sparse { ip, buf_idx, experts: active_experts.to_vec() })
            .expect("offload coordinator channel closed");
    }

    /// Wait for an in-flight async prefetch into `stg[buf_idx]` to finish. Returns `(bytes_read, io_elapsed)`.
    pub fn wait_layer(&mut self, buf_idx: usize) -> (u64, Duration) {
        assert!(buf_idx < 2);
        if !self.in_flight[buf_idx] {
            return (0, Duration::ZERO);
        }
        let (ref lock, ref cvar) = *self.done_slots;
        let mut slots = lock.lock().unwrap();
        while slots[buf_idx].is_none() {
            slots = cvar.wait(slots).unwrap();
        }
        let res = slots[buf_idx].take().unwrap();
        self.in_flight[buf_idx] = false;
        res
    }

    /// Synchronously load ONLY `active_experts` of layer `ip` via parallel `F_NOCACHE pread` into `stg[buf_idx]`.
    pub fn load_sparse_experts(&mut self, ip: usize, buf_idx: usize, active_experts: &[usize]) -> (u64, Duration) {
        assert!(buf_idx < 2 && ip < N_PHYS);
        assert!(!self.in_flight[buf_idx], "staging slot {buf_idx} is in-flight");
        if active_experts.is_empty() {
            return (0, Duration::ZERO);
        }
        let t0 = Instant::now();
        let tasks = build_sparse_tasks(&self.specs[ip], self.stg_ptrs[buf_idx], active_experts);
        let bytes = self.pool.execute(tasks).expect("sparse pread");
        (bytes, t0.elapsed())
    }

    /// Synchronously load full layer `ip` into `stg[buf_idx]`.
    #[allow(dead_code)]
    pub fn load_full_layer_sync(&mut self, ip: usize, buf_idx: usize) -> (u64, Duration) {
        assert!(buf_idx < 2 && ip < N_PHYS);
        assert!(!self.in_flight[buf_idx], "staging slot {buf_idx} is in-flight");
        let t0 = Instant::now();
        let tasks = build_full_layer_tasks(&self.specs[ip], self.stg_ptrs[buf_idx]);
        let bytes = self.pool.execute(tasks).expect("full-layer pread");
        (bytes, t0.elapsed())
    }

    /// Read a single expert `(gu, dn)` directly into caller-provided buffers using `F_NOCACHE pread`.
    fn read_expert_into(&self, ip: usize, e: usize, gu_dst: *mut u8, dn_dst: *mut u8) -> u64 {
        let s = &self.specs[ip];
        let mut tasks = Vec::with_capacity(4);
        push_chunked_tasks(
            &mut tasks,
            s.gu_file_off + e as u64 * s.gu_expert_bytes,
            gu_dst as usize,
            s.gu_expert_bytes as usize,
        );
        push_chunked_tasks(
            &mut tasks,
            s.dn_file_off + e as u64 * s.dn_expert_bytes,
            dn_dst as usize,
            s.dn_expert_bytes as usize,
        );
        self.pool.execute(tasks).expect("hot-cache pread")
    }
}

impl Drop for DiskExpertLoader {
    fn drop(&mut self) {
        // Shut down and join coordinator thread FIRST (which holds Arc<WorkerPool>),
        // guaranteeing NO background thread touches `stg` before Metal buffers are released!
        if let Some(tx) = self.coord_tx.take() {
            let _ = tx.send(CoordReq::Shutdown);
        }
        if let Some(h) = self.coord_thread.take() {
            let _ = h.join();
        }
        // `pool` is dropped next (joining all IO workers), and only then `stg` is dropped.
    }
}

// ----------------------------------------------------------------------------- Raw SSD F_NOCACHE Calibration

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct SsdCalibResult {
    pub best_threads: usize,
    pub best_seq_gibs: f64,
    pub best_rand_gibs: f64,
    pub per_thread: Vec<(usize, f64, f64)>, // (n_threads, seq_GiB/s, rand_expert_GiB/s)
}

/// Run a quick ~1-second raw `F_NOCACHE` multi-threaded `pread` calibration on the real GGUF
/// across `1 / 2 / 4 / 6 / 8` threads and automatically select the fastest thread count.
pub fn calibrate_ssd(eng: &Engine) -> SsdCalibResult {
    let path = path_from_fd(eng.g.file.as_raw_fd()).expect("resolve GGUF path from fd");
    let file = Arc::new(open_nocache_file(&path).expect("open GGUF with F_NOCACHE"));
    let (specs, alloc_bytes) = DiskExpertLoader::compute_specs(eng);

    // Single temporary page-aligned staging buffer (~709 MiB, dropped immediately after calibration)
    let tmp_buf = eng.gpu.buffer(alloc_bytes);
    let base = tmp_buf.contents().as_ptr() as usize;

    let thread_counts = [1usize, 2, 4, 6, 8];
    let mut per_thread = Vec::with_capacity(thread_counts.len());
    let mut best_threads = 6usize;
    let mut best_seq = 0.0f64;
    let mut best_rand = 0.0f64;

    println!("=== Raw F_NOCACHE NVMe Calibration ({}) ===", path.file_name().and_then(|s| s.to_str()).unwrap_or("gguf"));
    for (ti, &nt) in thread_counts.iter().enumerate() {
        let pool = WorkerPool::new(Arc::clone(&file), nt);

        // 1. Sequential full-layer read (cycle physical layers so SSD SLC/DRAM isn't re-hit on same LBA)
        let seq_ip = (ti * 4) % N_PHYS;
        let seq_tasks = build_full_layer_tasks(&specs[seq_ip], base);
        let t0 = Instant::now();
        let seq_bytes = pool.execute(seq_tasks).expect("calib seq pread");
        let seq_s = t0.elapsed().as_secs_f64().max(1e-9);
        let seq_gibs = seq_bytes as f64 / (1u64 << 30) as f64 / seq_s;

        // 2. Random-expert sparse read: 32 scattered experts from another layer (~170 MiB)
        let rand_ip = (seq_ip + 7) % N_PHYS;
        let rand_exps: Vec<usize> = (0..32).map(|k| (k * 37 + ti * 13) % NEXP).collect();
        let rand_tasks = build_sparse_tasks(&specs[rand_ip], base, &rand_exps);
        let t1 = Instant::now();
        let rand_bytes = pool.execute(rand_tasks).expect("calib rand pread");
        let rand_s = t1.elapsed().as_secs_f64().max(1e-9);
        let rand_gibs = rand_bytes as f64 / (1u64 << 30) as f64 / rand_s;

        println!(
            "  threads={nt}: sequential {:5.2} GiB/s ({:.0} MiB in {:5.1} ms) | sparse-32 random {:5.2} GiB/s ({:.0} MiB in {:5.1} ms)",
            seq_gibs,
            seq_bytes as f64 / 1048576.0,
            seq_s * 1e3,
            rand_gibs,
            rand_bytes as f64 / 1048576.0,
            rand_s * 1e3,
        );
        per_thread.push((nt, seq_gibs, rand_gibs));
        if seq_gibs > best_seq {
            best_seq = seq_gibs;
            best_rand = rand_gibs;
            best_threads = nt;
        }
        drop(pool);
    }
    drop(tmp_buf);
    println!("  -> selected optimal F_NOCACHE reader threads = {best_threads} ({best_seq:.2} GiB/s seq, {best_rand:.2} GiB/s sparse)\n");
    SsdCalibResult { best_threads, best_seq_gibs: best_seq, best_rand_gibs: best_rand, per_thread }
}

// ----------------------------------------------------------------------------- Shared Low-Level Split-Layer Metal Encoding

#[derive(Clone, Copy)]
enum ExpertWeightSrc<'a> {
    Resident,
    Staged { buf: &'a Buf, dn_off: u64 },
}

/// Encode stages 1..6a of `decode_layer`:
/// Attention inputs -> QKV -> Paged/Cascade Attention -> `wo` post-conv -> FFN inputs -> Router + TopK + Route4 + Gather.
/// Never touches `l.gu` or `l.dn`! Returns `mid` vector index (having pushed `cur` to `blocks` if `block_start`).
#[allow(clippy::too_many_arguments)]
fn encode_layer_pre_moe(
    be: &BatchEngine,
    e: &Enc,
    b_len: usize,
    max_pos: usize,
    min_suf: usize,
    slot: usize,
    ip: usize,
    cur: usize,
    blocks: &mut Vec<usize>,
    msg: Option<(usize, usize)>,
    mid: usize,
    cs_max_batch: usize,
) {
    let eng = be.eng;
    let l = &eng.layers[ip];
    let d = be.da;
    let ctl = msg.map(|_| &eng.loops[ip - LOOP_START]);
    let es = E as u32;

    // 1. Attention inputs (depth mix + optional loop controller -> half xh[0..3])
    let norms: [i32; 3] = if l.swa { [l.attn_norm; 3] } else { [l.q_inp_norm, l.k_inp_norm, l.v_inp_norm] };
    let mut st: Vec<u32> = blocks.iter().map(|&v| d.vec(v)).collect();
    st.push(d.vec(cur));
    let nq = if l.attn_res_query >= 0 { l.n_attn_q } else { 1 };
    if let Some(lw) = ctl {
        let (b0, b1) = msg.unwrap();
        let mut ja = eng.mix_job_off(&st, l.attn_res_query, nq, &d.inp[..nq as usize], &[], es);
        let mut jm = eng.mix_job_off(&[d.vec(b0), d.vec(b1)], lw.msg_query as i32, 4, &d.msg, &[], es);
        if l.swa {
            ja.hcopy[0] = [d.cath[0] as i32, d.cath[1] as i32, d.cath[2] as i32];
        } else {
            for j in 0..3 {
                ja.hcopy[j][0] = d.cath[j] as i32;
            }
        }
        for j in 0..4 {
            jm.hcopy[j][0] = (d.cath[j] + (E / 2) as u32) as i32;
        }
        be.mix_b(e, C_MIXA, &[ja, jm], b_len);
        let jobs: Vec<(usize, u32, u32, u32)> =
            (0..3).map(|j| (j, if l.swa { d.inp[0] } else { d.inp[j] }, d.msg[j], norms[j] as u32)).collect();
        be.ctrl_b(e, lw, &jobs, b_len, &[d.xh[0], d.xh[1], d.xh[2]]);
    } else {
        let nn: Vec<i32> = if l.swa { vec![l.attn_norm] } else { norms.to_vec() };
        let mut ja = eng.mix_job_off(&st, l.attn_res_query, nq, &d.inp[..nq as usize], &nn, es);
        for q in 0..nq as usize {
            ja.hout[q] = d.xh[q] as i32;
        }
        ja.nof32 = 0xF;
        be.mix_b(e, C_MIXA, &[ja], b_len);
    }

    // 2. QKV + Gate + Rel + Sink (+ dyn conv taps) fused segmented GEMM
    let xi: [usize; 3] = if ctl.is_some() || !l.swa { [0, 1, 2] } else { [0, 0, 0] };
    let mut hs = vec![
        Engine::hseg(&l.wq, d.xh[xi[0]], d.q),
        Engine::hseg(&l.wgate, d.xh[xi[0]], d.gate),
        Engine::hseg(&l.wrel, d.xh[xi[0]], d.rel),
        Engine::hseg(&l.wsink, d.xh[xi[0]], d.sink),
        Engine::hseg(&l.wk, d.xh[xi[1]], d.k),
        Engine::hseg(&l.wv, d.xh[xi[2]], d.v),
    ];
    hs[0].hout = 1;
    hs[1].hout = 1;
    if let (Some(kp), Some(vp)) = (&l.kconvp, &l.vconvp) {
        hs.push(Engine::hseg(kp, d.xh[xi[1]], d.kdyn));
        hs.push(Engine::hseg(vp, d.xh[xi[2]], d.vdyn));
    }
    be.gemm_h(e, C_QKV, &hs, b_len);

    // 3. Slot-major paged decode attention (with M5 neural-accelerator shared-prefix cascade when prefix visible)
    let (is_swa, slot_idx) = eng.slot_kind[slot];
    let pool_cap = if is_swa { be.pool.swa_cap } else { be.pool.full_cap };
    if is_swa {
        e.buf(&be.pool.swa, 0, 3);
    } else {
        e.buf(&be.pool.full, 0, 3);
    }
    let pt_off = if is_swa { be.ml.swa_pt } else { be.ml.full_pt };
    let pt_stride = if is_swa { SWA_RING_BLKS } else { be.max_full_blks };
    let prefix_len = be.prefix.as_ref().map_or(0, |p| p.prefix_len);
    let use_prefix = prefix_len > 0 && (!is_swa || min_suf < EXT);

    let pa_base = PagedAttnArgs {
        b: b_len as u32,
        pool_cap: pool_cap as u32,
        slot_idx: slot_idx as u32,
        is_swa: is_swa as u32,
        pt_stride: pt_stride as u32,
        pt_mod: SWA_RING_BLKS as u32,
        seq_pos_off: be.ml.seq_pos,
        seq_suf_off: be.ml.seq_suf,
        seq_cs_off: be.ml.seq_cs,
        pt_off,
        layer_slot: slot as u32,
        max_batch: cs_max_batch as u32,
        q_off: d.q,
        k_off: d.k,
        v_off: d.v,
        r_off: d.rel,
        sink_off: d.sink,
        kdyn_off: if l.swa { -1 } else { d.kdyn as i32 },
        vdyn_off: if l.swa { -1 } else { d.vdyn as i32 },
        kconv_t_w: l.k_conv_t,
        vconv_t_w: l.v_conv_t,
        qclip_w: l.q_clip,
        relclip_w: l.rel_clip,
        relbias_w: l.rel_bias,
        log_alpha: eng.log_alpha,
        log_floor: eng.log_floor,
        gate_off: d.gate,
        out_off: d.agh,
        part_off: d.pre_mlo,
        nsplit: 1,
        prefix_len: prefix_len as u32,
        qp_off: d.qp,
        rp_off: d.rp,
        pad0: 0,
    };

    if use_prefix {
        let pre = be.prefix.as_ref().unwrap();
        be.go(e, C_KVPREP, "k_kvprep_dec_b", &pa_base, (b_len, NKV, 1), 128);
        let (pre_buf, pre_n, pre_p0, pre_cap) = if is_swa {
            (&pre.pre_swa, pre.pre_swa_n, pre.pre_swa_p0, pre.pre_swa_cap)
        } else {
            (&pre.pre_full, pre.prefix_len, 0usize, pre.pre_full_cap)
        };
        e.buf(pre_buf, 0, 8);
        let k_base = (slot_idx * 2 * NKV * pre_cap * HD) as u32;
        let v_base = ((slot_idx * 2 + 1) * NKV * pre_cap * HD) as u32;
        let pfa = PrefixAttnArgs {
            b: b_len as u32,
            pre_n: pre_n as u32,
            pre_p0: pre_p0 as u32,
            pre_cap: pre_cap as u32,
            k_base,
            v_base,
            window: if is_swa { EXT as u32 } else { 0 },
            qp_off: d.qp,
            rp_off: d.rp,
            sink_off: d.sink,
            mlo_off: d.pre_mlo,
            relbias_w: l.rel_bias,
            seq_pos_off: be.ml.seq_pos,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        be.go(e, C_ATTN, "k_attn_prefix_h", &pfa, ((b_len * 3).div_ceil(32), NKV, 1), 256);
        be.go(e, C_COMB, "k_attn_suffix_paged_b", &pa_base, (NKV, b_len, 1), 128);
    } else {
        let nkeys_max = if is_swa { (max_pos + 1).min(EXT) } else { max_pos + 1 };
        let nsplit = if b_len * NKV >= 64 && nkeys_max <= 2048 {
            1usize
        } else {
            (64usize.div_ceil(b_len * NKV)).max(nkeys_max.div_ceil(1024)).min(d.max_split).max(1)
        };
        let pa = PagedAttnArgs { nsplit: nsplit as u32, ..pa_base };
        be.go(e, C_ATTN, "k_attn_paged_b", &pa, (nsplit, NKV, b_len), 128);
        if nsplit > 1 {
            let ca = CombineBArgs { part_off: d.pre_mlo, nsplit: nsplit as u32, gate_off: d.gate, out_off: d.agh };
            be.go(e, C_COMB, "k_attn_combine_b", &ca, (NH, b_len, 1), 128);
        }
    }

    // 4. Fused wo GEMM + vectorized SoA ring post-conv + optional residual -> vec(mid)
    let block_start = ip % BLOCK == 0;
    let wo = WoPostArgs {
        w_lo: l.wo.off as u32,
        w_hi: (l.wo.off >> 32) as u32,
        row_bytes: l.wo.row_bytes,
        wtype: l.wo.ty,
        nrows: l.wo.rows,
        k: l.wo.k,
        xh_off: d.agh,
        out_off: d.vec(mid),
        res_off: d.vec(cur),
        has_res: (!block_start) as u32,
        layer_slot: slot as u32,
        max_batch: cs_max_batch as u32,
        convw_t_off: l.attn_post_conv_t,
        b: b_len as u32,
        seq_cs_off: be.ml.seq_cs,
        seq_pos_off: be.ml.seq_pos,
    };
    if b_len > 64 {
        be.go(e, C_WO, "k_wo_post_sg8", &wo, (b_len.div_ceil(128), E / 64, 1), 256);
    } else {
        be.go(e, C_WO, "k_wo_post_sg4", &wo, (1, E / 64, 1), 128);
    }
    if block_start {
        blocks.push(cur);
    }

    // 5. FFN input (depth mix + optional loop controller -> half fh)
    let mut st: Vec<u32> = blocks.iter().map(|&v| d.vec(v)).collect();
    st.push(d.vec(mid));
    if let Some(lw) = ctl {
        let mut jf = eng.mix_job_off(&st, l.ffn_res_query as i32, 1, &[d.fin], &[], es);
        jf.hcopy[0][0] = d.cath[3] as i32;
        be.mix_b(e, C_MIXF, &[jf], b_len);
        be.ctrl_b(e, lw, &[(3, d.fin, d.msg[3], l.ffn_norm)], b_len, &[d.fh]);
    } else {
        let mut jf = eng.mix_job_off(&st, l.ffn_res_query as i32, 1, &[d.fin], &[l.ffn_norm as i32], es);
        jf.hout[0] = d.fh as i32;
        jf.nof32 = 1;
        be.mix_b(e, C_MIXF, &[jf], b_len);
    }

    // 6a. Router + TopK + Route4 + Gather (all read PB / act only, never expert weights!)
    be.gemm_h(e, C_ROUTER, &[Engine::hseg_pb(l.router, NEXP as u32, E as u32, d.fh, d.rlog)], b_len);
    let ta = TopkBArgs {
        logits_off: d.rlog,
        bias_off: l.exp_bias,
        sel_off: d.sel,
        t: b_len as u32,
        wscale: eng.wscale,
        pad: [0; 3],
    };
    be.go(e, C_TOPK, "k_topk_b", &ta, (b_len, 1, 1), 32);
    let ra = Route4Args {
        sel_off: d.sel,
        t: b_len as u32,
        perm_off: d.perm,
        inv_off: d.inv,
        aw_off: d.aw,
        ind_off: d.ind,
        gu_y: [(FF / 32) as u32, (FF / 32) as u32, (FF / 4) as u32, (FF / 4) as u32],
        dn_y: [(E / 64) as u32, (E / 64) as u32, (E / 8) as u32, (E / 8) as u32],
        tiles_off: d.tl,
    };
    be.go(e, C_TOPK, "k_moe_route4", &ra, (1, 1, 1), 1024);
    be.tohalf_gather(e, C_GATHER, d.fh, d.xgh, d.perm, b_len * TOPK);
}

/// Encode stages 6b + 7 of `decode_layer`:
/// MoE gate_up + down (zero-kernel-change buffer(0) swap for offloaded staging!) -> re-bind `eng.w` -> `k_moe_comb_post_b`.
#[allow(clippy::too_many_arguments)]
fn encode_layer_post_moe(
    be: &BatchEngine,
    e: &Enc,
    l: &Layer,
    b_len: usize,
    slot: usize,
    mid: usize,
    out: usize,
    cs_max_batch: usize,
    src: ExpertWeightSrc<'_>,
) {
    let eng = be.eng;
    let d = be.da;

    let (gu_off, dn_off) = match src {
        ExpertWeightSrc::Resident => (l.gu.off, l.dn.off),
        ExpertWeightSrc::Staged { buf, dn_off } => {
            e.buf(buf, 0, 0);
            (0u64, dn_off)
        }
    };

    let gu_cls: [(&str, usize); NTC_B] =
        [("k_moe_sg8", 256), ("k_moe_sg4", 128), ("k_moe_dir_gu2", 128), ("k_moe_dir_gu1", 128)];
    let dn_cls: [(&str, usize); NTC_B] =
        [("k_moe_sg8", 256), ("k_moe_sg4", 128), ("k_moe_dir_dn2", 128), ("k_moe_dir_dn1", 128)];

    if be.skip.get() & (1 << C_GU) == 0 {
        for (c, &(name, tg)) in gu_cls.iter().enumerate() {
            let g = MoeGemmHArgs {
                w_lo: gu_off as u32,
                w_hi: (gu_off >> 32) as u32,
                expert_bytes: l.gu.expert_bytes as u32,
                row_bytes: l.gu.row_bytes,
                wtype: l.gu.ty,
                nrows: (2 * FF) as u32,
                k: E as u32,
                xh_off: d.xgh,
                tiles_off: d.tl[c],
                ntiles_off: d.ind + (c * 4) as u32,
                out_off: d.acth,
                ldo: FF as u32,
                situ: 1,
                aw_off: d.aw,
                beta: eng.situ_beta,
                lbeta: eng.situ_lbeta,
            };
            e.pso(&eng.gpu.pso(name));
            e.bytes(&g, 5);
            e.dispatch_indirect(&be.act, (d.ind as usize + c * 4) * 4, tg);
        }
        e.barrier();
    }

    if be.skip.get() & (1 << C_DOWN) == 0 {
        for (c, &(name, tg)) in dn_cls.iter().enumerate() {
            let dn = MoeGemmHArgs {
                w_lo: dn_off as u32,
                w_hi: (dn_off >> 32) as u32,
                expert_bytes: l.dn.expert_bytes as u32,
                row_bytes: l.dn.row_bytes,
                wtype: l.dn.ty,
                nrows: E as u32,
                k: FF as u32,
                xh_off: d.acth,
                tiles_off: d.tl[c],
                ntiles_off: d.ind + (c * 4) as u32,
                out_off: d.yd,
                ldo: E as u32,
                situ: 0,
                aw_off: 0,
                beta: 0.0,
                lbeta: 0.0,
            };
            e.pso(&eng.gpu.pso(name));
            e.bytes(&dn, 5);
            e.dispatch_indirect(&be.act, (d.ind as usize + (NTC_B + c) * 4) * 4, tg);
        }
        e.barrier();
    }

    // Restore dense weights buffer(0) immediately after staged MoE kernels!
    if matches!(src, ExpertWeightSrc::Staged { .. }) {
        e.buf(&eng.w, 0, 0);
    }

    // 7. Fused MoE combine + vectorized SoA ring post-conv + residual -> vec(out)
    let cp = MoeCombPostArgs {
        yd_off: d.yd,
        inv_off: d.inv,
        res_off: d.vec(mid),
        out_off: d.vec(out),
        layer_slot: slot as u32,
        max_batch: cs_max_batch as u32,
        convw_t_off: l.ffn_post_conv_t,
        seq_cs_off: be.ml.seq_cs,
        seq_pos_off: be.ml.seq_pos,
        pad0: 0,
        pad1: 0,
        pad2: 0,
    };
    be.go(e, C_COMB2, "k_moe_comb_post_b", &cp, (b_len, E.div_ceil(1024), 1), 256);
}

/// Encode final depth mix (`out_mix`) -> half `d.fh[0..b_len]`.
fn encode_out_mix(be: &BatchEngine, e: &Enc, b_len: usize, blocks: &[usize], cur: usize) {
    let eng = be.eng;
    let d = be.da;
    let mut st: Vec<u32> = blocks.iter().map(|&v| d.vec(v)).collect();
    st.push(d.vec(cur));
    let mut jo = eng.mix_job_off(&st, eng.output_res_query as i32, 1, &[d.fin], &[eng.output_norm as i32], E as u32);
    jo.hout[0] = d.fh as i32;
    jo.nof32 = 1;
    be.mix_b(e, C_OMIX, &[jo], b_len);
}

/// Encode `lm_head` + greedy argmax OR vocab-sliced GPU top-k/top-p sampling writing to `tok_out_buf` at `out_base`.
fn encode_lmhead_sample(
    be: &BatchEngine,
    e: &Enc,
    b_len: usize,
    tok_out_buf: &Buf,
    out_base: u32,
    rng_off: u32,
    sample: Option<SampleCfg>,
) {
    let eng = be.eng;
    let d = be.da;
    e.buf(tok_out_buf, 0, 6);
    let w = &eng.output;
    match sample {
        Some(sc) if !sc.greedy() => {
            let mut vstart = 0usize;
            let mut first = true;
            while vstart < eng.n_vocab {
                let vcount = (eng.n_vocab - vstart).min(LM_VSLICE);
                let la = LmHeadArgs {
                    w_lo: w.off as u32,
                    w_hi: (w.off >> 32) as u32,
                    row_bytes: w.row_bytes,
                    wtype: w.ty,
                    n_vocab: eng.n_vocab as u32,
                    k: w.k,
                    xh_off: d.fh,
                    part_off: d.slice_buf,
                    b: b_len as u32,
                    nvch: 0,
                    vstart: vstart as u32,
                    vcount: vcount as u32,
                };
                let tiles = vcount.div_ceil(64);
                if b_len > 64 {
                    be.go(e, C_LMHEAD, "k_lmhead_slice_sg8", &la, (b_len.div_ceil(128), tiles, 1), 256);
                } else {
                    be.go(e, C_LMHEAD, "k_lmhead_slice_sg4", &la, (1, tiles, 1), 128);
                }
                let tm = TopkMergeArgs {
                    slice_off: d.slice_buf,
                    vstart: vstart as u32,
                    vcount: vcount as u32,
                    topk_off: d.topk_buf,
                    top_k: sc.top_k.clamp(1, 64) as u32,
                    first_slice: first as u32,
                    pad0: 0,
                    pad1: 0,
                };
                be.go(e, C_ARGMAX, "k_topk_merge", &tm, (b_len, 1, 1), 256);
                vstart += vcount;
                first = false;
            }
            e.buf(&be.rng_buf, 0, 7);
            let sa = SampleBatchArgs {
                topk_off: d.topk_buf,
                top_k: sc.top_k.clamp(1, 64) as u32,
                out_base,
                rng_off,
                temp: sc.temp,
                top_p: sc.top_p,
                softcap: eng.softcap,
                pad: 0.0,
            };
            be.go(e, C_ARGMAX, "k_sample_topk", &sa, (b_len, 1, 1), 64);
        }
        _ => {
            let nvch = eng.n_vocab.div_ceil(LM_VCHUNK);
            let la = LmHeadArgs {
                w_lo: w.off as u32,
                w_hi: (w.off >> 32) as u32,
                row_bytes: w.row_bytes,
                wtype: w.ty,
                n_vocab: eng.n_vocab as u32,
                k: w.k,
                xh_off: d.fh,
                part_off: d.am_part,
                b: b_len as u32,
                nvch: nvch as u32,
                vstart: 0,
                vcount: 0,
            };
            if b_len > 64 {
                be.go(e, C_LMHEAD, "k_lmhead_argmax_sg8", &la, (b_len.div_ceil(128), nvch, 1), 256);
            } else {
                be.go(e, C_LMHEAD, "k_lmhead_argmax_sg4", &la, (1, nvch, 1), 128);
            }
            let aa = ArgmaxBatchArgs { part_off: d.am_part, nvch: nvch as u32, out_base, pad: 0 };
            be.go(e, C_ARGMAX, "k_argmax_batch", &aa, (b_len, 1, 1), 128);
        }
    }
}

/// Read back the unique active expert IDs chosen by `k_topk_b` in `be.act[d.sel ..]`.
fn read_active_experts(be: &BatchEngine, b_len: usize, mut hist: Option<&mut [u64; NEXP]>) -> Vec<usize> {
    let mut seen = [false; NEXP];
    let p = unsafe { (be.act.contents().as_ptr() as *const f32).add(be.da.sel as usize) };
    for t in 0..b_len {
        for k in 0..TOPK {
            let e = unsafe { *p.add(t * 16 + k) as usize };
            if e < NEXP {
                seen[e] = true;
                if let Some(h) = hist.as_deref_mut() {
                    h[e] += 1;
                }
            }
        }
    }
    (0..NEXP).filter(|&e| seen[e]).collect()
}

/// Sum GPU execution time across a completed list of Metal command buffers.
fn sum_gpu_time(cbs: &[CmdBuf]) -> Duration {
    let mut secs = 0.0f64;
    for cb in cbs {
        let dt = cb.GPUEndTime() - cb.GPUStartTime();
        if dt.is_finite() && dt > 0.0 {
            secs += dt;
        }
    }
    Duration::from_secs_f64(secs)
}

// ----------------------------------------------------------------------------- 48-Step Layer Schedule Builder

#[derive(Clone, Debug)]
struct StepExec {
    slot: usize,
    ip: usize,
    cur: usize,
    blocks_before: Vec<usize>,
    msg: Option<(usize, usize)>,
    mid: usize,
    out: usize,
    load_vecs: Vec<usize>, // inter-layer live vectors read by this layer (for Mode C)
}

struct FullSchedule {
    execs: Vec<StepExec>,
    final_blocks: Vec<usize>,
    final_cur: usize,
}

/// Build the exact 48-step layer execution plan.
/// - `macro_mode = false`: uses `VecAlloc { next: 1, n: B_NVEC }` (exact mirror of `BatchEngine::encode_decode_step`).
/// - `macro_mode = true`: uses `VecAlloc { next: 1, n: B_NVEC_LIVE }` (`0..7`) for inter-layer states and
///   fixed `MACRO_MID_SLOT = 7` for intra-layer `mid`, so `macro_states` only stores `7` FP16 vectors/seq!
fn build_schedule(macro_mode: bool) -> FullSchedule {
    let mut va = VecAlloc { next: 1, n: if macro_mode { B_NVEC_LIVE } else { B_NVEC } };
    let mut blocks: Vec<usize> = Vec::new();
    let mut cur = 0usize;
    let mut execs = Vec::with_capacity(N_SLOT);

    let mut push_one = |slot: usize, ip: usize, cur: &mut usize, blocks: &mut Vec<usize>, msg: Option<(usize, usize)>, pinned: &[usize]| {
        let blocks_before = blocks.clone();
        let block_start = ip % BLOCK == 0;

        // Deduplicated live vectors read by this layer
        let mut load_vecs: Vec<usize> = Vec::with_capacity(B_NVEC_LIVE);
        for &v in &blocks_before {
            if !load_vecs.contains(&v) {
                load_vecs.push(v);
            }
        }
        if !load_vecs.contains(cur) {
            load_vecs.push(*cur);
        }
        if let Some((b0, b1)) = msg {
            if !load_vecs.contains(&b0) {
                load_vecs.push(b0);
            }
            if !load_vecs.contains(&b1) {
                load_vecs.push(b1);
            }
        }

        let mid = if macro_mode {
            MACRO_MID_SLOT
        } else {
            let mut live = blocks.clone();
            live.push(*cur);
            live.extend_from_slice(pinned);
            va.alloc(&live)
        };
        if block_start {
            blocks.push(*cur);
        }
        let mut live_out = blocks.clone();
        if macro_mode {
            // Avoid overwriting `cur` as well so inter-layer slots never alias within a layer
            live_out.push(*cur);
        } else {
            live_out.push(mid);
        }
        live_out.extend_from_slice(pinned);
        let out = va.alloc(&live_out);

        execs.push(StepExec { slot, ip, cur: *cur, blocks_before, msg, mid, out, load_vecs });
        *cur = out;
    };

    for ip in 0..LOOP_START {
        push_one(ip, ip, &mut cur, &mut blocks, None, &[]);
    }
    let prelude = cur;
    let prelude_blocks = blocks.clone();
    let mut bank = (0usize, 0usize);
    for pass in 0..N_PASS {
        if pass > 0 {
            cur = prelude;
            blocks = prelude_blocks.clone();
        }
        let msg = if pass > 0 { Some(bank) } else { None };
        let pinned: Vec<usize> = {
            let mut p = vec![prelude];
            if pass > 0 {
                p.push(bank.0);
                p.push(bank.1);
            }
            p
        };
        for ip in LOOP_START..LOOP_END {
            push_one(slot_of(pass, ip), ip, &mut cur, &mut blocks, msg, &pinned);
        }
        bank = (blocks[prelude_blocks.len() + 1], cur);
    }
    for ip in LOOP_END..N_PHYS {
        push_one(ip, ip, &mut cur, &mut blocks, None, &[]);
    }

    FullSchedule { execs, final_blocks: blocks, final_cur: cur }
}

fn next_offloaded_table(execs: &[StepExec], offload_mask: &[bool; N_PHYS]) -> Vec<Option<usize>> {
    let n = execs.len();
    let mut next = vec![None; n];
    let mut upcoming = None;
    for k in (0..n).rev() {
        next[k] = upcoming;
        if offload_mask[execs[k].ip] {
            upcoming = Some(execs[k].ip);
        }
    }
    next
}

// ----------------------------------------------------------------------------- Mode A: Per-Step Offloading (Async Overlap + Sparse Active-Expert)

#[derive(Clone, Copy, Debug, Default)]
pub struct OffloadStepStats {
    pub wall_time: Duration,
    pub gpu_time: Duration,
    pub ssd_io_time: Duration,
    pub ssd_wait_time: Duration,
    pub ssd_bytes: u64,
    pub unique_experts_read: u64,
    pub total_expert_slots: u64,
    pub overlap_pct: f64,
}

/// Core Mode A offloaded step runner: advances `seqs` through all 48 layers + `out_mix`, leaving `d.fh` ready.
/// Supports both:
/// - **Sparse active-expert mode** (`sparse_io = true`): runs router/topk, reads back `B * 8` expert IDs, and
///   `F_NOCACHE pread`s ONLY active experts into `stg[0]`.
/// - **Async double-buffered mode** (`sparse_io = false`): overlaps SSD prefetch of next offloaded layer into
///   `stg[cur ^ 1]` while GPU runs current layer's MoE + any subsequent resident layers + next layer's attn/router!
#[allow(clippy::too_many_arguments)]
fn run_offloaded_body(
    be: &mut BatchEngine,
    loader: &mut DiskExpertLoader,
    seqs: &mut [SeqSlot],
    offload_mask: &[bool; N_PHYS],
    sparse_io: bool,
    sample: Option<SampleCfg>,
    compute_next_tokens: bool,
    mut expert_hist: Option<&mut [[u64; NEXP]; N_PHYS]>,
) -> OffloadStepStats {
    let b_len = seqs.len();
    assert!(b_len >= 1 && b_len <= be.max_batch);
    let need_slice = !compute_next_tokens || sample.is_some_and(|s| !s.greedy());
    be.ensure_arena(b_len, need_slice, false);
    if let Some(sc) = sample {
        be.init_rng(sc.seed, b_len);
    }

    let (mut ns, mut nf) = (0usize, 0usize);
    for s in seqs.iter() {
        let (a, b) = s.pages_needed_for(s.suf);
        ns += a;
        nf += b;
    }
    let kv_bud = be.kv_budget();
    be.pool.ensure_free(&be.eng.gpu, ns, nf, kv_bud);
    for s in seqs.iter_mut() {
        s.ensure_pages_for(&mut be.pool, s.suf);
    }
    let tp = be.toks[0].contents().as_ptr() as *mut i32;
    for (b, s) in seqs.iter().enumerate() {
        unsafe { *tp.add(b) = s.next_tok as i32; }
    }
    be.upload_decode_meta(seqs, 0, 0);
    let max_pos = seqs.iter().map(|s| s.pos).max().unwrap();
    let min_suf = seqs.iter().map(|s| s.suf).min().unwrap();

    let sched = build_schedule(false);
    let next_off = next_offloaded_table(&sched.execs, offload_mask);
    let first_off = sched.execs.iter().find(|x| offload_mask[x.ip]).map(|x| x.ip);

    let eng = be.eng;
    let d = be.da;
    let cs_mb = be.max_batch;

    let mut stats = OffloadStepStats::default();
    let t_wall = Instant::now();
    let mut all_cbs: Vec<CmdBuf> = Vec::with_capacity(64);

    if !sparse_io {
        if let Some(first_ip) = first_off {
            loader.prefetch_layer(first_ip, 0);
        }
    }

    let mut cur_buf = 0usize;
    let mut pending_gpu: [Option<CmdBuf>; 2] = [None, None];

    // Open initial command buffer with embedding
    let mut cb = eng.gpu.cmd();
    let mut enc = Enc::new(&cb);
    be.bind_batch(&enc, 0, 0);
    let ea = EmbedArgs {
        emb_lo: eng.tok_embd.off as u32,
        emb_hi: (eng.tok_embd.off >> 32) as u32,
        row_bytes: eng.tok_embd.row_bytes,
        norm_off: eng.tok_norm,
        out0: d.vec(0),
        out1: d.vec(0),
        tok_index: 0,
        eps: eng.eps,
    };
    be.go(&enc, C_EMBED, "k_embed", &ea, (b_len, 1, 1), 256);

    for (k, ex) in sched.execs.iter().enumerate() {
        let l = &eng.layers[ex.ip];
        let mut blocks_mut = ex.blocks_before.clone();
        encode_layer_pre_moe(be, &enc, b_len, max_pos, min_suf, ex.slot, ex.ip, ex.cur, &mut blocks_mut, ex.msg, ex.mid, cs_mb);

        if !offload_mask[ex.ip] {
            // Resident layer: run MoE directly from eng.w inside the same encoder!
            encode_layer_post_moe(be, &enc, l, b_len, ex.slot, ex.mid, ex.out, cs_mb, ExpertWeightSrc::Resident);
            continue;
        }

        // Offloaded layer boundary: flush current encoder (Pre-MoE + any preceding resident layers)
        enc.end();
        cb.commit();

        if sparse_io {
            // Sparse active-expert path: wait for router/topk, read back active IDs, pread ONLY active experts!
            cb.waitUntilCompleted();
            all_cbs.push(cb);

            let active = read_active_experts(be, b_len, expert_hist.as_deref_mut().map(|h| &mut h[ex.ip]));
            stats.unique_experts_read += active.len() as u64;
            stats.total_expert_slots += NEXP as u64;

            let t_w = Instant::now();
            let (bytes, io_dur) = loader.load_sparse_experts(ex.ip, 0, &active);
            stats.ssd_wait_time += t_w.elapsed();
            stats.ssd_io_time += io_dur;
            stats.ssd_bytes += bytes;

            cb = eng.gpu.cmd();
            enc = Enc::new(&cb);
            be.bind_batch(&enc, 0, 0);
            let spec = loader.layer_spec(ex.ip);
            encode_layer_post_moe(
                be,
                &enc,
                l,
                b_len,
                ex.slot,
                ex.mid,
                ex.out,
                cs_mb,
                ExpertWeightSrc::Staged { buf: &loader.stg[0], dn_off: spec.dn_stg_off },
            );
        } else {
            // Async double-buffered full-layer overlap path!
            all_cbs.push(cb);

            let t_w = Instant::now();
            let (bytes, io_dur) = loader.wait_layer(cur_buf);
            stats.ssd_wait_time += t_w.elapsed();
            stats.ssd_io_time += io_dur;
            stats.ssd_bytes += bytes;
            stats.unique_experts_read += NEXP as u64;
            stats.total_expert_slots += NEXP as u64;

            // Submit layer k's Post-MoE immediately (reading stg[cur_buf]) so GPU stays 100% fed!
            let cb_moe = eng.gpu.cmd();
            let enc_moe = Enc::new(&cb_moe);
            be.bind_batch(&enc_moe, 0, 0);
            let spec = loader.layer_spec(ex.ip);
            encode_layer_post_moe(
                be,
                &enc_moe,
                l,
                b_len,
                ex.slot,
                ex.mid,
                ex.out,
                cs_mb,
                ExpertWeightSrc::Staged { buf: &loader.stg[cur_buf], dn_off: spec.dn_stg_off },
            );
            enc_moe.end();
            cb_moe.commit();

            // Wait for any previous GPU use of stg[next_buf] BEFORE starting SSD prefetch into stg[next_buf]!
            let next_buf = cur_buf ^ 1;
            if let Some(prev_cb) = pending_gpu[next_buf].take() {
                prev_cb.waitUntilCompleted();
            }
            if let Some(next_ip) = next_off[k] {
                loader.prefetch_layer(next_ip, next_buf);
            }
            pending_gpu[cur_buf] = Some(cb_moe.clone());
            all_cbs.push(cb_moe);
            cur_buf = next_buf;

            // Open fresh command buffer for subsequent resident layers + next offloaded layer's Pre-MoE
            cb = eng.gpu.cmd();
            enc = Enc::new(&cb);
            be.bind_batch(&enc, 0, 0);
        }
    }

    // Final output mix (+ lm_head/argmax when computing tokens)
    encode_out_mix(be, &enc, b_len, &sched.final_blocks, sched.final_cur);
    if compute_next_tokens {
        encode_lmhead_sample(be, &enc, b_len, &be.toks[1], 0, 0, sample);
    }
    enc.end();
    cb.commit();
    cb.waitUntilCompleted();
    all_cbs.push(cb);

    for p in pending_gpu.iter_mut() {
        if let Some(pcb) = p.take() {
            pcb.waitUntilCompleted();
        }
    }

    stats.wall_time = t_wall.elapsed();
    stats.gpu_time = sum_gpu_time(&all_cbs);
    let io_s = stats.ssd_io_time.as_secs_f64();
    let wait_s = stats.ssd_wait_time.as_secs_f64();
    stats.overlap_pct = if io_s > 1e-9 { ((io_s - wait_s) / io_s).clamp(0.0, 1.0) * 100.0 } else { 0.0 };
    stats
}

/// Advance `seqs` by ONE offloaded decode step (Mode A) and return `(next_tokens, OffloadStepStats)`.
pub fn step_offloaded_tokens(
    be: &mut BatchEngine,
    loader: &mut DiskExpertLoader,
    seqs: &mut [SeqSlot],
    offload_mask: &[bool; N_PHYS],
    sparse_io: bool,
    sample: Option<SampleCfg>,
) -> (Vec<u32>, OffloadStepStats) {
    let stats = run_offloaded_body(be, loader, seqs, offload_mask, sparse_io, sample, true, None);
    let out_p = be.toks[1].contents().as_ptr() as *const i32;
    let mut out = Vec::with_capacity(seqs.len());
    for (b, s) in seqs.iter_mut().enumerate() {
        let tok = unsafe { *out_p.add(b) as u32 };
        s.pos += 1;
        s.suf += 1;
        s.next_tok = tok;
        out.push(tok);
    }
    (out, stats)
}

/// Advance `seqs` by ONE offloaded decode step and return exact full-vocab f32 logits `[B][n_vocab]` (for exactness checks!).
pub fn step_offloaded_logits(
    be: &mut BatchEngine,
    loader: &mut DiskExpertLoader,
    seqs: &mut [SeqSlot],
    offload_mask: &[bool; N_PHYS],
    sparse_io: bool,
) -> (Vec<Vec<f32>>, OffloadStepStats) {
    let b_len = seqs.len();
    let stats = run_offloaded_body(be, loader, seqs, offload_mask, sparse_io, None, false, None);

    let eng = be.eng;
    let d = be.da;
    let mut out = vec![vec![0f32; eng.n_vocab]; b_len];
    let w = &eng.output;
    let mut vstart = 0usize;
    while vstart < eng.n_vocab {
        let vcount = (eng.n_vocab - vstart).min(LM_VSLICE);
        let cb2 = eng.gpu.cmd();
        let e2 = Enc::new(&cb2);
        be.bind_batch(&e2, 0, 0);
        let la = LmHeadArgs {
            w_lo: w.off as u32,
            w_hi: (w.off >> 32) as u32,
            row_bytes: w.row_bytes,
            wtype: w.ty,
            n_vocab: eng.n_vocab as u32,
            k: w.k,
            xh_off: d.fh,
            part_off: d.slice_buf,
            b: b_len as u32,
            nvch: 0,
            vstart: vstart as u32,
            vcount: vcount as u32,
        };
        let tiles = vcount.div_ceil(64);
        if b_len > 64 {
            be.go(&e2, C_LMHEAD, "k_lmhead_slice_sg8", &la, (b_len.div_ceil(128), tiles, 1), 256);
        } else {
            be.go(&e2, C_LMHEAD, "k_lmhead_slice_sg4", &la, (1, tiles, 1), 128);
        }
        e2.end();
        cb2.commit();
        cb2.waitUntilCompleted();
        let sp = unsafe { (be.act.contents().as_ptr() as *const f32).add(d.slice_buf as usize) };
        for b in 0..b_len {
            let sl = unsafe { std::slice::from_raw_parts(sp.add(b * vcount), vcount) };
            out[b][vstart..vstart + vcount].copy_from_slice(sl);
        }
        vstart += vcount;
    }
    for s in seqs.iter_mut() {
        s.pos += 1;
        s.suf += 1;
    }
    (out, stats)
}

// ----------------------------------------------------------------------------- Mode B: Zipf Hot/Cold Expert Cache

/// Fixed-RAM cache pinning the top `hot_per_layer` experts per offloaded layer, populated strictly via `F_NOCACHE pread`.
pub struct HotExpertCache {
    pub hot_per_layer: usize,
    pub offload_mask: [bool; N_PHYS],
    data: Vec<u8>,
    /// Per `(ip, expert)` byte offset `(gu_cache_off, dn_cache_off)` inside `data`, or `None` if cold.
    slots: [[Option<(usize, usize)>; NEXP]; N_PHYS],
    pub hot_fraction_profiled: f64,
}

impl HotExpertCache {
    /// Build a `HotExpertCache` from per-layer routing frequency histograms using `F_NOCACHE pread` (zero mmap touch!).
    /// Automatically clamps `hot_per_layer` if `max_cache_bytes` would be exceeded.
    pub fn build_from_histogram(
        loader: &DiskExpertLoader,
        offload_mask: &[bool; N_PHYS],
        hist: &[[u64; NEXP]; N_PHYS],
        requested_hot_per_layer: usize,
        max_cache_bytes: usize,
    ) -> Self {
        let mut per_expert_sum = 0usize;
        for (ip, &off) in offload_mask.iter().enumerate() {
            if off {
                let sp = loader.layer_spec(ip);
                per_expert_sum += (sp.gu_expert_bytes + sp.dn_expert_bytes) as usize;
            }
        }
        let max_hot = if per_expert_sum > 0 { max_cache_bytes / per_expert_sum } else { 0 };
        let hot_per_layer = requested_hot_per_layer.min(NEXP).min(max_hot);

        let total_bytes = hot_per_layer * per_expert_sum;
        let mut data = vec![0u8; total_bytes];
        let mut slots = [[None; NEXP]; N_PHYS];
        let mut cursor = 0usize;

        let mut total_picks = 0u64;
        let mut hot_picks = 0u64;

        for (ip, &off) in offload_mask.iter().enumerate() {
            if !off || hot_per_layer == 0 {
                continue;
            }
            let sp = loader.layer_spec(ip);
            let mut order: Vec<(usize, u64)> = (0..NEXP).map(|e| (e, hist[ip][e])).collect();
            order.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            total_picks += order.iter().map(|&(_, c)| c).sum::<u64>();
            for &(e, cnt) in order.iter().take(hot_per_layer) {
                hot_picks += cnt;
                let gu_off = cursor;
                cursor += sp.gu_expert_bytes as usize;
                let dn_off = cursor;
                cursor += sp.dn_expert_bytes as usize;
                let gu_dst = unsafe { data.as_mut_ptr().add(gu_off) };
                let dn_dst = unsafe { data.as_mut_ptr().add(dn_off) };
                loader.read_expert_into(ip, e, gu_dst, dn_dst);
                slots[ip][e] = Some((gu_off, dn_off));
            }
        }

        let hot_fraction_profiled = if total_picks > 0 { hot_picks as f64 / total_picks as f64 } else { 0.0 };
        HotExpertCache { hot_per_layer, offload_mask: *offload_mask, data, slots, hot_fraction_profiled }
    }

    #[inline]
    pub fn bytes(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_hot(&self, ip: usize, e: usize) -> bool {
        self.slots[ip][e].is_some()
    }

    /// Copy `hot_experts` of layer `ip` from RAM cache directly into their exact offsets in `stg[buf_idx]`.
    fn copy_hot_into_staging(&self, loader: &DiskExpertLoader, ip: usize, buf_idx: usize, hot_experts: &[usize]) {
        if hot_experts.is_empty() {
            return;
        }
        let sp = loader.layer_spec(ip);
        let stg_base = loader.stg[buf_idx].contents().as_ptr() as *mut u8;
        let gu_eb = sp.gu_expert_bytes as usize;
        let dn_eb = sp.dn_expert_bytes as usize;
        let dn_base = sp.dn_stg_off as usize;
        for &e in hot_experts {
            if let Some((gu_src, dn_src)) = self.slots[ip][e] {
                unsafe {
                    std::ptr::copy_nonoverlapping(self.data.as_ptr().add(gu_src), stg_base.add(e * gu_eb), gu_eb);
                    std::ptr::copy_nonoverlapping(
                        self.data.as_ptr().add(dn_src),
                        stg_base.add(dn_base + e * dn_eb),
                        dn_eb,
                    );
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct HotColdStats {
    pub step: OffloadStepStats,
    pub hot_hits: u64,
    pub cold_misses: u64,
    pub full_layer_bytes: u64,
    pub cache_bytes: usize,
}

/// Advance `seqs` by ONE Mode B (Zipf Hot/Cold) decode step!
/// Overlaps RAM `memcpy` of hot-cache hits WITH async `F_NOCACHE pread` of cold misses into disjoint offsets of `stg[0]`.
pub fn step_hot_cold_tokens(
    be: &mut BatchEngine,
    loader: &mut DiskExpertLoader,
    cache: &HotExpertCache,
    seqs: &mut [SeqSlot],
    sample: Option<SampleCfg>,
) -> (Vec<u32>, HotColdStats) {
    let b_len = seqs.len();
    assert!(b_len >= 1 && b_len <= be.max_batch);
    let need_slice = sample.is_some_and(|s| !s.greedy());
    be.ensure_arena(b_len, need_slice, false);
    if let Some(sc) = sample {
        be.init_rng(sc.seed, b_len);
    }

    let (mut ns, mut nf) = (0usize, 0usize);
    for s in seqs.iter() {
        let (a, b) = s.pages_needed_for(s.suf);
        ns += a;
        nf += b;
    }
    let kv_bud = be.kv_budget();
    be.pool.ensure_free(&be.eng.gpu, ns, nf, kv_bud);
    for s in seqs.iter_mut() {
        s.ensure_pages_for(&mut be.pool, s.suf);
    }
    let tp = be.toks[0].contents().as_ptr() as *mut i32;
    for (b, s) in seqs.iter().enumerate() {
        unsafe { *tp.add(b) = s.next_tok as i32; }
    }
    be.upload_decode_meta(seqs, 0, 0);
    let max_pos = seqs.iter().map(|s| s.pos).max().unwrap();
    let min_suf = seqs.iter().map(|s| s.suf).min().unwrap();

    let sched = build_schedule(false);
    let eng = be.eng;
    let d = be.da;
    let cs_mb = be.max_batch;

    let mut hc = HotColdStats { cache_bytes: cache.bytes(), ..Default::default() };
    let t_wall = Instant::now();
    let mut all_cbs: Vec<CmdBuf> = Vec::with_capacity(64);

    let mut cb = eng.gpu.cmd();
    let mut enc = Enc::new(&cb);
    be.bind_batch(&enc, 0, 0);
    let ea = EmbedArgs {
        emb_lo: eng.tok_embd.off as u32,
        emb_hi: (eng.tok_embd.off >> 32) as u32,
        row_bytes: eng.tok_embd.row_bytes,
        norm_off: eng.tok_norm,
        out0: d.vec(0),
        out1: d.vec(0),
        tok_index: 0,
        eps: eng.eps,
    };
    be.go(&enc, C_EMBED, "k_embed", &ea, (b_len, 1, 1), 256);

    for ex in &sched.execs {
        let l = &eng.layers[ex.ip];
        let mut blocks_mut = ex.blocks_before.clone();
        encode_layer_pre_moe(be, &enc, b_len, max_pos, min_suf, ex.slot, ex.ip, ex.cur, &mut blocks_mut, ex.msg, ex.mid, cs_mb);

        if !cache.offload_mask[ex.ip] {
            encode_layer_post_moe(be, &enc, l, b_len, ex.slot, ex.mid, ex.out, cs_mb, ExpertWeightSrc::Resident);
            continue;
        }

        enc.end();
        cb.commit();
        cb.waitUntilCompleted();
        all_cbs.push(cb);

        let sp = loader.layer_spec(ex.ip);
        hc.full_layer_bytes += sp.gu_total_bytes + sp.dn_total_bytes;

        let active = read_active_experts(be, b_len, None);
        let mut hot_hits = Vec::with_capacity(active.len());
        let mut cold_misses = Vec::with_capacity(active.len());
        for e in active {
            if cache.is_hot(ex.ip, e) {
                hot_hits.push(e);
            } else {
                cold_misses.push(e);
            }
        }
        hc.hot_hits += hot_hits.len() as u64;
        hc.cold_misses += cold_misses.len() as u64;
        hc.step.unique_experts_read += cold_misses.len() as u64;
        hc.step.total_expert_slots += NEXP as u64;

        // Overlap async SSD pread of cold misses with RAM memcpy of hot hits (disjoint expert offsets in stg[0]!)
        let t_w = Instant::now();
        if !cold_misses.is_empty() {
            loader.prefetch_sparse_experts(ex.ip, 0, &cold_misses);
        }
        cache.copy_hot_into_staging(loader, ex.ip, 0, &hot_hits);
        let (bytes, io_dur) = if !cold_misses.is_empty() { loader.wait_layer(0) } else { (0, Duration::ZERO) };
        hc.step.ssd_wait_time += t_w.elapsed();
        hc.step.ssd_io_time += io_dur;
        hc.step.ssd_bytes += bytes;

        cb = eng.gpu.cmd();
        enc = Enc::new(&cb);
        be.bind_batch(&enc, 0, 0);
        encode_layer_post_moe(
            be,
            &enc,
            l,
            b_len,
            ex.slot,
            ex.mid,
            ex.out,
            cs_mb,
            ExpertWeightSrc::Staged { buf: &loader.stg[0], dn_off: sp.dn_stg_off },
        );
    }

    encode_out_mix(be, &enc, b_len, &sched.final_blocks, sched.final_cur);
    encode_lmhead_sample(be, &enc, b_len, &be.toks[1], 0, 0, sample);
    enc.end();
    cb.commit();
    cb.waitUntilCompleted();
    all_cbs.push(cb);

    hc.step.wall_time = t_wall.elapsed();
    hc.step.gpu_time = sum_gpu_time(&all_cbs);
    let io_s = hc.step.ssd_io_time.as_secs_f64();
    let wait_s = hc.step.ssd_wait_time.as_secs_f64();
    hc.step.overlap_pct = if io_s > 1e-9 { ((io_s - wait_s) / io_s).clamp(0.0, 1.0) * 100.0 } else { 0.0 };

    let out_p = be.toks[1].contents().as_ptr() as *const i32;
    let mut out = Vec::with_capacity(b_len);
    for (b, s) in seqs.iter_mut().enumerate() {
        let tok = unsafe { *out_p.add(b) as u32 };
        s.pos += 1;
        s.suf += 1;
        s.next_tok = tok;
        out.push(tok);
    }
    (out, hc)
}

// ----------------------------------------------------------------------------- Mode C: Layer-Pipelined Macro-Batching

struct MacroPackKernels {
    pack_f32_to_f16: Pso,
    unpack_f16_to_f32: Pso,
}

impl MacroPackKernels {
    fn compile(gpu: &Gpu) -> Self {
        let opts = MTLCompileOptions::new();
        opts.setMathMode(MTLMathMode::Fast);
        opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
        let lib = gpu
            .dev
            .newLibraryWithSource_options_error(&NSString::from_str(MACRO_PACK_METAL_SRC), Some(&opts))
            .unwrap_or_else(|e| panic!("MacroPack Metal compile failed:\n{}", e.localizedDescription()));
        let make = |name: &str| {
            let f = lib
                .newFunctionWithName(&NSString::from_str(name))
                .unwrap_or_else(|| panic!("kernel {name} missing"));
            gpu.dev
                .newComputePipelineStateWithFunction_error(&f)
                .unwrap_or_else(|e| panic!("pso {name}: {}", e.localizedDescription()))
        };
        MacroPackKernels {
            pack_f32_to_f16: make("k_pack_f32_to_f16"),
            unpack_f16_to_f32: make("k_unpack_f16_to_f32"),
        }
    }

    #[inline]
    fn pack(&self, e: &Enc, act_f32_off: u32, macro_h_off: u64, count: usize) {
        let a = VecCopyArgs { src_off: act_f32_off as u64, dst_off: macro_h_off, count: count as u32, pad: 0 };
        e.pso(&self.pack_f32_to_f16);
        e.bytes(&a, 5);
        e.dispatch(((count / 4).div_ceil(256), 1, 1), 256);
    }

    #[inline]
    fn unpack(&self, e: &Enc, macro_h_off: u64, act_f32_off: u32, count: usize) {
        let a = VecCopyArgs { src_off: macro_h_off, dst_off: act_f32_off as u64, count: count as u32, pad: 0 };
        e.pso(&self.unpack_f16_to_f32);
        e.bytes(&a, 5);
        e.dispatch(((count / 4).div_ceil(256), 1, 1), 256);
    }
}

/// Build a compact synthetic shared `PrefixCache` (without touching offloaded mmap weights!) for realistic
/// M5 neural-accelerator FlashDecoding cascade attention in Mode C (`pre_len = 64` or `128`).
fn build_synthetic_prefix(eng: &Engine, pre_len: usize) -> PrefixCache {
    let pre_swa_n = pre_len.min(EXT);
    let pre_swa_p0 = pre_len - pre_swa_n;
    let pre_swa_cap = (pre_swa_n + 64).div_ceil(64) * 64;
    let pre_full_cap = (pre_len + 64).div_ceil(64) * 64;
    let pre_swa_bytes = N_SWA * 2 * NKV * pre_swa_cap * HD * eng.kvb;
    let pre_full_bytes = N_FULL * 2 * NKV * pre_full_cap * HD * eng.kvb;
    let pre_cs_bytes = CS_SEQ_ELEMS * 4;

    let pre_swa = eng.gpu.buffer(pre_swa_bytes);
    let pre_full = eng.gpu.buffer(pre_full_bytes);
    let pre_cs = eng.gpu.buffer(pre_cs_bytes);

    // Populate [0..pre_n) with finite, realistic small-magnitude values; keep guard rows [pre_n..cap) zeroed!
    let fill = |buf: &Buf, n_slots: usize, n_valid: usize, cap: usize| {
        if eng.kv8 {
            let p = buf.contents().as_ptr() as *mut i8;
            for s in 0..n_slots {
                for kv in 0..2 {
                    for g in 0..NKV {
                        let base = (((s * 2 + kv) * NKV + g) * cap) * HD;
                        for pos in 0..n_valid {
                            let row = base + pos * HD;
                            for d in 0..HD {
                                // Small deterministic int8 values with low-bit scale code (~16 -> small finite scale)
                                let v = if d < 8 {
                                    (((pos + g + d) & 1) as i8) * 2
                                } else {
                                    (((s * 7 + pos * 13 + d * 5) % 21) as i8) - 10
                                };
                                unsafe { *p.add(row + d) = v; }
                            }
                        }
                    }
                }
            }
        } else {
            // fp16: ±0.125 (0x3000 / 0xB000) and ±0.0625 (0x2C00 / 0xAC00)
            const VALS: [u16; 4] = [0x3000, 0xB000, 0x2C00, 0xAC00];
            let p = buf.contents().as_ptr() as *mut u16;
            for s in 0..n_slots {
                for kv in 0..2 {
                    for g in 0..NKV {
                        let base = (((s * 2 + kv) * NKV + g) * cap) * HD;
                        for pos in 0..n_valid {
                            let row = base + pos * HD;
                            for d in 0..HD {
                                unsafe { *p.add(row + d) = VALS[(s + pos + g + d) & 3]; }
                            }
                        }
                    }
                }
            }
        }
    };
    fill(&pre_swa, N_SWA, pre_swa_n, pre_swa_cap);
    fill(&pre_full, N_FULL, pre_len, pre_full_cap);

    PrefixCache {
        prefix_len: pre_len,
        pre_swa,
        pre_swa_n,
        pre_swa_p0,
        pre_swa_cap,
        pre_full,
        pre_full_cap,
        pre_cs,
        first_tok: 1000,
        first_logits: Vec::new(),
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MacroBatchStats {
    pub b_macro: usize,
    pub b_micro: usize,
    pub n_micro: usize,
    pub pool_seqs: usize,
    pub prefix_len: usize,
    pub exact_unique_kv: bool,
    pub wall_time: Duration,
    pub gpu_time: Duration,
    pub ssd_io_time: Duration,
    pub ssd_wait_time: Duration,
    pub ssd_bytes: u64,
    pub overlap_pct: f64,
    pub macro_state_bytes: usize,
    pub kv_bytes: usize,
    pub cs_bytes: usize,
    pub act_bytes: usize,
    pub staging_bytes: usize,
}

/// Run Mode C Layer-Pipelined Macro-Batching (`B_macro` sequences in `M = ceil(B_macro / B_micro)` micro-batches).
/// Streams each offloaded layer ONCE from SSD across all `B_macro` tokens while overlapping SSD I/O of layer `L+1`
/// with GPU compute of all `M` micro-batches on layer `L`!
///
/// Strictly OOM-safe: fits inside `working_budget_bytes` (excluding `loader.staging_bytes()`).
/// Uses 100% unique KV + CS slots per sequence whenever `B_macro` fits in budget, and cleanly falls back to
/// a `B_micro`-sized KV/CS pool + `pre_len = 64` shared prefix when `B_macro` (e.g. 2048–4096) exceeds tight `--max-mem`.
pub fn run_macro_batch_bench(
    eng: &Engine,
    loader: &mut DiskExpertLoader,
    policy: OffloadPolicy,
    b_macro: usize,
    b_micro: usize,
    steps: usize,
    working_budget_bytes: usize,
) -> Result<MacroBatchStats, String> {
    let b_macro = b_macro.max(1);
    let b_micro = b_micro.clamp(1, b_macro);
    let n_micro = b_macro.div_ceil(b_micro);
    let steps = steps.max(1);
    let prefix_len = 64usize;
    let max_ctx_macro = 256usize; // compact suffix window (64 prefix + up to 192 suffix)

    // 1. Compute compact FP16 macro-state size: [n_micro][B_NVEC_LIVE=7][b_micro][E] halfs
    let macro_state_elems = n_micro * B_NVEC_LIVE * b_micro * E;
    let macro_state_bytes = macro_state_elems * 2;

    // Prefix cache size (shared across all B_macro sequences!)
    let pre_cap = (prefix_len + 64).div_ceil(64) * 64;
    let prefix_bytes =
        (N_SWA + N_FULL) * 2 * NKV * pre_cap * HD * eng.kvb + CS_SEQ_ELEMS * 4;

    // Per-sequence KV (1 SWA page + 1 Full page = 32 suffix tokens) + CS slot bytes
    let kv_per_seq = PagedKvPool::swa_page_bytes(eng.kvb) + PagedKvPool::full_page_bytes(eng.kvb);
    let cs_per_seq = CS_SEQ_ELEMS * 4;
    let per_seq_bytes = kv_per_seq + cs_per_seq;

    // Reserve headroom for micro-batch activation arena (~85 MiB for B_micro=256) + macro_states + prefix + meta
    let fixed_overhead = macro_state_bytes + prefix_bytes + (160usize << 20);
    if working_budget_bytes <= fixed_overhead + b_micro * per_seq_bytes {
        return Err(format!(
            "B_macro={b_macro} B_micro={b_micro} needs > {:.2} GiB working budget (have {:.2} GiB)",
            (fixed_overhead + b_micro * per_seq_bytes) as f64 / (1u64 << 30) as f64,
            working_budget_bytes as f64 / (1u64 << 30) as f64,
        ));
    }

    let avail_pool = working_budget_bytes - fixed_overhead;
    let max_fit_seqs = avail_pool / per_seq_bytes;
    let exact_unique_kv = max_fit_seqs >= b_macro;
    let pool_seqs = if exact_unique_kv { b_macro } else { b_micro };

    // Allocate BatchEngine sized to `pool_seqs` for CS/KV, with `da` sized to `b_micro`!
    // Note: BatchEngine::try_new sizes `da` and `cs_pool` to `max_batch`. If `pool_seqs == b_macro`,
    // `da` for `b_macro` would waste activation RAM; instead when `exact_unique_kv`, we check if
    // `BatchEngine::try_new(..., pool_seqs, ...)` fits, else use `pool_seqs = b_micro` so `da` stays tiny (~85 MiB)!
    let init_swa = pool_seqs;
    let init_full = pool_seqs;
    let be_budget = working_budget_bytes.saturating_sub(macro_state_bytes + (32usize << 20));
    let (mut be, actual_pool_seqs, actual_exact_unique) =
        match BatchEngine::try_new(eng, pool_seqs, max_ctx_macro, init_swa, init_full, false, be_budget) {
            Ok(be) => (be, pool_seqs, exact_unique_kv),
            Err(_) => {
                // Fall back to B_micro-sized BatchEngine (compact pooled KV/CS + tiny B_micro arena!)
                let be = BatchEngine::try_new(eng, b_micro, max_ctx_macro, b_micro, b_micro, false, be_budget)?;
                (be, b_micro, false)
            }
        };

    // Attach shared prefix (never touches offloaded mmap!)
    be.prefix = Some(build_synthetic_prefix(eng, prefix_len));
    let cs_max_batch = be.max_batch;

    // Allocate compact FP16 inter-layer residual state buffer + macro token & per-microbatch metadata buffers
    let macro_states = eng.gpu.buffer(macro_state_bytes);
    let macro_toks = [eng.gpu.buffer(b_macro.max(16) * 4), eng.gpu.buffer(b_macro.max(16) * 4)];
    let be_meta_stride_bytes = (be.ml.total_u32 * 4).div_ceil(256) * 256;
    let macro_meta = eng.gpu.buffer(n_micro * be_meta_stride_bytes);

    // Initialize physical KV/CS slots (`actual_pool_seqs`)
    let mut phys_seqs = be.make_seq_slots(actual_pool_seqs, true);
    for s in phys_seqs.iter_mut() {
        s.ensure_pages_for(&mut be.pool, 0);
    }

    // Initialize per-sequence starting tokens (diverse across all B_macro sequences so routing diverges realistically!)
    let vocab_span = eng.n_vocab.saturating_sub(200).max(1);
    let tp0 = macro_toks[0].contents().as_ptr() as *mut i32;
    for i in 0..b_macro {
        unsafe { *tp0.add(i) = (100 + ((i * 7919 + 42) % vocab_span)) as i32; }
    }

    let pack = MacroPackKernels::compile(&eng.gpu);
    let sched = build_schedule(true);
    let offload_mask = policy.mask();
    let next_off = next_offloaded_table(&sched.execs, &offload_mask);
    let first_off = sched.execs.iter().find(|x| offload_mask[x.ip]).map(|x| x.ip);

    let d = be.da;
    let ml = be.ml;
    let mfb = be.max_full_blks;
    let macro_vec_off = |m: usize, v: usize| -> u64 { ((m * B_NVEC_LIVE + v) * b_micro * E) as u64 };

    // Run `1` warmup step + `steps` measured steps
    let total_iters = steps + 1;
    let mut agg_wall = Duration::ZERO;
    let mut agg_gpu = Duration::ZERO;
    let mut agg_io = Duration::ZERO;
    let mut agg_wait = Duration::ZERO;
    let mut agg_bytes = 0u64;

    let mut cur_tok_in = 0usize;
    for iter in 0..total_iters {
        let cur_tok_out = cur_tok_in ^ 1;
        let step_suf = iter % KV_BLK; // stays inside the 1 allocated suffix page per pooled slot
        let step_pos = prefix_len + step_suf;

        // Upload metadata for all M micro-batches ONCE before the layer loop!
        for m in 0..n_micro {
            let b_start = m * b_micro;
            let b_len = (b_macro - b_start).min(b_micro);
            let mp = unsafe { (macro_meta.contents().as_ptr() as *mut u8).add(m * be_meta_stride_bytes) as *mut u32 };
            for b in 0..b_len {
                let ps_idx = if actual_exact_unique { b_start + b } else { b };
                let s = &phys_seqs[ps_idx];
                unsafe {
                    *mp.add(ml.seq_pos as usize + b) = step_pos as u32;
                    *mp.add(ml.seq_suf as usize + b) = step_suf as u32;
                    *mp.add(ml.seq_cs as usize + b) = s.cs_slot;
                    let spt = mp.add(ml.swa_pt as usize + b * SWA_RING_BLKS);
                    for (k, &pid) in s.swa_pages.iter().enumerate() {
                        *spt.add(k) = pid;
                    }
                    let fpt = mp.add(ml.full_pt as usize + b * mfb);
                    for (k, &pid) in s.full_pages.iter().enumerate() {
                        *fpt.add(k) = pid;
                    }
                }
            }
        }

        let t_wall = Instant::now();
        let mut step_io = Duration::ZERO;
        let mut step_wait = Duration::ZERO;
        let mut step_bytes = 0u64;
        let mut all_cbs: Vec<CmdBuf> = Vec::with_capacity(N_SLOT + 2);

        // Kick off async prefetch of FIRST offloaded layer immediately while GPU runs embeddings!
        if let Some(first_ip) = first_off {
            loader.prefetch_layer(first_ip, 0);
        }

        // Stage 0: Embed all M micro-batches -> pack vec(0) to FP16 macro_states
        {
            let cb_emb = eng.gpu.cmd();
            let e = Enc::new(&cb_emb);
            be.bind_batch(&e, 0, 0);
            e.buf(&macro_toks[cur_tok_in], 0, 6);
            e.buf(&macro_states, 0, 9);
            for m in 0..n_micro {
                let b_start = m * b_micro;
                let b_len = (b_macro - b_start).min(b_micro);
                let ea = EmbedArgs {
                    emb_lo: eng.tok_embd.off as u32,
                    emb_hi: (eng.tok_embd.off >> 32) as u32,
                    row_bytes: eng.tok_embd.row_bytes,
                    norm_off: eng.tok_norm,
                    out0: d.vec(0),
                    out1: d.vec(0),
                    tok_index: b_start as u32,
                    eps: eng.eps,
                };
                be.go(&e, C_EMBED, "k_embed", &ea, (b_len, 1, 1), 256);
                pack.pack(&e, d.vec(0), macro_vec_off(m, 0), b_len * E);
                e.barrier();
            }
            e.end();
            cb_emb.commit();
            all_cbs.push(cb_emb);
        }

        // Stage 1: Walk ALL 48 layer executions layer-by-layer!
        let mut cur_buf = 0usize;
        let mut pending_gpu: [Option<CmdBuf>; 2] = [None, None];

        for (k, ex) in sched.execs.iter().enumerate() {
            let l = &eng.layers[ex.ip];
            let is_off = offload_mask[ex.ip];

            if is_off {
                let t_w = Instant::now();
                let (bytes, io_dur) = loader.wait_layer(cur_buf);
                step_wait += t_w.elapsed();
                step_io += io_dur;
                step_bytes += bytes;
            }

            let cb_l = eng.gpu.cmd();
            let e = Enc::new(&cb_l);
            be.bind_batch(&e, 0, 0);
            e.buf(&macro_states, 0, 9);

            let spec = loader.layer_spec(ex.ip);
            let wsrc = if is_off {
                ExpertWeightSrc::Staged { buf: &loader.stg[cur_buf], dn_off: spec.dn_stg_off }
            } else {
                ExpertWeightSrc::Resident
            };

            // Process ALL M micro-batches through layer k using ALREADY-STAGED weights (reading SSD ONCE!)
            for m in 0..n_micro {
                let b_start = m * b_micro;
                let b_len = (b_macro - b_start).min(b_micro);
                e.buf(&macro_meta, m * be_meta_stride_bytes, 7);

                // Unpack live FP16 inter-layer states -> FP32 DecodeArena
                for &v in &ex.load_vecs {
                    pack.unpack(&e, macro_vec_off(m, v), d.vec(v), b_len * E);
                }
                e.barrier();

                let mut blocks_mut = ex.blocks_before.clone();
                encode_layer_pre_moe(
                    &be,
                    &e,
                    b_len,
                    step_pos,
                    step_suf,
                    ex.slot,
                    ex.ip,
                    ex.cur,
                    &mut blocks_mut,
                    ex.msg,
                    ex.mid,
                    cs_max_batch,
                );
                encode_layer_post_moe(&be, &e, l, b_len, ex.slot, ex.mid, ex.out, cs_max_batch, wsrc);

                // Pack ONLY the newly produced layer output `ex.out` -> FP16 macro_states!
                pack.pack(&e, d.vec(ex.out), macro_vec_off(m, ex.out), b_len * E);
                e.barrier();
            }

            e.end();
            cb_l.commit();

            if is_off {
                // Immediately start prefetching NEXT offloaded layer into stg[cur_buf ^ 1] while GPU runs all M micro-batches!
                let next_buf = cur_buf ^ 1;
                if let Some(prev_cb) = pending_gpu[next_buf].take() {
                    prev_cb.waitUntilCompleted();
                }
                if let Some(next_ip) = next_off[k] {
                    loader.prefetch_layer(next_ip, next_buf);
                }
                pending_gpu[cur_buf] = Some(cb_l.clone());
                cur_buf = next_buf;
            }
            all_cbs.push(cb_l);
        }

        // Stage 2: Final output mix + lm_head + greedy argmax for all M micro-batches
        let mut final_vecs = sched.final_blocks.clone();
        if !final_vecs.contains(&sched.final_cur) {
            final_vecs.push(sched.final_cur);
        }
        {
            let cb_tail = eng.gpu.cmd();
            let e = Enc::new(&cb_tail);
            be.bind_batch(&e, 0, 0);
            e.buf(&macro_states, 0, 9);
            for m in 0..n_micro {
                let b_start = m * b_micro;
                let b_len = (b_macro - b_start).min(b_micro);
                for &v in &final_vecs {
                    pack.unpack(&e, macro_vec_off(m, v), d.vec(v), b_len * E);
                }
                e.barrier();
                encode_out_mix(&be, &e, b_len, &sched.final_blocks, sched.final_cur);
                encode_lmhead_sample(&be, &e, b_len, &macro_toks[cur_tok_out], b_start as u32, 0, None);
            }
            e.end();
            cb_tail.commit();
            cb_tail.waitUntilCompleted();
            all_cbs.push(cb_tail);
        }

        for p in pending_gpu.iter_mut() {
            if let Some(pcb) = p.take() {
                pcb.waitUntilCompleted();
            }
        }

        let step_wall = t_wall.elapsed();
        let step_gpu = sum_gpu_time(&all_cbs);

        // Skip iter 0 (warmup) from averages
        if iter > 0 {
            agg_wall += step_wall;
            agg_gpu += step_gpu;
            agg_io += step_io;
            agg_wait += step_wait;
            agg_bytes += step_bytes;
        }
        cur_tok_in = cur_tok_out;
    }

    for mut s in phys_seqs {
        s.free_pages(&mut be.pool);
        be.free_cs(s.cs_slot);
    }

    let div = steps as u32;
    let avg_wall = agg_wall / div;
    let avg_gpu = agg_gpu / div;
    let avg_io = agg_io / div;
    let avg_wait = agg_wait / div;
    let avg_bytes = agg_bytes / steps as u64;
    let io_s = avg_io.as_secs_f64();
    let wait_s = avg_wait.as_secs_f64();
    let overlap_pct = if io_s > 1e-9 { ((io_s - wait_s) / io_s).clamp(0.0, 1.0) * 100.0 } else { 0.0 };

    Ok(MacroBatchStats {
        b_macro,
        b_micro,
        n_micro,
        pool_seqs: actual_pool_seqs,
        prefix_len,
        exact_unique_kv: actual_exact_unique,
        wall_time: avg_wall,
        gpu_time: avg_gpu,
        ssd_io_time: avg_io,
        ssd_wait_time: avg_wait,
        ssd_bytes: avg_bytes,
        overlap_pct,
        macro_state_bytes,
        kv_bytes: be.kv_bytes(),
        cs_bytes: be.cs_bytes(),
        act_bytes: be.act_bytes(),
        staging_bytes: loader.staging_bytes(),
    })
}

// ----------------------------------------------------------------------------- Correctness Verification (Bit-Exact Parity vs Resident BatchEngine)

/// Verify bit-exact parity (`max_logit_diff` & argmax agreement at `B = 8`) for BOTH:
/// 1. Sparse active-expert offloading (`all24`, `sparse_io = true`)
/// 2. Double-buffered async full-layer offloading (`prelude_coda`, `sparse_io = false`)
/// against the resident `BatchEngine::step_logits`.
///
/// NOTE: Run this AFTER zero-page-cache RAM proofs because the resident reference touches all 24 layers via mmap!
pub fn verify_offload_exactness(eng: &Engine, loader: &mut DiskExpertLoader, working_budget: usize) -> bool {
    let b_test = 8usize;
    let n_steps = 2usize;
    let vocab_span = eng.n_vocab.saturating_sub(200).max(1);
    let init_pages = b_test * 2;

    println!("=== Correctness Verification: Offloaded vs Resident BatchEngine (B={b_test}, {n_steps} steps) ===");

    let run_case = |label: &str, policy: OffloadPolicy, sparse_io: bool, loader: &mut DiskExpertLoader| -> bool {
        // Run offloaded engine first
        let mut be_off = match BatchEngine::try_new(eng, b_test, 256, init_pages, init_pages, true, working_budget) {
            Ok(b) => b,
            Err(e) => {
                println!("  [{label}] SKIPPED: {e}");
                return false;
            }
        };
        let mut seqs_off = be_off.make_seq_slots(b_test, false);
        let mask = policy.mask();
        let mut off_logits_all = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            for (b, s) in seqs_off.iter_mut().enumerate() {
                s.next_tok = (100 + ((b * 997 + step * 31 + 17) % vocab_span)) as u32;
            }
            let (lgs, _) = step_offloaded_logits(&mut be_off, loader, &mut seqs_off, &mask, sparse_io);
            off_logits_all.push(lgs);
        }
        drop(be_off);

        // Run resident reference on identical inputs
        let mut be_ref = match BatchEngine::try_new(eng, b_test, 256, init_pages, init_pages, true, working_budget) {
            Ok(b) => b,
            Err(e) => {
                println!("  [{label}] SKIPPED ref: {e}");
                return false;
            }
        };
        let mut seqs_ref = be_ref.make_seq_slots(b_test, false);
        let mut max_diff = 0.0f32;
        let mut argmax_ok = 0usize;
        let mut argmax_tot = 0usize;
        let mut exact_bits = true;
        let am = |v: &[f32]| {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, &x)| if x > best.1 { (i, x) } else { best })
                .0
        };

        for step in 0..n_steps {
            for (b, s) in seqs_ref.iter_mut().enumerate() {
                s.next_tok = (100 + ((b * 997 + step * 31 + 17) % vocab_span)) as u32;
            }
            let ref_lgs = be_ref.step_logits(&mut seqs_ref);
            for b in 0..b_test {
                let a = &off_logits_all[step][b];
                let r = &ref_lgs[b];
                for (&x, &y) in a.iter().zip(r.iter()) {
                    if x.to_bits() != y.to_bits() {
                        exact_bits = false;
                    }
                    max_diff = max_diff.max((x - y).abs());
                }
                if am(a) == am(r) {
                    argmax_ok += 1;
                }
                argmax_tot += 1;
            }
        }
        drop(be_ref);

        let pass = max_diff == 0.0 && argmax_ok == argmax_tot;
        println!(
            "  [{label:<30}] max_logit_diff = {max_diff:.6e} | bit_exact = {exact_bits} | argmax = {argmax_ok}/{argmax_tot} -> {}",
            if pass { "PASS" } else { "FAIL" }
        );
        pass
    };

    let ok1 = run_case("Mode A sparse (all24)", OffloadPolicy::All24, true, loader);
    let ok2 = run_case("Mode A async (prelude_coda)", OffloadPolicy::PreludeCoda, false, loader);
    println!();
    ok1 && ok2
}

// ----------------------------------------------------------------------------- Full Benchmark Suite Entry Point

fn format_ram_tag(r: RamStats) -> String {
    format!(
        "phys {:5.0} MiB | mmap {:5.0} MiB (pin {:5.0}, off {:4.0})",
        r.phys_footprint_mib, r.mmap_resident_total_mib, r.mmap_resident_pinned_mib, r.mmap_resident_offloaded_mib
    )
}

/// Top-level `F_NOCACHE` disk expert offloading benchmark suite:
/// 1. Raw `F_NOCACHE` multi-threaded `pread` calibration (`1 / 2 / 4 / 6 / 8` threads) -> auto-picks fastest thread count.
/// 2. Evicts offloaded mmap ranges (`MADV_DONTNEED`) and reports baseline `proc_pid_rusage` + `mincore` RAM.
/// 3. Mode A Per-Step sweep (`B = 1, 8, 16, 32, 64, 128, 256`; sparse for `B <= 16`, async overlap for `B >= 32`).
/// 4. Mode B Zipf Hot/Cold Expert Cache sweep (`hot_per_layer = 32` & `64`; `B = 1, 8, 32, 64`).
/// 5. Mode C Layer-Pipelined Macro-Batching sweep (`B_macro = 512, 1024, 2048, 4096`, `B_micro = 256`).
/// 6. Bit-exactness verification (`B = 8` offloaded vs resident `BatchEngine`, run last so resident never pollutes `mincore`!).
pub fn run_bench_offload(eng: &Engine, policy: &str, max_mem_gb: Option<f64>) {
    assert!(eng.hgemm, "disk expert offloader requires the M5 fp16 tensor path (hgemm)");

    let policies: Vec<OffloadPolicy> = match policy.trim().to_ascii_lowercase().as_str() {
        "" | "all" | "sweep" | "auto" => vec![OffloadPolicy::PreludeCoda, OffloadPolicy::All24],
        other => vec![OffloadPolicy::parse(other).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            std::process::exit(2);
        })],
    };

    let safe_total_cap = BatchEngine::safe_mem_limit(eng, max_mem_gb);
    println!("================================================================================");
    println!(
        "inkling F_NOCACHE Disk Expert Offloader Benchmark | kv={} | mem_cap={:.2} GiB | policies={:?}",
        if eng.kv8 { "kv8" } else { "fp16" },
        safe_total_cap as f64 / (1u64 << 30) as f64,
        policies.iter().map(|p| p.name()).collect::<Vec<_>>(),
    );
    println!("================================================================================\n");

    // 1. Raw SSD calibration -> auto-pick fastest F_NOCACHE thread count
    let calib = calibrate_ssd(eng);

    // 2. Construct ONE reusable DiskExpertLoader (~1.42 GiB fixed double-buffered staging)
    let mut loader = DiskExpertLoader::new(eng, calib.best_threads).expect("create DiskExpertLoader");
    let staging_bytes = loader.staging_bytes();
    let working_budget = safe_total_cap.saturating_sub(staging_bytes).max(512usize << 20);
    println!(
        "DiskExpertLoader ready: staging = {:.2} GiB (2 x {:.0} MiB fixed) | working budget = {:.2} GiB | threads = {}",
        staging_bytes as f64 / (1u64 << 30) as f64,
        (staging_bytes / 2) as f64 / 1048576.0,
        working_budget as f64 / (1u64 << 30) as f64,
        loader.n_threads,
    );

    // Evict offloaded tensors for the widest policy first and print initial RAM proof
    let widest_mask = if policies.contains(&OffloadPolicy::All24) {
        OffloadPolicy::All24.mask()
    } else {
        policies[0].mask()
    };
    let evicted = evict_offloaded_mmap(eng, &widest_mask);
    let ram0 = measure_ram(eng, &widest_mask);
    println!(
        "Initial MADV_DONTNEED advised {:.2} GiB | {}",
        evicted as f64 / (1u64 << 30) as f64,
        format_ram_tag(ram0)
    );
    if ram0.mmap_resident_offloaded_mib > 64.0 {
        println!(
            "  note: {:.0} MiB of offloaded tensors were already warm in OS page cache (pass INK_NO_WARM=1 for 0 MiB cold-mmap proof)",
            ram0.mmap_resident_offloaded_mib
        );
    }
    println!();

    let vocab_span = eng.n_vocab.saturating_sub(200).max(1);

    // Store Mode A full-layer baselines per (policy, B) for Mode B speedup comparison
    let mut mode_a_full_ms: Vec<(OffloadPolicy, usize, f64)> = Vec::new();

    // ------------------------------------------------------------------------- 3. Mode A: Per-Step Sweep
    for &pol in &policies {
        let mask = pol.mask();
        evict_offloaded_mmap(eng, &mask);
        let ram_pol = measure_ram(eng, &mask);
        println!(
            "=== Mode A: Per-Step Offloading [policy={}, {}/24 layers = {:.2} GiB offloaded] ===",
            pol.name(),
            pol.count(),
            ram_pol.offloaded_logical_mib / 1024.0
        );
        let batch_sizes = [1usize, 8, 16, 32, 64, 128, 256];
        for &b in &batch_sizes {
            let sparse = b <= 16;
            let n_timed = if pol == OffloadPolicy::All24 && b >= 32 { 2usize } else { 3usize };
            let init_pages = b.max(1);
            let mut be = match BatchEngine::try_new(eng, b, 256, init_pages, init_pages, false, working_budget) {
                Ok(be) => be,
                Err(err) => {
                    println!("  B={b:3} | SKIPPED ({err})");
                    continue;
                }
            };
            let mut seqs = be.make_seq_slots(b, false);
            for (i, s) in seqs.iter_mut().enumerate() {
                s.next_tok = (100 + ((i * 7919 + 11) % vocab_span)) as u32;
            }
            // 1 warmup step + n_timed measured steps
            let _ = step_offloaded_tokens(&mut be, &mut loader, &mut seqs, &mask, sparse, None);
            let mut sum_wall = Duration::ZERO;
            let mut sum_gpu = Duration::ZERO;
            let mut sum_io = Duration::ZERO;
            let mut sum_wait = Duration::ZERO;
            let mut sum_bytes = 0u64;
            let mut sum_uniq = 0u64;
            let mut sum_tot = 0u64;
            for _ in 0..n_timed {
                let (_, st) = step_offloaded_tokens(&mut be, &mut loader, &mut seqs, &mask, sparse, None);
                sum_wall += st.wall_time;
                sum_gpu += st.gpu_time;
                sum_io += st.ssd_io_time;
                sum_wait += st.ssd_wait_time;
                sum_bytes += st.ssd_bytes;
                sum_uniq += st.unique_experts_read;
                sum_tot += st.total_expert_slots;
            }
            let ram = measure_ram(eng, &mask);
            let wall_ms = sum_wall.as_secs_f64() * 1e3 / n_timed as f64;
            let gpu_ms = sum_gpu.as_secs_f64() * 1e3 / n_timed as f64;
            let io_ms = sum_io.as_secs_f64() * 1e3 / n_timed as f64;
            let wait_ms = sum_wait.as_secs_f64() * 1e3 / n_timed as f64;
            let avg_bytes = sum_bytes as f64 / n_timed as f64;
            let ssd_gibs = if io_ms > 1e-6 { avg_bytes / (1u64 << 30) as f64 / (io_ms * 1e-3) } else { 0.0 };
            let overlap = if io_ms > 1e-6 { ((io_ms - wait_ms) / io_ms).clamp(0.0, 1.0) * 100.0 } else { 0.0 };
            let agg_tps = b as f64 / (wall_ms * 1e-3);
            let per_tps = 1e3 / wall_ms;
            let avg_exp = if sum_tot > 0 { (sum_uniq as f64 / sum_tot as f64) * NEXP as f64 } else { 0.0 };

            println!(
                "  B={b:3} [{:>6}] | wall {:7.1} ms ({:6.1} tok/s, {:5.2}/seq) | GPU {:6.1} ms | SSD {:6.0} MiB ({:4.2} GiB/s, wait {:6.1} ms, ovlp {:3.0}%, exp {:5.1}/128) | {}",
                if sparse { "sparse" } else { "async" },
                wall_ms,
                agg_tps,
                per_tps,
                gpu_ms,
                avg_bytes / 1048576.0,
                ssd_gibs,
                wait_ms,
                overlap,
                avg_exp,
                format_ram_tag(ram),
            );
            mode_a_full_ms.push((pol, b, wall_ms));
            for mut s in seqs {
                s.free_pages(&mut be.pool);
                be.free_cs(s.cs_slot);
            }
            drop(be);
        }
        println!();
    }

    // ------------------------------------------------------------------------- 4. Mode B: Zipf Hot/Cold Expert Cache
    {
        let hc_pol = policies[0];
        let mask = hc_pol.mask();
        evict_offloaded_mmap(eng, &mask);
        println!("=== Mode B: Zipf Hot/Cold Expert Cache [policy={}] ===", hc_pol.name());

        // Profile real expert selection frequencies using offloaded sparse decode (zero mmap touch!)
        let mut hist = [[0u64; NEXP]; N_PHYS];
        let prof_steps = if hc_pol == OffloadPolicy::All24 { 2usize } else { 3usize };
        {
            let b_prof = 32usize;
            if let Ok(mut be_p) = BatchEngine::try_new(eng, b_prof, 128, b_prof, b_prof, false, working_budget) {
                let mut seqs_p = be_p.make_seq_slots(b_prof, false);
                for (i, s) in seqs_p.iter_mut().enumerate() {
                    s.next_tok = (100 + ((i * 7919 + 5) % vocab_span)) as u32;
                }
                for _ in 0..prof_steps {
                    let _ = run_offloaded_body(&mut be_p, &mut loader, &mut seqs_p, &mask, true, None, true, Some(&mut hist));
                    let out_p = be_p.toks[1].contents().as_ptr() as *const i32;
                    for (b, s) in seqs_p.iter_mut().enumerate() {
                        let tok = unsafe { *out_p.add(b) as u32 };
                        s.pos += 1;
                        s.suf += 1;
                        s.next_tok = tok;
                    }
                }
                for mut s in seqs_p {
                    s.free_pages(&mut be_p.pool);
                    be_p.free_cs(s.cs_slot);
                }
            }
        }

        // Measure full-layer baseline for B = 1, 8, 32, 64 on hc_pol if not already full-layer
        let full_layer_ref_ms = |b: usize, loader: &mut DiskExpertLoader| -> f64 {
            if b >= 32 {
                if let Some(&(_, _, ms)) = mode_a_full_ms.iter().find(|&&(p, bb, _)| p == hc_pol && bb == b) {
                    return ms;
                }
            }
            if let Ok(mut be_r) = BatchEngine::try_new(eng, b, 128, b, b, false, working_budget) {
                let mut seqs_r = be_r.make_seq_slots(b, false);
                for (i, s) in seqs_r.iter_mut().enumerate() {
                    s.next_tok = (100 + ((i * 7919 + 11) % vocab_span)) as u32;
                }
                let (_, st) = step_offloaded_tokens(&mut be_r, loader, &mut seqs_r, &mask, false, None);
                for mut s in seqs_r {
                    s.free_pages(&mut be_r.pool);
                    be_r.free_cs(s.cs_slot);
                }
                st.wall_time.as_secs_f64() * 1e3
            } else {
                0.0
            }
        };

        let hc_batches = [1usize, 8, 32, 64];
        let mut full_refs = [0.0f64; 4];
        for (idx, &b) in hc_batches.iter().enumerate() {
            full_refs[idx] = full_layer_ref_ms(b, &mut loader);
        }

        for &hot_k in &[32usize, 64usize] {
            // Keep at least 1 GiB of working budget free for BatchEngine(B=64)
            let cache_budget = working_budget.saturating_sub(1usize << 30);
            let cache = HotExpertCache::build_from_histogram(&loader, &mask, &hist, hot_k, cache_budget);
            if cache.hot_per_layer == 0 {
                println!("  hot={hot_k}/128 | SKIPPED (exceeds memory cap)");
                continue;
            }
            let rem_budget = working_budget.saturating_sub(cache.bytes()).max(512usize << 20);
            println!(
                "  -- HotExpertCache: {}/128 experts/layer pinned ({:.2} GiB RAM, profiled Zipf hit share {:.1}%) --",
                cache.hot_per_layer,
                cache.bytes() as f64 / (1u64 << 30) as f64,
                cache.hot_fraction_profiled * 100.0
            );
            for (idx, &b) in hc_batches.iter().enumerate() {
                let mut be = match BatchEngine::try_new(eng, b, 128, b, b, false, rem_budget) {
                    Ok(be) => be,
                    Err(e) => {
                        println!("    B={b:2} | SKIPPED ({e})");
                        continue;
                    }
                };
                let mut seqs = be.make_seq_slots(b, false);
                for (i, s) in seqs.iter_mut().enumerate() {
                    s.next_tok = (100 + ((i * 7919 + 23) % vocab_span)) as u32;
                }
                // Warmup + 2 timed steps
                let _ = step_hot_cold_tokens(&mut be, &mut loader, &cache, &mut seqs, None);
                let n_timed = 2usize;
                let mut sum_wall = Duration::ZERO;
                let mut sum_hits = 0u64;
                let mut sum_miss = 0u64;
                let mut sum_ssd = 0u64;
                let mut sum_full = 0u64;
                for _ in 0..n_timed {
                    let (_, hc) = step_hot_cold_tokens(&mut be, &mut loader, &cache, &mut seqs, None);
                    sum_wall += hc.step.wall_time;
                    sum_hits += hc.hot_hits;
                    sum_miss += hc.cold_misses;
                    sum_ssd += hc.step.ssd_bytes;
                    sum_full += hc.full_layer_bytes;
                }
                let ram = measure_ram(eng, &mask);
                let wall_ms = sum_wall.as_secs_f64() * 1e3 / n_timed as f64;
                let tot_lookups = (sum_hits + sum_miss).max(1) as f64;
                let miss_pct = sum_miss as f64 / tot_lookups * 100.0;
                let saved_pct = if sum_full > 0 { (1.0 - sum_ssd as f64 / sum_full as f64) * 100.0 } else { 0.0 };
                let speedup = if full_refs[idx] > 0.0 { full_refs[idx] / wall_ms } else { 1.0 };
                println!(
                    "    B={b:2} | wall {:7.1} ms ({:6.1} tok/s) | vs full-stream {:4.2}x | miss {:5.1}% | SSD {:5.0} MiB (saved {:4.1}%) | {}",
                    wall_ms,
                    b as f64 / (wall_ms * 1e-3),
                    speedup,
                    miss_pct,
                    (sum_ssd as f64 / n_timed as f64) / 1048576.0,
                    saved_pct,
                    format_ram_tag(ram),
                );
                for mut s in seqs {
                    s.free_pages(&mut be.pool);
                    be.free_cs(s.cs_slot);
                }
                drop(be);
            }
            drop(cache);
        }
        println!();
    }

    // ------------------------------------------------------------------------- 5. Mode C: Layer-Pipelined Macro-Batching
    for &pol in &policies {
        let mask = pol.mask();
        evict_offloaded_mmap(eng, &mask);
        println!(
            "=== Mode C: Layer-Pipelined Macro-Batching [policy={}, {}/24 layers offloaded] ===",
            pol.name(),
            pol.count(),
        );
        // Pick largest b_micro in [256, 128, 64] that comfortably fits inside working_budget
        let per_seq_est = PagedKvPool::swa_page_bytes(eng.kvb) + PagedKvPool::full_page_bytes(eng.kvb) + CS_SEQ_ELEMS * 4;
        let b_micro = if working_budget >= 256 * per_seq_est + (400usize << 20) {
            256usize
        } else if working_budget >= 128 * per_seq_est + (256usize << 20) {
            128usize
        } else {
            64usize
        };
        let macro_sizes: &[usize] =
            if pol == OffloadPolicy::All24 { &[512, 1024, 2048] } else { &[512, 1024, 2048, 4096] };
        let steps = if pol == OffloadPolicy::All24 { 1usize } else { 2usize };
        for &b_macro in macro_sizes {
            match run_macro_batch_bench(eng, &mut loader, pol, b_macro, b_micro, steps, working_budget) {
                Ok(ms) => {
                    let ram = measure_ram(eng, &mask);
                    let wall_s = ms.wall_time.as_secs_f64();
                    let agg_tps = ms.b_macro as f64 / wall_s;
                    let per_tps = 1.0 / wall_s;
                    let io_s = ms.ssd_io_time.as_secs_f64();
                    let ssd_gibs = if io_s > 1e-9 { ms.ssd_bytes as f64 / (1u64 << 30) as f64 / io_s } else { 0.0 };
                    println!(
                        "  B_macro={:4} (M={:2}x{:3}, {:>6}) | wall {:7.1} ms | {:7.1} tok/s ({:4.2}/seq) | GPU {:7.1} ms | SSD {:5.2} GiB ({:4.2} GiB/s, raw {:6.1} ms, wait {:6.1} ms, ovlp {:3.0}%) | state {:4.0} MiB, KV {:4.0} + CS {:4.0} + Act {:3.0} MiB | {}",
                        ms.b_macro,
                        ms.n_micro,
                        ms.b_micro,
                        if ms.exact_unique_kv { "unique" } else { "pooled" },
                        wall_s * 1e3,
                        agg_tps,
                        per_tps,
                        ms.gpu_time.as_secs_f64() * 1e3,
                        ms.ssd_bytes as f64 / (1u64 << 30) as f64,
                        ssd_gibs,
                        io_s * 1e3,
                        ms.ssd_wait_time.as_secs_f64() * 1e3,
                        ms.overlap_pct,
                        ms.macro_state_bytes as f64 / 1048576.0,
                        ms.kv_bytes as f64 / 1048576.0,
                        ms.cs_bytes as f64 / 1048576.0,
                        ms.act_bytes as f64 / 1048576.0,
                        format_ram_tag(ram),
                    );
                }
                Err(err) => {
                    println!("  B_macro={b_macro:4} (B_micro={b_micro}) | SKIPPED ({err})");
                }
            }
        }
        println!();
    }

    // ------------------------------------------------------------------------- 6. Bit-Exactness Verification (run LAST so resident never pollutes mincore proofs!)
    let exact_ok = verify_offload_exactness(eng, &mut loader, working_budget);
    println!(
        "================================================================================\n\
         Offload benchmark complete | exactness_check = {}\n\
         ================================================================================",
        if exact_ok { "PASS (bit-exact)" } else { "CHECK FAILED" }
    );
}
