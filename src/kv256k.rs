//! 256K+ Streaming Compressed/Tiered Sparse-Retrieval KV Engine & High-Throughput Batched MoE + FP16 CS Optimizer.
//!
//! Architecture & strict OOM-safety guarantees:
//! - Every GPU allocation passes through `MemGuard` against a hard `max_mem_bytes` cap (`BatchEngine::safe_mem_limit`).
//! - Every NVMe cold-store & wave-swap file sets `F_NOCACHE` (`fcntl(fd, F_NOCACHE, 1)`) and unlinks immediately,
//!   so macOS unified buffer cache can NEVER balloon RAM and disk space is always reclaimed on exit.
//! - 64-bit safe page addressing throughout (supports 256K, 512K, 1M+ contexts without 32-bit overflow).
//! - WHT-128 + Rademacher orthogonal KV4 (4-bit) & KV2 (2-bit Lloyd-Max) compression with Quest page summaries,
//!   DuoAttention per-head retrieval masks, ref-counted Radix prefix page table, Hot RAM LRU + F_NOCACHE cold spill,
//!   and an async background-prefetching Wave-Swap manager for scaling to hundreds of 256K instances.

use crate::batch::{BatchEngine, PagedKvPool, KV_BLK, SWA_RING_BLKS};
use crate::engine::*;
use crate::metal::*;
use crate::tokenizer::Tokenizer;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLDevice};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

pub const NTC_OPT: usize = 6;
pub const N_SWA: usize = 40;
pub const N_FULL: usize = 8;
pub const CS_SEQ_ELEMS: usize = N_SLOT * NCH * 3; // 48 * 6656 * 3 = 958,464 elements (~1.83 MiB in fp16!)
pub const KV_PAGE_SUM_BYTES: usize = 528;  // sizeof(KvPageSummary) in batch_opt.metal (legacy Quest)
pub const KV_PAGE_WHT2_BYTES: usize = 416; // sizeof(KvPageWht2) in batch_opt.metal (8x16-tok WHT-2bit sketches + WHT-4bit centroids)
pub const WHT2_NSUB: usize = 8;
pub const SWA_INNER_BLKS: usize = 9;       // 9 x 32 = 257..288 newest SWA tokens (always covers d in 0..=256!)
pub const SWA_OUTER_BLKS: usize = 24;      // 24 x 32 = 736..768 older SWA tokens in WHT-K2V2 (9 + 24 = 33 = SWA_RING_BLKS!)
const LM_VCHUNK: usize = 16 * 64;          // 1024 vocab rows per TG in k_lmhead_argmax_sg
const LM_VSLICE_256K: usize = 16384;       // vocab slice size for E2E full-logit KL evaluation (B=1 only: 64 KiB)
const ALIGN_4K: usize = 4096;

/// Payload bytes per (K, V) pair (excluding the 2+2=4 bytes of half scales) for a given `kv_bits` mode:
/// - `4`  (`WHT-KV4`):  64 B K + 64 B V = 128 B (+ 4 B scales = 132 B/pair, 3.88x vs FP16)
/// - `3`  (`WHT-K3V2`): 48 B K + 32 B V =  80 B (+ 4 B scales =  84 B/pair, 6.10x vs FP16)
/// - `33` (`WHT-KV3`):  48 B K + 48 B V =  96 B (+ 4 B scales = 100 B/pair, 5.12x vs FP16)
/// - `2`  (`WHT-KV2`):  32 B K + 32 B V =  64 B (+ 4 B scales =  68 B/pair, 7.53x vs FP16)
#[inline]
pub fn kv_pair_payload_bytes(kv_bits: u32) -> usize {
    match kv_bits {
        4 => 128,
        3 => 80,
        33 => 96,
        2 => 64,
        _ => panic!("unsupported kv_bits={kv_bits} (expected 4, 3, 33, or 2)"),
    }
}

#[inline]
pub fn kv_mode_name(kv_bits: u32) -> &'static str {
    match kv_bits {
        4 => "WHT-KV4",
        3 => "WHT-K3V2",
        33 => "WHT-KV3",
        2 => "WHT-KV2",
        _ => "WHT-KV?",
    }
}

// ----------------------------------------------------------------------------- macOS physical memory footprint (OOM telemetry)
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RusageInfoV4 {
    ri_uuid: [u8; 16],
    ri_user_time: u64,
    ri_system_time: u64,
    ri_pkg_idle_wkups: u64,
    ri_interrupt_wkups: u64,
    ri_pageins: u64,
    ri_wired_size: u64,
    ri_resident_size: u64,
    ri_phys_footprint: u64,
    ri_proc_start_abstime: u64,
    ri_proc_exit_abstime: u64,
    ri_child_user_time: u64,
    ri_child_system_time: u64,
    ri_child_pkg_idle_wkups: u64,
    ri_child_interrupt_wkups: u64,
    ri_child_pageins: u64,
    ri_child_elapsed_abstime: u64,
    ri_diskio_bytesread: u64,
    ri_diskio_byteswritten: u64,
    ri_cpu_time_qos_default: u64,
    ri_cpu_time_qos_maintenance: u64,
    ri_cpu_time_qos_background: u64,
    ri_cpu_time_qos_utility: u64,
    ri_cpu_time_qos_legacy: u64,
    ri_cpu_time_qos_user_initiated: u64,
    ri_cpu_time_qos_user_interactive: u64,
    ri_billed_system_time: u64,
    ri_serviced_system_time: u64,
    ri_logical_writes: u64,
    ri_lifetime_max_phys_footprint: u64,
    ri_instructions: u64,
    ri_cycles: u64,
    ri_billed_energy: u64,
    ri_serviced_energy: u64,
    ri_interval_max_phys_footprint: u64,
    ri_runnable_time: u64,
}

extern "C" {
    fn proc_pid_rusage(pid: libc::c_int, flavor: libc::c_int, buffer: *mut libc::c_void) -> libc::c_int;
}

/// Current macOS process physical memory footprint in bytes (via `proc_pid_rusage(RUSAGE_INFO_V4)`).
pub fn proc_phys_footprint_bytes() -> u64 {
    let mut info = RusageInfoV4::default();
    let rc = unsafe { proc_pid_rusage(libc::getpid(), 4, &mut info as *mut RusageInfoV4 as *mut libc::c_void) };
    if rc == 0 { info.ri_phys_footprint } else { 0 }
}

/// Returns `(phys_footprint, lifetime_max_phys_footprint, disk_bytes_read, disk_bytes_written)`.
pub fn proc_rusage_summary() -> (u64, u64, u64, u64) {
    let mut info = RusageInfoV4::default();
    let rc = unsafe { proc_pid_rusage(libc::getpid(), 4, &mut info as *mut RusageInfoV4 as *mut libc::c_void) };
    if rc == 0 {
        (info.ri_phys_footprint, info.ri_lifetime_max_phys_footprint, info.ri_diskio_bytesread, info.ri_diskio_byteswritten)
    } else {
        (0, 0, 0, 0)
    }
}

/// Open an unlinked temporary NVMe file with `F_NOCACHE` enabled so macOS page cache can NEVER consume RAM.
fn open_nocache_temp_file(tag: &str) -> Result<File, String> {
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("inkling-{tag}-{}-{seq}.bin", std::process::id()));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)
        .map_err(|e| format!("open nocache temp file {}: {e}", path.display()))?;
    let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    let _ = std::fs::remove_file(&path); // immediately unlinked: guaranteed cleanup on exit/crash
    if rc < 0 {
        return Err(format!("fcntl(F_NOCACHE) failed on {}", path.display()));
    }
    Ok(file)
}

// ----------------------------------------------------------------------------- Hard-capped GPU memory allocator
#[derive(Clone, Copy, Debug)]
pub struct MemGuard {
    pub max_bytes: usize,
    pub used_bytes: usize,
}

impl MemGuard {
    pub fn new(max_bytes: usize) -> Self {
        Self { max_bytes, used_bytes: 0 }
    }
    pub fn remaining(&self) -> usize {
        self.max_bytes.saturating_sub(self.used_bytes)
    }
    pub fn try_alloc(&mut self, gpu: &Gpu, bytes: usize, tag: &str) -> Result<Buf, String> {
        let req = bytes.max(16);
        let max_buf = gpu.dev.maxBufferLength() as usize;
        if req > max_buf {
            return Err(format!(
                "buffer `{tag}` ({:.2} GiB) exceeds Metal maxBufferLength ({:.2} GiB)",
                req as f64 / (1u64 << 30) as f64,
                max_buf as f64 / (1u64 << 30) as f64
            ));
        }
        if self.used_bytes.saturating_add(req) > self.max_bytes {
            return Err(format!(
                "allocation `{tag}` (+{:.2} MiB) would exceed hard max_mem_bytes ({:.2} / {:.2} GiB)",
                req as f64 / 1048576.0,
                (self.used_bytes + req) as f64 / (1u64 << 30) as f64,
                self.max_bytes as f64 / (1u64 << 30) as f64
            ));
        }
        let buf = gpu.buffer(req);
        self.used_bytes += buf.length();
        Ok(buf)
    }
    pub fn reserve_host(&mut self, bytes: usize, tag: &str) -> Result<(), String> {
        if self.used_bytes.saturating_add(bytes) > self.max_bytes {
            return Err(format!(
                "host reservation `{tag}` (+{:.2} MiB) exceeds hard max_mem_bytes ({:.2} GiB)",
                bytes as f64 / 1048576.0,
                self.max_bytes as f64 / (1u64 << 30) as f64
            ));
        }
        self.used_bytes += bytes;
        Ok(())
    }
    pub fn release(&mut self, bytes: usize) {
        self.used_bytes = self.used_bytes.saturating_sub(bytes);
    }
}

// ----------------------------------------------------------------------------- Kernel arg structs (exact mirrors of batch_opt.metal & batch.metal)
#[repr(C)] #[derive(Clone, Copy, Default)]
struct Route4Args {
    sel_off: u32, t: u32, perm_off: u32, inv_off: u32, aw_off: u32, ind_off: u32,
    gu_y: [u32; 4], dn_y: [u32; 4], tiles_off: [u32; 4],
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct Route6Args {
    sel_off: u32, t: u32, perm_off: u32, inv_off: u32, aw_off: u32, ind_off: u32,
    gu_y: [u32; NTC_OPT], dn_y: [u32; NTC_OPT], tiles_off: [u32; NTC_OPT],
    c_dir_max: u32, c_sg1_max: u32, c_sg2_max: u32, pad0: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct WoPostArgs {
    w_lo: u32, w_hi: u32, row_bytes: u32, wtype: u32, nrows: u32, k: u32, xh_off: u32, out_off: u32,
    res_off: u32, has_res: u32, layer_slot: u32, max_batch: u32, convw_t_off: u32, b: u32, seq_cs_off: u32, seq_pos_off: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct MoeCombPostArgs {
    yd_off: u32, inv_off: u32, res_off: u32, out_off: u32, layer_slot: u32, max_batch: u32,
    convw_t_off: u32, seq_cs_off: u32, seq_pos_off: u32, pad0: u32, pad1: u32, pad2: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct PagedAttnArgs {
    b: u32, pool_cap: u32, slot_idx: u32, is_swa: u32, pt_stride: u32, pt_mod: u32,
    seq_pos_off: u32, seq_suf_off: u32, seq_cs_off: u32, pt_off: u32,
    layer_slot: u32, max_batch: u32, q_off: u32, k_off: u32, v_off: u32, r_off: u32, sink_off: u32,
    kdyn_off: i32, vdyn_off: i32,
    kconv_t_w: u32, vconv_t_w: u32, qclip_w: u32, relclip_w: u32, relbias_w: u32,
    log_alpha: f32, log_floor: f32,
    gate_off: u32, out_off: u32, part_off: u32, nsplit: u32, prefix_len: u32,
    qp_off: u32, rp_off: u32, pad0: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct CombineBArgs { part_off: u32, nsplit: u32, gate_off: u32, out_off: u32 }

#[repr(C)] #[derive(Clone, Copy, Default)]
struct CopyCsArgs { dst_off: u32, n: u32, max_batch: u32, prefix_len: u32 }

#[repr(C)] #[derive(Clone, Copy, Default)]
struct LmHeadArgs {
    w_lo: u32, w_hi: u32, row_bytes: u32, wtype: u32, n_vocab: u32, k: u32, xh_off: u32, part_off: u32,
    b: u32, nvch: u32, vstart: u32, vcount: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct ArgmaxBatchArgs { part_off: u32, nvch: u32, out_base: u32, pad: u32 }

#[repr(C)] #[derive(Clone, Copy, Default)]
struct ChunkConvHcsArgs {
    y_off: u32, ldy: u32, res_off: u32, out_off: u32,
    layer_slot: u32, branch: u32, max_batch: u32, convw_t_off: u32,
    has_res: u32, seq_cs: u32, p0: u32, t: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct StateHcsArgs {
    x_off: u32, ldx: u32, layer_slot: u32, branch: u32,
    max_batch: u32, nch: u32, seq_cs: u32, p0: u32,
    t: u32, pad0: u32, pad1: u32, pad2: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct ComputeDcMeanArgs {
    src_cap: u32, src_pos0: u32, n_toks: u32, n_slots: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct WhtKvPrepArgs {
    b: u32, pool_cap: u32, slot_idx: u32, page_toks: u32, kv_bits: u32,
    layer_slot: u32, max_batch: u32, is_prefill: u32, p0: u32,
    seq_pos_off: u32, seq_cs_off: u32, write_pid_off: u32,
    q_off: u32, k_off: u32, v_off: u32, r_off: u32,
    kdyn_off: i32, vdyn_off: i32,
    kconv_t_w: u32, vconv_t_w: u32, qclip_w: u32, relclip_w: u32,
    log_alpha: f32, log_floor: f32,
    qp_off: u32, rp_off: u32, sink_off: u32, eff_sink_off: u32,
    use_dc_mean: u32, pad0: u32, pad1: u32, pad2: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct Fp16ToWhtArgs {
    src_cap: u32, src_pos0: u32, slot_idx: u32, layer_slot: u32,
    pool_cap: u32, page_toks: u32, kv_bits: u32, dst_pos0: u32,
    pid_list_off: u32, n_toks: u32, use_dc_mean: u32, pad1: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct PageSumArgs {
    pool_cap: u32, slot_idx: u32, page_toks: u32, kv_bits: u32,
    n_pages: u32, pid_list_off: u32, spid_list_off: u32, cnt_list_off: u32,
    sum_pool_cap: u32, pad0: u32, pad1: u32, pad2: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct QuestScoreArgs {
    b: u32, slot_idx: u32, sum_pool_cap: u32, max_pages: u32,
    top_k: u32, topk_stride: u32, mask_words: u32, use_sum_pt: u32,
    qp_off: u32, sink_off: u32,
    seq_nfar_off: u32, sum_pt_off: u32, sum_pt_stride: u32, sum_base_off: u32,
    scores_off: u32, topk_off: u32, topk_cnt_off: u32, ret_mask_off: u32, head_mask_off: u32, pad0: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct SparseWhtAttnArgs {
    b: u32, pool_cap: u32, slot_idx: u32, layer_slot: u32,
    page_toks: u32, kv_bits: u32, sum_pool_cap: u32, use_sum_pt: u32,
    use_centroids: u32, local_pt_stride: u32, far_pt_stride: u32, sum_pt_stride: u32,
    topk_stride: u32, mask_words: u32,
    qp_off: u32, rp_off: u32, sink_off: u32, gate_off: u32, out_off: u32, relbias_w: u32,
    seq_pos_off: u32, seq_nfar_off: u32, local_pt_off: u32, far_pt_off: u32, sum_pt_off: u32, sum_base_off: u32,
    topk_off: u32, topk_cnt_off: u32, ret_mask_off: u32, head_mask_off: u32,
    sum_format: u32, pad1: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct ChunkWhtAttnArgs {
    t: u32, p0: u32, pool_cap: u32, slot_idx: u32, layer_slot: u32,
    page_toks: u32, kv_bits: u32, sum_pool_cap: u32, use_sum_pt: u32,
    qp_off: u32, rp_off: u32, sink_off: u32, gate_off: u32, out_off: u32, relbias_w: u32,
    local_p0_page: u32, local_pt_off: u32, sum_pt_off: u32, sum_base: u32, use_centroids: u32,
    sum_format: u32, pad0: u32, pad1: u32, pad2: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct SwaCompressArgs {
    n_jobs: u32, inner_cap: u32, outer_cap: u32, slot_start: u32,
    inner_fp16: u32, src_pid_off: u32, dst_pid_off: u32, pad0: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct SwaLoadEngineArgs {
    src_cap: u32, first_pos: u32, nkeys: u32, inner_p0: u32,
    inner_cap: u32, outer_cap: u32, inner_fp16: u32, slot_start: u32,
    inner_pt_off: u32, outer_pt_off: u32, pad0: u32, pad1: u32,
}

#[repr(C)] #[derive(Clone, Copy, Default)]
struct TieredSwaAttnArgs {
    b: u32, inner_cap: u32, outer_cap: u32, slot_idx: u32,
    inner_fp16: u32, layer_slot: u32, max_batch: u32, pad0: u32,
    seq_pos_off: u32, seq_cs_off: u32, inner_pt_off: u32, outer_pt_off: u32,
    q_off: u32, k_off: u32, v_off: u32, r_off: u32,
    sink_off: u32, gate_off: u32, out_off: u32, pad1: u32,
    kconv_t_w: u32, vconv_t_w: u32, qclip_w: u32, relclip_w: u32,
    relbias_w: u32, pad2: u32, pad3: u32, pad4: u32,
}

#[repr(C)] #[derive(Clone, Copy)]
struct KvPageSummaryHost {
    k_min: [i8; 128],
    k_max: [i8; 128],
    k_mean: [i8; 128],
    v_mean: [i8; 128],
    sc_kmin: u16,
    sc_kmax: u16,
    sc_kmean: u16,
    sc_vmean: u16,
    count: u32,
    page_id: u32,
}

#[repr(C)] #[derive(Clone, Copy)]
struct KvPageWht2Host {
    k_sk: [[u32; 8]; WHT2_NSUB],
    k_mean4: [u32; 16],
    v_mean4: [u32; 16],
    sc_sk: [u16; WHT2_NSUB],
    sc_kmean: u16,
    sc_vmean: u16,
    count: u32,
    page_id: u32,
    pad0: u32,
}

const _: () = assert!(std::mem::size_of::<KvPageSummaryHost>() == KV_PAGE_SUM_BYTES);
const _: () = assert!(std::mem::size_of::<KvPageWht2Host>() == KV_PAGE_WHT2_BYTES);

// ----------------------------------------------------------------------------- Lean Aliased Arena for 6-class MoE + 256K Sparse Attention
#[derive(Clone, Copy, Default)]
struct OptArena {
    cap: usize,
    vec0: u32,
    msg: [u32; 4],
    cath: [u32; 4],
    inp: [u32; 3],
    gain_o: [u32; 4],
    gh: [u32; 4],
    xh: [u32; 3],
    q: u32,
    gate: u32,
    k: u32,
    v: u32,
    rel: u32,
    sink: u32,
    eff_sink: u32,
    kdyn: u32,
    vdyn: u32,
    qp: u32,
    rp: u32,
    part: u32,
    agh: u32,
    wo_y: u32,
    fin: u32,
    fh: u32,
    rlog: u32,
    sel: u32,
    perm: u32,
    inv: u32,
    aw: u32,
    tl: [u32; NTC_OPT],
    ind: u32,
    xgh: u32,
    acth: u32,
    yd: u32,
    ffn_y: u32,
    am_part: u32,
    lm_slice: u32,
    max_split: usize,
    end: u32,
}

impl OptArena {
    fn new(cap: usize, n_vocab: usize, need_chunk_y: bool) -> Self {
        let cap = cap.max(1);
        let mut o = 0u32;
        let take = |o: &mut u32, n: usize| -> u32 {
            let r = *o;
            *o += (n.div_ceil(64) * 64) as u32;
            r
        };
        let mut d = OptArena { cap, max_split: if cap < 8 { 16 } else { 4 }, ..Default::default() };
        d.vec0 = take(&mut o, B_NVEC * cap * E);
        for i in 0..4 { d.msg[i] = take(&mut o, cap * E); }
        for i in 0..4 { d.cath[i] = take(&mut o, cap * E); }
        let scratch0 = o;

        // Attention phase
        let mut oa = scratch0;
        for i in 0..3 { d.inp[i] = take(&mut oa, cap * E); }
        for i in 0..4 { d.gain_o[i] = take(&mut oa, cap * E); }
        for i in 0..4 { d.gh[i] = take(&mut oa, cap * 32); }
        for i in 0..3 { d.xh[i] = take(&mut oa, cap * E / 2); }
        d.q = take(&mut oa, cap * NH * HD / 2);
        d.gate = take(&mut oa, cap * NH * HD / 2);
        d.k = take(&mut oa, cap * NKV * HD);
        d.v = take(&mut oa, cap * NKV * HD);
        d.rel = take(&mut oa, cap * NH * NREL);
        d.sink = take(&mut oa, cap * NH);
        d.eff_sink = take(&mut oa, cap * NH);
        d.kdyn = take(&mut oa, cap * 32);
        d.vdyn = take(&mut oa, cap * 32);
        d.qp = take(&mut oa, cap * NH * HD / 2);
        d.rp = take(&mut oa, cap * NH * NREL);
        d.part = take(&mut oa, cap * NH * d.max_split * (HD + 2));
        d.agh = take(&mut oa, cap * NH * HD / 2);
        d.wo_y = if need_chunk_y { take(&mut oa, cap * E) } else { 0 };

        // FFN / 6-class MoE phase (aliases attention scratch)
        let mut of = scratch0;
        d.fin = take(&mut of, cap * E);
        let ffn_gain_o = take(&mut of, cap * E);
        let ffn_gh = take(&mut of, cap * 32);
        d.fh = take(&mut of, cap * E / 2);
        d.rlog = take(&mut of, cap * NEXP);
        d.sel = take(&mut of, cap * 16);
        d.perm = take(&mut of, cap * TOPK);
        d.inv = take(&mut of, cap * TOPK);
        d.aw = take(&mut of, cap * TOPK);
        d.tl = [
            take(&mut of, (cap * TOPK / 128 + NEXP + 1) * 4),
            take(&mut of, (NEXP + 1) * 4),
            take(&mut of, (NEXP + 1) * 4),
            take(&mut of, (NEXP + 1) * 4),
            take(&mut of, (NEXP + 1) * 4),
            take(&mut of, (NEXP + 1) * 4),
        ];
        d.ind = take(&mut of, 64); // 2 * 6 * 4 = 48 uints
        d.acth = take(&mut of, cap * TOPK * FF / 2);
        d.xgh = take(&mut of, cap * TOPK * E / 2);
        d.yd = d.xgh; // xgh is dead before moe_down writes yd
        d.ffn_y = if need_chunk_y { take(&mut of, cap * E) } else { 0 };
        d.gain_o[3] = ffn_gain_o;
        d.gh[3] = ffn_gh;

        // LM head argmax / B=1 full-logit slice phase (aliases scratch after fh)
        let mut ol = d.fh + (cap * E / 2).div_ceil(64) as u32 * 64;
        let nvch = n_vocab.div_ceil(LM_VCHUNK);
        d.am_part = take(&mut ol, cap * nvch * 2);
        d.lm_slice = if cap == 1 { take(&mut ol, LM_VSLICE_256K) } else { 0 };

        d.end = oa.max(of).max(ol);
        d
    }
    #[inline]
    fn vec(&self, i: usize) -> u32 {
        self.vec0 + (i * self.cap * E) as u32
    }
}

// ----------------------------------------------------------------------------- Shared Low-Level GPU Dispatch Helpers
#[derive(Clone, Copy)]
pub struct MoeRouteCfg {
    pub use_6class: bool,
    pub c_dir_max: u32,
    pub c_sg1_max: u32,
    pub c_sg2_max: u32,
    pub use_dyn8: bool,
    pub use_hcs: bool,
}

impl MoeRouteCfg {
    pub fn baseline_4class() -> Self {
        Self { use_6class: false, c_dir_max: 2, c_sg1_max: 2, c_sg2_max: 2, use_dyn8: false, use_hcs: false }
    }
    pub fn opt_6class_dyn4(use_hcs: bool) -> Self {
        Self { use_6class: true, c_dir_max: 4, c_sg1_max: 16, c_sg2_max: 32, use_dyn8: false, use_hcs }
    }
    pub fn opt_6class_dyn8(use_hcs: bool) -> Self {
        Self { use_6class: true, c_dir_max: 8, c_sg1_max: 16, c_sg2_max: 32, use_dyn8: true, use_hcs }
    }
}

#[inline]
fn enc_go<T>(eng: &Engine, e: &Enc, skip: u32, cls: u32, pso: &str, args: &T, grid: (usize, usize, usize), tg: usize) {
    if skip & (1 << cls) != 0 { return; }
    e.pso(&eng.gpu.pso(pso));
    e.bytes(args, 5);
    e.dispatch(grid, tg);
    e.barrier();
}

fn enc_mix_b(eng: &Engine, e: &Enc, skip: u32, cls: u32, jobs: &[MixJob], t: usize) {
    if skip & (1 << cls) != 0 { return; }
    for j in jobs {
        let j = MixJob { ntok: t as u32, ..*j };
        let name = if j.q_off < 0 { "k_mix_b_id".to_string() } else { format!("k_mix_b_{}_{}", j.n_s, j.n_q) };
        e.pso(&eng.gpu.pso(&name));
        e.bytes(&j, 5);
        if t <= 64 { e.dispatch((t, 1, 1), 32); } else { e.dispatch((t.div_ceil(4), 1, 1), 128); }
    }
    e.barrier();
}

fn enc_gemm_h(eng: &Engine, e: &Enc, skip: u32, cls: u32, segs: &[HSeg], t: usize) {
    let mut a = GemmHArgs { nseg: segs.len() as u32, t: t as u32, ..Default::default() };
    let mut tiles = 0u32;
    for (i, sg) in segs.iter().enumerate() {
        a.seg[i] = *sg;
        a.seg[i].tile_start = tiles;
        tiles += sg.nrows.div_ceil(64);
    }
    if t > 64 { enc_go(eng, e, skip, cls, "k_gemm_sg8", &a, (t.div_ceil(128), tiles as usize, 1), 256); }
    else { enc_go(eng, e, skip, cls, "k_gemm_sg4", &a, (1, tiles as usize, 1), 128); }
}

fn enc_ctrl_b(eng: &Engine, e: &Enc, skip: u32, d: &OptArena, lw: &LoopW, jobs: &[(usize, u32, u32, u32)], t: usize, hout: &[u32]) {
    let down: Vec<HSeg> = jobs.iter().map(|&(c, _, _, _)| {
        let mut s = Engine::hseg_pb(lw.down[c], 64, (2 * E) as u32, d.cath[c], d.gh[c]);
        s.hout = 1;
        s
    }).collect();
    enc_gemm_h(eng, e, skip, C_CTRLD, &down, t);
    let mut ga = CtrlGfArgs { t: t as u32, njobs: jobs.len() as u32, ..Default::default() };
    for (i, &(c, u, m, _)) in jobs.iter().enumerate() {
        ga.job[i] = CtrlGfJob { gain_w: lw.gain[c], forget_w: lw.forget[c], gh_off: d.gh[c], u_off: u, m_off: m, raw_off: d.gain_o[c], pad: [0; 2] };
    }
    if t > 64 { enc_go(eng, e, skip, C_CTRLA, "k_ctrl_gf", &ga, (t.div_ceil(128), jobs.len() * (E / 32), 1), 256); }
    else { enc_go(eng, e, skip, C_CTRLA, "k_ctrl_gf4", &ga, (1, jobs.len() * (E / 32), 1), 128); }
    let mut a = CtrlBArgs { n: jobs.len() as u32, ..Default::default() };
    for (i, &(c, _, _, nw)) in jobs.iter().enumerate() {
        a.job[i] = CtrlBJob { u_off: d.gain_o[c], m_off: u32::MAX, g_off: 0, f_off: 0, out_off: 0, norm_w: nw, h_off: hout[i] as i32, nof32: 1 };
    }
    enc_go(eng, e, skip, C_CTRLN, "k_ctrl_apply_b", &a, (t, jobs.len(), 1), 256);
}

fn enc_moe_dispatch(eng: &Engine, e: &Enc, act: &Buf, skip: u32, d: &OptArena, l: &Layer, t: usize, cfg: MoeRouteCfg) {
    enc_gemm_h(eng, e, skip, C_ROUTER, &[Engine::hseg_pb(l.router, NEXP as u32, E as u32, d.fh, d.rlog)], t);
    let ta = TopkBArgs { logits_off: d.rlog, bias_off: l.exp_bias, sel_off: d.sel, t: t as u32, wscale: eng.wscale, pad: [0; 3] };
    enc_go(eng, e, skip, C_TOPK, "k_topk_b", &ta, (t, 1, 1), 32);

    if cfg.use_6class {
        let ra = Route6Args {
            sel_off: d.sel, t: t as u32, perm_off: d.perm, inv_off: d.inv, aw_off: d.aw, ind_off: d.ind,
            gu_y: [(FF / 32) as u32, (FF / 32) as u32, (FF / 32) as u32, (FF / 32) as u32, (FF / 4) as u32, (FF / 4) as u32],
            dn_y: [(E / 64) as u32, (E / 64) as u32, (E / 64) as u32, (E / 64) as u32, (E / 4) as u32, (E / 8) as u32],
            tiles_off: d.tl,
            c_dir_max: cfg.c_dir_max,
            c_sg1_max: cfg.c_sg1_max,
            c_sg2_max: cfg.c_sg2_max,
            pad0: 0,
        };
        enc_go(eng, e, skip, C_TOPK, "k_moe_route6", &ra, (1, 1, 1), 1024);
    } else {
        let ra = Route4Args {
            sel_off: d.sel, t: t as u32, perm_off: d.perm, inv_off: d.inv, aw_off: d.aw, ind_off: d.ind,
            gu_y: [(FF / 32) as u32, (FF / 32) as u32, (FF / 4) as u32, (FF / 4) as u32],
            dn_y: [(E / 64) as u32, (E / 64) as u32, (E / 8) as u32, (E / 8) as u32],
            tiles_off: [d.tl[0], d.tl[1], d.tl[2], d.tl[3]],
        };
        enc_go(eng, e, skip, C_TOPK, "k_moe_route4", &ra, (1, 1, 1), 1024);
    }

    // Gather FFN inputs into contiguous half rows
    if skip & (1 << C_GATHER) == 0 {
        let g = ToHalfArgs {
            src_off: d.fh, dst_off: d.xgh, k: E as u32, rows: (t * TOPK) as u32,
            perm_off: d.perm, gather: 3, src_stride: E as u32, dst_stride: E as u32,
        };
        e.pso(&eng.gpu.pso("k_tohalf"));
        e.bytes(&g, 5);
        e.dispatch((E.div_ceil(1024), t * TOPK, 1), 256);
        e.barrier();
    }

    if cfg.use_6class {
        let dir_gu = if cfg.use_dyn8 { "k_moe_dir_gu_dyn8" } else { "k_moe_dir_gu_dyn4" };
        let dir_dn = if cfg.use_dyn8 { "k_moe_dir_dn_dyn8" } else { "k_moe_dir_dn_dyn4" };
        let gu_cls: [(&str, usize); NTC_OPT] = [
            ("k_moe_sg8", 256), ("k_moe_sg4", 128), ("k_moe_sg2", 64), ("k_moe_sg1", 32), (dir_gu, 128), ("k_moe_dir_gu1", 128),
        ];
        let dn_cls: [(&str, usize); NTC_OPT] = [
            ("k_moe_sg8", 256), ("k_moe_sg4", 128), ("k_moe_sg2", 64), ("k_moe_sg1", 32), (dir_dn, 128), ("k_moe_dir_dn1", 128),
        ];
        if skip & (1 << C_GU) == 0 {
            for (c, &(name, tg)) in gu_cls.iter().enumerate() {
                let g = MoeGemmHArgs {
                    w_lo: l.gu.off as u32, w_hi: (l.gu.off >> 32) as u32, expert_bytes: l.gu.expert_bytes as u32, row_bytes: l.gu.row_bytes,
                    wtype: l.gu.ty, nrows: (2 * FF) as u32, k: E as u32, xh_off: d.xgh, tiles_off: d.tl[c], ntiles_off: d.ind + (c * 4) as u32,
                    out_off: d.acth, ldo: FF as u32, situ: 1, aw_off: d.aw, beta: eng.situ_beta, lbeta: eng.situ_lbeta,
                };
                e.pso(&eng.gpu.pso(name));
                e.bytes(&g, 5);
                e.dispatch_indirect(act, (d.ind as usize + c * 4) * 4, tg);
            }
            e.barrier();
        }
        if skip & (1 << C_DOWN) == 0 {
            for (c, &(name, tg)) in dn_cls.iter().enumerate() {
                let dn = MoeGemmHArgs {
                    w_lo: l.dn.off as u32, w_hi: (l.dn.off >> 32) as u32, expert_bytes: l.dn.expert_bytes as u32, row_bytes: l.dn.row_bytes,
                    wtype: l.dn.ty, nrows: E as u32, k: FF as u32, xh_off: d.acth, tiles_off: d.tl[c], ntiles_off: d.ind + (c * 4) as u32,
                    out_off: d.yd, ldo: E as u32, situ: 0, aw_off: 0, beta: 0.0, lbeta: 0.0,
                };
                e.pso(&eng.gpu.pso(name));
                e.bytes(&dn, 5);
                e.dispatch_indirect(act, (d.ind as usize + (NTC_OPT + c) * 4) * 4, tg);
            }
            e.barrier();
        }
    } else {
        let gu_cls: [(&str, usize); 4] = [("k_moe_sg8", 256), ("k_moe_sg4", 128), ("k_moe_dir_gu2", 128), ("k_moe_dir_gu1", 128)];
        let dn_cls: [(&str, usize); 4] = [("k_moe_sg8", 256), ("k_moe_sg4", 128), ("k_moe_dir_dn2", 128), ("k_moe_dir_dn1", 128)];
        if skip & (1 << C_GU) == 0 {
            for (c, &(name, tg)) in gu_cls.iter().enumerate() {
                let g = MoeGemmHArgs {
                    w_lo: l.gu.off as u32, w_hi: (l.gu.off >> 32) as u32, expert_bytes: l.gu.expert_bytes as u32, row_bytes: l.gu.row_bytes,
                    wtype: l.gu.ty, nrows: (2 * FF) as u32, k: E as u32, xh_off: d.xgh, tiles_off: d.tl[c], ntiles_off: d.ind + (c * 4) as u32,
                    out_off: d.acth, ldo: FF as u32, situ: 1, aw_off: d.aw, beta: eng.situ_beta, lbeta: eng.situ_lbeta,
                };
                e.pso(&eng.gpu.pso(name));
                e.bytes(&g, 5);
                e.dispatch_indirect(act, (d.ind as usize + c * 4) * 4, tg);
            }
            e.barrier();
        }
        if skip & (1 << C_DOWN) == 0 {
            for (c, &(name, tg)) in dn_cls.iter().enumerate() {
                let dn = MoeGemmHArgs {
                    w_lo: l.dn.off as u32, w_hi: (l.dn.off >> 32) as u32, expert_bytes: l.dn.expert_bytes as u32, row_bytes: l.dn.row_bytes,
                    wtype: l.dn.ty, nrows: E as u32, k: FF as u32, xh_off: d.acth, tiles_off: d.tl[c], ntiles_off: d.ind + (c * 4) as u32,
                    out_off: d.yd, ldo: E as u32, situ: 0, aw_off: 0, beta: 0.0, lbeta: 0.0,
                };
                e.pso(&eng.gpu.pso(name));
                e.bytes(&dn, 5);
                e.dispatch_indirect(act, (d.ind as usize + (4 + c) * 4) * 4, tg);
            }
            e.barrier();
        }
    }
}

#[inline]
fn f16_to_f32(h: u16) -> f32 {
    let sg = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 31) as i32;
    let m = (h & 1023) as f32;
    if e == 0 { sg * m * 2f32.powi(-24) } else { sg * (1.0 + m / 1024.0) * 2f32.powi(e - 15) }
}

#[inline]
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if v == 0.0 || exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7bff
    } else {
        let mant = (bits >> 13) & 0x3ff;
        let rem = bits & 0x1fff;
        let mut h = ((exp as u32) << 10 | mant) + if rem > 0x1000 || (rem == 0x1000 && (mant & 1) == 1) { 1 } else { 0 };
        if h >= 0x7c00 { h = 0x7bff; }
        sign | h as u16
    }
}

fn median_f64(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

// ============================================================================================================
// PART 1: MoEOptBench — 6-Class Small-Tile MoE Router & FP16 CS_POOL Verification + Benchmark
// ============================================================================================================

pub struct MoEOptBench;

impl MoEOptBench {
    /// Encode one complete 48-layer batched decode step using either baseline 4-class/FP32-CS or 6-class/FP16-CS.
    #[allow(clippy::too_many_arguments)]
    fn encode_step(
        eng: &Engine, cb: &CmdBuf, act: &Buf, kv: &PagedKvPool, cs: &Buf, meta: &Buf, toks_in: &Buf, toks_out: &Buf,
        da: &OptArena, b_len: usize, max_batch: usize, ctx_pos: usize,
        seq_pos_off: u32, seq_suf_off: u32, seq_cs_off: u32, swa_pt_off: u32, full_pt_off: u32, full_pt_stride: usize,
        skip: u32, cfg: MoeRouteCfg,
    ) {
        let e = Enc::new(cb);
        e.buf(&eng.w, 0, 0);
        e.buf(&eng.pb, 0, 1);
        e.buf(act, 0, 2);
        e.buf(&kv.swa, 0, 3);
        e.buf(cs, 0, 4);
        e.buf(toks_in, 0, 6);
        e.buf(meta, 0, 7);

        let ea = EmbedArgs {
            emb_lo: eng.tok_embd.off as u32, emb_hi: (eng.tok_embd.off >> 32) as u32, row_bytes: eng.tok_embd.row_bytes,
            norm_off: eng.tok_norm, out0: da.vec(0), out1: da.vec(0), tok_index: 0, eps: eng.eps,
        };
        enc_go(eng, &e, skip, C_EMBED, "k_embed", &ea, (b_len, 1, 1), 256);

        let mut va = VecAlloc { next: 1, n: B_NVEC };
        let mut blocks: Vec<usize> = Vec::new();
        let mut cur = 0usize;

        let run_layer = |slot: usize, ip: usize, cur_v: usize, blks: &mut Vec<usize>, msg: Option<(usize, usize)>, va: &mut VecAlloc, pinned: &[usize]| -> usize {
            let l = &eng.layers[ip];
            let ctl = msg.map(|_| &eng.loops[ip - LOOP_START]);
            let es = E as u32;

            // 1. Attention inputs
            let norms: [i32; 3] = if l.swa { [l.attn_norm; 3] } else { [l.q_inp_norm, l.k_inp_norm, l.v_inp_norm] };
            let mut st: Vec<u32> = blks.iter().map(|&v| da.vec(v)).collect();
            st.push(da.vec(cur_v));
            let nq = if l.attn_res_query >= 0 { l.n_attn_q } else { 1 };
            if let Some(lw) = ctl {
                let (b0, b1) = msg.unwrap();
                let mut ja = eng.mix_job_off(&st, l.attn_res_query, nq, &da.inp[..nq as usize], &[], es);
                let mut jm = eng.mix_job_off(&[da.vec(b0), da.vec(b1)], lw.msg_query as i32, 4, &da.msg, &[], es);
                if l.swa { ja.hcopy[0] = [da.cath[0] as i32, da.cath[1] as i32, da.cath[2] as i32]; }
                else { for j in 0..3 { ja.hcopy[j][0] = da.cath[j] as i32; } }
                for j in 0..4 { jm.hcopy[j][0] = (da.cath[j] + (E / 2) as u32) as i32; }
                enc_mix_b(eng, &e, skip, C_MIXA, &[ja, jm], b_len);
                let jobs: Vec<(usize, u32, u32, u32)> = (0..3).map(|j| (j, if l.swa { da.inp[0] } else { da.inp[j] }, da.msg[j], norms[j] as u32)).collect();
                enc_ctrl_b(eng, &e, skip, da, lw, &jobs, b_len, &[da.xh[0], da.xh[1], da.xh[2]]);
            } else {
                let nn: Vec<i32> = if l.swa { vec![l.attn_norm] } else { norms.to_vec() };
                let mut ja = eng.mix_job_off(&st, l.attn_res_query, nq, &da.inp[..nq as usize], &nn, es);
                for q in 0..nq as usize { ja.hout[q] = da.xh[q] as i32; }
                ja.nof32 = 0xF;
                enc_mix_b(eng, &e, skip, C_MIXA, &[ja], b_len);
            }

            // 2. QKV projections
            let xi: [usize; 3] = if ctl.is_some() || !l.swa { [0, 1, 2] } else { [0, 0, 0] };
            let mut hs = vec![
                Engine::hseg(&l.wq, da.xh[xi[0]], da.q),
                Engine::hseg(&l.wgate, da.xh[xi[0]], da.gate),
                Engine::hseg(&l.wrel, da.xh[xi[0]], da.rel),
                Engine::hseg(&l.wsink, da.xh[xi[0]], da.sink),
                Engine::hseg(&l.wk, da.xh[xi[1]], da.k),
                Engine::hseg(&l.wv, da.xh[xi[2]], da.v),
            ];
            hs[0].hout = 1;
            hs[1].hout = 1;
            if let (Some(kp), Some(vp)) = (&l.kconvp, &l.vconvp) {
                hs.push(Engine::hseg(kp, da.xh[xi[1]], da.kdyn));
                hs.push(Engine::hseg(vp, da.xh[xi[2]], da.vdyn));
            }
            enc_gemm_h(eng, &e, skip, C_QKV, &hs, b_len);

            // 3. Paged decode attention (FP32 or FP16 CS)
            let (is_swa, slot_idx) = eng.slot_kind[slot];
            let pool_cap = if is_swa { kv.swa_cap } else { kv.full_cap };
            if is_swa { e.buf(&kv.swa, 0, 3); } else { e.buf(&kv.full, 0, 3); }
            let nkeys_max = if is_swa { (ctx_pos + 1).min(EXT) } else { ctx_pos + 1 };
            let nsplit = if b_len * NKV >= 64 && nkeys_max <= 2048 {
                1usize
            } else {
                (64usize.div_ceil(b_len * NKV)).max(nkeys_max.div_ceil(1024)).min(da.max_split).max(1)
            };
            let pa = PagedAttnArgs {
                b: b_len as u32, pool_cap: pool_cap as u32, slot_idx: slot_idx as u32, is_swa: is_swa as u32,
                pt_stride: (if is_swa { SWA_RING_BLKS } else { full_pt_stride }) as u32, pt_mod: SWA_RING_BLKS as u32,
                seq_pos_off, seq_suf_off, seq_cs_off, pt_off: if is_swa { swa_pt_off } else { full_pt_off },
                layer_slot: slot as u32, max_batch: max_batch as u32,
                q_off: da.q, k_off: da.k, v_off: da.v, r_off: da.rel, sink_off: da.sink,
                kdyn_off: if l.swa { -1 } else { da.kdyn as i32 }, vdyn_off: if l.swa { -1 } else { da.vdyn as i32 },
                kconv_t_w: l.k_conv_t, vconv_t_w: l.v_conv_t, qclip_w: l.q_clip, relclip_w: l.rel_clip, relbias_w: l.rel_bias,
                log_alpha: eng.log_alpha, log_floor: eng.log_floor,
                gate_off: da.gate, out_off: da.agh, part_off: da.part, nsplit: nsplit as u32, prefix_len: 0,
                qp_off: da.qp, rp_off: da.rp, pad0: 0,
            };
            let attn_k = if cfg.use_hcs { "k_attn_paged_b_hcs" } else { "k_attn_paged_b" };
            enc_go(eng, &e, skip, C_ATTN, attn_k, &pa, (nsplit, NKV, b_len), 128);
            if nsplit > 1 {
                let ca = CombineBArgs { part_off: da.part, nsplit: nsplit as u32, gate_off: da.gate, out_off: da.agh };
                enc_go(eng, &e, skip, C_COMB, "k_attn_combine_b", &ca, (NH, b_len, 1), 128);
            }

            // 4. wo + post-conv + residual
            let block_start = ip % BLOCK == 0;
            let mut live: Vec<usize> = blks.clone(); live.push(cur_v); live.extend_from_slice(pinned);
            let mid = va.alloc(&live);
            let wo = WoPostArgs {
                w_lo: l.wo.off as u32, w_hi: (l.wo.off >> 32) as u32, row_bytes: l.wo.row_bytes, wtype: l.wo.ty,
                nrows: l.wo.rows, k: l.wo.k, xh_off: da.agh, out_off: da.vec(mid),
                res_off: da.vec(cur_v), has_res: (!block_start) as u32,
                layer_slot: slot as u32, max_batch: max_batch as u32, convw_t_off: l.attn_post_conv_t,
                b: b_len as u32, seq_cs_off, seq_pos_off,
            };
            if b_len > 64 {
                let wk = if cfg.use_hcs { "k_wo_post_sg8_hcs" } else { "k_wo_post_sg8" };
                enc_go(eng, &e, skip, C_WO, wk, &wo, (b_len.div_ceil(128), E / 64, 1), 256);
            } else {
                let wk = if cfg.use_hcs { "k_wo_post_sg4_hcs" } else { "k_wo_post_sg4" };
                enc_go(eng, &e, skip, C_WO, wk, &wo, (1, E / 64, 1), 128);
            }
            if block_start { blks.push(cur_v); }

            // 5. FFN input
            let mut st: Vec<u32> = blks.iter().map(|&v| da.vec(v)).collect();
            st.push(da.vec(mid));
            if let Some(lw) = ctl {
                let mut jf = eng.mix_job_off(&st, l.ffn_res_query as i32, 1, &[da.fin], &[], es);
                jf.hcopy[0][0] = da.cath[3] as i32;
                enc_mix_b(eng, &e, skip, C_MIXF, &[jf], b_len);
                enc_ctrl_b(eng, &e, skip, da, lw, &[(3, da.fin, da.msg[3], l.ffn_norm)], b_len, &[da.fh]);
            } else {
                let mut jf = eng.mix_job_off(&st, l.ffn_res_query as i32, 1, &[da.fin], &[l.ffn_norm as i32], es);
                jf.hout[0] = da.fh as i32;
                jf.nof32 = 1;
                enc_mix_b(eng, &e, skip, C_MIXF, &[jf], b_len);
            }

            // 6. MoE (4-class vs 6-class!)
            enc_moe_dispatch(eng, &e, act, skip, da, l, b_len, cfg);

            // 7. Fused MoE combine + post-conv + residual
            let mut live: Vec<usize> = blks.clone(); live.push(mid); live.extend_from_slice(pinned);
            let out = va.alloc(&live);
            let cp = MoeCombPostArgs {
                yd_off: da.yd, inv_off: da.inv, res_off: da.vec(mid), out_off: da.vec(out),
                layer_slot: slot as u32, max_batch: max_batch as u32, convw_t_off: l.ffn_post_conv_t,
                seq_cs_off, seq_pos_off, pad0: 0, pad1: 0, pad2: 0,
            };
            let ck = if cfg.use_hcs { "k_moe_comb_post_hcs" } else { "k_moe_comb_post_b" };
            enc_go(eng, &e, skip, C_COMB2, ck, &cp, (b_len, E.div_ceil(1024), 1), 256);
            out
        };

        for ip in 0..LOOP_START { cur = run_layer(ip, ip, cur, &mut blocks, None, &mut va, &[]); }
        let prelude = cur;
        let prelude_blocks = blocks.clone();
        let mut bank = (0usize, 0usize);
        for pass in 0..N_PASS {
            if pass > 0 { cur = prelude; blocks = prelude_blocks.clone(); }
            let msg = if pass > 0 { Some(bank) } else { None };
            let pinned: Vec<usize> = { let mut p = vec![prelude]; if pass > 0 { p.push(bank.0); p.push(bank.1); } p };
            for ip in LOOP_START..LOOP_END {
                cur = run_layer(slot_of(pass, ip), ip, cur, &mut blocks, msg, &mut va, &pinned);
            }
            bank = (blocks[prelude_blocks.len() + 1], cur);
        }
        for ip in LOOP_END..N_PHYS { cur = run_layer(ip, ip, cur, &mut blocks, None, &mut va, &[]); }

        // Final mix + fused lm_head argmax
        let mut st: Vec<u32> = blocks.iter().map(|&v| da.vec(v)).collect();
        st.push(da.vec(cur));
        let mut jo = eng.mix_job_off(&st, eng.output_res_query as i32, 1, &[da.fin], &[eng.output_norm as i32], E as u32);
        jo.hout[0] = da.fh as i32;
        jo.nof32 = 1;
        enc_mix_b(eng, &e, skip, C_OMIX, &[jo], b_len);

        e.buf(toks_out, 0, 6);
        let w = &eng.output;
        let nvch = eng.n_vocab.div_ceil(LM_VCHUNK);
        let la = LmHeadArgs {
            w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty,
            n_vocab: eng.n_vocab as u32, k: w.k, xh_off: da.fh, part_off: da.am_part,
            b: b_len as u32, nvch: nvch as u32, vstart: 0, vcount: 0,
        };
        if b_len > 64 { enc_go(eng, &e, skip, C_LMHEAD, "k_lmhead_argmax_sg8", &la, (b_len.div_ceil(128), nvch, 1), 256); }
        else { enc_go(eng, &e, skip, C_LMHEAD, "k_lmhead_argmax_sg4", &la, (1, nvch, 1), 128); }
        let aa = ArgmaxBatchArgs { part_off: da.am_part, nvch: nvch as u32, out_base: 0, pad: 0 };
        enc_go(eng, &e, skip, C_ARGMAX, "k_argmax_batch", &aa, (b_len, 1, 1), 128);
        e.end();
    }

    fn zero_cs_buf(eng: &Engine, cs: &Buf, meta: &Buf, b_len: usize, max_batch: usize, hcs: bool) {
        let cb = eng.gpu.cmd();
        let e = Enc::new(&cb);
        e.pso(&eng.gpu.pso(if hcs { "k_zero_cs_hcs" } else { "k_zero_cs_slots" }));
        e.buf(cs, 0, 4);
        e.buf(meta, 0, 7);
        let a = CopyCsArgs { dst_off: 0, n: b_len as u32, max_batch: max_batch as u32, prefix_len: 0 };
        e.bytes(&a, 5);
        e.dispatch((NCH.div_ceil(256), b_len, N_SLOT), 256);
        e.end();
        cb.commit();
        cb.waitUntilCompleted();
    }
}

/// Directly benchmark 6-class MoE router (`k_moe_route6` + `k_moe_dir_gu/dn_dyn4/8` + `k_moe_sg1/2/4/8`)
/// + FP16 `CS_POOL` vs baseline 4-class MoE (`k_moe_route4` + FP32 `CS_POOL`) across `batches`.
pub fn run_bench_moe_opt(eng: &Engine, batches: &[usize], max_mem_gb: Option<f64>) {
    assert!(eng.hgemm, "run_bench_moe_opt requires M5 fp16 tensor path (hgemm)");
    let max_mem = BatchEngine::safe_mem_limit(eng, max_mem_gb);
    let ctx_pos = 63usize; // 2 pages/seq (keeps focus sharply on MoE + FP16 CS, tiny KV RAM!)
    let full_pt_stride = 4usize;
    println!("=== MoE 6-Class Small-Tile + FP16 CS_POOL Benchmark (cap {:.2} GiB, {}) ===",
             max_mem as f64 / (1u64 << 30) as f64, if eng.kv8 { "kv8" } else { "fp16" });

    // ---- Step 1: Numerical parity check at B=64 (Layer 0 MoE + FP16 CS) ----
    {
        let b_chk = 64usize;
        let da = OptArena::new(b_chk, eng.n_vocab, false);
        let mut mg = MemGuard::new(max_mem);
        let act = mg.try_alloc(&eng.gpu, da.end as usize * 4, "chk_act").expect("chk_act");
        let cs_f32 = mg.try_alloc(&eng.gpu, b_chk * CS_SEQ_ELEMS * 4, "chk_cs32").expect("chk_cs32");
        let cs_f16 = mg.try_alloc(&eng.gpu, b_chk * CS_SEQ_ELEMS * 2, "chk_cs16").expect("chk_cs16");
        let meta = mg.try_alloc(&eng.gpu, 64 * 1024, "chk_meta").expect("chk_meta");
        let toks_in = mg.try_alloc(&eng.gpu, b_chk * 4, "chk_tin").expect("chk_tin");
        let toks_out = mg.try_alloc(&eng.gpu, b_chk * 4, "chk_tout").expect("chk_tout");
        let kv = PagedKvPool::try_new(&eng.gpu, eng.kvb, b_chk * 2 + 8, b_chk * 2 + 8, mg.remaining()).expect("chk_kv");

        // Fill deterministic tokens & metadata
        let tp = toks_in.contents().as_ptr() as *mut i32;
        let mp = meta.contents().as_ptr() as *mut u32;
        let seq_cs_off = 0u32;
        let seq_pos_off = 64u32;
        let seq_suf_off = 128u32;
        let swa_pt_off = 192u32;
        let full_pt_off = swa_pt_off + (b_chk * SWA_RING_BLKS).div_ceil(16) as u32 * 16;
        unsafe {
            for b in 0..b_chk {
                *tp.add(b) = (1000 + (b * 137) % 30000) as i32;
                *mp.add(seq_cs_off as usize + b) = b as u32;
                *mp.add(seq_pos_off as usize + b) = ctx_pos as u32;
                *mp.add(seq_suf_off as usize + b) = ctx_pos as u32;
                for k in 0..2 { *mp.add(swa_pt_off as usize + b * SWA_RING_BLKS + k) = (b * 2 + k) as u32; }
                for k in 0..2 { *mp.add(full_pt_off as usize + b * full_pt_stride + k) = (b * 2 + k) as u32; }
            }
        }

        // Run baseline vs opt_dyn4 vs opt_dyn8 on Layer 0 FFN input (`d.fh` populated by k_embed + k_mix_b)
        let run_l0_moe = |cfg: MoeRouteCfg, cs_buf: &Buf| -> (Vec<f32>, Vec<u32>) {
            MoEOptBench::zero_cs_buf(eng, cs_buf, &meta, b_chk, b_chk, cfg.use_hcs);
            let cb = eng.gpu.cmd();
            let e = Enc::new(&cb);
            e.buf(&eng.w, 0, 0); e.buf(&eng.pb, 0, 1); e.buf(&act, 0, 2);
            e.buf(&kv.swa, 0, 3); e.buf(cs_buf, 0, 4); e.buf(&toks_in, 0, 6); e.buf(&meta, 0, 7);
            let ea = EmbedArgs {
                emb_lo: eng.tok_embd.off as u32, emb_hi: (eng.tok_embd.off >> 32) as u32, row_bytes: eng.tok_embd.row_bytes,
                norm_off: eng.tok_norm, out0: da.vec(0), out1: da.vec(0), tok_index: 0, eps: eng.eps,
            };
            enc_go(eng, &e, 0, C_EMBED, "k_embed", &ea, (b_chk, 1, 1), 256);
            let l = &eng.layers[0];
            let mut jf = eng.mix_job_off(&[da.vec(0)], l.ffn_res_query as i32, 1, &[da.fin], &[l.ffn_norm as i32], E as u32);
            jf.hout[0] = da.fh as i32; jf.nof32 = 1;
            enc_mix_b(eng, &e, 0, C_MIXF, &[jf], b_chk);
            enc_moe_dispatch(eng, &e, &act, 0, &da, l, b_chk, cfg);
            let cp = MoeCombPostArgs {
                yd_off: da.yd, inv_off: da.inv, res_off: da.vec(0), out_off: da.vec(1),
                layer_slot: 0, max_batch: b_chk as u32, convw_t_off: l.ffn_post_conv_t,
                seq_cs_off, seq_pos_off, pad0: 0, pad1: 0, pad2: 0,
            };
            enc_go(eng, &e, 0, C_COMB2, if cfg.use_hcs { "k_moe_comb_post_hcs" } else { "k_moe_comb_post_b" }, &cp, (b_chk, E.div_ceil(1024), 1), 256);
            e.end();
            cb.commit();
            cb.waitUntilCompleted();
            let out = unsafe {
                std::slice::from_raw_parts((act.contents().as_ptr() as *const f32).add(da.vec(1) as usize), b_chk * E).to_vec()
            };
            let ind_p = unsafe { (act.contents().as_ptr() as *const u32).add(da.ind as usize) };
            let ncls = if cfg.use_6class { NTC_OPT } else { 4 };
            let tiles: Vec<u32> = (0..ncls).map(|c| unsafe { *ind_p.add(c * 4) }).collect();
            (out, tiles)
        };

        let (ref_out, ref_tiles) = run_l0_moe(MoeRouteCfg::baseline_4class(), &cs_f32);
        let (opt4_out, opt4_tiles) = run_l0_moe(MoeRouteCfg::opt_6class_dyn4(true), &cs_f16);
        let (opt8_out, opt8_tiles) = run_l0_moe(MoeRouteCfg::opt_6class_dyn8(true), &cs_f16);

        let err_stats = |a: &[f32], b: &[f32]| -> (f64, f64) {
            let (mut se, mut sr, mut mx) = (0.0f64, 0.0f64, 0.0f64);
            for (&x, &y) in a.iter().zip(b) {
                let d = (x - y) as f64;
                se += d * d;
                sr += (y as f64) * (y as f64);
                mx = mx.max(d.abs());
            }
            (mx, (se / sr.max(1e-30)).sqrt())
        };
        let (mx4, l2_4) = err_stats(&opt4_out, &ref_out);
        let (mx8, l2_8) = err_stats(&opt8_out, &ref_out);
        println!("  [parity B=64 L0 MoE + HCS] route4 tiles={ref_tiles:?}");
        println!("    route6(dyn4,4/16/32)+HCS: tiles={opt4_tiles:?} | max_abs={mx4:.5} rel_L2={l2_4:.6}");
        println!("    route6(dyn8,8/16/32)+HCS: tiles={opt8_tiles:?} | max_abs={mx8:.5} rel_L2={l2_8:.6}");
        let _ = toks_out;
    }

    // ---- Step 2: Sweep across requested batch sizes ----
    println!("\n  {:>4} | {:>9} {:>9} {:>6} | {:>9} {:>9} {:>6} | {:>9} {:>9} {:>6} | {:>8} {:>8} | {:>7}",
             "B", "gu_old", "gu_new", "sp", "dn_old", "dn_new", "sp", "step_old", "step_new", "sp", "old_tps", "new_tps", "CS_save");
    println!("  {}", "-".repeat(126));

    for &b_len in batches {
        let b_len = b_len.max(1);
        let da = OptArena::new(b_len, eng.n_vocab, false);
        let act_bytes = da.end as usize * 4;
        let cs32_bytes = b_len * CS_SEQ_ELEMS * 4;
        let cs16_bytes = b_len * CS_SEQ_ELEMS * 2;
        let pages_need = b_len * 2 + 8;
        let kv_bytes = pages_need * (PagedKvPool::swa_page_bytes(eng.kvb) + PagedKvPool::full_page_bytes(eng.kvb));
        let meta_u32 = 256 + b_len * (SWA_RING_BLKS + full_pt_stride + 4);
        let need_tot = act_bytes + cs32_bytes + cs16_bytes + kv_bytes + meta_u32 * 4 + b_len * 16;
        if need_tot > max_mem {
            println!("  {:>4} | SKIPPED (needs {:.2} GiB > {:.2} GiB cap)", b_len, need_tot as f64 / (1u64 << 30) as f64, max_mem as f64 / (1u64 << 30) as f64);
            continue;
        }

        let mut mg = MemGuard::new(max_mem);
        let act = mg.try_alloc(&eng.gpu, act_bytes, "act").unwrap();
        let cs32 = mg.try_alloc(&eng.gpu, cs32_bytes, "cs32").unwrap();
        let cs16 = mg.try_alloc(&eng.gpu, cs16_bytes, "cs16").unwrap();
        let meta = mg.try_alloc(&eng.gpu, meta_u32 * 4, "meta").unwrap();
        let toks_in = mg.try_alloc(&eng.gpu, b_len.max(16) * 4, "tin").unwrap();
        let toks_out = mg.try_alloc(&eng.gpu, b_len.max(16) * 4, "tout").unwrap();
        let kv = PagedKvPool::try_new(&eng.gpu, eng.kvb, pages_need, pages_need, mg.remaining()).unwrap();

        let seq_cs_off = 0u32;
        let seq_pos_off = (b_len.div_ceil(16) * 16) as u32;
        let seq_suf_off = seq_pos_off + (b_len.div_ceil(16) * 16) as u32;
        let swa_pt_off = seq_suf_off + (b_len.div_ceil(16) * 16) as u32;
        let full_pt_off = swa_pt_off + (b_len * SWA_RING_BLKS).div_ceil(16) as u32 * 16;
        unsafe {
            let tp = toks_in.contents().as_ptr() as *mut i32;
            let mp = meta.contents().as_ptr() as *mut u32;
            for b in 0..b_len {
                *tp.add(b) = (1000 + (b * 137) % 30000) as i32;
                *mp.add(seq_cs_off as usize + b) = b as u32;
                *mp.add(seq_pos_off as usize + b) = ctx_pos as u32;
                *mp.add(seq_suf_off as usize + b) = ctx_pos as u32;
                for k in 0..2 { *mp.add(swa_pt_off as usize + b * SWA_RING_BLKS + k) = (b * 2 + k) as u32; }
                for k in 0..2 { *mp.add(full_pt_off as usize + b * full_pt_stride + k) = (b * 2 + k) as u32; }
            }
        }

        let measure_cfg = |cfg: MoeRouteCfg, cs_buf: &Buf| -> (f64, f64, f64, Vec<u32>) {
            let run_ms = |skip: u32| -> f64 {
                let cb = eng.gpu.cmd();
                MoEOptBench::encode_step(
                    eng, &cb, &act, &kv, cs_buf, &meta, &toks_in, &toks_out, &da,
                    b_len, b_len, ctx_pos, seq_pos_off, seq_suf_off, seq_cs_off, swa_pt_off, full_pt_off, full_pt_stride,
                    skip, cfg,
                );
                cb.commit();
                cb.waitUntilCompleted();
                (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
            };
            // Warmup + full step median (5 reps)
            MoEOptBench::zero_cs_buf(eng, cs_buf, &meta, b_len, b_len, cfg.use_hcs);
            let _ = run_ms(0);
            let full_ms = median_f64((0..5).map(|_| {
                MoEOptBench::zero_cs_buf(eng, cs_buf, &meta, b_len, b_len, cfg.use_hcs);
                run_ms(0)
            }).collect());

            // Populate valid FFN/MoE routing scratch (skip lm_head so d.ind/d.tl stay intact!)
            MoEOptBench::zero_cs_buf(eng, cs_buf, &meta, b_len, b_len, cfg.use_hcs);
            let _ = run_ms((1 << C_LMHEAD) | (1 << C_ARGMAX));
            let ind_p = unsafe { (act.contents().as_ptr() as *const u32).add(da.ind as usize) };
            let ncls = if cfg.use_6class { NTC_OPT } else { 4 };
            let tiles: Vec<u32> = (0..ncls).map(|c| unsafe { *ind_p.add(c * 4) }).collect();

            let empty_ms = median_f64((0..3).map(|_| run_ms(u32::MAX)).collect());
            let gu_ms = (median_f64((0..3).map(|_| run_ms(!(1u32 << C_GU))).collect()) - empty_ms).max(0.01);
            let dn_ms = (median_f64((0..3).map(|_| run_ms(!(1u32 << C_DOWN))).collect()) - empty_ms).max(0.01);
            (gu_ms, dn_ms, full_ms, tiles)
        };

        let (gu_old, dn_old, step_old, _) = measure_cfg(MoeRouteCfg::baseline_4class(), &cs32);
        let (gu_4, dn_4, step_4, tiles_4) = measure_cfg(MoeRouteCfg::opt_6class_dyn4(true), &cs16);
        let (gu_8, dn_8, step_8, tiles_8) = measure_cfg(MoeRouteCfg::opt_6class_dyn8(true), &cs16);
        let (gu_new, dn_new, step_new, best_tag, best_tiles) = if step_4 <= step_8 {
            (gu_4, dn_4, step_4, "dyn4(4/16/32)", tiles_4)
        } else {
            (gu_8, dn_8, step_8, "dyn8(8/16/32)", tiles_8)
        };

        let old_tps = b_len as f64 / (step_old * 1e-3);
        let new_tps = b_len as f64 / (step_new * 1e-3);
        let cs_save_mib = (cs32_bytes - cs16_bytes) as f64 / 1048576.0;
        println!("  {:>4} | {:>7.2}ms {:>7.2}ms {:>5.2}x | {:>7.2}ms {:>7.2}ms {:>5.2}x | {:>7.2}ms {:>7.2}ms {:>5.2}x | {:>8.1} {:>8.1} | -{:.1}MiB",
                 b_len, gu_old, gu_new, gu_old / gu_new, dn_old, dn_new, dn_old / dn_new,
                 step_old, step_new, step_old / step_new, old_tps, new_tps, cs_save_mib);
        println!("       best={best_tag} tiles[sg8,sg4,sg2,sg1,dirN,dir1]={best_tiles:?} (dyn4={step_4:.2}ms vs dyn8={step_8:.2}ms)");
    }
    let (fp, peak, _, _) = proc_rusage_summary();
    println!("  [rusage] phys_footprint={:.2} GiB | lifetime_peak={:.2} GiB\n",
             fp as f64 / (1u64 << 30) as f64, peak as f64 / (1u64 << 30) as f64);
}

// ============================================================================================================
// PART 2: 256K+ Streaming Compressed / Tiered Sparse-Retrieval KV Engine
// ============================================================================================================

/// Per-head DuoAttention mode table across all 64 full-attention KV heads (8 full slots x 8 KV heads).
/// `true` = retrieval head (Quest scoring + Top-K far pages + diffuse centroids);
/// `false` = streaming head (local 1K window + optional diffuse anchor, zero far-page reads!).
#[derive(Clone, Copy, Debug)]
pub struct DuoHeadConfig {
    pub retrieval_mask: [bool; N_FULL * NKV],
}

impl DuoHeadConfig {
    pub fn all_retrieval() -> Self {
        Self { retrieval_mask: [true; N_FULL * NKV] }
    }

    pub fn from_fraction(frac: f64) -> Self {
        let keep = ((frac.clamp(0.0, 1.0)) * (N_FULL * NKV) as f64).round() as usize;
        let mut mask = [false; N_FULL * NKV];
        for i in 0..keep {
            // spread evenly across the 8 full slots
            let idx = (i * N_FULL * NKV) / keep.max(1);
            mask[idx.min(N_FULL * NKV - 1)] = true;
        }
        Self { retrieval_mask: mask }
    }

    /// Parse `INK_DUO_HEADS` env var:
    /// - 64-char `0`/`1` bitstring (e.g. `1101...`)
    /// - comma-separated head indices in `0..64` (e.g. `0,1,8,9,56,63`)
    /// - float fraction in `0.0..=1.0` (e.g. `0.5`)
    /// Defaults to all 64 heads enabled until empirical Probe 1 head scores are plugged in.
    pub fn from_env() -> Self {
        let Ok(raw) = std::env::var("INK_DUO_HEADS") else { return Self::all_retrieval(); };
        let s = raw.trim();
        if s.len() == N_FULL * NKV && s.bytes().all(|b| b == b'0' || b == b'1') {
            let mut mask = [false; N_FULL * NKV];
            for (i, b) in s.bytes().enumerate() { mask[i] = b == b'1'; }
            return Self { retrieval_mask: mask };
        }
        if s.contains(',') {
            let mut mask = [false; N_FULL * NKV];
            for tok in s.split(',') {
                if let Ok(idx) = tok.trim().parse::<usize>() {
                    if idx < N_FULL * NKV { mask[idx] = true; }
                }
            }
            return Self { retrieval_mask: mask };
        }
        if let Ok(frac) = s.parse::<f64>() {
            return Self::from_fraction(frac);
        }
        Self::all_retrieval()
    }

    pub fn retrieval_count(&self) -> usize {
        self.retrieval_mask.iter().filter(|&&b| b).count()
    }

    pub fn to_u32_words(&self) -> [u32; 2] {
        let mut w = [0u32; 2];
        for (i, &on) in self.retrieval_mask.iter().enumerate() {
            if on { w[i >> 5] |= 1u32 << (i & 31); }
        }
        w
    }
}

// ----------------------------------------------------------------------------- Ref-Counted Radix Prefix Page Table
#[derive(Clone, Copy, Debug)]
pub struct RadixPage {
    pub hash: u64,
    pub logical_id: u64,
    pub ref_count: usize,
    pub sum_pid: u32,
    pub tok_count: u32,
}

pub struct RadixPageTable {
    map: HashMap<u64, RadixPage>,
    next_logical_id: u64,
    total_refs: usize,
}

impl RadixPageTable {
    pub fn new() -> Self {
        Self { map: HashMap::new(), next_logical_id: 1, total_refs: 0 }
    }

    /// Deterministic rolling 64-bit SplitMix hash chaining previous prefix hash with the next block of token IDs.
    pub fn rolling_page_hash(prev_hash: u64, toks: &[u32]) -> u64 {
        let mut h = prev_hash ^ 0x9E37_79B9_7F4A_7C15;
        for &t in toks {
            let mut z = h.wrapping_add((t as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)) ^ 0x94D0_49BB_1331_11EB;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            h = z ^ (z >> 31);
        }
        h ^ ((toks.len() as u64) << 48)
    }

    pub fn alloc_logical_id(&mut self) -> u64 {
        let id = self.next_logical_id;
        self.next_logical_id += 1;
        id
    }

    /// Acquire an existing shared prefix page if present (bumps ref_count; 0x extra RAM/NVMe!).
    pub fn acquire_existing(&mut self, hash: u64) -> Option<(u64, u32)> {
        let entry = self.map.get_mut(&hash)?;
        entry.ref_count += 1;
        self.total_refs += 1;
        Some((entry.logical_id, entry.sum_pid))
    }

    pub fn insert_new(&mut self, hash: u64, logical_id: u64, sum_pid: u32, tok_count: u32) {
        if let Some(e) = self.map.get_mut(&hash) {
            e.ref_count += 1;
        } else {
            self.map.insert(hash, RadixPage { hash, logical_id, ref_count: 1, sum_pid, tok_count });
        }
        self.total_refs += 1;
    }

    pub fn release(&mut self, hash: u64) -> Option<RadixPage> {
        let remove = if let Some(e) = self.map.get_mut(&hash) {
            e.ref_count = e.ref_count.saturating_sub(1);
            self.total_refs = self.total_refs.saturating_sub(1);
            e.ref_count == 0
        } else {
            false
        };
        if remove { self.map.remove(&hash) } else { None }
    }

    pub fn unique_pages(&self) -> usize { self.map.len() }
    pub fn total_refs(&self) -> usize { self.total_refs }
    pub fn dedup_ratio(&self) -> f64 {
        if self.map.is_empty() { 1.0 } else { self.total_refs as f64 / self.map.len() as f64 }
    }
}

// ----------------------------------------------------------------------------- F_NOCACHE NVMe Cold Page Store
pub struct NvmeColdStore {
    file: File,
    pub page_toks: usize,
    pub kv_bits: u32,
    pub payload_slot_bytes: usize, // bytes of payload for 1 full slot x 1 page
    pub scale_slot_bytes: usize,   // bytes of scales for 1 full slot x 1 page
    pub aligned_page_bytes: usize, // 4 KiB-aligned record size on disk (all N_FULL=8 slots)
    free_slots: Vec<u64>,
    next_slot: u64,
    scratch: Vec<u8>,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub write_time: Duration,
    pub read_time: Duration,
}

impl NvmeColdStore {
    pub fn new(page_toks: usize, kv_bits: u32) -> Result<Self, String> {
        assert!(matches!(kv_bits, 4 | 3 | 33 | 2), "kv_bits must be 4, 3 (K3V2), 33 (KV3), or 2");
        let file = open_nocache_temp_file("kv256k-cold")?;
        let payload_slot_bytes = NKV * page_toks * kv_pair_payload_bytes(kv_bits);
        let scale_slot_bytes = 2 * NKV * page_toks * 2;
        let raw_bytes = N_FULL * (payload_slot_bytes + scale_slot_bytes);
        let aligned_page_bytes = raw_bytes.div_ceil(ALIGN_4K) * ALIGN_4K;
        Ok(Self {
            file,
            page_toks,
            kv_bits,
            payload_slot_bytes,
            scale_slot_bytes,
            aligned_page_bytes,
            free_slots: Vec::new(),
            next_slot: 0,
            scratch: vec![0u8; aligned_page_bytes],
            bytes_written: 0,
            bytes_read: 0,
            write_time: Duration::ZERO,
            read_time: Duration::ZERO,
        })
    }

    pub fn alloc_slot(&mut self) -> u64 {
        if let Some(s) = self.free_slots.pop() { s } else {
            let s = self.next_slot;
            self.next_slot += 1;
            s
        }
    }

    pub fn free_slot(&mut self, slot: u64) {
        self.free_slots.push(slot);
    }

    /// Spill one physical hot page `pid` (all 8 full-attention slots) from GPU buffers to `F_NOCACHE` NVMe.
    pub fn spill_hot_page(&mut self, cold_slot: u64, pid: u32, kv_comp: &Buf, kv_scales: &Buf, hot_cap: usize) -> Result<(), String> {
        let t0 = Instant::now();
        let p_src = kv_comp.contents().as_ptr() as *const u8;
        let s_src = kv_scales.contents().as_ptr() as *const u8;
        let psb = self.payload_slot_bytes;
        let ssb = self.scale_slot_bytes;
        let mut off = 0usize;
        for slot in 0..N_FULL {
            let pidx = slot * hot_cap + pid as usize;
            unsafe {
                std::ptr::copy_nonoverlapping(p_src.add(pidx * psb), self.scratch.as_mut_ptr().add(off), psb);
                off += psb;
                std::ptr::copy_nonoverlapping(s_src.add(pidx * ssb), self.scratch.as_mut_ptr().add(off), ssb);
                off += ssb;
            }
        }
        let foff = cold_slot * self.aligned_page_bytes as u64;
        self.file.write_all_at(&self.scratch, foff).map_err(|e| format!("cold spill write_all_at: {e}"))?;
        self.bytes_written += self.aligned_page_bytes as u64;
        self.write_time += t0.elapsed();
        Ok(())
    }

    /// Parallel uncached `pread` of multiple `(cold_slot, dst_hot_pid)` pages directly into `StorageModeShared` GPU buffers.
    pub fn fetch_pages_to_gpu(&mut self, jobs: &[(u64, u32)], kv_comp: &Buf, kv_scales: &Buf, hot_cap: usize) -> Result<(), String> {
        if jobs.is_empty() { return Ok(()); }
        let t0 = Instant::now();
        let p_dst_addr = kv_comp.contents().as_ptr() as usize;
        let s_dst_addr = kv_scales.contents().as_ptr() as usize;
        let psb = self.payload_slot_bytes;
        let ssb = self.scale_slot_bytes;
        let rec_bytes = self.aligned_page_bytes;
        let file = &self.file;
        let nthreads = jobs.len().min(4);

        let next = std::sync::atomic::AtomicUsize::new(0);
        let err_flag = std::sync::atomic::AtomicBool::new(false);
        thread::scope(|sc| {
            for _ in 0..nthreads {
                let next = &next;
                let err_flag = &err_flag;
                sc.spawn(move || {
                    let mut local_buf = vec![0u8; rec_bytes];
                    loop {
                        let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if k >= jobs.len() { break; }
                        let (cold_slot, pid) = jobs[k];
                        let foff = cold_slot * rec_bytes as u64;
                        if file.read_exact_at(&mut local_buf, foff).is_err() {
                            err_flag.store(true, std::sync::atomic::Ordering::Relaxed);
                            return;
                        }
                        let mut off = 0usize;
                        let p_dst = p_dst_addr as *mut u8;
                        let s_dst = s_dst_addr as *mut u8;
                        for slot in 0..N_FULL {
                            let pidx = slot * hot_cap + pid as usize;
                            unsafe {
                                std::ptr::copy_nonoverlapping(local_buf.as_ptr().add(off), p_dst.add(pidx * psb), psb);
                                off += psb;
                                std::ptr::copy_nonoverlapping(local_buf.as_ptr().add(off), s_dst.add(pidx * ssb), ssb);
                                off += ssb;
                            }
                        }
                    }
                });
            }
        });
        if err_flag.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("cold store parallel read_exact_at failed".into());
        }
        self.bytes_read += (jobs.len() * rec_bytes) as u64;
        self.read_time += t0.elapsed();
        Ok(())
    }
}

// ----------------------------------------------------------------------------- Hot RAM LRU Cache (Strict Cap + Pinning)
pub struct HotRamLruCache {
    pub hot_cap: usize,
    free_pids: Vec<u32>,
    hot_to_logical: Vec<Option<u64>>,
    logical_to_hot: HashMap<u64, u32>,
    logical_to_cold: HashMap<u64, u64>,
    pin_count: Vec<u32>,
    dirty: Vec<bool>,
    last_used: Vec<u64>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl HotRamLruCache {
    pub fn new(hot_cap: usize) -> Self {
        let hot_cap = hot_cap.max(1);
        Self {
            hot_cap,
            free_pids: (0..hot_cap as u32).rev().collect(),
            hot_to_logical: vec![None; hot_cap],
            logical_to_hot: HashMap::with_capacity(hot_cap * 2),
            logical_to_cold: HashMap::new(),
            pin_count: vec![0; hot_cap],
            dirty: vec![false; hot_cap],
            last_used: vec![0; hot_cap],
            clock: 1,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    #[inline]
    pub fn lookup_hot(&mut self, logical_id: u64) -> Option<u32> {
        let &pid = self.logical_to_hot.get(&logical_id)?;
        self.clock += 1;
        self.last_used[pid as usize] = self.clock;
        Some(pid)
    }

    #[inline]
    pub fn peek_hot(&self, logical_id: u64) -> Option<u32> {
        self.logical_to_hot.get(&logical_id).copied()
    }

    pub fn pin(&mut self, pid: u32) {
        self.pin_count[pid as usize] = self.pin_count[pid as usize].saturating_add(1);
        self.clock += 1;
        self.last_used[pid as usize] = self.clock;
    }

    pub fn unpin(&mut self, pid: u32) {
        self.pin_count[pid as usize] = self.pin_count[pid as usize].saturating_sub(1);
    }

    pub fn mark_dirty(&mut self, pid: u32) {
        self.dirty[pid as usize] = true;
    }

    /// Allocate a hot `pid` for `logical_id`, evicting the LRU unpinned page to `NvmeColdStore` if full.
    pub fn alloc_for_logical(
        &mut self, logical_id: u64, pin_now: bool, dirty: bool,
        cold: &mut NvmeColdStore, kv_comp: &Buf, kv_scales: &Buf,
    ) -> Result<u32, String> {
        if let Some(pid) = self.lookup_hot(logical_id) {
            if pin_now { self.pin(pid); }
            if dirty { self.dirty[pid as usize] = true; }
            return Ok(pid);
        }
        let pid = if let Some(p) = self.free_pids.pop() {
            p
        } else {
            // Find LRU unpinned victim
            let mut victim: Option<(usize, u64)> = None;
            for p in 0..self.hot_cap {
                if self.pin_count[p] == 0 {
                    let ts = self.last_used[p];
                    if victim.map_or(true, |(_, best_ts)| ts < best_ts) {
                        victim = Some((p, ts));
                    }
                }
            }
            let (vp, _) = victim.ok_or_else(|| "HotRamLruCache exhausted: all hot pages are pinned!".to_string())?;
            if let Some(old_lid) = self.hot_to_logical[vp].take() {
                if self.dirty[vp] || !self.logical_to_cold.contains_key(&old_lid) {
                    let cslot = *self.logical_to_cold.entry(old_lid).or_insert_with(|| cold.alloc_slot());
                    cold.spill_hot_page(cslot, vp as u32, kv_comp, kv_scales, self.hot_cap)?;
                }
                self.logical_to_hot.remove(&old_lid);
                self.evictions += 1;
            }
            vp as u32
        };

        self.hot_to_logical[pid as usize] = Some(logical_id);
        self.logical_to_hot.insert(logical_id, pid);
        self.dirty[pid as usize] = dirty;
        self.clock += 1;
        self.last_used[pid as usize] = self.clock;
        if pin_now { self.pin_count[pid as usize] = self.pin_count[pid as usize].saturating_add(1); }
        Ok(pid)
    }

    /// Demand-fetch a set of cold logical pages into Hot RAM LRU (batched parallel `pread`).
    pub fn demand_fetch_batch(
        &mut self, logical_ids: &[u64], cold: &mut NvmeColdStore, kv_comp: &Buf, kv_scales: &Buf,
    ) -> Result<(usize, usize), String> {
        let mut jobs: Vec<(u64, u32)> = Vec::new();
        let (mut hit, mut miss) = (0usize, 0usize);
        for &lid in logical_ids {
            if self.lookup_hot(lid).is_some() {
                hit += 1;
                self.hits += 1;
            } else if let Some(&cslot) = self.logical_to_cold.get(&lid) {
                let pid = self.alloc_for_logical(lid, false, false, cold, kv_comp, kv_scales)?;
                jobs.push((cslot, pid));
                miss += 1;
                self.misses += 1;
            }
        }
        if !jobs.is_empty() {
            cold.fetch_pages_to_gpu(&jobs, kv_comp, kv_scales, self.hot_cap)?;
        }
        Ok((hit, miss))
    }

    pub fn free_logical(&mut self, logical_id: u64, cold: &mut NvmeColdStore) {
        if let Some(pid) = self.logical_to_hot.remove(&logical_id) {
            self.hot_to_logical[pid as usize] = None;
            self.pin_count[pid as usize] = 0;
            self.dirty[pid as usize] = false;
            self.free_pids.push(pid);
        }
        if let Some(cslot) = self.logical_to_cold.remove(&logical_id) {
            cold.free_slot(cslot);
        }
    }
}

// ----------------------------------------------------------------------------- Wave-Swap Manager (F_NOCACHE + Async Prefetcher)
struct PrefetchReq {
    seq_id: usize,
    offset: u64,
    bytes: usize,
}

struct PrefetchDone {
    seq_id: usize,
    data: Vec<u8>,
    elapsed: Duration,
}

pub struct WaveSwapRecord {
    pub offset: u64,
    pub total_bytes: usize,
    pub swa_bytes: usize,
    pub cs_bytes: usize,
    pub sum_bytes: usize,
    pub n_swa_pages: usize,
    pub n_sum_pages: usize,
}

pub struct WaveSwapManager {
    file: File,
    next_offset: u64,
    records: HashMap<usize, WaveSwapRecord>,
    req_tx: Option<mpsc::Sender<PrefetchReq>>,
    done_rx: mpsc::Receiver<PrefetchDone>,
    worker: Option<thread::JoinHandle<()>>,
    prefetched: HashMap<usize, Vec<u8>>,
    pub max_staging_bytes: usize,
    pub staged_bytes: usize,
    pub bytes_swapped_out: u64,
    pub bytes_swapped_in: u64,
    pub swap_out_time: Duration,
    pub swap_in_time: Duration,
}

impl WaveSwapManager {
    pub fn new(max_staging_bytes: usize) -> Result<Self, String> {
        let file = open_nocache_temp_file("kv256k-waveswap")?;
        let bg_file = file.try_clone().map_err(|e| format!("clone wave-swap fd: {e}"))?;
        let (req_tx, req_rx) = mpsc::channel::<PrefetchReq>();
        let (done_tx, done_rx) = mpsc::channel::<PrefetchDone>();

        let worker = thread::spawn(move || {
            while let Ok(req) = req_rx.recv() {
                let t0 = Instant::now();
                let mut data = vec![0u8; req.bytes];
                if bg_file.read_exact_at(&mut data, req.offset).is_ok() {
                    let _ = done_tx.send(PrefetchDone { seq_id: req.seq_id, data, elapsed: t0.elapsed() });
                }
            }
        });

        Ok(Self {
            file,
            next_offset: 0,
            records: HashMap::new(),
            req_tx: Some(req_tx),
            done_rx,
            worker: Some(worker),
            prefetched: HashMap::new(),
            max_staging_bytes,
            staged_bytes: 0,
            bytes_swapped_out: 0,
            bytes_swapped_in: 0,
            swap_out_time: Duration::ZERO,
            swap_in_time: Duration::ZERO,
        })
    }

    fn drain_done(&mut self) {
        while let Ok(d) = self.done_rx.try_recv() {
            self.bytes_swapped_in += d.data.len() as u64;
            self.swap_in_time += d.elapsed;
            self.staged_bytes += d.data.len();
            self.prefetched.insert(d.seq_id, d.data);
        }
    }

    /// Serialize an instance's live SWA pages + FP16 CS slot + private page summaries to `F_NOCACHE` NVMe.
    #[allow(clippy::too_many_arguments)]
    pub fn swap_out_instance(
        &mut self, seq_id: usize,
        swa_buf: &Buf, swa_cap: usize, swa_pages: &[u32], kvb: usize,
        cs_hcs: &Buf, cs_slot: u32, max_batch: usize,
        sum_buf: &Buf, sum_pool_cap: usize, sum_pids: &[u32],
    ) -> Result<Duration, String> {
        let t0 = Instant::now();
        let swa_slot_pb = PagedKvPool::slot_page_bytes(kvb);
        let swa_bytes = swa_pages.len() * N_SWA * swa_slot_pb;
        let cs_bytes = CS_SEQ_ELEMS * 2; // FP16 CS = 1.83 MiB
        let sum_page_all_bytes = N_FULL * NKV * KV_PAGE_SUM_BYTES;
        let sum_bytes = sum_pids.len() * sum_page_all_bytes;
        let total_bytes = (swa_bytes + cs_bytes + sum_bytes).div_ceil(ALIGN_4K) * ALIGN_4K;

        let mut blob = vec![0u8; total_bytes];
        let mut off = 0usize;

        // 1. SWA pages (slot-major)
        let swa_ptr = swa_buf.contents().as_ptr() as *const u8;
        for &pid in swa_pages {
            for s in 0..N_SWA {
                let src_off = (s * swa_cap + pid as usize) * swa_slot_pb;
                unsafe { std::ptr::copy_nonoverlapping(swa_ptr.add(src_off), blob.as_mut_ptr().add(off), swa_slot_pb); }
                off += swa_slot_pb;
            }
        }

        // 2. FP16 CS slot (slot-major, 4 branches)
        let cs_ptr = cs_hcs.contents().as_ptr() as *const u8;
        let branch_dims = [NKV * HD, NKV * HD, E, E];
        let mb = max_batch;
        for slot in 0..N_SLOT {
            let mut b_base = slot * mb * 3 * NCH;
            for &dim in &branch_dims {
                let src_elems = b_base + (cs_slot as usize) * 3 * dim;
                let nbytes = 3 * dim * 2;
                unsafe { std::ptr::copy_nonoverlapping(cs_ptr.add(src_elems * 2), blob.as_mut_ptr().add(off), nbytes); }
                off += nbytes;
                b_base += mb * 3 * dim;
            }
        }

        // 3. Page summaries
        let sum_ptr = sum_buf.contents().as_ptr() as *const u8;
        let head_row_bytes = NKV * KV_PAGE_SUM_BYTES;
        for &spid in sum_pids {
            for s in 0..N_FULL {
                let src_off = (s * sum_pool_cap + spid as usize) * head_row_bytes;
                unsafe { std::ptr::copy_nonoverlapping(sum_ptr.add(src_off), blob.as_mut_ptr().add(off), head_row_bytes); }
                off += head_row_bytes;
            }
        }

        let offset = if let Some(prev) = self.records.get(&seq_id) {
            if prev.total_bytes >= total_bytes { prev.offset } else {
                let o = self.next_offset;
                self.next_offset += total_bytes as u64;
                o
            }
        } else {
            let o = self.next_offset;
            self.next_offset += total_bytes as u64;
            o
        };

        self.file.write_all_at(&blob, offset).map_err(|e| format!("wave swap_out write: {e}"))?;
        let elapsed = t0.elapsed();
        self.bytes_swapped_out += total_bytes as u64;
        self.swap_out_time += elapsed;
        self.records.insert(seq_id, WaveSwapRecord {
            offset, total_bytes, swa_bytes, cs_bytes, sum_bytes,
            n_swa_pages: swa_pages.len(), n_sum_pages: sum_pids.len(),
        });
        Ok(elapsed)
    }

    /// Enqueue async background prefetch for wave `w + 1` (strictly respects `max_staging_bytes`!).
    pub fn request_prefetch(&mut self, seq_id: usize) -> bool {
        self.drain_done();
        if self.prefetched.contains_key(&seq_id) { return true; }
        let Some(rec) = self.records.get(&seq_id) else { return false; };
        if self.staged_bytes + rec.total_bytes > self.max_staging_bytes {
            return false; // respect hard RAM cap; will load synchronously when swapped in
        }
        if let Some(tx) = &self.req_tx {
            let _ = tx.send(PrefetchReq { seq_id, offset: rec.offset, bytes: rec.total_bytes });
            true
        } else {
            false
        }
    }

    /// Restore a swapped-out instance into active GPU buffers (consumes prefetched buffer if ready!).
    #[allow(clippy::too_many_arguments)]
    pub fn swap_in_instance(
        &mut self, seq_id: usize,
        swa_buf: &Buf, swa_cap: usize, swa_pages: &[u32], kvb: usize,
        cs_hcs: &Buf, cs_slot: u32, max_batch: usize,
        sum_buf: &Buf, sum_pool_cap: usize, sum_pids: &[u32],
    ) -> Result<Duration, String> {
        let t0 = Instant::now();
        self.drain_done();
        let rec = self.records.get(&seq_id).ok_or_else(|| format!("seq {seq_id} not found in wave swap"))?;
        let offset = rec.offset;
        let total_bytes = rec.total_bytes;

        let blob = if let Some(b) = self.prefetched.remove(&seq_id) {
            self.staged_bytes = self.staged_bytes.saturating_sub(b.len());
            b
        } else {
            let mut b = vec![0u8; total_bytes];
            self.file.read_exact_at(&mut b, offset).map_err(|e| format!("wave swap_in read: {e}"))?;
            self.bytes_swapped_in += total_bytes as u64;
            b
        };

        let mut off = 0usize;
        let swa_slot_pb = PagedKvPool::slot_page_bytes(kvb);
        let swa_ptr = swa_buf.contents().as_ptr() as *mut u8;
        for &pid in swa_pages {
            for s in 0..N_SWA {
                let dst_off = (s * swa_cap + pid as usize) * swa_slot_pb;
                unsafe { std::ptr::copy_nonoverlapping(blob.as_ptr().add(off), swa_ptr.add(dst_off), swa_slot_pb); }
                off += swa_slot_pb;
            }
        }

        let cs_ptr = cs_hcs.contents().as_ptr() as *mut u8;
        let branch_dims = [NKV * HD, NKV * HD, E, E];
        let mb = max_batch;
        for slot in 0..N_SLOT {
            let mut b_base = slot * mb * 3 * NCH;
            for &dim in &branch_dims {
                let dst_elems = b_base + (cs_slot as usize) * 3 * dim;
                let nbytes = 3 * dim * 2;
                unsafe { std::ptr::copy_nonoverlapping(blob.as_ptr().add(off), cs_ptr.add(dst_elems * 2), nbytes); }
                off += nbytes;
                b_base += mb * 3 * dim;
            }
        }

        let sum_ptr = sum_buf.contents().as_ptr() as *mut u8;
        let head_row_bytes = NKV * KV_PAGE_SUM_BYTES;
        for &spid in sum_pids {
            for s in 0..N_FULL {
                let dst_off = (s * sum_pool_cap + spid as usize) * head_row_bytes;
                unsafe { std::ptr::copy_nonoverlapping(blob.as_ptr().add(off), sum_ptr.add(dst_off), head_row_bytes); }
                off += head_row_bytes;
            }
        }
        let elapsed = t0.elapsed();
        self.swap_in_time += elapsed;
        Ok(elapsed)
    }

    /// Serialize an instance's compact Tiered SWA (inner 9 blks + outer 24 blks WHT-K2V2) + FP16 CS + KvPageWht2/Summary to NVMe.
    #[allow(clippy::too_many_arguments)]
    pub fn swap_out_tiered(
        &mut self, seq_id: usize,
        swa_inner: &Buf, swa_inner_sc: &Buf, swa_inner_cap: usize, inner_pages: &[u32], inner_fp16: bool,
        swa_outer_comp: &Buf, swa_outer_sc: &Buf, swa_outer_cap: usize, outer_pages: &[u32],
        cs_hcs: &Buf, cs_slot: u32, max_batch: usize,
        sum_buf: &Buf, sum_pool_cap: usize, sum_pids: &[u32], sum_elem_bytes: usize,
    ) -> Result<Duration, String> {
        let t0 = Instant::now();
        let inner_elem_b = if inner_fp16 { 2 } else { 1 };
        let inner_slot_pb = 2 * NKV * KV_BLK * HD * inner_elem_b;
        let inner_sc_pb = if inner_fp16 { 0 } else { 2 * NKV * KV_BLK * 2 };
        let outer_comp_pb = 2 * NKV * KV_BLK * 32;
        let outer_sc_pb = 2 * NKV * KV_BLK * 2;
        let swa_bytes = inner_pages.len() * N_SWA * (inner_slot_pb + inner_sc_pb)
            + outer_pages.len() * N_SWA * (outer_comp_pb + outer_sc_pb);
        let cs_bytes = CS_SEQ_ELEMS * 2;
        let sum_page_all_bytes = N_FULL * NKV * sum_elem_bytes;
        let sum_bytes = sum_pids.len() * sum_page_all_bytes;
        let total_bytes = (swa_bytes + cs_bytes + sum_bytes).div_ceil(ALIGN_4K) * ALIGN_4K;

        let mut blob = vec![0u8; total_bytes];
        let mut off = 0usize;

        // 1a. Tiered SWA inner pages
        let in_ptr = swa_inner.contents().as_ptr() as *const u8;
        let insc_ptr = swa_inner_sc.contents().as_ptr() as *const u8;
        for &pid in inner_pages {
            for s in 0..N_SWA {
                let idx = s * swa_inner_cap + pid as usize;
                unsafe {
                    std::ptr::copy_nonoverlapping(in_ptr.add(idx * inner_slot_pb), blob.as_mut_ptr().add(off), inner_slot_pb);
                    off += inner_slot_pb;
                    if inner_sc_pb > 0 {
                        std::ptr::copy_nonoverlapping(insc_ptr.add(idx * inner_sc_pb), blob.as_mut_ptr().add(off), inner_sc_pb);
                        off += inner_sc_pb;
                    }
                }
            }
        }
        // 1b. Tiered SWA outer WHT-K2V2 pages
        let out_ptr = swa_outer_comp.contents().as_ptr() as *const u8;
        let outsc_ptr = swa_outer_sc.contents().as_ptr() as *const u8;
        for &pid in outer_pages {
            for s in 0..N_SWA {
                let idx = s * swa_outer_cap + pid as usize;
                unsafe {
                    std::ptr::copy_nonoverlapping(out_ptr.add(idx * outer_comp_pb), blob.as_mut_ptr().add(off), outer_comp_pb);
                    off += outer_comp_pb;
                    std::ptr::copy_nonoverlapping(outsc_ptr.add(idx * outer_sc_pb), blob.as_mut_ptr().add(off), outer_sc_pb);
                    off += outer_sc_pb;
                }
            }
        }
        // 2. FP16 CS slot
        let cs_ptr = cs_hcs.contents().as_ptr() as *const u8;
        let branch_dims = [NKV * HD, NKV * HD, E, E];
        for slot in 0..N_SLOT {
            let mut b_base = slot * max_batch * 3 * NCH;
            for &dim in &branch_dims {
                let src_elems = b_base + (cs_slot as usize) * 3 * dim;
                let nbytes = 3 * dim * 2;
                unsafe { std::ptr::copy_nonoverlapping(cs_ptr.add(src_elems * 2), blob.as_mut_ptr().add(off), nbytes); }
                off += nbytes;
                b_base += max_batch * 3 * dim;
            }
        }
        // 3. Page index (KvPageWht2 or KvPageSummary)
        let sum_ptr = sum_buf.contents().as_ptr() as *const u8;
        let head_row_bytes = NKV * sum_elem_bytes;
        for &spid in sum_pids {
            for s in 0..N_FULL {
                let src_off = (s * sum_pool_cap + spid as usize) * head_row_bytes;
                unsafe { std::ptr::copy_nonoverlapping(sum_ptr.add(src_off), blob.as_mut_ptr().add(off), head_row_bytes); }
                off += head_row_bytes;
            }
        }

        let offset = if let Some(prev) = self.records.get(&seq_id) {
            if prev.total_bytes >= total_bytes { prev.offset } else {
                let o = self.next_offset; self.next_offset += total_bytes as u64; o
            }
        } else {
            let o = self.next_offset; self.next_offset += total_bytes as u64; o
        };

        self.file.write_all_at(&blob, offset).map_err(|e| format!("wave swap_out_tiered write: {e}"))?;
        let elapsed = t0.elapsed();
        self.bytes_swapped_out += total_bytes as u64;
        self.swap_out_time += elapsed;
        self.records.insert(seq_id, WaveSwapRecord {
            offset, total_bytes, swa_bytes, cs_bytes, sum_bytes,
            n_swa_pages: inner_pages.len() + outer_pages.len(), n_sum_pages: sum_pids.len(),
        });
        Ok(elapsed)
    }

    /// Restore a swapped-out Tiered-SWA + KvPageWht2 instance into active GPU buffers (consumes prefetched buffer if ready!).
    #[allow(clippy::too_many_arguments)]
    pub fn swap_in_tiered(
        &mut self, seq_id: usize,
        swa_inner: &Buf, swa_inner_sc: &Buf, swa_inner_cap: usize, inner_pages: &[u32], inner_fp16: bool,
        swa_outer_comp: &Buf, swa_outer_sc: &Buf, swa_outer_cap: usize, outer_pages: &[u32],
        cs_hcs: &Buf, cs_slot: u32, max_batch: usize,
        sum_buf: &Buf, sum_pool_cap: usize, sum_pids: &[u32], sum_elem_bytes: usize,
    ) -> Result<Duration, String> {
        let t0 = Instant::now();
        self.drain_done();
        let rec = self.records.get(&seq_id).ok_or_else(|| format!("seq {seq_id} not found in wave swap"))?;
        let offset = rec.offset;
        let total_bytes = rec.total_bytes;

        let blob = if let Some(b) = self.prefetched.remove(&seq_id) {
            self.staged_bytes = self.staged_bytes.saturating_sub(b.len());
            b
        } else {
            let mut b = vec![0u8; total_bytes];
            self.file.read_exact_at(&mut b, offset).map_err(|e| format!("wave swap_in_tiered read: {e}"))?;
            self.bytes_swapped_in += total_bytes as u64;
            b
        };

        let mut off = 0usize;
        let inner_elem_b = if inner_fp16 { 2 } else { 1 };
        let inner_slot_pb = 2 * NKV * KV_BLK * HD * inner_elem_b;
        let inner_sc_pb = if inner_fp16 { 0 } else { 2 * NKV * KV_BLK * 2 };
        let outer_comp_pb = 2 * NKV * KV_BLK * 32;
        let outer_sc_pb = 2 * NKV * KV_BLK * 2;

        let in_ptr = swa_inner.contents().as_ptr() as *mut u8;
        let insc_ptr = swa_inner_sc.contents().as_ptr() as *mut u8;
        for &pid in inner_pages {
            for s in 0..N_SWA {
                let idx = s * swa_inner_cap + pid as usize;
                unsafe {
                    std::ptr::copy_nonoverlapping(blob.as_ptr().add(off), in_ptr.add(idx * inner_slot_pb), inner_slot_pb);
                    off += inner_slot_pb;
                    if inner_sc_pb > 0 {
                        std::ptr::copy_nonoverlapping(blob.as_ptr().add(off), insc_ptr.add(idx * inner_sc_pb), inner_sc_pb);
                        off += inner_sc_pb;
                    }
                }
            }
        }
        let out_ptr = swa_outer_comp.contents().as_ptr() as *mut u8;
        let outsc_ptr = swa_outer_sc.contents().as_ptr() as *mut u8;
        for &pid in outer_pages {
            for s in 0..N_SWA {
                let idx = s * swa_outer_cap + pid as usize;
                unsafe {
                    std::ptr::copy_nonoverlapping(blob.as_ptr().add(off), out_ptr.add(idx * outer_comp_pb), outer_comp_pb);
                    off += outer_comp_pb;
                    std::ptr::copy_nonoverlapping(blob.as_ptr().add(off), outsc_ptr.add(idx * outer_sc_pb), outer_sc_pb);
                    off += outer_sc_pb;
                }
            }
        }
        let cs_ptr = cs_hcs.contents().as_ptr() as *mut u8;
        let branch_dims = [NKV * HD, NKV * HD, E, E];
        for slot in 0..N_SLOT {
            let mut b_base = slot * max_batch * 3 * NCH;
            for &dim in &branch_dims {
                let dst_elems = b_base + (cs_slot as usize) * 3 * dim;
                let nbytes = 3 * dim * 2;
                unsafe { std::ptr::copy_nonoverlapping(blob.as_ptr().add(off), cs_ptr.add(dst_elems * 2), nbytes); }
                off += nbytes;
                b_base += max_batch * 3 * dim;
            }
        }
        let sum_ptr = sum_buf.contents().as_ptr() as *mut u8;
        let head_row_bytes = NKV * sum_elem_bytes;
        for &spid in sum_pids {
            for s in 0..N_FULL {
                let dst_off = (s * sum_pool_cap + spid as usize) * head_row_bytes;
                unsafe { std::ptr::copy_nonoverlapping(blob.as_ptr().add(off), sum_ptr.add(dst_off), head_row_bytes); }
                off += head_row_bytes;
            }
        }
        let elapsed = t0.elapsed();
        self.swap_in_time += elapsed;
        Ok(elapsed)
    }
}

impl Drop for WaveSwapManager {
    fn drop(&mut self) {
        self.req_tx.take();
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

// ----------------------------------------------------------------------------- Kv256kPool & Kv256kSession
#[derive(Clone, Debug)]
pub struct Kv256kConfig {
    pub max_active_batch: usize,
    pub max_ctx: usize,
    pub page_toks: usize,   // 64 or 128
    pub kv_bits: u32,       // 4 (WHT-KV4), 3 (WHT-K3V2), 33 (WHT-KV3), or 2 (WHT-KV2)
    pub top_k_pages: usize, // retrieved far pages per head (<= 128)
    pub use_centroids: bool,
    pub use_wht2_sketch: bool, // true: KvPageWht2 (416 B) + k_score_pages_wht2; false: legacy Quest KvPageSummary (528 B)
    pub use_tiered_swa: bool,  // true: inner-256 (9 blks) + outer-768 WHT-K2V2 (24 blks) (~38.8 MiB/inst!)
    pub swa_inner_fp16: bool,  // true: FP16 inner-256 (60.9 MiB/inst); false: int8+half inner-256 (38.8 MiB/inst)
    pub use_dc_mean: bool,     // true: per-head K_DC_MEAN subtraction before WHT + sink-exact dc_bias
    pub duo: DuoHeadConfig,
    pub max_mem_bytes: usize,
    pub sum_pool_pages: usize,
    pub hot_full_pages: usize,
    pub chunk_cap: usize,   // max streaming prefill chunk size (0 = decode-only)
}

#[derive(Clone, Copy)]
struct SparseAuxLayout {
    scores_off: u32,
    topk_off: u32,
    topk_cnt_off: u32,
    ret_mask_off: u32,
    head_mask_off: u32,
    max_pages: usize,
    topk_stride: usize,
    mask_words: usize,
    total_u32: usize,
}

impl SparseAuxLayout {
    fn new(max_batch: usize, max_pages: usize, top_k: usize) -> Self {
        let mut o = 0u32;
        let mut take = |n: usize| { let r = o; o += (n.div_ceil(16) * 16) as u32; r };
        let topk_stride = top_k.clamp(1, 128).div_ceil(16) * 16;
        let mask_words = max_pages.max(32).div_ceil(32);
        let bg = max_batch * NKV;
        let scores_off = take(bg * max_pages.max(1));
        let topk_off = take(bg * topk_stride);
        let topk_cnt_off = take(bg);
        let ret_mask_off = take(bg * mask_words);
        let head_mask_off = take(16);
        Self { scores_off, topk_off, topk_cnt_off, ret_mask_off, head_mask_off, max_pages, topk_stride, mask_words, total_u32: o as usize }
    }
}

#[derive(Clone, Copy)]
struct SparseMetaLayout {
    seq_pos: u32,
    seq_suf: u32,
    seq_cs: u32,
    seq_nfar: u32,
    write_pid: u32,
    swa_pt: u32,
    swa_inner_pt: u32,
    swa_outer_pt: u32,
    swa_grad_src: u32,
    swa_grad_dst: u32,
    local_pt: u32,
    far_pt: u32,
    sum_pt: u32,
    sum_base: u32,
    local_pt_stride: usize,
    far_pt_stride: usize,
    total_u32: usize,
}

impl SparseMetaLayout {
    fn new(max_batch: usize, max_pages: usize, page_toks: usize) -> Self {
        let mut o = 0u32;
        let mut take = |n: usize| { let r = o; o += (n.div_ceil(16) * 16) as u32; r };
        let local_pt_stride = (EXT / page_toks + 2).max(4);
        let far_pt_stride = max_pages.max(1);
        let seq_pos = take(max_batch);
        let seq_suf = take(max_batch);
        let seq_cs = take(max_batch);
        let seq_nfar = take(max_batch);
        let write_pid = take(max_batch.max(64));
        let swa_pt = take(max_batch * SWA_RING_BLKS);
        let swa_inner_pt = take(max_batch * SWA_INNER_BLKS);
        let swa_outer_pt = take(max_batch * SWA_OUTER_BLKS);
        let swa_grad_src = take(max_batch.max(16));
        let swa_grad_dst = take(max_batch.max(16));
        let local_pt = take(max_batch * local_pt_stride);
        let far_pt = take(max_batch * far_pt_stride);
        let sum_pt = take(max_batch * far_pt_stride);
        let sum_base = take(max_batch);
        Self {
            seq_pos, seq_suf, seq_cs, seq_nfar, write_pid,
            swa_pt, swa_inner_pt, swa_outer_pt, swa_grad_src, swa_grad_dst,
            local_pt, far_pt, sum_pt, sum_base,
            local_pt_stride, far_pt_stride, total_u32: o as usize,
        }
    }
}

pub struct Kv256kSession {
    pub id: usize,
    pub pos: usize,
    pub cs_slot: Option<u32>,
    /// Legacy SWA pages (33 blks) when `!use_tiered_swa`, OR Tiered SWA inner pages (9 blks) when `use_tiered_swa`.
    pub swa_pages: Vec<u32>,
    /// Tiered SWA outer WHT-K2V2 pages (24 blks) when `use_tiered_swa`.
    pub swa_outer_pages: Vec<u32>,
    /// Logical page ID for each page `0..n_pages` (maps into HotRamLruCache / NvmeColdStore).
    pub logical_pages: Vec<u64>,
    /// Summary pool page ID for each completed page `0..n_pages` (0xFFFFFFFF if not yet summarized).
    pub sum_pages: Vec<u32>,
    /// Radix hashes for deduplicated completed pages (shared across sessions with identical prefixes!).
    pub page_hashes: Vec<Option<u64>>,
    pub last_prefix_hash: u64,
    pub swapped_out: bool,
    pub next_tok: u32,
    pub generated: Vec<u32>,
}

impl Kv256kSession {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            pos: 0,
            cs_slot: None,
            swa_pages: Vec::new(),
            swa_outer_pages: Vec::new(),
            logical_pages: Vec::new(),
            sum_pages: Vec::new(),
            page_hashes: Vec::new(),
            last_prefix_hash: 0,
            swapped_out: false,
            next_tok: 0,
            generated: Vec::new(),
        }
    }
}

pub struct Kv256kPool<'a> {
    pub eng: &'a Engine,
    pub cfg: Kv256kConfig,
    pub mem: MemGuard,
    pub kv_comp: Buf,      // [N_FULL][hot_cap] packed K+V (KV4 / K3V2 / KV3 / KV2)
    pub kv_scales: Buf,    // [N_FULL][hot_cap][2][NKV][page_toks] half
    pub k_dc_mean: Buf,    // [N_FULL][NKV][HD] half (16 KiB static per-head DC key mean!)
    pub use_dc_mean: bool, // active only once calibrated (or enabled with non-zero k_dc_mean)
    chunk_swa_comp: Option<Buf>,   // 1-slot SWA scratch for streaming chunked prefill
    chunk_swa_scales: Option<Buf>,
    chunk_swa_cap: usize,
    pub kv_sum: Buf,       // [N_FULL][sum_pool_cap][NKV] KvPageWht2 (416 B) or KvPageSummary (528 B)
    pub swa_pool: Buf,     // legacy SWA pool (16 B dummy when use_tiered_swa == true)
    pub swa_cap: usize,
    swa_free: Vec<u32>,
    pub swa_inner: Buf,       // Tiered SWA inner ring: [N_SWA][swa_inner_cap][2][NKV][32][128] half or char
    pub swa_inner_sc: Buf,    // Tiered SWA inner scales: [N_SWA][swa_inner_cap][2][NKV][32] half
    pub swa_outer_comp: Buf,  // Tiered SWA outer WHT-K2V2: [N_SWA][swa_outer_cap][2][NKV][32][8 uint32]
    pub swa_outer_sc: Buf,    // Tiered SWA outer scales: [N_SWA][swa_outer_cap][2][NKV][32] half
    pub swa_inner_cap: usize,
    pub swa_outer_cap: usize,
    swa_inner_free: Vec<u32>,
    swa_outer_free: Vec<u32>,
    pub cs_pool_hcs: Buf,  // FP16 slot-major ring conv state (2x smaller!)
    free_cs: Vec<u32>,
    free_sum: Vec<u32>,
    pub act: Buf,
    da: OptArena,
    pub meta: [Buf; 2],
    ml: SparseMetaLayout,
    pub aux: Buf,
    al: SparseAuxLayout,
    pub toks: [Buf; 2],
    pub lru: HotRamLruCache,
    pub radix: RadixPageTable,
    pub cold: NvmeColdStore,
    pub swap: WaveSwapManager,
    pub moe_cfg: MoeRouteCfg,
}

impl<'a> Kv256kPool<'a> {
    pub fn full_comp_page_bytes(page_toks: usize, kv_bits: u32) -> usize {
        N_FULL * NKV * page_toks * (kv_pair_payload_bytes(kv_bits) + 4)
    }

    pub fn summary_page_bytes() -> usize {
        N_FULL * NKV * KV_PAGE_SUM_BYTES // 8 * 8 * 528 = 33,792 B (legacy Quest)
    }

    pub fn summary_page_bytes_for(use_wht2_sketch: bool) -> usize {
        N_FULL * NKV * if use_wht2_sketch { KV_PAGE_WHT2_BYTES } else { KV_PAGE_SUM_BYTES }
    }

    pub fn tiered_swa_inst_bytes(inner_fp16: bool) -> usize {
        let inner_elem = if inner_fp16 { 2 } else { 1 };
        let inner_payload = SWA_INNER_BLKS * N_SWA * 2 * NKV * KV_BLK * HD * inner_elem;
        let inner_sc = if inner_fp16 { 0 } else { SWA_INNER_BLKS * N_SWA * 2 * NKV * KV_BLK * 2 };
        let outer_payload = SWA_OUTER_BLKS * N_SWA * 2 * NKV * KV_BLK * 32;
        let outer_sc = SWA_OUTER_BLKS * N_SWA * 2 * NKV * KV_BLK * 2;
        inner_payload + inner_sc + outer_payload + outer_sc
    }

    pub fn try_new(eng: &'a Engine, mut cfg: Kv256kConfig) -> Result<Self, String> {
        assert!(eng.hgemm, "Kv256kPool requires M5 fp16 tensor path");
        assert!(cfg.page_toks == 64 || cfg.page_toks == 128, "page_toks must be 64 or 128");
        assert!(matches!(cfg.kv_bits, 4 | 3 | 33 | 2), "kv_bits must be 4, 3 (K3V2), 33 (KV3), or 2");
        cfg.max_active_batch = cfg.max_active_batch.max(1);
        let max_pages = cfg.max_ctx.div_ceil(cfg.page_toks).max(1);

        let mut mem = MemGuard::new(cfg.max_mem_bytes);

        // 1. Activation arena (decode batch or streaming prefill chunk)
        let arena_cap = cfg.max_active_batch.max(cfg.chunk_cap);
        let da = OptArena::new(arena_cap, eng.n_vocab, cfg.chunk_cap > 0);
        let act = mem.try_alloc(&eng.gpu, da.end as usize * 4, "kv256k_act")?;

        // 2. FP16 CS_POOL (half!) + 16 KiB static per-head DC key mean buffer
        let cs_bytes = cfg.max_active_batch * CS_SEQ_ELEMS * 2;
        let cs_pool_hcs = mem.try_alloc(&eng.gpu, cs_bytes, "kv256k_cs_hcs")?;
        let k_dc_mean = mem.try_alloc(&eng.gpu, N_FULL * NKV * HD * 2, "kv256k_dc_mean")?;
        unsafe { std::ptr::write_bytes(k_dc_mean.contents().as_ptr() as *mut u8, 0, k_dc_mean.length()); }

        // 3. Metadata & Sparse AUX buffers
        let ml = SparseMetaLayout::new(cfg.max_active_batch, max_pages, cfg.page_toks);
        let al = SparseAuxLayout::new(cfg.max_active_batch, max_pages, cfg.top_k_pages);
        let meta = [
            mem.try_alloc(&eng.gpu, ml.total_u32 * 4, "kv256k_meta0")?,
            mem.try_alloc(&eng.gpu, ml.total_u32 * 4, "kv256k_meta1")?,
        ];
        let aux = mem.try_alloc(&eng.gpu, al.total_u32 * 4, "kv256k_aux")?;
        let toks = [
            mem.try_alloc(&eng.gpu, arena_cap.max(16) * 4, "kv256k_toks0")?,
            mem.try_alloc(&eng.gpu, arena_cap.max(16) * 4, "kv256k_toks1")?,
        ];

        // 4. SWA paged pool: either compact Tiered SWA (inner 256 + outer 768 WHT-K2V2) or legacy 33-blk SWA
        let (swa_pool, swa_cap, swa_free, swa_inner, swa_inner_sc, swa_outer_comp, swa_outer_sc, swa_inner_cap, swa_outer_cap, swa_inner_free, swa_outer_free) =
            if cfg.use_tiered_swa {
                let icap = cfg.max_active_batch * SWA_INNER_BLKS;
                let ocap = cfg.max_active_batch * SWA_OUTER_BLKS;
                let inner_elem = if cfg.swa_inner_fp16 { 2 } else { 1 };
                let s_in = mem.try_alloc(&eng.gpu, icap * N_SWA * 2 * NKV * KV_BLK * HD * inner_elem, "kv256k_swa_inner")?;
                let s_insc = mem.try_alloc(
                    &eng.gpu,
                    if cfg.swa_inner_fp16 { 16 } else { icap * N_SWA * 2 * NKV * KV_BLK * 2 },
                    "kv256k_swa_inner_sc",
                )?;
                let s_oc = mem.try_alloc(&eng.gpu, ocap * N_SWA * 2 * NKV * KV_BLK * 32, "kv256k_swa_outer_comp")?;
                let s_osc = mem.try_alloc(&eng.gpu, ocap * N_SWA * 2 * NKV * KV_BLK * 2, "kv256k_swa_outer_sc")?;
                let dummy = mem.try_alloc(&eng.gpu, 16, "kv256k_swa_dummy")?;
                (
                    dummy, 0usize, Vec::new(),
                    s_in, s_insc, s_oc, s_osc,
                    icap, ocap,
                    (0..icap as u32).rev().collect(),
                    (0..ocap as u32).rev().collect(),
                )
            } else {
                let cap = cfg.max_active_batch * SWA_RING_BLKS;
                let sp = mem.try_alloc(&eng.gpu, cap * PagedKvPool::swa_page_bytes(eng.kvb), "kv256k_swa")?;
                let d0 = mem.try_alloc(&eng.gpu, 16, "kv256k_swa_in_dummy")?;
                let d1 = mem.try_alloc(&eng.gpu, 16, "kv256k_swa_insc_dummy")?;
                let d2 = mem.try_alloc(&eng.gpu, 16, "kv256k_swa_oc_dummy")?;
                let d3 = mem.try_alloc(&eng.gpu, 16, "kv256k_swa_osc_dummy")?;
                (
                    sp, cap, (0..cap as u32).rev().collect(),
                    d0, d1, d2, d3,
                    0usize, 0usize, Vec::new(), Vec::new(),
                )
            };

        // 5. Summary / WHT-2bit-sketch page index pool
        let sum_pb = Self::summary_page_bytes_for(cfg.use_wht2_sketch);
        let sum_cap = cfg.sum_pool_pages.max(1);
        let kv_sum = mem.try_alloc(&eng.gpu, sum_cap * sum_pb, "kv256k_sum")?;

        // 5b. Dedicated 1-slot SWA scratch for streaming chunked prefill (only if chunk_cap > 0)
        let pair_payload = kv_pair_payload_bytes(cfg.kv_bits);
        let (chunk_swa_comp, chunk_swa_scales, chunk_swa_cap) = if cfg.chunk_cap > 0 {
            let ccap = (EXT + cfg.chunk_cap).div_ceil(cfg.page_toks) + 2;
            let c_comp = mem.try_alloc(&eng.gpu, ccap * NKV * cfg.page_toks * pair_payload, "kv256k_chunk_swa_comp")?;
            let c_sc = mem.try_alloc(&eng.gpu, ccap * 2 * NKV * cfg.page_toks * 2, "kv256k_chunk_swa_scales")?;
            (Some(c_comp), Some(c_sc), ccap)
        } else {
            (None, None, 0)
        };

        // 6. Hot RAM compressed Full KV pool (KV4 / K3V2 / KV3 / KV2), capped strictly to remaining budget!
        let swap_stage_reserve = (128usize << 20).min(mem.remaining() / 5);
        let comp_pb = Self::full_comp_page_bytes(cfg.page_toks, cfg.kv_bits);
        let avail_for_hot = mem.remaining().saturating_sub(swap_stage_reserve);
        let max_hot_fit = avail_for_hot / comp_pb;
        let min_local_pages = cfg.max_active_batch * (EXT / cfg.page_toks + 1);
        let hot_cap = if cfg.hot_full_pages > 0 {
            cfg.hot_full_pages.min(max_hot_fit)
        } else {
            max_hot_fit.min(min_local_pages + cfg.max_active_batch * cfg.top_k_pages)
        };
        if hot_cap < min_local_pages {
            return Err(format!(
                "Hot compressed KV cap ({hot_cap} pages) < minimum local window ({min_local_pages} pages) within {:.2} GiB budget; reduce batch or use kv3/kv2",
                cfg.max_mem_bytes as f64 / (1u64 << 30) as f64
            ));
        }
        cfg.hot_full_pages = hot_cap;

        let payload_bytes = N_FULL * hot_cap * NKV * cfg.page_toks * pair_payload;
        let scales_bytes = N_FULL * hot_cap * 2 * NKV * cfg.page_toks * 2;
        let kv_comp = mem.try_alloc(&eng.gpu, payload_bytes, "kv256k_comp_payload")?;
        let kv_scales = mem.try_alloc(&eng.gpu, scales_bytes, "kv256k_comp_scales")?;

        // Write DuoAttention head mask into AUX
        let hwords = cfg.duo.to_u32_words();
        unsafe {
            let ap = aux.contents().as_ptr() as *mut u32;
            *ap.add(al.head_mask_off as usize) = hwords[0];
            *ap.add(al.head_mask_off as usize + 1) = hwords[1];
        }

        let cold = NvmeColdStore::new(cfg.page_toks, cfg.kv_bits)?;
        let swap = WaveSwapManager::new(swap_stage_reserve)?;
        mem.reserve_host(swap_stage_reserve.min(mem.remaining()), "wave_swap_staging")?;

        let max_active = cfg.max_active_batch as u32;
        let use_dc_mean = cfg.use_dc_mean;
        Ok(Self {
            eng,
            cfg,
            mem,
            kv_comp,
            kv_scales,
            k_dc_mean,
            use_dc_mean,
            chunk_swa_comp,
            chunk_swa_scales,
            chunk_swa_cap,
            kv_sum,
            swa_pool,
            swa_cap,
            swa_free,
            swa_inner,
            swa_inner_sc,
            swa_outer_comp,
            swa_outer_sc,
            swa_inner_cap,
            swa_outer_cap,
            swa_inner_free,
            swa_outer_free,
            cs_pool_hcs,
            free_cs: (0..max_active).rev().collect(),
            free_sum: (0..sum_cap as u32).rev().collect(),
            act,
            da,
            meta,
            ml,
            aux,
            al,
            toks,
            lru: HotRamLruCache::new(hot_cap),
            radix: RadixPageTable::new(),
            cold,
            swap,
            moe_cfg: MoeRouteCfg::opt_6class_dyn4(true),
        })
    }

    pub fn set_duo_heads(&self, duo: DuoHeadConfig) {
        let w = duo.to_u32_words();
        unsafe {
            let ap = self.aux.contents().as_ptr() as *mut u32;
            *ap.add(self.al.head_mask_off as usize) = w[0];
            *ap.add(self.al.head_mask_off as usize + 1) = w[1];
        }
    }

    /// Compute static per-head key DC mean `[N_FULL][NKV][HD]` directly on GPU from `self.eng.kv_full[0..n_toks)`.
    pub fn calibrate_dc_mean_from_engine(&mut self, n_toks: usize) {
        if n_toks == 0 { return; }
        let cb = self.eng.gpu.cmd();
        let e = Enc::new(&cb);
        e.pso(&self.eng.gpu.pso("k_compute_k_dc_mean"));
        e.buf(&self.eng.kv_full.borrow(), 0, 3);
        e.buf(&self.k_dc_mean, 0, 10);
        let a = ComputeDcMeanArgs {
            src_cap: self.eng.full_cap.get() as u32,
            src_pos0: 0,
            n_toks: n_toks as u32,
            n_slots: N_FULL as u32,
        };
        e.bytes(&a, 5);
        e.dispatch((NKV, N_FULL, 1), 128);
        e.end();
        cb.commit();
        cb.waitUntilCompleted();
        self.use_dc_mean = true;
    }

    pub fn clear_dc_mean(&mut self) {
        unsafe { std::ptr::write_bytes(self.k_dc_mean.contents().as_ptr() as *mut u8, 0, self.k_dc_mean.length()); }
        self.use_dc_mean = false;
    }

    pub fn alloc_cs(&mut self) -> Result<u32, String> {
        self.free_cs.pop().ok_or_else(|| "FP16 CS slot pool exhausted".to_string())
    }
    pub fn free_cs(&mut self, slot: u32) { self.free_cs.push(slot); }

    pub fn alloc_swa_page(&mut self) -> Result<u32, String> {
        if self.cfg.use_tiered_swa {
            self.swa_inner_free.pop().ok_or_else(|| "Tiered SWA inner page pool exhausted".to_string())
        } else {
            self.swa_free.pop().ok_or_else(|| "SWA page pool exhausted".to_string())
        }
    }
    pub fn free_swa_page(&mut self, pid: u32) {
        if self.cfg.use_tiered_swa { self.swa_inner_free.push(pid); } else { self.swa_free.push(pid); }
    }

    pub fn alloc_swa_outer_page(&mut self) -> Result<u32, String> {
        self.swa_outer_free.pop().ok_or_else(|| "Tiered SWA outer page pool exhausted".to_string())
    }
    pub fn free_swa_outer_page(&mut self, pid: u32) { self.swa_outer_free.push(pid); }

    pub fn alloc_sum_page(&mut self) -> Result<u32, String> {
        self.free_sum.pop().ok_or_else(|| "KvPageSummary/Wht2 pool exhausted".to_string())
    }
    pub fn free_sum_page(&mut self, spid: u32) { self.free_sum.push(spid); }

    pub fn zero_cs_slots(&self, cs_slots: &[u32]) {
        if cs_slots.is_empty() { return; }
        let mp = self.meta[0].contents().as_ptr() as *mut u32;
        for (i, &s) in cs_slots.iter().enumerate() { unsafe { *mp.add(i) = s; } }
        let cb = self.eng.gpu.cmd();
        let e = Enc::new(&cb);
        e.pso(&self.eng.gpu.pso("k_zero_cs_hcs"));
        e.buf(&self.cs_pool_hcs, 0, 4);
        e.buf(&self.meta[0], 0, 7);
        let a = CopyCsArgs { dst_off: 0, n: cs_slots.len() as u32, max_batch: self.cfg.max_active_batch as u32, prefix_len: 0 };
        e.bytes(&a, 5);
        e.dispatch((NCH.div_ceil(256), cs_slots.len(), N_SLOT), 256);
        e.end();
        cb.commit();
        cb.waitUntilCompleted();
    }

    /// Pin local window pages for `sess` and ensure the active write page & SWA ring pages exist.
    fn ensure_session_decode_pages(&mut self, sess: &mut Kv256kSession) -> Result<u32, String> {
        if self.cfg.use_tiered_swa {
            let in_idx = (sess.pos / KV_BLK) % SWA_INNER_BLKS;
            while sess.swa_pages.len() <= in_idx {
                let p = self.alloc_swa_page()?;
                sess.swa_pages.push(p);
            }
            if sess.pos >= SWA_INNER_BLKS * KV_BLK {
                let grad_blk = (sess.pos / KV_BLK) - SWA_INNER_BLKS;
                let out_idx = grad_blk % SWA_OUTER_BLKS;
                while sess.swa_outer_pages.len() <= out_idx {
                    let p = self.alloc_swa_outer_page()?;
                    sess.swa_outer_pages.push(p);
                }
            }
        } else {
            let swa_idx = (sess.pos / KV_BLK) % SWA_RING_BLKS;
            while sess.swa_pages.len() <= swa_idx {
                let p = self.alloc_swa_page()?;
                sess.swa_pages.push(p);
            }
        }
        // Full compressed page for `sess.pos`
        let p_idx = sess.pos / self.cfg.page_toks;
        while sess.logical_pages.len() <= p_idx {
            let lid = self.radix.alloc_logical_id();
            sess.logical_pages.push(lid);
            sess.sum_pages.push(u32::MAX);
            sess.page_hashes.push(None);
        }
        let write_lid = sess.logical_pages[p_idx];
        let write_pid = self.lru.alloc_for_logical(write_lid, true, true, &mut self.cold, &self.kv_comp, &self.kv_scales)?;
        Ok(write_pid)
    }

    /// Summarize a newly completed compressed full page across all 8 full slots on the GPU
    /// (using `k_page_summarize_wht2` when `use_wht2_sketch == true`, else legacy `k_page_summarize_kv`).
    pub fn summarize_completed_page(&mut self, hot_pid: u32, spid: u32, count: u32) {
        let mp = self.meta[1].contents().as_ptr() as *mut u32;
        unsafe {
            *mp.add(0) = hot_pid;
            *mp.add(16) = spid;
            *mp.add(32) = count;
        }
        let cb = self.eng.gpu.cmd();
        let e = Enc::new(&cb);
        let kname = if self.cfg.use_wht2_sketch { "k_page_summarize_wht2" } else { "k_page_summarize_kv" };
        e.pso(&self.eng.gpu.pso(kname));
        e.buf(&self.kv_comp, 0, 3);
        e.buf(&self.kv_scales, 0, 6);
        e.buf(&self.meta[1], 0, 7);
        e.buf(&self.kv_sum, 0, 8);
        let a = PageSumArgs {
            pool_cap: self.lru.hot_cap as u32,
            slot_idx: 0,
            page_toks: self.cfg.page_toks as u32,
            kv_bits: self.cfg.kv_bits,
            n_pages: 1,
            pid_list_off: 0,
            spid_list_off: 16,
            cnt_list_off: 32,
            sum_pool_cap: self.cfg.sum_pool_pages as u32,
            pad0: 0, pad1: 0, pad2: 0,
        };
        e.bytes(&a, 5);
        e.dispatch((1, NKV, N_FULL), 128);
        e.end();
        cb.commit();
        cb.waitUntilCompleted();
    }

    /// Load a live session's complete post-prefill state (`0..n_toks`) directly from `self.eng`'s
    /// exact `kv_full`, `kv_swa`, and `cs` buffers on GPU (used for exact E2E 48-layer logit KL verification!).
    pub fn load_session_from_engine(&mut self, sess: &mut Kv256kSession, n_toks: usize, next_tok: u32) -> Result<(), String> {
        assert!(n_toks >= 1 && n_toks < self.cfg.max_ctx);
        let pt = self.cfg.page_toks;

        // 1. Allocate & broadcast FP16 CS slot from Engine's fp32 CS (`prefix_len = n_toks`)
        let cs_slot = match sess.cs_slot {
            Some(c) => c,
            None => {
                let c = self.alloc_cs()?;
                sess.cs_slot = Some(c);
                c
            }
        };
        unsafe { *(self.meta[1].contents().as_ptr() as *mut u32) = cs_slot; }
        {
            let cb = self.eng.gpu.cmd();
            let e = Enc::new(&cb);
            e.pso(&self.eng.gpu.pso("k_broadcast_cs_hcs"));
            e.buf(&self.eng.cs, 0, 4);
            e.buf(&self.meta[1], 0, 7);
            e.buf(&self.cs_pool_hcs, 0, 8);
            let ca = CopyCsArgs {
                dst_off: 0, n: 1, max_batch: self.cfg.max_active_batch as u32, prefix_len: n_toks as u32,
            };
            e.bytes(&ca, 5);
            e.dispatch((NCH.div_ceil(256), 1, N_SLOT), 256);
            e.end();
            cb.commit();
            cb.waitUntilCompleted();
        }

        // 2. Load SWA window (`[first_pos .. n_toks)`) from `eng.kv_swa` into Tiered SWA or legacy SWA
        let nkeys_swa = n_toks.min(EXT);
        let first_pos_swa = n_toks - nkeys_swa;
        if self.cfg.use_tiered_swa {
            while sess.swa_pages.len() < SWA_INNER_BLKS {
                let p = self.alloc_swa_page()?;
                sess.swa_pages.push(p);
            }
            while sess.swa_outer_pages.len() < SWA_OUTER_BLKS {
                let p = self.alloc_swa_outer_page()?;
                sess.swa_outer_pages.push(p);
            }
            let last_blk = (n_toks - 1) / KV_BLK;
            let inner_start_blk = last_blk.saturating_sub(SWA_INNER_BLKS - 1);
            let inner_p0 = inner_start_blk * KV_BLK;
            let mp = self.meta[1].contents().as_ptr() as *mut u32;
            let in_off = 0u32;
            let out_off = 16u32;
            unsafe {
                for (k, &p) in sess.swa_pages.iter().enumerate() { *mp.add(in_off as usize + k) = p; }
                for (k, &p) in sess.swa_outer_pages.iter().enumerate() { *mp.add(out_off as usize + k) = p; }
            }
            let cb = self.eng.gpu.cmd();
            let e = Enc::new(&cb);
            e.pso(&self.eng.gpu.pso("k_swa_load_from_engine"));
            e.buf(&self.eng.kv_swa.borrow(), 0, 1);
            e.buf(&self.swa_inner, 0, 3);
            e.buf(&self.swa_inner_sc, 0, 6);
            e.buf(&self.meta[1], 0, 7);
            e.buf(&self.swa_outer_comp, 0, 8);
            e.buf(&self.swa_outer_sc, 0, 9);
            let la = SwaLoadEngineArgs {
                src_cap: self.eng.swa_cap.get() as u32,
                first_pos: first_pos_swa as u32,
                nkeys: nkeys_swa as u32,
                inner_p0: inner_p0 as u32,
                inner_cap: self.swa_inner_cap as u32,
                outer_cap: self.swa_outer_cap as u32,
                inner_fp16: self.cfg.swa_inner_fp16 as u32,
                slot_start: 0,
                inner_pt_off: in_off,
                outer_pt_off: out_off,
                pad0: 0, pad1: 0,
            };
            e.bytes(&la, 5);
            e.dispatch((nkeys_swa, NKV, N_SWA), 128);
            e.end();
            cb.commit();
            cb.waitUntilCompleted();
        } else {
            while sess.swa_pages.len() < SWA_RING_BLKS {
                let p = self.alloc_swa_page()?;
                sess.swa_pages.push(p);
            }
            let mp = self.meta[1].contents().as_ptr() as *mut u32;
            unsafe { for (k, &p) in sess.swa_pages.iter().enumerate() { *mp.add(k) = p; } }
            #[repr(C)] #[derive(Clone, Copy)]
            struct CopyToPagedArgsLocal { n_pos: u32, pos0: u32, src_cap: u32, is_swa: u32, pool_cap: u32, pt_off: u32, pt_mod: u32, suf0: u32 }
            let cb = self.eng.gpu.cmd();
            let e = Enc::new(&cb);
            e.pso(&self.eng.gpu.pso("k_copy_kv_to_paged"));
            e.buf(&self.eng.kv_swa.borrow(), 0, 3);
            e.buf(&self.meta[1], 0, 7);
            e.buf(&self.swa_pool, 0, 8);
            let ca = CopyToPagedArgsLocal {
                n_pos: nkeys_swa as u32, pos0: first_pos_swa as u32, src_cap: self.eng.swa_cap.get() as u32,
                is_swa: 1, pool_cap: self.swa_cap as u32, pt_off: 0, pt_mod: SWA_RING_BLKS as u32, suf0: first_pos_swa as u32,
            };
            e.bytes(&ca, 5);
            e.dispatch((nkeys_swa, NKV, N_SWA), 32);
            e.end();
            cb.commit();
            cb.waitUntilCompleted();
        }

        // 3. Compress Engine's full KV [0..n_toks) into WHT pages across all 8 full slots (with K_DC_MEAN subtraction!)
        let n_pages = n_toks.div_ceil(pt);
        let mut hot_pids: Vec<u32> = Vec::with_capacity(n_pages);
        for lp in 0..n_pages {
            while sess.logical_pages.len() <= lp {
                let lid = self.radix.alloc_logical_id();
                sess.logical_pages.push(lid);
                sess.sum_pages.push(u32::MAX);
                sess.page_hashes.push(None);
            }
            let pid = self.lru.alloc_for_logical(sess.logical_pages[lp], true, true, &mut self.cold, &self.kv_comp, &self.kv_scales)?;
            hot_pids.push(pid);
        }
        let mp = self.meta[1].contents().as_ptr() as *mut u32;
        unsafe { for (i, &p) in hot_pids.iter().enumerate() { *mp.add(i) = p; } }
        {
            let cb = self.eng.gpu.cmd();
            let e = Enc::new(&cb);
            e.pso(&self.eng.gpu.pso("k_fp16_to_wht_compress"));
            e.buf(&self.eng.kv_full.borrow(), 0, 3);
            e.buf(&self.kv_comp, 0, 6);
            e.buf(&self.meta[1], 0, 7);
            e.buf(&self.kv_scales, 0, 8);
            e.buf(&self.k_dc_mean, 0, 10);
            for layer_slot in 0..N_SLOT {
                let (is_swa, fi) = self.eng.slot_kind[layer_slot];
                if is_swa { continue; }
                let fa = Fp16ToWhtArgs {
                    src_cap: self.eng.full_cap.get() as u32,
                    src_pos0: 0,
                    slot_idx: fi as u32,
                    layer_slot: layer_slot as u32,
                    pool_cap: self.lru.hot_cap as u32,
                    page_toks: pt as u32,
                    kv_bits: self.cfg.kv_bits,
                    dst_pos0: 0,
                    pid_list_off: 0,
                    n_toks: n_toks as u32,
                    use_dc_mean: self.use_dc_mean as u32,
                    pad1: 0,
                };
                e.bytes(&fa, 5);
                e.dispatch((n_toks, NKV, 1), 128);
            }
            e.end();
            cb.commit();
            cb.waitUntilCompleted();
        }

        // 4. Summarize all completed full pages (`0..n_toks / pt`) on GPU
        let n_completed = n_toks / pt;
        if n_completed > 0 {
            let pid_off = 0usize;
            let spid_off = n_completed.div_ceil(16) * 16;
            for lp in 0..n_completed {
                let spid = self.alloc_sum_page()?;
                sess.sum_pages[lp] = spid;
                unsafe {
                    *mp.add(pid_off + lp) = hot_pids[lp];
                    *mp.add(spid_off + lp) = spid;
                }
            }
            let cb = self.eng.gpu.cmd();
            let e = Enc::new(&cb);
            let kname = if self.cfg.use_wht2_sketch { "k_page_summarize_wht2" } else { "k_page_summarize_kv" };
            e.pso(&self.eng.gpu.pso(kname));
            e.buf(&self.kv_comp, 0, 3);
            e.buf(&self.kv_scales, 0, 6);
            e.buf(&self.meta[1], 0, 7);
            e.buf(&self.kv_sum, 0, 8);
            let sa = PageSumArgs {
                pool_cap: self.lru.hot_cap as u32,
                slot_idx: 0,
                page_toks: pt as u32,
                kv_bits: self.cfg.kv_bits,
                n_pages: n_completed as u32,
                pid_list_off: pid_off as u32,
                spid_list_off: spid_off as u32,
                cnt_list_off: u32::MAX,
                sum_pool_cap: self.cfg.sum_pool_pages as u32,
                pad0: 0, pad1: 0, pad2: 0,
            };
            e.bytes(&sa, 5);
            e.dispatch((n_completed, NKV, N_FULL), 128);
            e.end();
            cb.commit();
            cb.waitUntilCompleted();
        }

        for pid in hot_pids { self.lru.unpin(pid); }
        sess.pos = n_toks;
        sess.next_tok = next_tok;
        Ok(())
    }

    /// Streaming chunked 256K+ prefill! Processes arbitrary `toks.len()` in `chunk_size` chunks,
    /// deduplicating prefix blocks through `RadixPageTable`, summarizing completed pages on GPU,
    /// and evicting cold far pages directly to `F_NOCACHE` NVMe so RAM NEVER exceeds `max_mem_bytes`!
    pub fn prefill_streaming(&mut self, sess: &mut Kv256kSession, toks: &[u32], chunk_size: usize) -> Result<Duration, String> {
        assert!(!toks.is_empty());
        assert!(self.cfg.chunk_cap >= 64, "Kv256kConfig::chunk_cap must be >= 64 for streaming prefill");
        let t0 = Instant::now();
        let chunk_lim = chunk_size.clamp(64, self.cfg.chunk_cap).min(self.cfg.max_ctx);
        if sess.cs_slot.is_none() {
            let cs = self.alloc_cs()?;
            sess.cs_slot = Some(cs);
            self.zero_cs_slots(&[cs]);
        }
        let cs_slot = sess.cs_slot.unwrap();
        let pt = self.cfg.page_toks;

        let mut p0 = 0usize;
        while p0 < toks.len() {
            // Check Radix Prefix Page Table for exact full-page hits at page boundaries (when pos == 0..prefix)
            if sess.pos % pt == 0 && p0 + pt <= toks.len() && sess.pos == p0 {
                let h = RadixPageTable::rolling_page_hash(sess.last_prefix_hash, &toks[p0..p0 + pt]);
                if let Some((lid, spid)) = self.radix.acquire_existing(h) {
                    sess.logical_pages.push(lid);
                    sess.sum_pages.push(spid);
                    sess.page_hashes.push(Some(h));
                    sess.last_prefix_hash = h;
                    sess.pos += pt;
                    p0 += pt;
                    continue;
                }
            }

            let t = (toks.len() - p0).min(chunk_lim);
            let tp = self.toks[0].contents().as_ptr() as *mut i32;
            for (i, &tk) in toks[p0..p0 + t].iter().enumerate() {
                unsafe { *tp.add(i) = tk as i32; }
            }

            let first_page = sess.pos / pt;
            let last_page = (sess.pos + t - 1) / pt;
            let local_p0_page = if sess.pos + 1 > EXT { (sess.pos + 1 - EXT) / pt } else { 0 };
            let mut pinned_now: Vec<u32> = Vec::new();
            let mut chunk_pids: Vec<u32> = Vec::new();
            let mut local_pids: Vec<u32> = Vec::new();

            for lp in local_p0_page..=last_page {
                while sess.logical_pages.len() <= lp {
                    let lid = self.radix.alloc_logical_id();
                    sess.logical_pages.push(lid);
                    sess.sum_pages.push(u32::MAX);
                    sess.page_hashes.push(None);
                }
                let lid = sess.logical_pages[lp];
                let pid = self.lru.alloc_for_logical(lid, true, true, &mut self.cold, &self.kv_comp, &self.kv_scales)?;
                pinned_now.push(pid);
                local_pids.push(pid);
                if lp >= first_page { chunk_pids.push(pid); }
            }
            for p in (sess.pos..sess.pos + t).step_by(KV_BLK) {
                if self.cfg.use_tiered_swa {
                    let in_idx = (p / KV_BLK) % SWA_INNER_BLKS;
                    while sess.swa_pages.len() <= in_idx {
                        let sp = self.alloc_swa_page()?;
                        sess.swa_pages.push(sp);
                    }
                    if p >= SWA_INNER_BLKS * KV_BLK {
                        let out_idx = ((p / KV_BLK) - SWA_INNER_BLKS) % SWA_OUTER_BLKS;
                        while sess.swa_outer_pages.len() <= out_idx {
                            let op = self.alloc_swa_outer_page()?;
                            sess.swa_outer_pages.push(op);
                        }
                    }
                } else {
                    let swa_idx = (p / KV_BLK) % SWA_RING_BLKS;
                    while sess.swa_pages.len() <= swa_idx {
                        let sp = self.alloc_swa_page()?;
                        sess.swa_pages.push(sp);
                    }
                }
            }

            let mp = self.meta[0].contents().as_ptr() as *mut u32;
            unsafe {
                *mp.add(self.ml.seq_cs as usize) = cs_slot;
                for (i, &pid) in chunk_pids.iter().enumerate() { *mp.add(self.ml.write_pid as usize + i) = pid; }
                for (i, &pid) in local_pids.iter().enumerate() {
                    *mp.add(self.ml.local_pt as usize + i) = pid;
                    *mp.add(self.ml.far_pt as usize + i) = i as u32;
                }
                for lp in 0..local_p0_page { *mp.add(self.ml.sum_pt as usize + lp) = sess.sum_pages[lp]; }
            }

            let cb = self.eng.gpu.cmd();
            let e = Enc::new(&cb);
            e.buf(&self.eng.w, 0, 0);
            e.buf(&self.eng.pb, 0, 1);
            e.buf(&self.act, 0, 2);
            e.buf(&self.kv_comp, 0, 3);
            e.buf(&self.cs_pool_hcs, 0, 4);
            e.buf(&self.toks[0], 0, 6);
            e.buf(&self.meta[0], 0, 7);
            e.buf(&self.kv_sum, 0, 8);
            e.buf(&self.k_dc_mean, 0, 10);

            let da = &self.da;
            let ea = EmbedArgs {
                emb_lo: self.eng.tok_embd.off as u32, emb_hi: (self.eng.tok_embd.off >> 32) as u32, row_bytes: self.eng.tok_embd.row_bytes,
                norm_off: self.eng.tok_norm, out0: da.vec(0), out1: da.vec(0), tok_index: 0, eps: self.eng.eps,
            };
            enc_go(self.eng, &e, 0, C_EMBED, "k_embed", &ea, (t, 1, 1), 256);

            let mut va = VecAlloc { next: 1, n: B_NVEC };
            let mut blocks: Vec<usize> = Vec::new();
            let mut cur = 0usize;
            let cur_p0 = sess.pos;

            let run_chunk_layer = |slot: usize, ip: usize, cur_v: usize, blks: &mut Vec<usize>, msg: Option<(usize, usize)>, va: &mut VecAlloc, pinned: &[usize]| -> usize {
                let l = &self.eng.layers[ip];
                let ctl = msg.map(|_| &self.eng.loops[ip - LOOP_START]);
                let es = E as u32;

                // 1. Attention input mix
                let norms: [i32; 3] = if l.swa { [l.attn_norm; 3] } else { [l.q_inp_norm, l.k_inp_norm, l.v_inp_norm] };
                let mut st: Vec<u32> = blks.iter().map(|&v| da.vec(v)).collect();
                st.push(da.vec(cur_v));
                let nq = if l.attn_res_query >= 0 { l.n_attn_q } else { 1 };
                if let Some(lw) = ctl {
                    let (b0, b1) = msg.unwrap();
                    let mut ja = self.eng.mix_job_off(&st, l.attn_res_query, nq, &da.inp[..nq as usize], &[], es);
                    let mut jm = self.eng.mix_job_off(&[da.vec(b0), da.vec(b1)], lw.msg_query as i32, 4, &da.msg, &[], es);
                    if l.swa { ja.hcopy[0] = [da.cath[0] as i32, da.cath[1] as i32, da.cath[2] as i32]; }
                    else { for j in 0..3 { ja.hcopy[j][0] = da.cath[j] as i32; } }
                    for j in 0..4 { jm.hcopy[j][0] = (da.cath[j] + (E / 2) as u32) as i32; }
                    enc_mix_b(self.eng, &e, 0, C_MIXA, &[ja, jm], t);
                    let jobs: Vec<(usize, u32, u32, u32)> = (0..3).map(|j| (j, if l.swa { da.inp[0] } else { da.inp[j] }, da.msg[j], norms[j] as u32)).collect();
                    enc_ctrl_b(self.eng, &e, 0, da, lw, &jobs, t, &[da.xh[0], da.xh[1], da.xh[2]]);
                } else {
                    let nn: Vec<i32> = if l.swa { vec![l.attn_norm] } else { norms.to_vec() };
                    let mut ja = self.eng.mix_job_off(&st, l.attn_res_query, nq, &da.inp[..nq as usize], &nn, es);
                    for q in 0..nq as usize { ja.hout[q] = da.xh[q] as i32; }
                    ja.nof32 = 0xF;
                    enc_mix_b(self.eng, &e, 0, C_MIXA, &[ja], t);
                }

                // 2. QKV projections
                let xi: [usize; 3] = if ctl.is_some() || !l.swa { [0, 1, 2] } else { [0, 0, 0] };
                let mut hs = vec![
                    Engine::hseg(&l.wq, da.xh[xi[0]], da.q),
                    Engine::hseg(&l.wgate, da.xh[xi[0]], da.gate),
                    Engine::hseg(&l.wrel, da.xh[xi[0]], da.rel),
                    Engine::hseg(&l.wsink, da.xh[xi[0]], da.sink),
                    Engine::hseg(&l.wk, da.xh[xi[1]], da.k),
                    Engine::hseg(&l.wv, da.xh[xi[2]], da.v),
                ];
                hs[0].hout = 1; hs[1].hout = 1;
                if let (Some(kp), Some(vp)) = (&l.kconvp, &l.vconvp) {
                    hs.push(Engine::hseg(kp, da.xh[xi[1]], da.kdyn));
                    hs.push(Engine::hseg(vp, da.xh[xi[2]], da.vdyn));
                }
                enc_gemm_h(self.eng, &e, 0, C_QKV, &hs, t);

                // 3. WHT compressed KV write + causal chunked WHT attention
                let (is_swa, slot_idx) = self.eng.slot_kind[slot];
                let (comp_buf, sc_buf, pool_cap, eff_slot, wp_off, lpt_off) = if is_swa {
                    (
                        self.chunk_swa_comp.as_ref().expect("chunk_swa_comp"),
                        self.chunk_swa_scales.as_ref().expect("chunk_swa_scales"),
                        self.chunk_swa_cap,
                        0usize,
                        self.ml.far_pt + (first_page - local_p0_page) as u32,
                        self.ml.far_pt,
                    )
                } else {
                    (&self.kv_comp, &self.kv_scales, self.lru.hot_cap, slot_idx, self.ml.write_pid, self.ml.local_pt)
                };
                let dc_on = !is_swa && self.use_dc_mean;
                let sink_for_attn = if dc_on { da.eff_sink } else { da.sink };
                e.buf(comp_buf, 0, 3);
                e.buf(sc_buf, 0, 6);
                let kw = WhtKvPrepArgs {
                    b: t as u32, pool_cap: pool_cap as u32, slot_idx: eff_slot as u32,
                    page_toks: pt as u32, kv_bits: self.cfg.kv_bits,
                    layer_slot: slot as u32, max_batch: self.cfg.max_active_batch as u32,
                    is_prefill: 1, p0: cur_p0 as u32,
                    seq_pos_off: 0, seq_cs_off: self.ml.seq_cs, write_pid_off: wp_off,
                    q_off: da.q, k_off: da.k, v_off: da.v, r_off: da.rel,
                    kdyn_off: if l.swa { -1 } else { da.kdyn as i32 }, vdyn_off: if l.swa { -1 } else { da.vdyn as i32 },
                    kconv_t_w: l.k_conv_t, vconv_t_w: l.v_conv_t, qclip_w: l.q_clip, relclip_w: l.rel_clip,
                    log_alpha: self.eng.log_alpha, log_floor: self.eng.log_floor,
                    qp_off: da.qp, rp_off: da.rp, sink_off: da.sink, eff_sink_off: sink_for_attn,
                    use_dc_mean: dc_on as u32, pad0: 0, pad1: 0, pad2: 0,
                };
                enc_go(self.eng, &e, 0, C_KVPREP, "k_kvprep_wht_write", &kw, (t, NKV, 1), 128);

                for br in 0..2u32 {
                    let su = StateHcsArgs {
                        x_off: if br == 0 { da.k } else { da.v }, ldx: (NKV * HD) as u32,
                        layer_slot: slot as u32, branch: br, max_batch: self.cfg.max_active_batch as u32,
                        nch: (NKV * HD) as u32, seq_cs: cs_slot, p0: cur_p0 as u32, t: t as u32, pad0: 0, pad1: 0, pad2: 0,
                    };
                    enc_go(self.eng, &e, 0, C_KVPREP, "k_state_upd_hcs", &su, (NKV * HD / 256, 1, 1), 256);
                }

                let ca = ChunkWhtAttnArgs {
                    t: t as u32, p0: cur_p0 as u32, pool_cap: pool_cap as u32,
                    slot_idx: eff_slot as u32, layer_slot: slot as u32,
                    page_toks: pt as u32, kv_bits: self.cfg.kv_bits,
                    sum_pool_cap: self.cfg.sum_pool_pages as u32, use_sum_pt: 1,
                    qp_off: da.qp, rp_off: da.rp, sink_off: sink_for_attn, gate_off: da.gate, out_off: da.agh, relbias_w: l.rel_bias,
                    local_p0_page: local_p0_page as u32,
                    local_pt_off: lpt_off, sum_pt_off: self.ml.sum_pt, sum_base: 0,
                    use_centroids: (!is_swa) as u32,
                    sum_format: self.cfg.use_wht2_sketch as u32, pad0: 0, pad1: 0, pad2: 0,
                };
                enc_go(self.eng, &e, 0, C_ATTN, "k_attn_wht_chunk", &ca, (NKV, t, 1), 128);

                // 4. wo + FP16 CS chunk post-conv + ring state update
                enc_gemm_h(self.eng, &e, 0, C_WO, &[Engine::hseg(&l.wo, da.agh, da.wo_y)], t);
                let block_start = ip % BLOCK == 0;
                let mut live: Vec<usize> = blks.clone(); live.push(cur_v); live.extend_from_slice(pinned);
                let mid = va.alloc(&live);
                let cv = ChunkConvHcsArgs {
                    y_off: da.wo_y, ldy: es, res_off: da.vec(cur_v), out_off: da.vec(mid),
                    layer_slot: slot as u32, branch: 2, max_batch: self.cfg.max_active_batch as u32,
                    convw_t_off: l.attn_post_conv_t, has_res: (!block_start) as u32,
                    seq_cs: cs_slot, p0: cur_p0 as u32, t: t as u32,
                };
                enc_go(self.eng, &e, 0, C_WOPOST, "k_postconv_hcs_chunk", &cv, (E / 256, t, 1), 256);
                let su = StateHcsArgs {
                    x_off: da.wo_y, ldx: es, layer_slot: slot as u32, branch: 2,
                    max_batch: self.cfg.max_active_batch as u32, nch: es, seq_cs: cs_slot, p0: cur_p0 as u32, t: t as u32,
                    pad0: 0, pad1: 0, pad2: 0,
                };
                enc_go(self.eng, &e, 0, C_WOPOST, "k_state_upd_hcs", &su, (E / 256, 1, 1), 256);
                if block_start { blks.push(cur_v); }

                // 5. FFN input + 6-class MoE + combine + FP16 CS chunk post-conv
                let mut st: Vec<u32> = blks.iter().map(|&v| da.vec(v)).collect(); st.push(da.vec(mid));
                if let Some(lw) = ctl {
                    let mut jf = self.eng.mix_job_off(&st, l.ffn_res_query as i32, 1, &[da.fin], &[], es);
                    jf.hcopy[0][0] = da.cath[3] as i32;
                    enc_mix_b(self.eng, &e, 0, C_MIXF, &[jf], t);
                    enc_ctrl_b(self.eng, &e, 0, da, lw, &[(3, da.fin, da.msg[3], l.ffn_norm)], t, &[da.fh]);
                } else {
                    let mut jf = self.eng.mix_job_off(&st, l.ffn_res_query as i32, 1, &[da.fin], &[l.ffn_norm as i32], es);
                    jf.hout[0] = da.fh as i32; jf.nof32 = 1;
                    enc_mix_b(self.eng, &e, 0, C_MIXF, &[jf], t);
                }
                enc_moe_dispatch(self.eng, &e, &self.act, 0, da, l, t, self.moe_cfg);
                #[repr(C)] #[derive(Clone, Copy)]
                struct MoeCombArgsLocal { yd_off: u32, inv_off: u32, sel_off: u32, out_off: u32 }
                let mca = MoeCombArgsLocal { yd_off: da.yd, inv_off: da.inv, sel_off: da.sel, out_off: da.ffn_y };
                enc_go(self.eng, &e, 0, C_COMB2, "k_moe_comb_h", &mca, (t, E.div_ceil(1024), 1), 256);
                let mut live: Vec<usize> = blks.clone(); live.push(mid); live.extend_from_slice(pinned);
                let out = va.alloc(&live);
                let cv = ChunkConvHcsArgs {
                    y_off: da.ffn_y, ldy: es, res_off: da.vec(mid), out_off: da.vec(out),
                    layer_slot: slot as u32, branch: 3, max_batch: self.cfg.max_active_batch as u32,
                    convw_t_off: l.ffn_post_conv_t, has_res: 1, seq_cs: cs_slot, p0: cur_p0 as u32, t: t as u32,
                };
                enc_go(self.eng, &e, 0, C_FFNPOST, "k_postconv_hcs_chunk", &cv, (E / 256, t, 1), 256);
                let su = StateHcsArgs {
                    x_off: da.ffn_y, ldx: es, layer_slot: slot as u32, branch: 3,
                    max_batch: self.cfg.max_active_batch as u32, nch: es, seq_cs: cs_slot, p0: cur_p0 as u32, t: t as u32,
                    pad0: 0, pad1: 0, pad2: 0,
                };
                enc_go(self.eng, &e, 0, C_FFNPOST, "k_state_upd_hcs", &su, (E / 256, 1, 1), 256);
                out
            };

            for ip in 0..LOOP_START { cur = run_chunk_layer(ip, ip, cur, &mut blocks, None, &mut va, &[]); }
            let prelude = cur;
            let prelude_blocks = blocks.clone();
            let mut bank = (0usize, 0usize);
            for pass in 0..N_PASS {
                if pass > 0 { cur = prelude; blocks = prelude_blocks.clone(); }
                let msg = if pass > 0 { Some(bank) } else { None };
                let pinned: Vec<usize> = { let mut p = vec![prelude]; if pass > 0 { p.push(bank.0); p.push(bank.1); } p };
                for ip in LOOP_START..LOOP_END {
                    cur = run_chunk_layer(slot_of(pass, ip), ip, cur, &mut blocks, msg, &mut va, &pinned);
                }
                bank = (blocks[prelude_blocks.len() + 1], cur);
            }
            for ip in LOOP_END..N_PHYS { cur = run_chunk_layer(ip, ip, cur, &mut blocks, None, &mut va, &[]); }
            e.end();
            cb.commit();
            cb.waitUntilCompleted();

            let new_pos = sess.pos + t;
            let completed_end = new_pos / pt;
            for lp in first_page..completed_end {
                if sess.sum_pages[lp] == u32::MAX {
                    let spid = self.alloc_sum_page()?;
                    let hot_pid = self.lru.peek_hot(sess.logical_pages[lp]).unwrap();
                    self.summarize_completed_page(hot_pid, spid, pt as u32);
                    sess.sum_pages[lp] = spid;
                    let p_start = lp * pt;
                    if p_start >= p0 && p_start + pt <= p0 + t {
                        let h = RadixPageTable::rolling_page_hash(sess.last_prefix_hash, &toks[p_start..p_start + pt]);
                        self.radix.insert_new(h, sess.logical_pages[lp], spid, pt as u32);
                        sess.page_hashes[lp] = Some(h);
                        sess.last_prefix_hash = h;
                    }
                }
            }

            for pid in pinned_now { self.lru.unpin(pid); }
            sess.pos = new_pos;
            p0 += t;
        }
        Ok(t0.elapsed())
    }

    /// Internal shared 48-layer decode encoder used by BOTH `upload_and_encode_decode_step` (batched argmax)
    /// and `step_single_logits` (B=1 full-vocab KL verification).
    fn encode_decode_step_impl(
        &mut self, sessions: &mut [Kv256kSession], meta_idx: usize, tok_in_idx: usize, tok_out_idx: usize,
        skip: u32, run_argmax: bool,
    ) -> Result<(CmdBuf, Vec<u32>), String> {
        let b_len = sessions.len();
        assert!(b_len >= 1 && b_len <= self.cfg.max_active_batch);
        let pt = self.cfg.page_toks;
        let mut pinned_pids: Vec<u32> = Vec::new();
        let mut write_pids: Vec<u32> = Vec::with_capacity(b_len);
        let mut grad_jobs: Vec<(u32, u32)> = Vec::new();

        for s in sessions.iter_mut() {
            let wp = self.ensure_session_decode_pages(s)?;
            pinned_pids.push(wp);
            write_pids.push(wp);
            // If Tiered SWA crosses a 32-token block boundary past inner capacity (pos >= 288 && pos % 32 == 0),
            // graduate the oldest inner block `(cur_blk - 9)` into outer WHT-K2V2 BEFORE overwriting its inner slot!
            if self.cfg.use_tiered_swa && s.pos >= SWA_INNER_BLKS * KV_BLK && (s.pos % KV_BLK) == 0 {
                let cur_blk = s.pos / KV_BLK;
                let grad_blk = cur_blk - SWA_INNER_BLKS;
                let src_pid = s.swa_pages[grad_blk % SWA_INNER_BLKS];
                let dst_pid = s.swa_outer_pages[grad_blk % SWA_OUTER_BLKS];
                grad_jobs.push((src_pid, dst_pid));
            }
            let nfar = if s.pos + 1 > EXT { (s.pos + 1 - EXT) / pt } else { 0 };
            let cur_p = s.pos / pt;
            for lp in nfar..=cur_p {
                if lp < s.logical_pages.len() {
                    let pid = self.lru.alloc_for_logical(s.logical_pages[lp], true, false, &mut self.cold, &self.kv_comp, &self.kv_scales)?;
                    pinned_pids.push(pid);
                }
            }
        }

        let mp = self.meta[meta_idx].contents().as_ptr() as *mut u32;
        let tp = self.toks[tok_in_idx].contents().as_ptr() as *mut i32;
        let ml = self.ml;
        unsafe {
            for (j, &(sp, dp)) in grad_jobs.iter().enumerate() {
                *mp.add(ml.swa_grad_src as usize + j) = sp;
                *mp.add(ml.swa_grad_dst as usize + j) = dp;
            }
            for (b, s) in sessions.iter().enumerate() {
                *tp.add(b) = s.next_tok as i32;
                *mp.add(ml.seq_pos as usize + b) = s.pos as u32;
                *mp.add(ml.seq_suf as usize + b) = s.pos as u32;
                *mp.add(ml.seq_cs as usize + b) = s.cs_slot.expect("session missing active cs_slot");
                let nfar = if s.pos + 1 > EXT { (s.pos + 1 - EXT) / pt } else { 0 };
                *mp.add(ml.seq_nfar as usize + b) = nfar as u32;
                *mp.add(ml.write_pid as usize + b) = write_pids[b];

                if self.cfg.use_tiered_swa {
                    let ipt = mp.add(ml.swa_inner_pt as usize + b * SWA_INNER_BLKS);
                    for (k, &pid) in s.swa_pages.iter().take(SWA_INNER_BLKS).enumerate() { *ipt.add(k) = pid; }
                    let opt = mp.add(ml.swa_outer_pt as usize + b * SWA_OUTER_BLKS);
                    for (k, &pid) in s.swa_outer_pages.iter().take(SWA_OUTER_BLKS).enumerate() { *opt.add(k) = pid; }
                } else {
                    let spt = mp.add(ml.swa_pt as usize + b * SWA_RING_BLKS);
                    for (k, &pid) in s.swa_pages.iter().take(SWA_RING_BLKS).enumerate() { *spt.add(k) = pid; }
                }

                let lpt = mp.add(ml.local_pt as usize + b * ml.local_pt_stride);
                let cur_p = s.pos / pt;
                for (k, lp) in (nfar..=cur_p).enumerate() {
                    if k < ml.local_pt_stride && lp < s.logical_pages.len() {
                        *lpt.add(k) = self.lru.peek_hot(s.logical_pages[lp]).unwrap_or(write_pids[b]);
                    }
                }

                let fpt = mp.add(ml.far_pt as usize + b * ml.far_pt_stride);
                let sumpt = mp.add(ml.sum_pt as usize + b * ml.far_pt_stride);
                for lp in 0..nfar.min(ml.far_pt_stride) {
                    *fpt.add(lp) = self.lru.peek_hot(s.logical_pages[lp]).unwrap_or(u32::MAX);
                    *sumpt.add(lp) = s.sum_pages[lp];
                }
            }
        }

        let cb = self.eng.gpu.cmd();
        let e = Enc::new(&cb);
        e.buf(&self.eng.w, 0, 0);
        e.buf(&self.eng.pb, 0, 1);
        e.buf(&self.act, 0, 2);
        e.buf(&self.cs_pool_hcs, 0, 4);
        e.buf(&self.meta[meta_idx], 0, 7);
        e.buf(&self.k_dc_mean, 0, 10);

        // Graduate any Tiered-SWA blocks crossing age 256 into outer WHT-K2V2 BEFORE layer 0 overwrites inner!
        if !grad_jobs.is_empty() && (skip & (1 << C_KVPREP) == 0) {
            e.buf(&self.swa_inner, 0, 3);
            e.buf(&self.swa_inner_sc, 0, 6);
            e.buf(&self.swa_outer_comp, 0, 8);
            e.buf(&self.swa_outer_sc, 0, 9);
            let ca = SwaCompressArgs {
                n_jobs: grad_jobs.len() as u32,
                inner_cap: self.swa_inner_cap as u32,
                outer_cap: self.swa_outer_cap as u32,
                slot_start: 0,
                inner_fp16: self.cfg.swa_inner_fp16 as u32,
                src_pid_off: ml.swa_grad_src,
                dst_pid_off: ml.swa_grad_dst,
                pad0: 0,
            };
            enc_go(self.eng, &e, skip, C_KVPREP, "k_swa_compress_outer", &ca, (grad_jobs.len() * KV_BLK, NKV, N_SWA), 128);
        }

        // Bind input tokens at buffer(6) right before k_embed!
        e.buf(&self.toks[tok_in_idx], 0, 6);
        let da = &self.da;
        let ea = EmbedArgs {
            emb_lo: self.eng.tok_embd.off as u32, emb_hi: (self.eng.tok_embd.off >> 32) as u32, row_bytes: self.eng.tok_embd.row_bytes,
            norm_off: self.eng.tok_norm, out0: da.vec(0), out1: da.vec(0), tok_index: 0, eps: self.eng.eps,
        };
        enc_go(self.eng, &e, skip, C_EMBED, "k_embed", &ea, (b_len, 1, 1), 256);

        let mut va = VecAlloc { next: 1, n: B_NVEC };
        let mut blocks: Vec<usize> = Vec::new();
        let mut cur = 0usize;
        let max_pos = sessions.iter().map(|s| s.pos).max().unwrap_or(0);
        let use_wht2 = self.cfg.use_wht2_sketch;
        let use_tiered_swa = self.cfg.use_tiered_swa;
        let dc_on = self.use_dc_mean;

        let run_dec_layer = |slot: usize, ip: usize, cur_v: usize, blks: &mut Vec<usize>, msg: Option<(usize, usize)>, va: &mut VecAlloc, pinned: &[usize]| -> usize {
            let l = &self.eng.layers[ip];
            let ctl = msg.map(|_| &self.eng.loops[ip - LOOP_START]);
            let es = E as u32;

            // 1. Attention inputs
            let norms: [i32; 3] = if l.swa { [l.attn_norm; 3] } else { [l.q_inp_norm, l.k_inp_norm, l.v_inp_norm] };
            let mut st: Vec<u32> = blks.iter().map(|&v| da.vec(v)).collect();
            st.push(da.vec(cur_v));
            let nq = if l.attn_res_query >= 0 { l.n_attn_q } else { 1 };
            if let Some(lw) = ctl {
                let (b0, b1) = msg.unwrap();
                let mut ja = self.eng.mix_job_off(&st, l.attn_res_query, nq, &da.inp[..nq as usize], &[], es);
                let mut jm = self.eng.mix_job_off(&[da.vec(b0), da.vec(b1)], lw.msg_query as i32, 4, &da.msg, &[], es);
                if l.swa { ja.hcopy[0] = [da.cath[0] as i32, da.cath[1] as i32, da.cath[2] as i32]; }
                else { for j in 0..3 { ja.hcopy[j][0] = da.cath[j] as i32; } }
                for j in 0..4 { jm.hcopy[j][0] = (da.cath[j] + (E / 2) as u32) as i32; }
                enc_mix_b(self.eng, &e, skip, C_MIXA, &[ja, jm], b_len);
                let jobs: Vec<(usize, u32, u32, u32)> = (0..3).map(|j| (j, if l.swa { da.inp[0] } else { da.inp[j] }, da.msg[j], norms[j] as u32)).collect();
                enc_ctrl_b(self.eng, &e, skip, da, lw, &jobs, b_len, &[da.xh[0], da.xh[1], da.xh[2]]);
            } else {
                let nn: Vec<i32> = if l.swa { vec![l.attn_norm] } else { norms.to_vec() };
                let mut ja = self.eng.mix_job_off(&st, l.attn_res_query, nq, &da.inp[..nq as usize], &nn, es);
                for q in 0..nq as usize { ja.hout[q] = da.xh[q] as i32; }
                ja.nof32 = 0xF;
                enc_mix_b(self.eng, &e, skip, C_MIXA, &[ja], b_len);
            }

            // 2. QKV projections
            let xi: [usize; 3] = if ctl.is_some() || !l.swa { [0, 1, 2] } else { [0, 0, 0] };
            let mut hs = vec![
                Engine::hseg(&l.wq, da.xh[xi[0]], da.q),
                Engine::hseg(&l.wgate, da.xh[xi[0]], da.gate),
                Engine::hseg(&l.wrel, da.xh[xi[0]], da.rel),
                Engine::hseg(&l.wsink, da.xh[xi[0]], da.sink),
                Engine::hseg(&l.wk, da.xh[xi[1]], da.k),
                Engine::hseg(&l.wv, da.xh[xi[2]], da.v),
            ];
            hs[0].hout = 1; hs[1].hout = 1;
            if let (Some(kp), Some(vp)) = (&l.kconvp, &l.vconvp) {
                hs.push(Engine::hseg(kp, da.xh[xi[1]], da.kdyn));
                hs.push(Engine::hseg(vp, da.xh[xi[2]], da.vdyn));
            }
            enc_gemm_h(self.eng, &e, skip, C_QKV, &hs, b_len);

            // 3. Attention:
            //    - SWA (40 slots): Tiered SWA (inner-256 + outer-768 WHT-K2V2) or legacy paged SWA
            //    - Full (8 slots): WHT-K3V2/KV4/KV2 + K_DC_MEAN (sink-exact) + WHT-2bit-sketch/Quest + Sparse WHT Attn!
            let (is_swa, slot_idx) = self.eng.slot_kind[slot];
            if is_swa {
                if use_tiered_swa {
                    e.buf(&self.swa_inner, 0, 3);
                    e.buf(&self.swa_inner_sc, 0, 6);
                    e.buf(&self.swa_outer_comp, 0, 8);
                    e.buf(&self.swa_outer_sc, 0, 9);
                    let ta = TieredSwaAttnArgs {
                        b: b_len as u32,
                        inner_cap: self.swa_inner_cap as u32,
                        outer_cap: self.swa_outer_cap as u32,
                        slot_idx: slot_idx as u32,
                        inner_fp16: self.cfg.swa_inner_fp16 as u32,
                        layer_slot: slot as u32,
                        max_batch: self.cfg.max_active_batch as u32,
                        pad0: 0,
                        seq_pos_off: ml.seq_pos,
                        seq_cs_off: ml.seq_cs,
                        inner_pt_off: ml.swa_inner_pt,
                        outer_pt_off: ml.swa_outer_pt,
                        q_off: da.q, k_off: da.k, v_off: da.v, r_off: da.rel,
                        sink_off: da.sink, gate_off: da.gate, out_off: da.agh, pad1: 0,
                        kconv_t_w: l.k_conv_t, vconv_t_w: l.v_conv_t, qclip_w: l.q_clip, relclip_w: l.rel_clip,
                        relbias_w: l.rel_bias, pad2: 0, pad3: 0, pad4: 0,
                    };
                    enc_go(self.eng, &e, skip, C_ATTN, "k_attn_swa_tiered_b", &ta, (NKV, b_len, 1), 128);
                } else {
                    e.buf(&self.swa_pool, 0, 3);
                    let _ = (max_pos + 1).min(EXT);
                    let nsplit = if b_len * NKV >= 64 { 1usize } else { 64usize.div_ceil(b_len * NKV).min(da.max_split).max(1) };
                    let pa = PagedAttnArgs {
                        b: b_len as u32, pool_cap: self.swa_cap as u32, slot_idx: slot_idx as u32, is_swa: 1,
                        pt_stride: SWA_RING_BLKS as u32, pt_mod: SWA_RING_BLKS as u32,
                        seq_pos_off: ml.seq_pos, seq_suf_off: ml.seq_suf, seq_cs_off: ml.seq_cs, pt_off: ml.swa_pt,
                        layer_slot: slot as u32, max_batch: self.cfg.max_active_batch as u32,
                        q_off: da.q, k_off: da.k, v_off: da.v, r_off: da.rel, sink_off: da.sink,
                        kdyn_off: -1, vdyn_off: -1,
                        kconv_t_w: l.k_conv_t, vconv_t_w: l.v_conv_t, qclip_w: l.q_clip, relclip_w: l.rel_clip, relbias_w: l.rel_bias,
                        log_alpha: self.eng.log_alpha, log_floor: self.eng.log_floor,
                        gate_off: da.gate, out_off: da.agh, part_off: da.part, nsplit: nsplit as u32, prefix_len: 0,
                        qp_off: da.qp, rp_off: da.rp, pad0: 0,
                    };
                    enc_go(self.eng, &e, skip, C_ATTN, "k_attn_paged_b_hcs", &pa, (nsplit, NKV, b_len), 128);
                    if nsplit > 1 {
                        let ca = CombineBArgs { part_off: da.part, nsplit: nsplit as u32, gate_off: da.gate, out_off: da.agh };
                        enc_go(self.eng, &e, skip, C_COMB, "k_attn_combine_b", &ca, (NH, b_len, 1), 128);
                    }
                }
            } else {
                e.buf(&self.kv_comp, 0, 3);
                e.buf(&self.kv_scales, 0, 6);
                e.buf(&self.kv_sum, 0, 8);
                e.buf(&self.aux, 0, 9);
                let eff_sink = if dc_on { da.eff_sink } else { da.sink };
                let kw = WhtKvPrepArgs {
                    b: b_len as u32, pool_cap: self.lru.hot_cap as u32, slot_idx: slot_idx as u32,
                    page_toks: pt as u32, kv_bits: self.cfg.kv_bits,
                    layer_slot: slot as u32, max_batch: self.cfg.max_active_batch as u32,
                    is_prefill: 0, p0: 0,
                    seq_pos_off: ml.seq_pos, seq_cs_off: ml.seq_cs, write_pid_off: ml.write_pid,
                    q_off: da.q, k_off: da.k, v_off: da.v, r_off: da.rel,
                    kdyn_off: da.kdyn as i32, vdyn_off: da.vdyn as i32,
                    kconv_t_w: l.k_conv_t, vconv_t_w: l.v_conv_t, qclip_w: l.q_clip, relclip_w: l.rel_clip,
                    log_alpha: self.eng.log_alpha, log_floor: self.eng.log_floor,
                    qp_off: da.qp, rp_off: da.rp, sink_off: da.sink, eff_sink_off: eff_sink,
                    use_dc_mean: dc_on as u32, pad0: 0, pad1: 0, pad2: 0,
                };
                enc_go(self.eng, &e, skip, C_KVPREP, "k_kvprep_wht_write", &kw, (b_len, NKV, 1), 128);

                let qs = QuestScoreArgs {
                    b: b_len as u32, slot_idx: slot_idx as u32, sum_pool_cap: self.cfg.sum_pool_pages as u32,
                    max_pages: self.al.max_pages as u32, top_k: self.cfg.top_k_pages as u32,
                    topk_stride: self.al.topk_stride as u32, mask_words: self.al.mask_words as u32, use_sum_pt: 1,
                    qp_off: da.qp, sink_off: eff_sink,
                    seq_nfar_off: ml.seq_nfar, sum_pt_off: ml.sum_pt, sum_pt_stride: ml.far_pt_stride as u32, sum_base_off: ml.sum_base,
                    scores_off: self.al.scores_off, topk_off: self.al.topk_off, topk_cnt_off: self.al.topk_cnt_off,
                    ret_mask_off: self.al.ret_mask_off, head_mask_off: self.al.head_mask_off, pad0: 0,
                };
                let score_k = if use_wht2 { "k_score_pages_wht2" } else { "k_score_pages_quest" };
                // C_FFNPOST is unused in fused decode -> gives 100% isolated scorer timing across all 8 full layers!
                enc_go(self.eng, &e, skip, C_FFNPOST, score_k, &qs, (NKV, b_len, 1), 256);

                let sa = SparseWhtAttnArgs {
                    b: b_len as u32, pool_cap: self.lru.hot_cap as u32, slot_idx: slot_idx as u32, layer_slot: slot as u32,
                    page_toks: pt as u32, kv_bits: self.cfg.kv_bits, sum_pool_cap: self.cfg.sum_pool_pages as u32, use_sum_pt: 1,
                    use_centroids: self.cfg.use_centroids as u32,
                    local_pt_stride: ml.local_pt_stride as u32, far_pt_stride: ml.far_pt_stride as u32, sum_pt_stride: ml.far_pt_stride as u32,
                    topk_stride: self.al.topk_stride as u32, mask_words: self.al.mask_words as u32,
                    qp_off: da.qp, rp_off: da.rp, sink_off: eff_sink, gate_off: da.gate, out_off: da.agh, relbias_w: l.rel_bias,
                    seq_pos_off: ml.seq_pos, seq_nfar_off: ml.seq_nfar,
                    local_pt_off: ml.local_pt, far_pt_off: ml.far_pt, sum_pt_off: ml.sum_pt, sum_base_off: ml.sum_base,
                    topk_off: self.al.topk_off, topk_cnt_off: self.al.topk_cnt_off,
                    ret_mask_off: self.al.ret_mask_off, head_mask_off: self.al.head_mask_off,
                    sum_format: use_wht2 as u32, pad1: 0,
                };
                // C_WOPOST is unused in fused decode -> gives 100% isolated 8-full-layer sparse WHT attention timing!
                enc_go(self.eng, &e, skip, C_WOPOST, "k_attn_sparse_wht_b", &sa, (NKV, b_len, 1), 128);
            }

            // 4. wo + FP16 CS post-conv + residual
            let block_start = ip % BLOCK == 0;
            let mut live: Vec<usize> = blks.clone(); live.push(cur_v); live.extend_from_slice(pinned);
            let mid = va.alloc(&live);
            let wo = WoPostArgs {
                w_lo: l.wo.off as u32, w_hi: (l.wo.off >> 32) as u32, row_bytes: l.wo.row_bytes, wtype: l.wo.ty,
                nrows: l.wo.rows, k: l.wo.k, xh_off: da.agh, out_off: da.vec(mid),
                res_off: da.vec(cur_v), has_res: (!block_start) as u32,
                layer_slot: slot as u32, max_batch: self.cfg.max_active_batch as u32, convw_t_off: l.attn_post_conv_t,
                b: b_len as u32, seq_cs_off: ml.seq_cs, seq_pos_off: ml.seq_pos,
            };
            if b_len > 64 { enc_go(self.eng, &e, skip, C_WO, "k_wo_post_sg8_hcs", &wo, (b_len.div_ceil(128), E / 64, 1), 256); }
            else { enc_go(self.eng, &e, skip, C_WO, "k_wo_post_sg4_hcs", &wo, (1, E / 64, 1), 128); }
            if block_start { blks.push(cur_v); }

            // 5. FFN input + 6-class MoE + FP16 CS combine
            let mut st: Vec<u32> = blks.iter().map(|&v| da.vec(v)).collect(); st.push(da.vec(mid));
            if let Some(lw) = ctl {
                let mut jf = self.eng.mix_job_off(&st, l.ffn_res_query as i32, 1, &[da.fin], &[], es);
                jf.hcopy[0][0] = da.cath[3] as i32;
                enc_mix_b(self.eng, &e, skip, C_MIXF, &[jf], b_len);
                enc_ctrl_b(self.eng, &e, skip, da, lw, &[(3, da.fin, da.msg[3], l.ffn_norm)], b_len, &[da.fh]);
            } else {
                let mut jf = self.eng.mix_job_off(&st, l.ffn_res_query as i32, 1, &[da.fin], &[l.ffn_norm as i32], es);
                jf.hout[0] = da.fh as i32; jf.nof32 = 1;
                enc_mix_b(self.eng, &e, skip, C_MIXF, &[jf], b_len);
            }
            enc_moe_dispatch(self.eng, &e, &self.act, skip, da, l, b_len, self.moe_cfg);
            let mut live: Vec<usize> = blks.clone(); live.push(mid); live.extend_from_slice(pinned);
            let out = va.alloc(&live);
            let cp = MoeCombPostArgs {
                yd_off: da.yd, inv_off: da.inv, res_off: da.vec(mid), out_off: da.vec(out),
                layer_slot: slot as u32, max_batch: self.cfg.max_active_batch as u32, convw_t_off: l.ffn_post_conv_t,
                seq_cs_off: ml.seq_cs, seq_pos_off: ml.seq_pos, pad0: 0, pad1: 0, pad2: 0,
            };
            enc_go(self.eng, &e, skip, C_COMB2, "k_moe_comb_post_hcs", &cp, (b_len, E.div_ceil(1024), 1), 256);
            out
        };

        for ip in 0..LOOP_START { cur = run_dec_layer(ip, ip, cur, &mut blocks, None, &mut va, &[]); }
        let prelude = cur;
        let prelude_blocks = blocks.clone();
        let mut bank = (0usize, 0usize);
        for pass in 0..N_PASS {
            if pass > 0 { cur = prelude; blocks = prelude_blocks.clone(); }
            let msg = if pass > 0 { Some(bank) } else { None };
            let pinned: Vec<usize> = { let mut p = vec![prelude]; if pass > 0 { p.push(bank.0); p.push(bank.1); } p };
            for ip in LOOP_START..LOOP_END {
                cur = run_dec_layer(slot_of(pass, ip), ip, cur, &mut blocks, msg, &mut va, &pinned);
            }
            bank = (blocks[prelude_blocks.len() + 1], cur);
        }
        for ip in LOOP_END..N_PHYS { cur = run_dec_layer(ip, ip, cur, &mut blocks, None, &mut va, &[]); }

        let mut st: Vec<u32> = blocks.iter().map(|&v| da.vec(v)).collect(); st.push(da.vec(cur));
        let mut jo = self.eng.mix_job_off(&st, self.eng.output_res_query as i32, 1, &[da.fin], &[self.eng.output_norm as i32], E as u32);
        jo.hout[0] = da.fh as i32; jo.nof32 = 1;
        enc_mix_b(self.eng, &e, skip, C_OMIX, &[jo], b_len);

        if run_argmax {
            e.buf(&self.toks[tok_out_idx], 0, 6);
            let w = &self.eng.output;
            let nvch = self.eng.n_vocab.div_ceil(LM_VCHUNK);
            let la = LmHeadArgs {
                w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty,
                n_vocab: self.eng.n_vocab as u32, k: w.k, xh_off: da.fh, part_off: da.am_part,
                b: b_len as u32, nvch: nvch as u32, vstart: 0, vcount: 0,
            };
            if b_len > 64 { enc_go(self.eng, &e, skip, C_LMHEAD, "k_lmhead_argmax_sg8", &la, (b_len.div_ceil(128), nvch, 1), 256); }
            else { enc_go(self.eng, &e, skip, C_LMHEAD, "k_lmhead_argmax_sg4", &la, (1, nvch, 1), 128); }
            let aa = ArgmaxBatchArgs { part_off: da.am_part, nvch: nvch as u32, out_base: 0, pad: 0 };
            enc_go(self.eng, &e, skip, C_ARGMAX, "k_argmax_batch", &aa, (b_len, 1, 1), 128);
        }
        e.end();
        Ok((cb, pinned_pids))
    }

    /// Upload metadata for a batch of active 256K sessions and encode ONE complete 48-layer sparse-WHT decode step
    /// (6-class MoE + FP16 CS + Tiered SWA + DC-mean + WHT-2bit-sketch/Quest + WHT-K3V2/KV4/KV2 sparse attention + fused argmax).
    pub fn upload_and_encode_decode_step(
        &mut self, sessions: &mut [Kv256kSession], meta_idx: usize, tok_in_idx: usize, tok_out_idx: usize, skip: u32,
    ) -> Result<(CmdBuf, Vec<u32>), String> {
        self.encode_decode_step_impl(sessions, meta_idx, tok_in_idx, tok_out_idx, skip, true)
    }

    /// Execute one complete 48-layer GPU decode step for a single session (`max_active_batch == 1`),
    /// extract full-vocab un-softcapped logits via `k_lmhead_slice_sg4`, advance `sess.pos += 1`,
    /// and summarize any newly completed full page!
    pub fn step_single_logits(&mut self, sess: &mut Kv256kSession) -> Result<Vec<f32>, String> {
        assert_eq!(self.da.cap, 1, "step_single_logits requires arena cap == 1");
        let (cb, pinned) = self.encode_decode_step_impl(std::slice::from_mut(sess), 0, 0, 1, 0, false)?;
        cb.commit();
        cb.waitUntilCompleted();
        self.unpin_pids(&pinned);

        // Extract full n_vocab logits in slices of LM_VSLICE_256K via k_lmhead_slice_sg4
        let nv = self.eng.n_vocab;
        let w = &self.eng.output;
        let mut out = vec![0.0f32; nv];
        let slice_off = self.da.lm_slice;
        let mut v0 = 0usize;
        while v0 < nv {
            let vc = (nv - v0).min(LM_VSLICE_256K);
            let cb2 = self.eng.gpu.cmd();
            let e2 = Enc::new(&cb2);
            e2.buf(&self.eng.w, 0, 0);
            e2.buf(&self.act, 0, 2);
            let la = LmHeadArgs {
                w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty,
                n_vocab: nv as u32, k: w.k, xh_off: self.da.fh, part_off: slice_off,
                b: 1, nvch: 0, vstart: v0 as u32, vcount: vc as u32,
            };
            e2.pso(&self.eng.gpu.pso("k_lmhead_slice_sg4"));
            e2.bytes(&la, 5);
            e2.dispatch((1, vc.div_ceil(64), 1), 128);
            e2.end();
            cb2.commit();
            cb2.waitUntilCompleted();
            unsafe {
                let src = (self.act.contents().as_ptr() as *const f32).add(slice_off as usize);
                std::ptr::copy_nonoverlapping(src, out.as_mut_ptr().add(v0), vc);
            }
            v0 += vc;
        }

        sess.pos += 1;
        let pt = self.cfg.page_toks;
        if sess.pos % pt == 0 {
            let lp = sess.pos / pt - 1;
            if lp < sess.sum_pages.len() && sess.sum_pages[lp] == u32::MAX {
                if let Some(hot_pid) = self.lru.peek_hot(sess.logical_pages[lp]) {
                    let spid = self.alloc_sum_page()?;
                    self.summarize_completed_page(hot_pid, spid, pt as u32);
                    sess.sum_pages[lp] = spid;
                }
            }
        }
        Ok(out)
    }

    pub fn unpin_pids(&mut self, pids: &[u32]) {
        for &p in pids { self.lru.unpin(p); }
    }
}

// ----------------------------------------------------------------------------- KL / Softcap / Needle Helpers for E2E Verification
#[inline]
fn softcap_256k(x: f32, c: f32) -> f32 {
    if c > 0.0 { c * (x / c).tanh() } else { x }
}

#[inline]
fn argmax_256k(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0
}

fn logit_kl_256k(ref_lg: &[f32], test_lg: &[f32]) -> f64 {
    let lse = |v: &[f32]| -> f64 {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        let s: f64 = v.iter().map(|&x| (x as f64 - m).exp()).sum();
        m + s.ln()
    };
    let (lr, lt) = (lse(ref_lg), lse(test_lg));
    let mut kl = 0.0f64;
    for (&r, &t) in ref_lg.iter().zip(test_lg.iter()) {
        let log_pr = r as f64 - lr;
        let log_pt = t as f64 - lt;
        kl += log_pr.exp() * (log_pr - log_pt);
    }
    kl.max(0.0)
}

/// Build a non-repetitive realistic needle-in-a-haystack prompt targeting `target_toks` tokens
/// (planted facts sit in positions `< 600` = far zone so sparse retrieval is genuinely tested!).
fn build_e2e_prompt_256k(tk: &Tokenizer, target_toks: usize) -> Vec<u32> {
    let header = "<|message_system|><|content_text|>Thinking effort level: 0.2<|end_message|>\
<|message_user|><|content_text|>Carefully read the archived telemetry dossier below and answer the final question using exact records.\n\n\
[ARCHIVE RECORD A-01] Facility Aurora-7 vault passkey: COBALT-91482-DELTA. Chief engineer: Dr. Helena Vance. Primary coolant pressure limit: 418.7 kPa.\n\
[ARCHIVE RECORD A-02] Orbital relay station Meridian-4 frequency allocation: 14.725 GHz, backup transponder code KESTREL-603.\n\
[ARCHIVE RECORD A-03] Subsurface sensor array Borealis-9 calibration offset: -0.0427 mV, commissioned on October 14, 2031.\n\n";
    let footer = "\n\nQuestion: State the exact vault passkey for Facility Aurora-7, the primary coolant pressure limit, and the backup transponder code for Meridian-4.<|end_message|>";

    let head_toks = tk.encode(header, true);
    let foot_toks = tk.encode(footer, true);
    let body_budget = target_toks.saturating_sub(head_toks.len() + foot_toks.len());

    let topics = [
        "hydrological survey", "spectral interferometry", "cryogenic valve timing",
        "metallurgical fatigue", "geothermal heat exchange", "photonic routing",
        "atmospheric lidar", "acoustic beamforming", "magnetospheric drift",
        "pelagic sediment cores", "seismic waveguide", "orbital debris tracking",
    ];
    let regions = [
        "Sector North-3", "Basin East-11", "Ridge West-6", "Shelf South-9",
        "Plateau Central-2", "Trench Outer-14", "Corridor High-8", "Atoll Far-5",
    ];
    let mut body = String::with_capacity(body_budget * 5);
    let mut idx = 0usize;
    let mut body_toks = Vec::new();
    while body_toks.len() < body_budget {
        body.clear();
        for _ in 0..8 {
            let t = topics[(idx * 5 + 3) % topics.len()];
            let r = regions[(idx * 7 + 1) % regions.len()];
            let val = 100.0 + ((idx * 37 + 19) % 890) as f64 * 0.13;
            let code = 1000 + (idx * 733 + 191) % 8999;
            body.push_str(&format!(
                "Log #{idx:04} ({r}): Routine {t} telemetry recorded nominal baseline {val:.2} units under protocol R-{code}; variance remained within seasonal tolerance bands.\n"
            ));
            idx += 1;
        }
        let chunk = tk.encode(&body, false);
        let take = chunk.len().min(body_budget - body_toks.len());
        body_toks.extend_from_slice(&chunk[..take]);
    }
    let mut out = Vec::with_capacity(head_toks.len() + body_toks.len() + foot_toks.len());
    out.extend_from_slice(&head_toks);
    out.extend_from_slice(&body_toks);
    out.extend_from_slice(&foot_toks);
    out
}

// ============================================================================================================
// PART 3: Built-in End-to-End 256K Verification & Benchmark Suite (`run_bench_256k`)
// ============================================================================================================

/// End-to-end verification & benchmark of the upgraded 256K+ Tiered Sparse-Retrieval KV Engine:
/// 1. WHT-128 KV4 / KV3 / K3V2 / KV2 + Per-Head DC Key Mean GPU compression & cosine verification on real tokens.
/// 2. WHT-2bit-sketch (`k_score_pages_wht2`) vs Quest (`k_score_pages_quest`) side-by-side at full `ctx_len` (256K!).
/// 3. Real end-to-end 48-layer Logit KL & Argmax verification on real prompt tokens (`n = 2048..4096`) vs exact FP16!
/// 4. Radix Prefix Page Table dedup + Hot RAM LRU + `F_NOCACHE` NVMe Cold Store spill & reload test.
/// 5. End-to-end 48-layer batched decode at `ctx_len` across `B = 1, 16, 64, 96, 128` inside `--max-mem` (12 GiB!).
/// 6. Wave-Swap multi-instance overhead (`F_NOCACHE` + Async Prefetcher) with compact Tiered SWA + `KvPageWht2`!
pub fn run_bench_256k(
    eng: &Engine, tk: &Tokenizer, n_instances: usize, ctx_len: usize,
    top_k_pages: usize, kv_bits: u32, max_mem_gb: Option<f64>,
) {
    assert!(eng.hgemm, "run_bench_256k requires M5 fp16 tensor path");
    let kv_bits = match kv_bits {
        4 | 3 | 33 | 2 => kv_bits,
        _ => 3, // default to winning WHT-K3V2 recipe
    };
    let ctx_len = ctx_len.max(2048);
    let top_k_pages = top_k_pages.clamp(1, 128);
    let max_mem = BatchEngine::safe_mem_limit(eng, max_mem_gb);
    let page_toks: usize = std::env::var("INK_PAGE_TOKS").ok().and_then(|v| v.parse().ok()).filter(|&p| p == 64 || p == 128)
        .unwrap_or(if ctx_len >= 131_072 { 128 } else { 64 });
    let duo = DuoHeadConfig::from_env();
    let n_pages_256k = ctx_len.div_ceil(page_toks);

    println!("============================================================================================");
    println!("  Inkling 256K+ Tiered Sparse-Retrieval KV Engine (WHT-K3V2 + WHT-2bit-Sketch + Tiered SWA)");
    println!("  ctx={} ({} pages x {} tok) | {} (kv_bits={}) | Top-K={} pages ({} far tok) | Duo={}/64 ret",
             ctx_len, n_pages_256k, page_toks, kv_mode_name(kv_bits), kv_bits, top_k_pages, top_k_pages * page_toks, duo.retrieval_count());
    println!("  n_instances={} | Tiered-SWA (inner 256 int8 + outer 768 WHT-K2V2 = {:.1} MiB) | cap={:.2} GiB",
             n_instances, Kv256kPool::tiered_swa_inst_bytes(false) as f64 / 1048576.0, max_mem as f64 / (1u64 << 30) as f64);
    println!("============================================================================================");

    // ---- Stage [1/6]: WHT-128 KV4 / KV3 / K3V2 / KV2 + Per-Head DC Key Mean GPU Compression Accuracy ----
    {
        let sample_text = "The quick brown fox jumps over the lazy dog while computing Walsh-Hadamard rotated key-value attention with per-head DC mean subtraction and Lloyd-Max bitplane quantization across 256K tokens on Apple Silicon Metal 4. In addition, structured records like COBALT-91482-DELTA and pressure 418.7 kPa demonstrate sharp attention peaks.";
        let toks = tk.encode(sample_text, false);
        let n_tok = toks.len().min(64).max(24);
        println!("\n[1/6] WHT-128 KV4 / KV3 / K3V2 / KV2 + Per-Head DC Mean GPU compression vs exact FP16 (n={n_tok} real tokens):");

        eng.reset();
        eng.reserve(n_tok + 1, false);
        eng.ensure_batch(n_tok);
        eng.set_tokens(0, &toks[..n_tok]);
        let cb = eng.gpu.cmd();
        eng.encode_prefill(&cb, 0, n_tok, true, true);
        cb.commit();
        cb.waitUntilCompleted();

        let mut mg = MemGuard::new(max_mem);
        let hot_cap = 4usize;
        let dc_buf = mg.try_alloc(&eng.gpu, N_FULL * NKV * HD * 2, "t1_dc").unwrap();
        let meta = mg.try_alloc(&eng.gpu, 1024, "t1_meta").unwrap();
        unsafe { *(meta.contents().as_ptr() as *mut u32) = 0; }

        // Compute GPU per-head DC key mean across all 8 full slots
        {
            let cb_dc = eng.gpu.cmd();
            let e_dc = Enc::new(&cb_dc);
            e_dc.pso(&eng.gpu.pso("k_compute_k_dc_mean"));
            e_dc.buf(&eng.kv_full.borrow(), 0, 3);
            e_dc.buf(&dc_buf, 0, 10);
            let ca = ComputeDcMeanArgs {
                src_cap: eng.full_cap.get() as u32, src_pos0: 0, n_toks: n_tok as u32, n_slots: N_FULL as u32,
            };
            e_dc.bytes(&ca, 5);
            e_dc.dispatch((NKV, N_FULL, 1), 128);
            e_dc.end();
            cb_dc.commit();
            cb_dc.waitUntilCompleted();
        }

        let full_layer_slot = eng.slot_kind.iter().position(|&(is_swa, _)| !is_swa).unwrap_or(5);
        let dc_ptr = dc_buf.contents().as_ptr() as *const u16;
        let mut mu_h0 = [0.0f32; HD];
        for d in 0..HD { mu_h0[d] = f16_to_f32(unsafe { *dc_ptr.add(d) }); }

        // Read original K & V (slot 0, head 0) + measure DC energy fraction
        let kv_full_ref = eng.kv_full.borrow();
        let src_ptr_u16 = kv_full_ref.contents().as_ptr() as *const u16;
        let src_ptr_i8 = kv_full_ref.contents().as_ptr() as *const i8;
        let src_cap = eng.full_cap.get();
        let read_orig = |is_v: usize, t: usize| -> [f32; HD] {
            let mut out = [0.0f32; HD];
            let base = (is_v * NKV) * src_cap * HD + t * HD;
            if !eng.kv8 {
                for d in 0..HD { out[d] = f16_to_f32(unsafe { *src_ptr_u16.add(base + d) }); }
            } else {
                let mut code = 0u32;
                for i in 0..8 {
                    let b = unsafe { *src_ptr_i8.add(base + i) as u8 };
                    code |= ((b & 1) as u32) << (7 - i);
                }
                let ks = (8 + (code & 7)) as f32 * f32::from_bits(((code >> 3) + 107) << 23);
                for d in 0..HD { out[d] = unsafe { *src_ptr_i8.add(base + d) as f32 } * ks; }
            }
            out
        };
        let (mut tot_e, mut dc_e) = (0.0f64, 0.0f64);
        for t in 0..n_tok {
            let k = read_orig(0, t);
            for d in 0..HD {
                tot_e += (k[d] as f64) * (k[d] as f64);
                dc_e += (mu_h0[d] as f64) * (mu_h0[d] as f64);
            }
        }
        println!("  Per-head DC Key Mean energy fraction (slot 0, head 0): {:.1}% of total key energy!", 100.0 * dc_e / tot_e.max(1e-12));

        let lm2 = [-1.5104176f32, -0.45278004, 0.45278004, 1.5104176];
        let lm3 = [-2.1519456f32, -1.3439093, -0.7560053, -0.24509418, 0.24509418, 0.7560053, 1.3439093, 2.1519456];
        let mut sgn_h0 = [1.0f32; HD];
        for d in 0..HD {
            let mut hh = ((full_layer_slot as u32).wrapping_mul(2654435761)) ^ ((d as u32).wrapping_mul(3266489917));
            hh ^= hh >> 16; hh = hh.wrapping_mul(0x85ebca6b); hh ^= hh >> 13;
            sgn_h0[d] = if (hh & 1) != 0 { -1.0 } else { 1.0 };
        }
        let inv_wht_unrot = |rot: &mut [f32; HD]| {
            let mut h = 1usize;
            while h < HD {
                for i in (0..HD).step_by(h * 2) {
                    for j in i..i + h {
                        let (x, y) = (rot[j], rot[j + h]);
                        rot[j] = x + y;
                        rot[j + h] = x - y;
                    }
                }
                h <<= 1;
            }
            let inv_sqrt = 1.0f32 / (HD as f32).sqrt();
            for d in 0..HD { rot[d] = sgn_h0[d] * rot[d] * inv_sqrt; }
        };

        for &bits in &[4u32, 33, 3, 2] {
            let pair_b = kv_pair_payload_bytes(bits);
            let cbuf = mg.try_alloc(&eng.gpu, N_FULL * hot_cap * NKV * 64 * pair_b, "t1_c").unwrap();
            let sbuf = mg.try_alloc(&eng.gpu, N_FULL * hot_cap * 2 * NKV * 64 * 2, "t1_s").unwrap();
            let (kb, vb): (u32, u32) = match bits { 4 => (4, 4), 33 => (3, 3), 3 => (3, 2), _ => (2, 2) };
            let (kw, vw): (usize, usize) = (if kb == 4 { 16 } else if kb == 3 { 12 } else { 8 }, if vb == 4 { 16 } else if vb == 3 { 12 } else { 8 });

            let mut cos_k_nodc = 0.0f64;
            let mut cos_k_dc = 0.0f64;
            let mut cos_v = 0.0f64;
            for use_dc in [false, true] {
                let cb2 = eng.gpu.cmd();
                let e2 = Enc::new(&cb2);
                e2.pso(&eng.gpu.pso("k_fp16_to_wht_compress"));
                e2.buf(&eng.kv_full.borrow(), 0, 3);
                e2.buf(&cbuf, 0, 6);
                e2.buf(&meta, 0, 7);
                e2.buf(&sbuf, 0, 8);
                e2.buf(&dc_buf, 0, 10);
                let fa = Fp16ToWhtArgs {
                    src_cap: src_cap as u32, src_pos0: 0, slot_idx: 0, layer_slot: full_layer_slot as u32,
                    pool_cap: hot_cap as u32, page_toks: 64, kv_bits: bits, dst_pos0: 0,
                    pid_list_off: 0, n_toks: n_tok as u32, use_dc_mean: use_dc as u32, pad1: 0,
                };
                e2.bytes(&fa, 5);
                e2.dispatch((n_tok, NKV, 1), 128);
                e2.end();
                cb2.commit();
                cb2.waitUntilCompleted();

                let c_ptr = cbuf.contents().as_ptr() as *const u32;
                let s_ptr = sbuf.contents().as_ptr() as *const u16;
                let unpack_vec = |is_v: usize, vbits: u32, words: usize, t: usize| -> [f32; HD] {
                    let nkp = NKV * 64;
                    let woff = if is_v == 0 { t * words } else { nkp * kw + t * words };
                    let vidx = if is_v == 0 { t } else { nkp + t };
                    let sc = f16_to_f32(unsafe { *s_ptr.add(vidx) });
                    let mut rot = [0.0f32; HD];
                    for d in 0..HD {
                        rot[d] = if vbits == 4 {
                            let w = unsafe { *c_ptr.add(woff + (d >> 3)) };
                            (((w >> ((d & 7) * 4)) & 0xF) as i32 - 8) as f32 * sc
                        } else if vbits == 3 {
                            let ch = d >> 5;
                            let bt = d & 31;
                            let b0 = (unsafe { *c_ptr.add(woff + ch) } >> bt) & 1;
                            let b1 = (unsafe { *c_ptr.add(woff + 4 + ch) } >> bt) & 1;
                            let b2 = (unsafe { *c_ptr.add(woff + 8 + ch) } >> bt) & 1;
                            lm3[(b0 | (b1 << 1) | (b2 << 2)) as usize] * sc
                        } else {
                            let w = unsafe { *c_ptr.add(woff + (d >> 4)) };
                            lm2[((w >> ((d & 15) * 2)) & 3) as usize] * sc
                        };
                    }
                    inv_wht_unrot(&mut rot);
                    rot
                };
                let (mut dk, mut nak, mut nbk) = (0.0f64, 0.0f64, 0.0f64);
                let (mut dv, mut nav, mut nbv) = (0.0f64, 0.0f64, 0.0f64);
                for t in 0..n_tok {
                    let orig_k = read_orig(0, t);
                    let mut rec_k = unpack_vec(0, kb, kw, t);
                    if use_dc { for d in 0..HD { rec_k[d] += mu_h0[d]; } }
                    for d in 0..HD {
                        dk += orig_k[d] as f64 * rec_k[d] as f64;
                        nak += orig_k[d] as f64 * orig_k[d] as f64;
                        nbk += rec_k[d] as f64 * rec_k[d] as f64;
                    }
                    if !use_dc {
                        let orig_v = read_orig(1, t);
                        let rec_v = unpack_vec(1, vb, vw, t);
                        for d in 0..HD {
                            dv += orig_v[d] as f64 * rec_v[d] as f64;
                            nav += orig_v[d] as f64 * orig_v[d] as f64;
                            nbv += rec_v[d] as f64 * rec_v[d] as f64;
                        }
                    }
                }
                let ck = if nak > 0.0 && nbk > 0.0 { dk / (nak.sqrt() * nbk.sqrt()) } else { 1.0 };
                if use_dc { cos_k_dc = ck; } else { cos_k_nodc = ck; cos_v = if nav > 0.0 && nbv > 0.0 { dv / (nav.sqrt() * nbv.sqrt()) } else { 1.0 }; }
            }
            let total_pair_b = pair_b + 4;
            let comp_ratio = 512.0 / total_pair_b as f64;
            let cold_256k_mib = (262_144usize * N_FULL * NKV * total_pair_b) as f64 / 1048576.0;
            println!("  {:>8}: K_cos={cos_k_nodc:.5} -> +DC_mean={cos_k_dc:.5} | V_cos={cos_v:.5} | {:>3} B/pair ({comp_ratio:.2}x vs FP16; 256K cold={cold_256k_mib:.0} MiB)",
                     kv_mode_name(bits), total_pair_b);
            mg.release(cbuf.length() + sbuf.length());
        }
        drop(kv_full_ref);
        eng.release_batch();
        eng.shrink_kv();
    }

    // ---- Stage [2/6]: WHT-2bit-Sketch vs Quest Page Scoring & Needle Recall Side-by-Side at Full 256K ----
    {
        println!("\n[2/6] WHT-2bit-Sketch vs Quest Page Scoring & Needle Recall at ctx={ctx_len} ({n_pages_256k} pages):");
        let mut mg = MemGuard::new(max_mem);
        let sum_q_buf = mg.try_alloc(&eng.gpu, n_pages_256k * Kv256kPool::summary_page_bytes_for(false), "t2_quest").unwrap();
        let sum_w_buf = mg.try_alloc(&eng.gpu, n_pages_256k * Kv256kPool::summary_page_bytes_for(true), "t2_wht2").unwrap();
        let al = SparseAuxLayout::new(1, n_pages_256k, top_k_pages);
        let aux = mg.try_alloc(&eng.gpu, al.total_u32 * 4, "t2_aux").unwrap();
        let act = mg.try_alloc(&eng.gpu, 65536, "t2_act").unwrap();
        let meta = mg.try_alloc(&eng.gpu, 65536, "t2_meta").unwrap();

        let needles = [17usize, n_pages_256k / 3, (2 * n_pages_256k) / 3, n_pages_256k.saturating_sub(23)];
        let qp_off = 0u32;
        let sink_off = (NH * HD) as u32;
        // Pack needle pattern (+1.51, -1.51 alternating -> 2-bit codes 3, 0, 3, 0 ... = 0x33333333 -> wait: (3<<0)|(0<<2)|(3<<4)|(0<<6)... = 0x33333333!)
        let needle_word_2b: u32 = 0x3333_3333u32; // dims 0,2,4.. -> code 3 (+1.51), dims 1,3,5.. -> code 0 (-1.51)
        unsafe {
            let qp = act.contents().as_ptr() as *mut u16;
            for h in 0..NH {
                for d in 0..HD {
                    *qp.add(h * HD + d) = f32_to_f16(if d % 2 == 0 { 0.5 } else { -0.5 });
                }
            }
            let sp = (act.contents().as_ptr() as *mut f32).add(sink_off as usize);
            for h in 0..NH { *sp.add(h) = 0.0; }
            let mp = meta.contents().as_ptr() as *mut u32;
            *mp.add(0) = n_pages_256k as u32;
            *mp.add(16) = 0;
            let ap = aux.contents().as_ptr() as *mut u32;
            *ap.add(al.head_mask_off as usize) = u32::MAX;
            *ap.add(al.head_mask_off as usize + 1) = u32::MAX;

            // Populate both Quest KvPageSummary and KvPageWht2 with identical planted needles + background noise!
            let q_p = sum_q_buf.contents().as_ptr() as *mut KvPageSummaryHost;
            let w_p = sum_w_buf.contents().as_ptr() as *mut KvPageWht2Host;
            for p in 0..n_pages_256k {
                let is_needle = needles.contains(&p);
                for g in 0..NKV {
                    let qd = &mut *q_p.add(p * NKV + g);
                    qd.sc_kmin = f32_to_f16(if is_needle { 0.02 } else { 0.004 });
                    qd.sc_kmax = f32_to_f16(if is_needle { 0.02 } else { 0.004 });
                    qd.sc_kmean = f32_to_f16(0.01);
                    qd.sc_vmean = f32_to_f16(0.01);
                    qd.count = page_toks as u32;
                    qd.page_id = p as u32;
                    for d in 0..HD {
                        let sgn: i8 = if d % 2 == 0 { 1 } else { -1 };
                        if is_needle {
                            qd.k_min[d] = 90 * sgn;
                            qd.k_max[d] = 120 * sgn;
                        } else {
                            let noise = (((p * 31 + d * 17 + g * 13) % 41) as i8) - 20;
                            qd.k_min[d] = noise.saturating_sub(10);
                            qd.k_max[d] = noise.saturating_add(10);
                        }
                        qd.k_mean[d] = 0;
                        qd.v_mean[d] = 0;
                    }

                    let wd = &mut *w_p.add(p * NKV + g);
                    wd.sc_kmean = f32_to_f16(0.01);
                    wd.sc_vmean = f32_to_f16(0.01);
                    wd.count = page_toks as u32;
                    wd.page_id = p as u32;
                    wd.pad0 = 0;
                    for i in 0..16 { wd.k_mean4[i] = 0x8888_8888; wd.v_mean4[i] = 0x8888_8888; }
                    for sb in 0..WHT2_NSUB {
                        if is_needle && sb == 3 {
                            // Needle resides in ONE 16-token sub-block (sb == 3) while other 7 sub-blocks are noise!
                            wd.sc_sk[sb] = f32_to_f16(1.0);
                            for w in 0..8 { wd.k_sk[sb][w] = needle_word_2b; }
                        } else {
                            wd.sc_sk[sb] = f32_to_f16(0.25);
                            for w in 0..8 {
                                let mut h = ((p * 2654435761) ^ (sb * 2246822519) ^ (w * 3266489917) ^ (g * 668265263)) as u32;
                                h ^= h >> 16; h = h.wrapping_mul(0x85ebca6b); h ^= h >> 13;
                                wd.k_sk[sb][w] = h;
                            }
                        }
                    }
                }
            }
        }

        let qs = QuestScoreArgs {
            b: 1, slot_idx: 0, sum_pool_cap: n_pages_256k as u32, max_pages: n_pages_256k as u32,
            top_k: top_k_pages as u32, topk_stride: al.topk_stride as u32, mask_words: al.mask_words as u32, use_sum_pt: 0,
            qp_off, sink_off, seq_nfar_off: 0, sum_pt_off: 0, sum_pt_stride: 0, sum_base_off: 16,
            scores_off: al.scores_off, topk_off: al.topk_off, topk_cnt_off: al.topk_cnt_off,
            ret_mask_off: al.ret_mask_off, head_mask_off: al.head_mask_off, pad0: 0,
        };
        let bench_scorer = |label: &str, kname: &str, buf: &Buf, mib: f64| {
            let mut times = Vec::new();
            for _ in 0..7 {
                let cb = eng.gpu.cmd();
                let e = Enc::new(&cb);
                e.pso(&eng.gpu.pso(kname));
                e.buf(&act, 0, 2); e.buf(&meta, 0, 7); e.buf(buf, 0, 8); e.buf(&aux, 0, 9);
                e.bytes(&qs, 5);
                e.dispatch((NKV, 1, 1), 256);
                e.end();
                cb.commit();
                cb.waitUntilCompleted();
                times.push((cb.GPUEndTime() - cb.GPUStartTime()) * 1e3);
            }
            let ap = aux.contents().as_ptr() as *const u32;
            let cnt = unsafe { *ap.add(al.topk_cnt_off as usize) } as usize;
            let topk_g0: Vec<usize> = (0..cnt).map(|i| unsafe { *ap.add(al.topk_off as usize + i) as usize }).collect();
            let recalled = needles.iter().filter(|n| topk_g0.contains(n)).count();
            let ms = median_f64(times);
            println!("  {label:<24}: {ms:.3} ms/slot (8-slot {:.3} ms) | index={mib:.1} MiB | needle recall={recalled}/{} (top-4={:?})",
                     ms * N_FULL as f64, needles.len(), &topk_g0[..cnt.min(4)]);
        };
        let quest_mib = (n_pages_256k * Kv256kPool::summary_page_bytes_for(false)) as f64 / 1048576.0;
        let wht2_mib = (n_pages_256k * Kv256kPool::summary_page_bytes_for(true)) as f64 / 1048576.0;
        let wht2_sk_only_mib = (n_pages_256k * N_FULL * NKV * (WHT2_NSUB * 34)) as f64 / 1048576.0;
        bench_scorer("Quest (k_min/k_max)", "k_score_pages_quest", &sum_q_buf, quest_mib);
        bench_scorer(&format!("WHT-2bit-sketch ({wht2_sk_only_mib:.0}M sk)"), "k_score_pages_wht2", &sum_w_buf, wht2_mib);
    }

    // ---- Stage [3/6]: Real End-to-End 48-Layer Logit KL & Argmax Verification vs Exact FP16 ----
    {
        let e2e_target: usize = std::env::var("INK_E2E_N").ok().and_then(|v| v.parse().ok())
            .unwrap_or(if eng.max_ctx >= 4200 { 4096 } else { 2048 })
            .min(eng.max_ctx.saturating_sub(64))
            .max(1536);
        let e2e_steps: usize = 16;
        let e2e_pt: usize = 64;
        let pt_toks = build_e2e_prompt_256k(tk, e2e_target);
        let n = pt_toks.len();
        let nfar_e2e = n.saturating_sub(EXT) / e2e_pt;
        let e2e_topk = (nfar_e2e / 4).clamp(2, 8).min(nfar_e2e.max(1));
        println!("\n[3/6] Real End-to-End 48-Layer Logit KL & Argmax Verification vs Exact Engine (n={n} tok, {nfar_e2e} far pages x {e2e_pt}, Top-K={e2e_topk} [{:.1}% active], {e2e_steps} steps):",
                 100.0 * e2e_topk as f64 / nfar_e2e.max(1) as f64);

        // Exact Engine prefill + reference 16-step decode
        let prev_tail = eng.tail.get();
        eng.tail.set(false);
        *eng.prune.borrow_mut() = None;
        *eng.qcap.borrow_mut() = None;
        eng.reset();
        eng.reserve(n + e2e_steps + 2, false);
        eng.ensure_batch(n.min(eng.chunk));
        eng.set_tokens(0, &pt_toks);
        let mut p0 = 0usize;
        while p0 < n {
            let t = (n - p0).min(eng.chunk);
            let cb = eng.gpu.cmd();
            eng.encode_prefill(&cb, p0, t, p0 + t == n, false);
            cb.commit();
            cb.waitUntilCompleted();
            p0 += t;
        }
        let first_tok = argmax_256k(eng.logits()) as u32;
        let snap = eng.snapshot();
        eng.release_batch();

        let mut ref_toks = Vec::with_capacity(e2e_steps);
        let mut ref_logits = Vec::with_capacity(e2e_steps);
        let mut cur_tok = first_tok;
        for i in 0..e2e_steps {
            eng.set_tokens(n + i, &[cur_tok]);
            let cb = eng.gpu.cmd();
            eng.encode_step(&cb, n + i, true, false);
            cb.commit();
            cb.waitUntilCompleted();
            let lg: Vec<f32> = eng.logits().iter().map(|&x| softcap_256k(x, eng.softcap)).collect();
            ref_toks.push(cur_tok);
            cur_tok = argmax_256k(&lg) as u32;
            ref_logits.push(lg);
        }
        eng.restore(&snap);

        println!("  {:>38} | {:>8} {:>9} {:>9} | {:>7}",
                 "configuration", "payload", "mean KL", "worst KL", "argmax");
        println!("  {}", "-".repeat(86));

        let mut eval_configs: Vec<(&str, u32, bool, bool, bool, bool)> = vec![
            // (label, bits, use_wht2_sketch, use_dc_mean, use_tiered_swa, swa_inner_fp16)
            ("WHT-K3V2 + WHT2-sk + DC + TierSWA(i8)",  3, true,  true,  true,  false),
            ("WHT-K3V2 + WHT2-sk + DC + TierSWA(f16)", 3, true,  true,  true,  true),
            ("WHT-K3V2 + Quest   + DC + TierSWA(i8)",  3, false, true,  true,  false),
            ("WHT-K3V2 + WHT2-sk (noDC) + TierSWA",    3, true,  false, true,  false),
        ];
        if kv_bits != 3 {
            eval_configs.push(("CLI kv_bits + WHT2-sk + DC + TierSWA", kv_bits, true, true, true, false));
        }

        for &(label, bits, use_wht2, use_dc, use_tswa, in_fp16) in &eval_configs {
            let n_pages_need = (n + e2e_steps + 64).div_ceil(e2e_pt);
            let cfg = Kv256kConfig {
                max_active_batch: 1,
                max_ctx: n + e2e_steps + 64,
                page_toks: e2e_pt,
                kv_bits: bits,
                top_k_pages: e2e_topk,
                use_centroids: true,
                use_wht2_sketch: use_wht2,
                use_tiered_swa: use_tswa,
                swa_inner_fp16: in_fp16,
                use_dc_mean: use_dc,
                duo: DuoHeadConfig::all_retrieval(),
                max_mem_bytes: max_mem,
                sum_pool_pages: n_pages_need + 4,
                hot_full_pages: n_pages_need + 4,
                chunk_cap: 0,
            };
            match Kv256kPool::try_new(eng, cfg) {
                Ok(mut pool) => {
                    if use_dc { pool.calibrate_dc_mean_from_engine(n); } else { pool.clear_dc_mean(); }
                    let mut sess = Kv256kSession::new(0);
                    pool.load_session_from_engine(&mut sess, n, ref_toks[0]).expect("load_session_from_engine");
                    let (mut sum_kl, mut worst_kl, mut agree) = (0.0f64, 0.0f64, 0usize);
                    for i in 0..e2e_steps {
                        sess.next_tok = ref_toks[i];
                        let raw = pool.step_single_logits(&mut sess).expect("step_single_logits");
                        let lg: Vec<f32> = raw.iter().map(|&x| softcap_256k(x, eng.softcap)).collect();
                        let kl = logit_kl_256k(&ref_logits[i], &lg);
                        if argmax_256k(&lg) == argmax_256k(&ref_logits[i]) { agree += 1; }
                        sum_kl += kl;
                        if kl > worst_kl { worst_kl = kl; }
                    }
                    let mean_kl = sum_kl / e2e_steps as f64;
                    println!("  {:>38} | {:>8} {:>9.5} {:>9.5} | {:>2}/{:<2}",
                             label, kv_mode_name(bits), mean_kl, worst_kl, agree, e2e_steps);
                }
                Err(err) => println!("  {:>38} | SKIPPED ({err})", label),
            }
        }
        eng.tail.set(prev_tail);
        eng.shrink_kv();
    }

    // ---- Stage [4/6]: Radix Dedup + Hot RAM LRU + F_NOCACHE NVMe Cold Store Spill & Reload ----
    {
        println!("\n[4/6] Radix Prefix Dedup + Hot RAM LRU + F_NOCACHE NVMe Cold Spill/Reload ({}):", kv_mode_name(kv_bits));
        let fp_before = proc_phys_footprint_bytes();
        let mut radix = RadixPageTable::new();
        let prefix_toks: Vec<u32> = (0..page_toks * 64).map(|i| 500 + (i as u32 * 29) % 40000).collect();
        let mut shared_lids = Vec::new();
        for inst in 0..128usize {
            let mut h = 0u64;
            for p in 0..64usize {
                h = RadixPageTable::rolling_page_hash(h, &prefix_toks[p * page_toks..(p + 1) * page_toks]);
                if let Some((lid, _)) = radix.acquire_existing(h) {
                    if inst == 0 { shared_lids.push(lid); }
                } else {
                    let lid = radix.alloc_logical_id();
                    radix.insert_new(h, lid, p as u32, page_toks as u32);
                    if inst == 0 { shared_lids.push(lid); }
                }
            }
        }
        let mut cold = NvmeColdStore::new(page_toks, kv_bits).expect("nvme cold store");
        let hot_cap = 32usize;
        let mut lru = HotRamLruCache::new(hot_cap);
        let mut mg = MemGuard::new(max_mem);
        let pair_b = kv_pair_payload_bytes(kv_bits);
        let kv_comp = mg.try_alloc(&eng.gpu, N_FULL * hot_cap * NKV * page_toks * pair_b, "t4_comp").unwrap();
        let kv_scales = mg.try_alloc(&eng.gpu, N_FULL * hot_cap * 2 * NKV * page_toks * 2, "t4_sc").unwrap();
        for i in 0..96u64 {
            let _ = lru.alloc_for_logical(1000 + i, false, true, &mut cold, &kv_comp, &kv_scales).unwrap();
        }
        let cold_req: Vec<u64> = (1000..1016u64).collect();
        let (hits, misses) = lru.demand_fetch_batch(&cold_req, &mut cold, &kv_comp, &kv_scales).unwrap();
        let fp_after = proc_phys_footprint_bytes();
        let wr_gbs = cold.bytes_written as f64 / cold.write_time.as_secs_f64().max(1e-6) / 1e9;
        let rd_gbs = cold.bytes_read as f64 / cold.read_time.as_secs_f64().max(1e-6) / 1e9;
        println!("  Radix dedup (128 inst x 64 prefix pages): unique={} total_refs={} ({:.1}x dedup!)",
                 radix.unique_pages(), radix.total_refs(), radix.dedup_ratio());
        println!("  LRU cold spill/reload: evictions={} | demand hits={hits} misses={misses} | F_NOCACHE write={wr_gbs:.2} GB/s read={rd_gbs:.2} GB/s",
                 lru.evictions);
        println!("  OOM check (F_NOCACHE): phys_footprint before={:.2} GiB -> after={:.2} GiB (delta {:+.1} MiB)",
                 fp_before as f64 / (1u64 << 30) as f64, fp_after as f64 / (1u64 << 30) as f64,
                 (fp_after as i64 - fp_before as i64) as f64 / 1048576.0);
    }

    // ---- Stage [5/6]: End-to-End 48-Layer Batched Decode at ctx = 256K across B = 1, 16, 64, 96, 128 ----
    println!("\n[5/6] End-to-End 48-Layer Batched Decode at ctx={ctx_len} ({}, WHT-2bit-sketch, Tiered-SWA, Top-K={top_k_pages}):",
             kv_mode_name(kv_bits));
    let swa_tiered_inst = Kv256kPool::tiered_swa_inst_bytes(false);
    let swa_legacy_fp16 = SWA_RING_BLKS * PagedKvPool::swa_page_bytes(2);
    let cs_per_inst = CS_SEQ_ELEMS * 2;
    let sum_wht2_inst = n_pages_256k * Kv256kPool::summary_page_bytes_for(true);
    let hot_local_per_inst = (EXT / page_toks + 1) * Kv256kPool::full_comp_page_bytes(page_toks, kv_bits);
    let hot_ret_16_inst = top_k_pages.min(16).min(n_pages_256k) * Kv256kPool::full_comp_page_bytes(page_toks, kv_bits);
    let active_inst = swa_tiered_inst + cs_per_inst + sum_wht2_inst + hot_local_per_inst + hot_ret_16_inst;
    let dense_fp16_inst = swa_legacy_fp16 + CS_SEQ_ELEMS * 4 + ctx_len * N_FULL * 2 * NKV * HD * 2;
    println!("  Per-256K-instance RAM (independent): TierSWA={:.1} MiB (vs {:.1} FP16) | FP16_CS={:.2} MiB | WHT2_Idx={:.1} MiB | HotLoc={:.1} MiB | HotRet={:.1} MiB => {:.1} MiB active ({:.1}x vs {:.0} MiB FP16!)",
             swa_tiered_inst as f64 / 1048576.0, swa_legacy_fp16 as f64 / 1048576.0,
             cs_per_inst as f64 / 1048576.0, sum_wht2_inst as f64 / 1048576.0,
             hot_local_per_inst as f64 / 1048576.0, hot_ret_16_inst as f64 / 1048576.0,
             active_inst as f64 / 1048576.0, dense_fp16_inst as f64 / active_inst as f64,
             dense_fp16_inst as f64 / 1048576.0);
    println!("  Estimated live independent 256K capacity inside {:.1} GiB: ~{} instances!",
             max_mem as f64 / (1u64 << 30) as f64,
             max_mem.saturating_sub(256 << 20) / (swa_tiered_inst + cs_per_inst + sum_wht2_inst + hot_local_per_inst + (4 * Kv256kPool::full_comp_page_bytes(page_toks, kv_bits))));

    println!("\n  {:>4} {:>12} | {:>7} {:>7} {:>7} {:>8} | {:>9} {:>8} | {:>9} {:>9}",
             "B", "mode", "wht2_ms", "satn_ms", "tswa_ms", "step_ms", "agg_tps", "per_tps", "pool_GiB", "rusage_GiB");
    println!("  {}", "-".repeat(108));

    for &b_len in &[1usize, 16, 64, 96, 128] {
        let min_local_pages = b_len * (EXT / page_toks + 2);
        let indep_sum_pages = b_len * n_pages_256k;
        let radix_sum_pages = n_pages_256k + b_len * 64;

        let est_base = |sum_p: usize, hot_p: usize| -> usize {
            let da = OptArena::new(b_len, eng.n_vocab, false);
            let ml = SparseMetaLayout::new(b_len, n_pages_256k, page_toks);
            let al = SparseAuxLayout::new(b_len, n_pages_256k, top_k_pages);
            da.end as usize * 4
                + b_len * cs_per_inst
                + (ml.total_u32 * 2 + al.total_u32 + b_len.max(16) * 2) * 4
                + b_len * swa_tiered_inst
                + sum_p * Kv256kPool::summary_page_bytes_for(true)
                + hot_p * Kv256kPool::full_comp_page_bytes(page_toks, kv_bits)
                + (128usize << 20)
        };

        // Try independent mode with 16 -> 8 -> 4 -> 2 hot retrieved pages/inst before falling back to radix-shared!
        let mut chosen: Option<(&str, bool, usize, usize, usize)> = None;
        for &ret_p in &[top_k_pages.min(16), top_k_pages.min(8), top_k_pages.min(4), top_k_pages.min(2), 0] {
            if est_base(indep_sum_pages, min_local_pages + b_len * ret_p) <= max_mem {
                chosen = Some(("independent", false, indep_sum_pages, min_local_pages + b_len * ret_p, ret_p));
                break;
            }
        }
        if chosen.is_none() {
            for &ret_p in &[top_k_pages.min(64), top_k_pages.min(16), 0] {
                if est_base(radix_sum_pages, min_local_pages + ret_p) <= max_mem {
                    chosen = Some(("radix-shared", true, radix_sum_pages, min_local_pages + ret_p, ret_p));
                    break;
                }
            }
        }
        let Some((mode_tag, use_radix, sum_pages, hot_pages, hot_ret_each)) = chosen else {
            let need_gb = est_base(radix_sum_pages, min_local_pages) as f64 / (1u64 << 30) as f64;
            println!("  {:>4} {:>12} | SKIPPED (TierSWA+CS needs {:.2} GiB > {:.2} GiB cap; use wave-swap)",
                     b_len, "over-cap", need_gb, max_mem as f64 / (1u64 << 30) as f64);
            continue;
        };

        let cfg = Kv256kConfig {
            max_active_batch: b_len,
            max_ctx: ctx_len + 64,
            page_toks,
            kv_bits,
            top_k_pages,
            use_centroids: true,
            use_wht2_sketch: true,
            use_tiered_swa: true,
            swa_inner_fp16: false,
            use_dc_mean: true,
            duo,
            max_mem_bytes: max_mem,
            sum_pool_pages: sum_pages,
            hot_full_pages: hot_pages,
            chunk_cap: 0,
        };
        let mut pool = match Kv256kPool::try_new(eng, cfg) {
            Ok(p) => p,
            Err(err) => {
                println!("  {:>4} {:>12} | SKIPPED ({err})", b_len, mode_tag);
                continue;
            }
        };

        let nfar = (ctx_len - EXT) / page_toks;
        let mut shared_lids: Vec<u64> = Vec::new();
        let mut shared_spids: Vec<u32> = Vec::new();
        if use_radix {
            for _ in 0..nfar {
                shared_lids.push(pool.radix.alloc_logical_id());
                shared_spids.push(pool.alloc_sum_page().unwrap());
            }
            for lp in 0..hot_ret_each.min(nfar) {
                let _ = pool.lru.alloc_for_logical(shared_lids[lp], false, false, &mut pool.cold, &pool.kv_comp, &pool.kv_scales);
            }
        }

        let mut sessions: Vec<Kv256kSession> = Vec::with_capacity(b_len);
        let mut cs_ids: Vec<u32> = Vec::with_capacity(b_len);
        for b in 0..b_len {
            let mut s = Kv256kSession::new(b);
            let cs = pool.alloc_cs().unwrap();
            cs_ids.push(cs);
            s.cs_slot = Some(cs);
            s.pos = ctx_len - 1;
            s.next_tok = (1000 + (b * 137) % 30000) as u32;
            for _ in 0..SWA_INNER_BLKS { s.swa_pages.push(pool.alloc_swa_page().unwrap()); }
            for _ in 0..SWA_OUTER_BLKS { s.swa_outer_pages.push(pool.alloc_swa_outer_page().unwrap()); }
            let total_p = (ctx_len - 1) / page_toks + 1;
            for lp in 0..total_p {
                if use_radix && lp < nfar {
                    s.logical_pages.push(shared_lids[lp]);
                    s.sum_pages.push(shared_spids[lp]);
                } else {
                    let lid = pool.radix.alloc_logical_id();
                    let spid = if lp < nfar { pool.alloc_sum_page().unwrap() } else { u32::MAX };
                    s.logical_pages.push(lid);
                    s.sum_pages.push(spid);
                    if !use_radix && lp < hot_ret_each {
                        let _ = pool.lru.alloc_for_logical(lid, false, false, &mut pool.cold, &pool.kv_comp, &pool.kv_scales);
                    }
                }
                s.page_hashes.push(None);
            }
            sessions.push(s);
        }
        pool.zero_cs_slots(&cs_ids);

        // Initialize non-zero KvPageWht2 scales & counts so scorer and background centroids execute real math
        unsafe {
            let sum_p = pool.kv_sum.contents().as_ptr() as *mut KvPageWht2Host;
            let sc_h = f32_to_f16(0.25);
            let scm_h = f32_to_f16(0.05);
            for slot in 0..N_FULL {
                for sp in 0..sum_pages {
                    for g in 0..NKV {
                        let dst = &mut *sum_p.add((slot * sum_pages + sp) * NKV + g);
                        for sb in 0..WHT2_NSUB { dst.sc_sk[sb] = sc_h; }
                        dst.sc_kmean = scm_h;
                        dst.sc_vmean = scm_h;
                        dst.count = page_toks as u32;
                        dst.page_id = sp as u32;
                    }
                }
            }
        }

        let run_ms = |pool: &mut Kv256kPool, sessions: &mut [Kv256kSession], skip: u32| -> f64 {
            let (cb, pinned) = pool.upload_and_encode_decode_step(sessions, 0, 0, 1, skip).unwrap();
            cb.commit();
            cb.waitUntilCompleted();
            pool.unpin_pids(&pinned);
            (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
        };

        let _ = run_ms(&mut pool, &mut sessions, 0);
        let step_ms = median_f64((0..5).map(|_| {
            pool.zero_cs_slots(&cs_ids);
            run_ms(&mut pool, &mut sessions, 0)
        }).collect());

        let _ = run_ms(&mut pool, &mut sessions, (1 << C_LMHEAD) | (1 << C_ARGMAX));
        let empty_ms = median_f64((0..3).map(|_| run_ms(&mut pool, &mut sessions, u32::MAX)).collect());
        let wht2_ms = (median_f64((0..3).map(|_| run_ms(&mut pool, &mut sessions, !(1u32 << C_FFNPOST))).collect()) - empty_ms).max(0.01);
        let sattn_ms = (median_f64((0..3).map(|_| run_ms(&mut pool, &mut sessions, !(1u32 << C_WOPOST))).collect()) - empty_ms).max(0.01);
        let tswa_ms = (median_f64((0..3).map(|_| run_ms(&mut pool, &mut sessions, !(1u32 << C_ATTN))).collect()) - empty_ms).max(0.01);

        let agg_tps = b_len as f64 / (step_ms * 1e-3);
        let per_tps = 1e3 / step_ms;
        let fp_gb = proc_phys_footprint_bytes() as f64 / (1u64 << 30) as f64;
        let pool_gb = pool.mem.used_bytes as f64 / (1u64 << 30) as f64;

        println!("  {:>4} {:>12} | {:>5.2}ms {:>5.2}ms {:>5.2}ms {:>6.2}ms | {:>9.1} {:>8.2} | {:>7.2}GiB {:>7.2}GiB",
                 b_len, mode_tag, wht2_ms, sattn_ms, tswa_ms, step_ms, agg_tps, per_tps, pool_gb, fp_gb);
        drop(pool);
    }

    // ---- Stage [6/6]: Wave-Swap Multi-Instance Overhead (Tiered SWA + FP16 CS + KvPageWht2) ----
    {
        let wave_test_n = n_instances.clamp(2, 8);
        println!("\n[6/6] Wave-Swap Manager Benchmark (simulating {n_instances} loaded 256K instances via F_NOCACHE swap, testing {wave_test_n} live swaps):");
        let cfg = Kv256kConfig {
            max_active_batch: 2,
            max_ctx: ctx_len + 64,
            page_toks,
            kv_bits,
            top_k_pages,
            use_centroids: true,
            use_wht2_sketch: true,
            use_tiered_swa: true,
            swa_inner_fp16: false,
            use_dc_mean: true,
            duo,
            max_mem_bytes: max_mem,
            sum_pool_pages: n_pages_256k * 2,
            hot_full_pages: 64,
            chunk_cap: 0,
        };
        if let Ok(mut pool) = Kv256kPool::try_new(eng, cfg) {
            let in_pids: Vec<u32> = (0..SWA_INNER_BLKS).map(|_| pool.alloc_swa_page().unwrap()).collect();
            let out_pids: Vec<u32> = (0..SWA_OUTER_BLKS).map(|_| pool.alloc_swa_outer_page().unwrap()).collect();
            let sum_pids: Vec<u32> = (0..n_pages_256k).map(|_| pool.alloc_sum_page().unwrap()).collect();
            let cs0 = pool.alloc_cs().unwrap();

            let fp0 = proc_phys_footprint_bytes();
            let mut out_times = Vec::new();
            for seq_id in 0..wave_test_n {
                let dt = pool.swap.swap_out_tiered(
                    seq_id,
                    &pool.swa_inner, &pool.swa_inner_sc, pool.swa_inner_cap, &in_pids, false,
                    &pool.swa_outer_comp, &pool.swa_outer_sc, pool.swa_outer_cap, &out_pids,
                    &pool.cs_pool_hcs, cs0, pool.cfg.max_active_batch,
                    &pool.kv_sum, pool.cfg.sum_pool_pages, &sum_pids, KV_PAGE_WHT2_BYTES,
                ).unwrap();
                out_times.push(dt.as_secs_f64() * 1e3);
            }
            let _ = pool.swap.request_prefetch(0);
            thread::sleep(Duration::from_millis(20));
            let mut in_times = Vec::new();
            for seq_id in 0..wave_test_n {
                if seq_id + 1 < wave_test_n { let _ = pool.swap.request_prefetch(seq_id + 1); }
                let dt = pool.swap.swap_in_tiered(
                    seq_id,
                    &pool.swa_inner, &pool.swa_inner_sc, pool.swa_inner_cap, &in_pids, false,
                    &pool.swa_outer_comp, &pool.swa_outer_sc, pool.swa_outer_cap, &out_pids,
                    &pool.cs_pool_hcs, cs0, pool.cfg.max_active_batch,
                    &pool.kv_sum, pool.cfg.sum_pool_pages, &sum_pids, KV_PAGE_WHT2_BYTES,
                ).unwrap();
                in_times.push(dt.as_secs_f64() * 1e3);
            }
            let fp1 = proc_phys_footprint_bytes();
            let per_inst_mb = (swa_tiered_inst + cs_per_inst + sum_wht2_inst) as f64 / 1048576.0;
            let med_out = median_f64(out_times);
            let med_in = median_f64(in_times);
            println!("  Swap record size = {:.1} MiB/instance (TierSWA {:.1} + FP16_CS {:.2} + 256K WHT2_Idx {:.1} MiB)",
                     per_inst_mb, swa_tiered_inst as f64 / 1048576.0, cs_per_inst as f64 / 1048576.0, sum_wht2_inst as f64 / 1048576.0);
            println!("  swap_out = {med_out:.2} ms ({:.2} GB/s) | swap_in (prefetched) = {med_in:.2} ms ({:.2} GB/s)",
                     (per_inst_mb / 1024.0) / (med_out * 1e-3).max(1e-6),
                     (per_inst_mb / 1024.0) / (med_in * 1e-3).max(1e-6));
            println!("  OOM check: phys_footprint before={:.2} GiB -> after={:.2} GiB (delta {:+.1} MiB)",
                     fp0 as f64 / (1u64 << 30) as f64, fp1 as f64 / (1u64 << 30) as f64,
                     (fp1 as i64 - fp0 as i64) as f64 / 1048576.0);
        } else {
            println!("  Wave-Swap test skipped: insufficient budget for 2 active 256K slots.");
        }
    }
    let (fp, peak, dr, dw) = proc_rusage_summary();
    println!("\n[256K summary] phys_footprint={:.2} GiB | lifetime_peak={:.2} GiB | disk_read={:.1} MiB | disk_written={:.1} MiB",
             fp as f64 / (1u64 << 30) as f64, peak as f64 / (1u64 << 30) as f64,
             dr as f64 / 1048576.0, dw as f64 / 1048576.0);
}
