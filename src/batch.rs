//! High-throughput multi-sequence batched inference engine:
//! - Paged KV cache (KV_BLK = 32 tokens/page, 64-bit safe addressing, SWA 33-page ring recycling)
//! - Shared-prefix zero-copy KV cache + M5 neural-accelerator GQA-packed FlashDecoding cascade attention
//! - Packed multi-prompt batched prefill directly into the paged KV cache
//! - 4-class MoE routing (sg8 / sg4 / direct-2 / direct-1) + fused wo/MoE post-conv & state update
//! - Fused register-argmax lm_head (zero logits buffer) & vocab-sliced GPU top-k/top-p sampler
//! - 2-step GPU pipelining (zero CPU-GPU sync stalls) & continuous batching

use crate::engine::*;
use crate::metal::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLDevice};
use std::cell::Cell;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub const KV_BLK: usize = 32;
pub const SWA_RING_BLKS: usize = EXT / KV_BLK + 1; // 33
pub const NTC_B: usize = 4;
const LM_VCHUNK: usize = 16 * 64; // 1024 vocab rows per TG in k_lmhead_argmax_sg
const LM_VSLICE: usize = 12800;   // f32 vocab slice size for GPU top-k / full-logits extraction
const N_SWA: usize = 40;
const N_FULL: usize = 8;
const CS_SEQ_ELEMS: usize = N_SLOT * NCH * 3; // 48 * 6656 * 3 = 958,464 floats (~3.83 MiB)

// ----------------------------------------------------------------------------- kernel arg structs (mirror batch.metal)
#[repr(C)] #[derive(Clone, Copy, Default)]
struct Route4Args {
    sel_off: u32, t: u32, perm_off: u32, inv_off: u32, aw_off: u32, ind_off: u32,
    gu_y: [u32; NTC_B], dn_y: [u32; NTC_B], tiles_off: [u32; NTC_B],
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
struct PrefixAttnArgs {
    b: u32, pre_n: u32, pre_p0: u32, pre_cap: u32, k_base: u32, v_base: u32, window: u32,
    qp_off: u32, rp_off: u32, sink_off: u32, mlo_off: u32, relbias_w: u32, seq_pos_off: u32, pad0: u32, pad1: u32, pad2: u32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct LmHeadArgs {
    w_lo: u32, w_hi: u32, row_bytes: u32, wtype: u32, n_vocab: u32, k: u32, xh_off: u32, part_off: u32,
    b: u32, nvch: u32, vstart: u32, vcount: u32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct ArgmaxBatchArgs { part_off: u32, nvch: u32, out_base: u32, pad: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct TopkMergeArgs { slice_off: u32, vstart: u32, vcount: u32, topk_off: u32, top_k: u32, first_slice: u32, pad0: u32, pad1: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct SampleBatchArgs { topk_off: u32, top_k: u32, out_base: u32, rng_off: u32, temp: f32, top_p: f32, softcap: f32, pad: f32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct CopyPreKvArgs { pre_n: u32, pre_p0: u32, pre_cap: u32, src_cap: u32, is_swa: u32, pad0: u32, pad1: u32, pad2: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct CopyToPagedArgs { n_pos: u32, pos0: u32, src_cap: u32, is_swa: u32, pool_cap: u32, pt_off: u32, pt_mod: u32, suf0: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct CopyCsArgs { dst_off: u32, n: u32, max_batch: u32, prefix_len: u32 }
#[repr(C)] #[derive(Clone, Copy, Default)]
struct PackedConvArgs {
    y_off: u32, ldy: u32, res_off: u32, out_off: u32, layer_slot: u32, branch: u32, max_batch: u32, convw_t_off: u32,
    has_res: u32, seq_start_off: u32, seq_p0_off: u32, tok_seq_off: u32, seq_cs_off: u32, t: u32, pad0: u32, pad1: u32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct PackedStateArgs {
    x_off: u32, ldx: u32, layer_slot: u32, branch: u32, max_batch: u32, nch: u32,
    seq_start_off: u32, seq_len_off: u32, seq_p0_off: u32, seq_cs_off: u32, pad0: u32, pad1: u32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct PackedKvPrepArgs {
    t: u32, pool_cap: u32, slot_idx: u32, is_swa: u32, pt_stride: u32, pt_mod: u32,
    tok_pos_off: u32, tok_seq_off: u32, seq_start_off: u32, seq_p0_off: u32, seq_cs_off: u32, pt_off: u32,
    layer_slot: u32, max_batch: u32,
    q_off: u32, k_off: u32, v_off: u32, r_off: u32, kdyn_off: i32, vdyn_off: i32,
    qp_off: u32, rp_off: u32, kconv_t_w: u32, vconv_t_w: u32, qclip_w: u32, relclip_w: u32,
    log_alpha: f32, log_floor: f32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct PackedAttnArgs {
    pool_cap: u32, slot_idx: u32, is_swa: u32, pt_stride: u32, pt_mod: u32,
    pt_off: u32, qtiles_off: u32, qp_off: u32, rp_off: u32, sink_off: u32, gate_off: u32, out_off: u32, relbias_w: u32, pad0: u32, pad1: u32, pad2: u32,
}
#[repr(C)] #[derive(Clone, Copy, Default)]
struct OutMixGatherArgs {
    n_s: u32, s: u32, s_off: [u32; 6],
    q_off: u32, norm_off: u32, hout_off: u32, last_idx_off: u32,
    eps: f32, pad0: u32, pad1: u32, pad2: u32,
}

// ----------------------------------------------------------------------------- Sampling & stats
#[derive(Clone, Copy)]
pub struct SampleCfg {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}
impl SampleCfg {
    pub fn greedy(&self) -> bool { self.temp <= 0.0 }
}

#[derive(Default, Clone, Copy)]
pub struct BatchStats {
    pub prompt_tokens: usize,
    pub prompt_time: Duration,
    pub gen_tokens: usize,
    pub gen_time: Duration,
    pub decode_steps: usize,
    pub peak_batch: usize,
    pub kv_bytes: usize,
    pub cs_bytes: usize,
    pub act_bytes: usize,
}

// ----------------------------------------------------------------------------- Lean aliased activation arena
#[derive(Clone, Copy, Default)]
pub(crate) struct DecodeArena {
    pub(crate) cap: usize,
    pub(crate) vec0: u32, pub(crate) msg: [u32; 4], pub(crate) cath: [u32; 4],
    // attn-phase scratch
    pub(crate) inp: [u32; 3], pub(crate) gain_o: [u32; 4], pub(crate) gh: [u32; 4], pub(crate) xh: [u32; 3],
    pub(crate) q: u32, pub(crate) gate: u32, pub(crate) k: u32, pub(crate) v: u32, pub(crate) rel: u32, pub(crate) sink: u32, pub(crate) kdyn: u32, pub(crate) vdyn: u32,
    pub(crate) qp: u32, pub(crate) rp: u32, pub(crate) pre_mlo: u32, pub(crate) agh: u32, pub(crate) wo_y: u32,
    // ffn-phase scratch (aliases attn-phase!)
    pub(crate) fin: u32, pub(crate) fh: u32, pub(crate) rlog: u32, pub(crate) sel: u32, pub(crate) perm: u32, pub(crate) inv: u32, pub(crate) aw: u32,
    pub(crate) tl: [u32; NTC_B], pub(crate) ind: u32, pub(crate) xgh: u32, pub(crate) acth: u32, pub(crate) yd: u32, pub(crate) ffn_y: u32,
    // lm_head scratch (aliases scratch!)
    pub(crate) am_part: u32, pub(crate) topk_buf: u32, pub(crate) slice_buf: u32,
    pub(crate) max_split: usize,
    pub(crate) end: u32,
}

impl DecodeArena {
    fn new(cap: usize, n_vocab: usize, need_slice: bool, need_packed: bool) -> DecodeArena {
        let cap = cap.max(1);
        let mut o = 0u32;
        let take = |o: &mut u32, n: usize| -> u32 { let r = *o; *o += (n.div_ceil(64) * 64) as u32; r };
        let mut d = DecodeArena { cap, max_split: if cap < 8 { 16 } else { 4 }, ..Default::default() };
        // persistent across layer phases
        d.vec0 = take(&mut o, B_NVEC * cap * E);
        for i in 0..4 { d.msg[i] = take(&mut o, cap * E); }
        for i in 0..4 { d.cath[i] = take(&mut o, cap * E); } // 2*E halfs = E f32 words
        let scratch0 = o;

        // Branch 1: attention phase
        let mut oa = scratch0;
        for i in 0..3 { d.inp[i] = take(&mut oa, cap * E); }
        for i in 0..4 { d.gain_o[i] = take(&mut oa, cap * E); }
        for i in 0..4 { d.gh[i] = take(&mut oa, cap * 32); } // 64 halfs = 32 f32 words
        for i in 0..3 { d.xh[i] = take(&mut oa, cap * E / 2); }
        d.q = take(&mut oa, cap * NH * HD / 2);
        d.gate = take(&mut oa, cap * NH * HD / 2);
        d.k = take(&mut oa, cap * NKV * HD);
        d.v = take(&mut oa, cap * NKV * HD);
        d.rel = take(&mut oa, cap * NH * NREL);
        d.sink = take(&mut oa, cap * NH);
        d.kdyn = take(&mut oa, cap * 32);
        d.vdyn = take(&mut oa, cap * 32);
        d.qp = take(&mut oa, cap * NH * HD / 2);
        d.rp = take(&mut oa, cap * NH * NREL);
        d.pre_mlo = take(&mut oa, cap * NH * d.max_split * (HD + 2));
        d.agh = take(&mut oa, cap * NH * HD / 2);
        d.wo_y = if need_packed { take(&mut oa, cap * E) } else { 0 };

        // Branch 2: FFN phase (aliases attention scratch; msg[3] & cath[3] are persistent!)
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
        ];
        d.ind = take(&mut of, 64);
        d.acth = take(&mut of, cap * TOPK * FF / 2);
        d.xgh = take(&mut of, cap * TOPK * E / 2);
        d.yd = d.xgh; // xgh is dead once moe_gu finishes, before moe_down writes yd!
        d.ffn_y = if need_packed { take(&mut of, cap * E) } else { 0 };
        // Keep gain_o[3] and gh[3] cleanly separated from both attn & ffn
        // (attn only uses 0..3; ffn only uses index 3!)
        d.gain_o[3] = ffn_gain_o;
        d.gh[3] = ffn_gh;

        // Branch 3: lm_head phase (aliases scratch after fh!)
        let mut ol = d.fh + (cap * E / 2).div_ceil(64) as u32 * 64;
        let nvch = n_vocab.div_ceil(LM_VCHUNK);
        d.am_part = take(&mut ol, cap * nvch * 2);
        d.topk_buf = take(&mut ol, cap * 128);
        d.slice_buf = if need_slice { take(&mut ol, cap * LM_VSLICE) } else { 0 };

        d.end = oa.max(of).max(ol);
        d
    }
    #[inline] pub(crate) fn vec(&self, i: usize) -> u32 { self.vec0 + (i * self.cap * E) as u32 }
}

// ----------------------------------------------------------------------------- Slot-major Paged KV pool
// Layout: [slot_in_kind][cap_pages][2(K,V)][NKV][KV_BLK=32][HD]
// Slot-major layout keeps all pages of a layer contiguous (40x fewer TLB misses than page-major!).
pub struct PagedKvPool {
    pub swa: Buf,
    pub full: Buf,
    pub swa_cap: usize,  // pages
    pub full_cap: usize, // pages
    swa_free: Vec<u32>,
    full_free: Vec<u32>,
    kvb: usize,
}

impl PagedKvPool {
    pub fn slot_page_bytes(kvb: usize) -> usize { 2 * NKV * KV_BLK * HD * kvb }
    pub fn swa_page_bytes(kvb: usize) -> usize { N_SWA * Self::slot_page_bytes(kvb) }
    pub fn full_page_bytes(kvb: usize) -> usize { N_FULL * Self::slot_page_bytes(kvb) }

    pub fn try_new(gpu: &Gpu, kvb: usize, init_swa: usize, init_full: usize, max_kv_bytes: usize) -> Result<PagedKvPool, String> {
        let init_swa = init_swa.max(1);
        let init_full = init_full.max(1);
        let max_buf = gpu.dev.maxBufferLength() as usize;
        let swa_bytes = init_swa * Self::swa_page_bytes(kvb);
        let full_bytes = init_full * Self::full_page_bytes(kvb);
        if swa_bytes > max_buf {
            return Err(format!("SWA KV pool ({:.2} GiB) exceeds Metal maxBufferLength ({:.2} GiB)",
                               swa_bytes as f64 / (1u64 << 30) as f64, max_buf as f64 / (1u64 << 30) as f64));
        }
        if full_bytes > max_buf {
            return Err(format!("Full KV pool ({:.2} GiB) exceeds Metal maxBufferLength ({:.2} GiB)",
                               full_bytes as f64 / (1u64 << 30) as f64, max_buf as f64 / (1u64 << 30) as f64));
        }
        if swa_bytes + full_bytes > max_kv_bytes {
            return Err(format!("KV pool ({:.2} GiB = SWA {:.2} + Full {:.2}) exceeds available KV budget ({:.2} GiB)",
                               (swa_bytes + full_bytes) as f64 / (1u64 << 30) as f64,
                               swa_bytes as f64 / (1u64 << 30) as f64,
                               full_bytes as f64 / (1u64 << 30) as f64,
                               max_kv_bytes as f64 / (1u64 << 30) as f64));
        }
        let swa = gpu.buffer(swa_bytes);
        let full = gpu.buffer(full_bytes);
        let swa_free = (0..init_swa as u32).rev().collect();
        let full_free = (0..init_full as u32).rev().collect();
        Ok(PagedKvPool { swa, full, swa_cap: init_swa, full_cap: init_full, swa_free, full_free, kvb })
    }

    pub fn bytes(&self) -> usize { self.swa.length() + self.full.length() }
    pub fn free_swa_pages(&self) -> usize { self.swa_free.len() }
    pub fn free_full_pages(&self) -> usize { self.full_free.len() }

    /// Ensure at least `need_swa` and `need_full` free pages exist within `max_kv_bytes`. GPU MUST be idle if growth occurs.
    pub fn ensure_free(&mut self, gpu: &Gpu, need_swa: usize, need_full: usize, max_kv_bytes: usize) {
        let max_buf = gpu.dev.maxBufferLength() as usize;
        let spb = Self::slot_page_bytes(self.kvb);
        if self.swa_free.len() < need_swa {
            let pb = N_SWA * spb;
            let min_add = need_swa - self.swa_free.len();
            let max_add_mem = max_kv_bytes.saturating_sub(self.bytes()) / pb;
            assert!(min_add <= max_add_mem,
                    "SWA KV pool growth (+{min_add} pages = +{:.2} GiB) exceeds memory cap ({:.2} GiB KV budget); reduce -b/-n, use --kv8, or raise --max-mem",
                    min_add as f64 * pb as f64 / (1u64 << 30) as f64, max_kv_bytes as f64 / (1u64 << 30) as f64);
            let add = min_add.max(self.swa_cap / 4).max(32).min(max_add_mem);
            let new_cap = self.swa_cap + add;
            assert!(new_cap * pb <= max_buf, "SWA KV pool growth ({:.2} GiB) exceeds maxBufferLength", new_cap as f64 * pb as f64 / (1u64 << 30) as f64);
            let nb = gpu.buffer(new_cap * pb);
            for s in 0..N_SWA {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (self.swa.contents().as_ptr() as *const u8).add(s * self.swa_cap * spb),
                        (nb.contents().as_ptr() as *mut u8).add(s * new_cap * spb),
                        self.swa_cap * spb,
                    );
                }
            }
            for p in (self.swa_cap as u32..new_cap as u32).rev() { self.swa_free.push(p); }
            self.swa = nb;
            self.swa_cap = new_cap;
        }
        if self.full_free.len() < need_full {
            let pb = N_FULL * spb;
            let min_add = need_full - self.full_free.len();
            let max_add_mem = max_kv_bytes.saturating_sub(self.bytes()) / pb;
            assert!(min_add <= max_add_mem,
                    "Full KV pool growth (+{min_add} pages = +{:.2} GiB) exceeds memory cap ({:.2} GiB KV budget); reduce -b/-n, use --kv8, or raise --max-mem",
                    min_add as f64 * pb as f64 / (1u64 << 30) as f64, max_kv_bytes as f64 / (1u64 << 30) as f64);
            let add = min_add.max(self.full_cap / 4).max(32).min(max_add_mem);
            let new_cap = self.full_cap + add;
            assert!(new_cap * pb <= max_buf, "Full KV pool growth ({:.2} GiB) exceeds maxBufferLength", new_cap as f64 * pb as f64 / (1u64 << 30) as f64);
            let nb = gpu.buffer(new_cap * pb);
            for s in 0..N_FULL {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (self.full.contents().as_ptr() as *const u8).add(s * self.full_cap * spb),
                        (nb.contents().as_ptr() as *mut u8).add(s * new_cap * spb),
                        self.full_cap * spb,
                    );
                }
            }
            for p in (self.full_cap as u32..new_cap as u32).rev() { self.full_free.push(p); }
            self.full = nb;
            self.full_cap = new_cap;
        }
    }

    pub fn alloc_swa(&mut self) -> u32 { self.swa_free.pop().expect("SWA page pool exhausted") }
    pub fn alloc_full(&mut self) -> u32 { self.full_free.pop().expect("Full page pool exhausted") }
    pub fn free_swa(&mut self, pid: u32) { self.swa_free.push(pid); }
    pub fn free_full(&mut self, pid: u32) { self.full_free.push(pid); }
}

// ----------------------------------------------------------------------------- Shared Prefix Cache
pub struct PrefixCache {
    pub prefix_len: usize,
    pub pre_swa: Buf,
    pub pre_swa_n: usize,
    pub pre_swa_p0: usize,
    pub pre_swa_cap: usize,
    pub pre_full: Buf,
    pub pre_full_cap: usize,
    pub pre_cs: Buf,
    #[allow(dead_code)]
    pub first_tok: u32,
    pub first_logits: Vec<f32>,
}

impl PrefixCache {
    pub fn bytes(&self) -> usize { self.pre_swa.length() + self.pre_full.length() + self.pre_cs.length() }
}

// ----------------------------------------------------------------------------- Per-sequence runtime state
pub struct SeqSlot {
    pub id: usize,
    pub cs_slot: u32,
    pub pos: usize,         // position of the NEXT token to process (== prefix_len + suf)
    pub suf: usize,         // private suffix index of the NEXT token to process
    pub swa_pages: Vec<u32>,  // up to SWA_RING_BLKS (33)
    pub full_pages: Vec<u32>, // grows as suf / KV_BLK
    pub generated: Vec<u32>,
    pub max_new: usize,
    pub next_tok: u32,
    pub done: bool,
}

impl SeqSlot {
    pub(crate) fn ensure_pages_for(&mut self, pool: &mut PagedKvPool, suf_pos: usize) {
        let blk = suf_pos / KV_BLK;
        let swa_idx = blk % SWA_RING_BLKS;
        while self.swa_pages.len() <= swa_idx { self.swa_pages.push(pool.alloc_swa()); }
        while self.full_pages.len() <= blk { self.full_pages.push(pool.alloc_full()); }
    }
    pub(crate) fn pages_needed_for(&self, suf_pos: usize) -> (usize, usize) {
        let blk = suf_pos / KV_BLK;
        let swa_idx = blk % SWA_RING_BLKS;
        (
            if swa_idx >= self.swa_pages.len() { swa_idx + 1 - self.swa_pages.len() } else { 0 },
            if blk >= self.full_pages.len() { blk + 1 - self.full_pages.len() } else { 0 },
        )
    }
    pub(crate) fn free_pages(&mut self, pool: &mut PagedKvPool) {
        for p in self.swa_pages.drain(..) { pool.free_swa(p); }
        for p in self.full_pages.drain(..) { pool.free_full(p); }
    }
}

// ----------------------------------------------------------------------------- Metadata buffer layout
#[derive(Clone, Copy)]
pub(crate) struct MetaLayout {
    pub(crate) seq_pos: u32,
    pub(crate) seq_suf: u32,
    pub(crate) seq_cs: u32,
    pub(crate) swa_pt: u32,
    pub(crate) full_pt: u32,
    pub(crate) total_u32: usize,
}
impl MetaLayout {
    fn new(max_batch: usize, max_full_blks: usize) -> MetaLayout {
        let mut o = 0u32;
        let mut take = |n: usize| { let r = o; o += (n.div_ceil(16) * 16) as u32; r };
        let seq_pos = take(max_batch);
        let seq_suf = take(max_batch);
        let seq_cs = take(max_batch);
        let swa_pt = take(max_batch * SWA_RING_BLKS);
        let full_pt = take(max_batch * max_full_blks);
        MetaLayout { seq_pos, seq_suf, seq_cs, swa_pt, full_pt, total_u32: o as usize }
    }
}

// ----------------------------------------------------------------------------- BatchEngine
pub struct BatchEngine<'a> {
    pub eng: &'a Engine,
    pub pool: PagedKvPool,
    pub cs_pool: Buf,
    pub act: Buf,
    pub(crate) da: DecodeArena,
    pub(crate) meta: [Buf; 2],
    pub(crate) ml: MetaLayout,
    pub(crate) toks: [Buf; 2],
    pub(crate) rng_buf: Buf,
    pub max_batch: usize,
    pub max_ctx: usize,
    pub max_mem_bytes: usize,
    pub(crate) max_full_blks: usize,
    pub prefix: Option<PrefixCache>,
    pub skip: Cell<u32>,
    free_cs_slots: Vec<u32>,
}

impl<'a> BatchEngine<'a> {
    /// Compute a safe hard memory cap for BatchEngine working GPU memory (KV + CS + Act + Meta).
    /// Always leaves generous unified-memory headroom below Metal's recommendedMaxWorkingSetSize so macOS can NEVER OOM.
    pub fn safe_mem_limit(eng: &Engine, requested_gb: Option<f64>) -> usize {
        let rec = eng.gpu.dev.recommendedMaxWorkingSetSize() as usize;
        let weights = eng.g.file_len as usize + eng.pb.length();
        // Absolute hard ceiling: leave at least 6 GiB below recommendedMaxWorkingSetSize
        let hard_ceil = rec.saturating_sub(weights + (6usize << 30)).max(512usize << 20);
        // Safe default: leave at least 10 GiB below recommendedMaxWorkingSetSize, capped at 10 GiB
        let default_cap = rec.saturating_sub(weights + (10usize << 30)).min(10usize << 30).max(512usize << 20);
        match requested_gb {
            Some(gb) => ((gb.max(0.5) * (1u64 << 30) as f64) as usize).min(hard_ceil),
            None => default_cap,
        }
    }

    pub fn try_new(
        eng: &'a Engine, max_batch: usize, max_ctx: usize,
        init_swa_pages: usize, init_full_pages: usize, need_slice: bool, max_mem_bytes: usize,
    ) -> Result<BatchEngine<'a>, String> {
        assert!(eng.hgemm, "batched engine requires the M5 fp16 tensor path (hgemm)");
        let max_batch = max_batch.max(1);
        let max_full_blks = max_ctx.div_ceil(KV_BLK).max(1);
        let da = DecodeArena::new(max_batch, eng.n_vocab, need_slice, false);
        let ml = MetaLayout::new(max_batch, max_full_blks);

        let cs_bytes = max_batch * CS_SEQ_ELEMS * 4;
        let act_bytes = da.end as usize * 4;
        let meta_bytes = 2 * ml.total_u32 * 4 + 2 * max_batch.max(16) * 4 + max_batch.max(16) * 8;
        let init_swa = init_swa_pages.max(1);
        let init_full = init_full_pages.max(1);
        let kv_bytes = init_swa * PagedKvPool::swa_page_bytes(eng.kvb) + init_full * PagedKvPool::full_page_bytes(eng.kvb);
        let total = kv_bytes + cs_bytes + act_bytes + meta_bytes;
        if total > max_mem_bytes {
            return Err(format!(
                "requested B={max_batch} needs {:.2} GiB (KV {:.2} + CS {:.2} + Act {:.2} GiB) > memory cap {:.2} GiB; reduce -b/-n, use --kv8, or raise --max-mem",
                total as f64 / (1u64 << 30) as f64,
                kv_bytes as f64 / (1u64 << 30) as f64,
                cs_bytes as f64 / (1u64 << 30) as f64,
                act_bytes as f64 / (1u64 << 30) as f64,
                max_mem_bytes as f64 / (1u64 << 30) as f64,
            ));
        }

        let max_kv_bytes = max_mem_bytes.saturating_sub(cs_bytes + act_bytes + meta_bytes);
        let pool = PagedKvPool::try_new(&eng.gpu, eng.kvb, init_swa, init_full, max_kv_bytes)?;
        let cs_pool = eng.gpu.buffer(cs_bytes);
        let act = eng.gpu.buffer(act_bytes);
        let meta = [eng.gpu.buffer(ml.total_u32 * 4), eng.gpu.buffer(ml.total_u32 * 4)];
        let toks = [eng.gpu.buffer(max_batch.max(16) * 4), eng.gpu.buffer(max_batch.max(16) * 4)];
        let rng_buf = eng.gpu.buffer(max_batch.max(16) * 8);
        let free_cs_slots = (0..max_batch as u32).rev().collect();
        Ok(BatchEngine {
            eng, pool, cs_pool, act, da, meta, ml, toks, rng_buf,
            max_batch, max_ctx, max_mem_bytes, max_full_blks, prefix: None, skip: Cell::new(0), free_cs_slots,
        })
    }

    pub fn kv_bytes(&self) -> usize {
        self.pool.bytes() + self.prefix.as_ref().map_or(0, |p| p.bytes())
    }
    pub fn cs_bytes(&self) -> usize { self.cs_pool.length() }
    pub fn act_bytes(&self) -> usize { self.act.length() }
    pub fn meta_bytes(&self) -> usize {
        self.meta[0].length() + self.meta[1].length() + self.toks[0].length() + self.toks[1].length() + self.rng_buf.length()
    }
    pub fn total_bytes(&self) -> usize { self.kv_bytes() + self.cs_bytes() + self.act_bytes() + self.meta_bytes() }
    pub(crate) fn kv_budget(&self) -> usize {
        let non_pool = self.prefix.as_ref().map_or(0, |p| p.bytes()) + self.cs_bytes() + self.act_bytes() + self.meta_bytes();
        self.max_mem_bytes.saturating_sub(non_pool)
    }

    pub(crate) fn ensure_arena(&mut self, cap: usize, need_slice: bool, need_packed: bool) {
        let need_grow = cap > self.da.cap
            || (need_slice && self.da.slice_buf == 0)
            || (need_packed && self.da.wo_y == 0);
        if !need_grow { return; }
        let new_cap = cap.max(self.da.cap);
        let da = DecodeArena::new(new_cap, self.eng.n_vocab, need_slice || self.da.slice_buf != 0, need_packed || self.da.wo_y != 0);
        let new_bytes = da.end as usize * 4;
        let other = self.kv_bytes() + self.cs_bytes() + self.meta_bytes();
        assert!(other + new_bytes <= self.max_mem_bytes,
                "activation arena ({:.2} GiB) + other ({:.2} GiB) exceeds memory cap ({:.2} GiB)",
                new_bytes as f64 / (1u64 << 30) as f64, other as f64 / (1u64 << 30) as f64,
                self.max_mem_bytes as f64 / (1u64 << 30) as f64);
        self.act = self.eng.gpu.buffer(new_bytes);
        self.da = da;
    }

    /// Shrink activation arena back down to decode size (drops temporary packed-prefill buffer immediately).
    fn shrink_arena(&mut self, cap: usize, need_slice: bool) {
        let cap = cap.max(1);
        if self.da.cap <= cap && self.da.wo_y == 0 && (need_slice || self.da.slice_buf == 0) { return; }
        let da = DecodeArena::new(cap, self.eng.n_vocab, need_slice, false);
        self.act = self.eng.gpu.buffer(da.end as usize * 4);
        self.da = da;
    }

    pub(crate) fn init_rng(&self, seed: u64, n: usize) {
        let p = self.rng_buf.contents().as_ptr() as *mut u64;
        for i in 0..n {
            let s = seed.wrapping_add((i as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)) ^ 0x9E37_79B9_7F4A_7C15;
            unsafe { *p.add(i) = s; }
        }
    }

    pub(crate) fn alloc_cs(&mut self) -> u32 { self.free_cs_slots.pop().expect("CS slot pool exhausted") }
    pub(crate) fn free_cs(&mut self, slot: u32) { self.free_cs_slots.push(slot); }

    /// Prefill a shared prefix ONCE using Engine's fast prefill path, copy its K/V into contiguous guard-padded
    /// prefix buffers for M5 neural-accelerator cascade attention, and save its conv state.
    pub fn set_prefix(&mut self, prefix_toks: &[u32]) -> (u32, Duration) {
        assert!(!prefix_toks.is_empty());
        let p_len = prefix_toks.len();
        assert!(p_len < self.max_ctx);
        let pre_swa_n = p_len.min(EXT);
        let pre_swa_p0 = p_len - pre_swa_n;
        let pre_swa_cap = (pre_swa_n + 64).div_ceil(64) * 64;
        let pre_full_cap = (p_len + 64).div_ceil(64) * 64;
        let pre_swa_bytes = N_SWA * 2 * NKV * pre_swa_cap * HD * self.eng.kvb;
        let pre_full_bytes = N_FULL * 2 * NKV * pre_full_cap * HD * self.eng.kvb;
        let pre_cs_bytes = CS_SEQ_ELEMS * 4;
        let non_pre = self.pool.bytes() + self.cs_bytes() + self.act_bytes() + self.meta_bytes();
        assert!(non_pre + pre_swa_bytes + pre_full_bytes + pre_cs_bytes <= self.max_mem_bytes,
                "shared prefix ({:.2} GiB) + batch state ({:.2} GiB) exceeds memory cap ({:.2} GiB)",
                (pre_swa_bytes + pre_full_bytes + pre_cs_bytes) as f64 / (1u64 << 30) as f64,
                non_pre as f64 / (1u64 << 30) as f64,
                self.max_mem_bytes as f64 / (1u64 << 30) as f64);

        self.prefix = None; // drop any old prefix first
        let t0 = Instant::now();
        self.eng.reset();
        self.eng.reserve(p_len + 1, false);
        self.eng.ensure_batch(p_len.min(self.eng.chunk));
        self.eng.set_tokens(0, prefix_toks);
        let mut p0 = 0;
        let mut last = None;
        while p0 < p_len {
            let t = (p_len - p0).min(self.eng.chunk);
            let cb = self.eng.gpu.cmd();
            self.eng.encode_prefill(&cb, p0, t, p0 + t == p_len, true);
            cb.commit();
            last = Some(cb);
            p0 += t;
        }
        last.unwrap().waitUntilCompleted();
        let first_tok = self.eng.token_at(p_len);
        let first_logits = self.eng.logits().to_vec();

        // Linearize SWA & Full prefix K/V into contiguous buffers with 64-key guard zero padding
        let pre_swa = self.eng.gpu.buffer(pre_swa_bytes);
        let pre_full = self.eng.gpu.buffer(pre_full_bytes);
        let pre_cs = self.eng.gpu.buffer(pre_cs_bytes);
        unsafe {
            std::ptr::copy_nonoverlapping(self.eng.cs.contents().as_ptr() as *const u8, pre_cs.contents().as_ptr() as *mut u8, pre_cs_bytes);
        }

        let cb = self.eng.gpu.cmd();
        let e = Enc::new(&cb);
        e.pso(&self.eng.gpu.pso("k_copy_prefix_kv"));
        let a_swa = CopyPreKvArgs { pre_n: pre_swa_n as u32, pre_p0: pre_swa_p0 as u32, pre_cap: pre_swa_cap as u32, src_cap: self.eng.swa_cap.get() as u32, is_swa: 1, pad0: 0, pad1: 0, pad2: 0 };
        e.buf(&self.eng.kv_swa.borrow(), 0, 3);
        e.buf(&pre_swa, 0, 8);
        e.bytes(&a_swa, 5);
        e.dispatch((pre_swa_cap, NKV, N_SWA), 32);
        e.barrier();
        let a_full = CopyPreKvArgs { pre_n: p_len as u32, pre_p0: 0, pre_cap: pre_full_cap as u32, src_cap: self.eng.full_cap.get() as u32, is_swa: 0, pad0: 0, pad1: 0, pad2: 0 };
        e.buf(&self.eng.kv_full.borrow(), 0, 3);
        e.buf(&pre_full, 0, 8);
        e.bytes(&a_full, 5);
        e.dispatch((pre_full_cap, NKV, N_FULL), 32);
        e.end();
        cb.commit();
        cb.waitUntilCompleted();

        // Free engine's temporary prefill arena and KV cache immediately
        self.eng.release_batch();
        self.eng.shrink_kv();

        self.prefix = Some(PrefixCache {
            prefix_len: p_len, pre_swa, pre_swa_n, pre_swa_p0, pre_swa_cap, pre_full, pre_full_cap, pre_cs, first_tok, first_logits,
        });
        (first_tok, t0.elapsed())
    }

    /// Broadcast prefix conv state (AoS -> slot-major SoA ring at prefix_len % 3) to a list of sequence slots on the GPU.
    fn init_cs_from_prefix(&self, cs_slots: &[u32]) {
        if cs_slots.is_empty() { return; }
        let pre = self.prefix.as_ref().expect("set_prefix first");
        let meta = &self.meta[0];
        let mp = meta.contents().as_ptr() as *mut u32;
        for (i, &s) in cs_slots.iter().enumerate() { unsafe { *mp.add(i) = s; } }
        let cb = self.eng.gpu.cmd();
        let e = Enc::new(&cb);
        e.pso(&self.eng.gpu.pso("k_broadcast_cs"));
        e.buf(&pre.pre_cs, 0, 4);
        e.buf(&self.cs_pool, 0, 8);
        e.buf(meta, 0, 7);
        let a = CopyCsArgs { dst_off: 0, n: cs_slots.len() as u32, max_batch: self.max_batch as u32, prefix_len: pre.prefix_len as u32 };
        e.bytes(&a, 5);
        e.dispatch((NCH.div_ceil(256), cs_slots.len(), N_SLOT), 256);
        e.end();
        cb.commit();
        cb.waitUntilCompleted();
    }

    /// Zero conv state for freshly allocated independent sequence slots in slot-major CS_POOL on the GPU.
    fn zero_cs(&self, cs_slots: &[u32]) {
        if cs_slots.is_empty() { return; }
        let meta = &self.meta[0];
        let mp = meta.contents().as_ptr() as *mut u32;
        for (i, &s) in cs_slots.iter().enumerate() { unsafe { *mp.add(i) = s; } }
        let cb = self.eng.gpu.cmd();
        let e = Enc::new(&cb);
        e.pso(&self.eng.gpu.pso("k_zero_cs_slots"));
        e.buf(&self.cs_pool, 0, 4);
        e.buf(meta, 0, 7);
        let a = CopyCsArgs { dst_off: 0, n: cs_slots.len() as u32, max_batch: self.max_batch as u32, prefix_len: 0 };
        e.bytes(&a, 5);
        e.dispatch((NCH.div_ceil(256), cs_slots.len(), N_SLOT), 256);
        e.end();
        cb.commit();
        cb.waitUntilCompleted();
    }

    // ------------------------------------------------------------------------- Low-level encode helpers
    #[inline]
    pub(crate) fn go<T>(&self, e: &Enc, cls: u32, pso: &str, args: &T, grid: (usize, usize, usize), tg: usize) {
        if self.skip.get() & (1 << cls) != 0 { return; }
        e.pso(&self.eng.gpu.pso(pso));
        e.bytes(args, 5);
        e.dispatch(grid, tg);
        e.barrier();
    }

    pub(crate) fn bind_batch(&self, e: &Enc, meta_idx: usize, tok_in_idx: usize) {
        e.buf(&self.eng.w, 0, 0);
        e.buf(&self.eng.pb, 0, 1);
        e.buf(&self.act, 0, 2);
        e.buf(&self.pool.swa, 0, 3);
        e.buf(&self.cs_pool, 0, 4);
        e.buf(&self.toks[tok_in_idx], 0, 6);
        e.buf(&self.meta[meta_idx], 0, 7);
    }

    pub(crate) fn mix_b(&self, e: &Enc, cls: u32, jobs: &[MixJob], t: usize) {
        if self.skip.get() & (1 << cls) != 0 { return; }
        for j in jobs {
            let j = MixJob { ntok: t as u32, ..*j };
            let name = if j.q_off < 0 { "k_mix_b_id".to_string() } else { format!("k_mix_b_{}_{}", j.n_s, j.n_q) };
            e.pso(&self.eng.gpu.pso(&name));
            e.bytes(&j, 5);
            if t <= 64 { e.dispatch((t, 1, 1), 32); } else { e.dispatch((t.div_ceil(4), 1, 1), 128); }
        }
        e.barrier();
    }

    pub(crate) fn gemm_h(&self, e: &Enc, cls: u32, segs: &[HSeg], t: usize) {
        let mut a = GemmHArgs { nseg: segs.len() as u32, t: t as u32, ..Default::default() };
        let mut tiles = 0u32;
        for (i, sg) in segs.iter().enumerate() { a.seg[i] = *sg; a.seg[i].tile_start = tiles; tiles += sg.nrows.div_ceil(64); }
        if t > 64 { self.go(e, cls, "k_gemm_sg8", &a, (t.div_ceil(128), tiles as usize, 1), 256); }
        else { self.go(e, cls, "k_gemm_sg4", &a, (1, tiles as usize, 1), 128); }
    }

    pub(crate) fn ctrl_b(&self, e: &Enc, lw: &LoopW, jobs: &[(usize, u32, u32, u32)], t: usize, hout: &[u32]) {
        let d = self.da;
        let down: Vec<HSeg> = jobs.iter().map(|&(c, _, _, _)| {
            let mut s = Engine::hseg_pb(lw.down[c], 64, (2 * E) as u32, d.cath[c], d.gh[c]);
            s.hout = 1;
            s
        }).collect();
        self.gemm_h(e, C_CTRLD, &down, t);
        let mut ga = CtrlGfArgs { t: t as u32, njobs: jobs.len() as u32, ..Default::default() };
        for (i, &(c, u, m, _)) in jobs.iter().enumerate() {
            ga.job[i] = CtrlGfJob { gain_w: lw.gain[c], forget_w: lw.forget[c], gh_off: d.gh[c], u_off: u, m_off: m, raw_off: d.gain_o[c], pad: [0; 2] };
        }
        if t > 64 { self.go(e, C_CTRLA, "k_ctrl_gf", &ga, (t.div_ceil(128), jobs.len() * (E / 32), 1), 256); }
        else { self.go(e, C_CTRLA, "k_ctrl_gf4", &ga, (1, jobs.len() * (E / 32), 1), 128); }
        let mut a = CtrlBArgs { n: jobs.len() as u32, ..Default::default() };
        for (i, &(c, _, _, nw)) in jobs.iter().enumerate() {
            a.job[i] = CtrlBJob { u_off: d.gain_o[c], m_off: u32::MAX, g_off: 0, f_off: 0, out_off: 0, norm_w: nw, h_off: hout[i] as i32, nof32: 1 };
        }
        self.go(e, C_CTRLN, "k_ctrl_apply_b", &a, (t, jobs.len(), 1), 256);
    }

    pub(crate) fn tohalf_gather(&self, e: &Enc, cls: u32, src: u32, dst: u32, perm: u32, rows: usize) {
        if self.skip.get() & (1 << cls) != 0 { return; }
        let a = ToHalfArgs { src_off: src, dst_off: dst, k: E as u32, rows: rows as u32, perm_off: perm, gather: 3, src_stride: E as u32, dst_stride: E as u32 };
        e.pso(&self.eng.gpu.pso("k_tohalf"));
        e.bytes(&a, 5);
        e.dispatch((E.div_ceil(1024), rows, 1), 256);
        e.barrier();
    }

    fn moe_4class(&self, e: &Enc, l: &Layer, t: usize) {
        let d = self.da;
        // router -> rlog
        self.gemm_h(e, C_ROUTER, &[Engine::hseg_pb(l.router, NEXP as u32, E as u32, d.fh, d.rlog)], t);
        let ta = TopkBArgs { logits_off: d.rlog, bias_off: l.exp_bias, sel_off: d.sel, t: t as u32, wscale: self.eng.wscale, pad: [0; 3] };
        self.go(e, C_TOPK, "k_topk_b", &ta, (t, 1, 1), 32);
        let ra = Route4Args {
            sel_off: d.sel, t: t as u32, perm_off: d.perm, inv_off: d.inv, aw_off: d.aw, ind_off: d.ind,
            gu_y: [(FF / 32) as u32, (FF / 32) as u32, (FF / 4) as u32, (FF / 4) as u32],
            dn_y: [(E / 64) as u32, (E / 64) as u32, (E / 8) as u32, (E / 8) as u32],
            tiles_off: d.tl,
        };
        self.go(e, C_TOPK, "k_moe_route4", &ra, (1, 1, 1), 1024);
        self.tohalf_gather(e, C_GATHER, d.fh, d.xgh, d.perm, t * TOPK);

        let gu_cls: [(&str, usize); NTC_B] = [("k_moe_sg8", 256), ("k_moe_sg4", 128), ("k_moe_dir_gu2", 128), ("k_moe_dir_gu1", 128)];
        let dn_cls: [(&str, usize); NTC_B] = [("k_moe_sg8", 256), ("k_moe_sg4", 128), ("k_moe_dir_dn2", 128), ("k_moe_dir_dn1", 128)];

        if self.skip.get() & (1 << C_GU) == 0 {
            for (c, &(name, tg)) in gu_cls.iter().enumerate() {
                let g = MoeGemmHArgs {
                    w_lo: l.gu.off as u32, w_hi: (l.gu.off >> 32) as u32, expert_bytes: l.gu.expert_bytes as u32, row_bytes: l.gu.row_bytes,
                    wtype: l.gu.ty, nrows: (2 * FF) as u32, k: E as u32, xh_off: d.xgh, tiles_off: d.tl[c], ntiles_off: d.ind + (c * 4) as u32,
                    out_off: d.acth, ldo: FF as u32, situ: 1, aw_off: d.aw, beta: self.eng.situ_beta, lbeta: self.eng.situ_lbeta,
                };
                e.pso(&self.eng.gpu.pso(name));
                e.bytes(&g, 5);
                e.dispatch_indirect(&self.act, (d.ind as usize + c * 4) * 4, tg);
            }
            e.barrier();
        }
        if self.skip.get() & (1 << C_DOWN) == 0 {
            for (c, &(name, tg)) in dn_cls.iter().enumerate() {
                let dn = MoeGemmHArgs {
                    w_lo: l.dn.off as u32, w_hi: (l.dn.off >> 32) as u32, expert_bytes: l.dn.expert_bytes as u32, row_bytes: l.dn.row_bytes,
                    wtype: l.dn.ty, nrows: E as u32, k: FF as u32, xh_off: d.acth, tiles_off: d.tl[c], ntiles_off: d.ind + (c * 4) as u32,
                    out_off: d.yd, ldo: E as u32, situ: 0, aw_off: 0, beta: 0.0, lbeta: 0.0,
                };
                e.pso(&self.eng.gpu.pso(name));
                e.bytes(&dn, 5);
                e.dispatch_indirect(&self.act, (d.ind as usize + (NTC_B + c) * 4) * 4, tg);
            }
            e.barrier();
        }
    }

    // ------------------------------------------------------------------------- Upload metadata for decode step
    pub(crate) fn upload_decode_meta(&self, seqs: &[SeqSlot], meta_idx: usize, pos_delta: usize) {
        let mp = self.meta[meta_idx].contents().as_ptr() as *mut u32;
        let ml = self.ml;
        let mfb = self.max_full_blks;
        unsafe {
            for (b, s) in seqs.iter().enumerate() {
                *mp.add(ml.seq_pos as usize + b) = (s.pos + pos_delta) as u32;
                *mp.add(ml.seq_suf as usize + b) = (s.suf + pos_delta) as u32;
                *mp.add(ml.seq_cs as usize + b) = s.cs_slot;
                let spt = mp.add(ml.swa_pt as usize + b * SWA_RING_BLKS);
                for (k, &pid) in s.swa_pages.iter().enumerate() { *spt.add(k) = pid; }
                let fpt = mp.add(ml.full_pt as usize + b * mfb);
                for (k, &pid) in s.full_pages.iter().enumerate() { *fpt.add(k) = pid; }
            }
        }
    }

    // ------------------------------------------------------------------------- Encode one full batched decode step
    pub fn encode_decode_step(
        &self, cb: &CmdBuf, b_len: usize, max_pos: usize, min_suf: usize,
        meta_idx: usize, tok_in_idx: usize, tok_out_idx: usize,
        sample: Option<SampleCfg>,
    ) {
        assert!(b_len >= 1 && b_len <= self.da.cap);
        let e = Enc::new(cb);
        self.bind_batch(&e, meta_idx, tok_in_idx);
        let d = self.da;
        let eng = self.eng;

        // 1. Embedding -> vec(0)
        let ea = EmbedArgs {
            emb_lo: eng.tok_embd.off as u32, emb_hi: (eng.tok_embd.off >> 32) as u32, row_bytes: eng.tok_embd.row_bytes,
            norm_off: eng.tok_norm, out0: d.vec(0), out1: d.vec(0), tok_index: 0, eps: eng.eps,
        };
        self.go(&e, C_EMBED, "k_embed", &ea, (b_len, 1, 1), 256);

        // 2. 48 layer executions (0..6, 3x 6..18, 18..24)
        let mut va = VecAlloc { next: 1, n: B_NVEC };
        let mut blocks: Vec<usize> = Vec::new();
        let mut cur = 0usize;
        for ip in 0..LOOP_START { cur = self.decode_layer(&e, b_len, max_pos, min_suf, ip, ip, cur, &mut blocks, None, &mut va, &[]); }
        let prelude = cur;
        let prelude_blocks = blocks.clone();
        let mut bank = (0usize, 0usize);
        for pass in 0..N_PASS {
            if pass > 0 { cur = prelude; blocks = prelude_blocks.clone(); }
            let msg = if pass > 0 { Some(bank) } else { None };
            let pinned: Vec<usize> = { let mut p = vec![prelude]; if pass > 0 { p.push(bank.0); p.push(bank.1); } p };
            for ip in LOOP_START..LOOP_END {
                cur = self.decode_layer(&e, b_len, max_pos, min_suf, slot_of(pass, ip), ip, cur, &mut blocks, msg, &mut va, &pinned);
            }
            bank = (blocks[prelude_blocks.len() + 1], cur);
        }
        for ip in LOOP_END..N_PHYS { cur = self.decode_layer(&e, b_len, max_pos, min_suf, ip, ip, cur, &mut blocks, None, &mut va, &[]); }

        // 3. Final output depth mix -> half d.fh[0..b_len]
        let mut st: Vec<u32> = blocks.iter().map(|&v| d.vec(v)).collect();
        st.push(d.vec(cur));
        let mut jo = eng.mix_job_off(&st, eng.output_res_query as i32, 1, &[d.fin], &[eng.output_norm as i32], E as u32);
        jo.hout[0] = d.fh as i32;
        jo.nof32 = 1;
        self.mix_b(&e, C_OMIX, &[jo], b_len);

        // 4. lm_head + greedy argmax OR vocab-sliced GPU top-k/top-p sampler -> writes next token ids to toks[tok_out_idx]
        e.buf(&self.toks[tok_out_idx], 0, 6);
        let w = &eng.output;
        match sample {
            Some(sc) if !sc.greedy() => {
                let mut vstart = 0usize;
                let mut first = true;
                while vstart < eng.n_vocab {
                    let vcount = (eng.n_vocab - vstart).min(LM_VSLICE);
                    let la = LmHeadArgs {
                        w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty,
                        n_vocab: eng.n_vocab as u32, k: w.k, xh_off: d.fh, part_off: d.slice_buf,
                        b: b_len as u32, nvch: 0, vstart: vstart as u32, vcount: vcount as u32,
                    };
                    let tiles = vcount.div_ceil(64);
                    if b_len > 64 { self.go(&e, C_LMHEAD, "k_lmhead_slice_sg8", &la, (b_len.div_ceil(128), tiles, 1), 256); }
                    else { self.go(&e, C_LMHEAD, "k_lmhead_slice_sg4", &la, (1, tiles, 1), 128); }
                    let tm = TopkMergeArgs {
                        slice_off: d.slice_buf, vstart: vstart as u32, vcount: vcount as u32, topk_off: d.topk_buf,
                        top_k: sc.top_k.clamp(1, 64) as u32, first_slice: first as u32, pad0: 0, pad1: 0,
                    };
                    self.go(&e, C_ARGMAX, "k_topk_merge", &tm, (b_len, 1, 1), 256);
                    vstart += vcount;
                    first = false;
                }
                e.buf(&self.rng_buf, 0, 7);
                let sa = SampleBatchArgs {
                    topk_off: d.topk_buf, top_k: sc.top_k.clamp(1, 64) as u32, out_base: 0, rng_off: 0,
                    temp: sc.temp, top_p: sc.top_p, softcap: eng.softcap, pad: 0.0,
                };
                self.go(&e, C_ARGMAX, "k_sample_topk", &sa, (b_len, 1, 1), 64);
            }
            _ => {
                let nvch = eng.n_vocab.div_ceil(LM_VCHUNK);
                let la = LmHeadArgs {
                    w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty,
                    n_vocab: eng.n_vocab as u32, k: w.k, xh_off: d.fh, part_off: d.am_part,
                    b: b_len as u32, nvch: nvch as u32, vstart: 0, vcount: 0,
                };
                if b_len > 64 { self.go(&e, C_LMHEAD, "k_lmhead_argmax_sg8", &la, (b_len.div_ceil(128), nvch, 1), 256); }
                else { self.go(&e, C_LMHEAD, "k_lmhead_argmax_sg4", &la, (1, nvch, 1), 128); }
                let aa = ArgmaxBatchArgs { part_off: d.am_part, nvch: nvch as u32, out_base: 0, pad: 0 };
                self.go(&e, C_ARGMAX, "k_argmax_batch", &aa, (b_len, 1, 1), 128);
            }
        }
        e.end();
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_layer(
        &self, e: &Enc, b_len: usize, max_pos: usize, min_suf: usize,
        slot: usize, ip: usize, cur: usize, blocks: &mut Vec<usize>,
        msg: Option<(usize, usize)>, va: &mut VecAlloc, pinned: &[usize],
    ) -> usize {
        let eng = self.eng;
        let l = &eng.layers[ip];
        let d = self.da;
        let ctl = msg.map(|_| &eng.loops[ip - LOOP_START]);
        let es = E as u32;

        // ---- 1. Attention inputs (depth mix + optional loop controller -> half xh[0..3])
        let norms: [i32; 3] = if l.swa { [l.attn_norm; 3] } else { [l.q_inp_norm, l.k_inp_norm, l.v_inp_norm] };
        let mut st: Vec<u32> = blocks.iter().map(|&v| d.vec(v)).collect();
        st.push(d.vec(cur));
        let nq = if l.attn_res_query >= 0 { l.n_attn_q } else { 1 };
        if let Some(lw) = ctl {
            let (b0, b1) = msg.unwrap();
            let mut ja = eng.mix_job_off(&st, l.attn_res_query, nq, &d.inp[..nq as usize], &[], es);
            let mut jm = eng.mix_job_off(&[d.vec(b0), d.vec(b1)], lw.msg_query as i32, 4, &d.msg, &[], es);
            if l.swa { ja.hcopy[0] = [d.cath[0] as i32, d.cath[1] as i32, d.cath[2] as i32]; }
            else { for j in 0..3 { ja.hcopy[j][0] = d.cath[j] as i32; } }
            for j in 0..4 { jm.hcopy[j][0] = (d.cath[j] + (E / 2) as u32) as i32; }
            self.mix_b(e, C_MIXA, &[ja, jm], b_len);
            let jobs: Vec<(usize, u32, u32, u32)> = (0..3).map(|j| (j, if l.swa { d.inp[0] } else { d.inp[j] }, d.msg[j], norms[j] as u32)).collect();
            self.ctrl_b(e, lw, &jobs, b_len, &[d.xh[0], d.xh[1], d.xh[2]]);
        } else {
            let nn: Vec<i32> = if l.swa { vec![l.attn_norm] } else { norms.to_vec() };
            let mut ja = eng.mix_job_off(&st, l.attn_res_query, nq, &d.inp[..nq as usize], &nn, es);
            for q in 0..nq as usize { ja.hout[q] = d.xh[q] as i32; }
            ja.nof32 = 0xF;
            self.mix_b(e, C_MIXA, &[ja], b_len);
        }

        // ---- 2. QKV + Gate + Rel + Sink (+ dyn conv taps) fused segmented GEMM
        let xi: [usize; 3] = if ctl.is_some() || !l.swa { [0, 1, 2] } else { [0, 0, 0] };
        let mut hs = vec![
            Engine::hseg(&l.wq, d.xh[xi[0]], d.q),
            Engine::hseg(&l.wgate, d.xh[xi[0]], d.gate),
            Engine::hseg(&l.wrel, d.xh[xi[0]], d.rel),
            Engine::hseg(&l.wsink, d.xh[xi[0]], d.sink),
            Engine::hseg(&l.wk, d.xh[xi[1]], d.k),
            Engine::hseg(&l.wv, d.xh[xi[2]], d.v),
        ];
        hs[0].hout = 1; // q as half
        hs[1].hout = 1; // gate as half
        if let (Some(kp), Some(vp)) = (&l.kconvp, &l.vconvp) {
            hs.push(Engine::hseg(kp, d.xh[xi[1]], d.kdyn));
            hs.push(Engine::hseg(vp, d.xh[xi[2]], d.vdyn));
        }
        self.gemm_h(e, C_QKV, &hs, b_len);

        // ---- 3. Slot-major paged decode attention (with M5 neural-accelerator shared-prefix cascade if prefix visible!)
        let (is_swa, slot_idx) = eng.slot_kind[slot];
        let pool_cap = if is_swa { self.pool.swa_cap } else { self.pool.full_cap };
        if is_swa { e.buf(&self.pool.swa, 0, 3); } else { e.buf(&self.pool.full, 0, 3); }
        let pt_off = if is_swa { self.ml.swa_pt } else { self.ml.full_pt };
        let pt_stride = if is_swa { SWA_RING_BLKS } else { self.max_full_blks };
        let prefix_len = self.prefix.as_ref().map_or(0, |p| p.prefix_len);
        let use_prefix = prefix_len > 0 && (!is_swa || min_suf < EXT);

        let pa_base = PagedAttnArgs {
            b: b_len as u32, pool_cap: pool_cap as u32, slot_idx: slot_idx as u32, is_swa: is_swa as u32,
            pt_stride: pt_stride as u32, pt_mod: SWA_RING_BLKS as u32,
            seq_pos_off: self.ml.seq_pos, seq_suf_off: self.ml.seq_suf, seq_cs_off: self.ml.seq_cs, pt_off,
            layer_slot: slot as u32, max_batch: self.max_batch as u32,
            q_off: d.q, k_off: d.k, v_off: d.v, r_off: d.rel, sink_off: d.sink,
            kdyn_off: if l.swa { -1 } else { d.kdyn as i32 }, vdyn_off: if l.swa { -1 } else { d.vdyn as i32 },
            kconv_t_w: l.k_conv_t, vconv_t_w: l.v_conv_t, qclip_w: l.q_clip, relclip_w: l.rel_clip, relbias_w: l.rel_bias,
            log_alpha: eng.log_alpha, log_floor: eng.log_floor,
            gate_off: d.gate, out_off: d.agh, part_off: d.pre_mlo, nsplit: 1, prefix_len: prefix_len as u32,
            qp_off: d.qp, rp_off: d.rp, pad0: 0,
        };

        if use_prefix {
            let pre = self.prefix.as_ref().unwrap();
            // Step 1: K/V conv + ring CS update + paged write + GQA-packed qp/rp
            self.go(e, C_KVPREP, "k_kvprep_dec_b", &pa_base, (b_len, NKV, 1), 128);
            // Step 2: M5 neural-accelerator GQA-packed shared prefix attention -> pre_mlo
            let (pre_buf, pre_n, pre_p0, pre_cap) = if is_swa {
                (&pre.pre_swa, pre.pre_swa_n, pre.pre_swa_p0, pre.pre_swa_cap)
            } else {
                (&pre.pre_full, pre.prefix_len, 0usize, pre.pre_full_cap)
            };
            e.buf(pre_buf, 0, 8);
            let k_base = (slot_idx * 2 * NKV * pre_cap * HD) as u32;
            let v_base = ((slot_idx * 2 + 1) * NKV * pre_cap * HD) as u32;
            let pfa = PrefixAttnArgs {
                b: b_len as u32, pre_n: pre_n as u32, pre_p0: pre_p0 as u32, pre_cap: pre_cap as u32,
                k_base, v_base, window: if is_swa { EXT as u32 } else { 0 },
                qp_off: d.qp, rp_off: d.rp, sink_off: d.sink, mlo_off: d.pre_mlo, relbias_w: l.rel_bias,
                seq_pos_off: self.ml.seq_pos, pad0: 0, pad1: 0, pad2: 0,
            };
            self.go(e, C_ATTN, "k_attn_prefix_h", &pfa, ((b_len * 3).div_ceil(32), NKV, 1), 256);
            // Step 3: Private paged suffix attention + exact online softmax merge + sigmoid gate -> half agh
            self.go(e, C_COMB, "k_attn_suffix_paged_b", &pa_base, (NKV, b_len, 1), 128);
        } else {
            // Single-pass fused paged decode attention (handles both prefix_len==0 and SWA beyond prefix window!)
            let nkeys_max = if is_swa { (max_pos + 1).min(EXT) } else { max_pos + 1 };
            let nsplit = if b_len * NKV >= 64 && nkeys_max <= 2048 {
                1usize
            } else {
                (64usize.div_ceil(b_len * NKV)).max(nkeys_max.div_ceil(1024)).min(d.max_split).max(1)
            };
            let pa = PagedAttnArgs { nsplit: nsplit as u32, ..pa_base };
            self.go(e, C_ATTN, "k_attn_paged_b", &pa, (nsplit, NKV, b_len), 128);
            if nsplit > 1 {
                let ca = CombineBArgs { part_off: d.pre_mlo, nsplit: nsplit as u32, gate_off: d.gate, out_off: d.agh };
                self.go(e, C_COMB, "k_attn_combine_b", &ca, (NH, b_len, 1), 128);
            }
        }

        // ---- 4. Fused wo GEMM + vectorized SoA ring post-conv + optional residual -> vec(mid)
        let block_start = ip % BLOCK == 0;
        let mut live: Vec<usize> = blocks.clone();
        live.push(cur);
        live.extend_from_slice(pinned);
        let mid = va.alloc(&live);
        let wo = WoPostArgs {
            w_lo: l.wo.off as u32, w_hi: (l.wo.off >> 32) as u32, row_bytes: l.wo.row_bytes, wtype: l.wo.ty,
            nrows: l.wo.rows, k: l.wo.k, xh_off: d.agh, out_off: d.vec(mid),
            res_off: d.vec(cur), has_res: (!block_start) as u32,
            layer_slot: slot as u32, max_batch: self.max_batch as u32, convw_t_off: l.attn_post_conv_t,
            b: b_len as u32, seq_cs_off: self.ml.seq_cs, seq_pos_off: self.ml.seq_pos,
        };
        if b_len > 64 { self.go(e, C_WO, "k_wo_post_sg8", &wo, (b_len.div_ceil(128), E / 64, 1), 256); }
        else { self.go(e, C_WO, "k_wo_post_sg4", &wo, (1, E / 64, 1), 128); }
        if block_start { blocks.push(cur); }

        // ---- 5. FFN input (depth mix + optional loop controller -> half fh)
        let mut st: Vec<u32> = blocks.iter().map(|&v| d.vec(v)).collect();
        st.push(d.vec(mid));
        if let Some(lw) = ctl {
            let mut jf = eng.mix_job_off(&st, l.ffn_res_query as i32, 1, &[d.fin], &[], es);
            jf.hcopy[0][0] = d.cath[3] as i32;
            self.mix_b(e, C_MIXF, &[jf], b_len);
            self.ctrl_b(e, lw, &[(3, d.fin, d.msg[3], l.ffn_norm)], b_len, &[d.fh]);
        } else {
            let mut jf = eng.mix_job_off(&st, l.ffn_res_query as i32, 1, &[d.fin], &[l.ffn_norm as i32], es);
            jf.hout[0] = d.fh as i32;
            jf.nof32 = 1;
            self.mix_b(e, C_MIXF, &[jf], b_len);
        }

        // ---- 6. 4-class MoE (router + topk + route4 + gather + gate_up + down)
        self.moe_4class(e, l, b_len);

        // ---- 7. Fused MoE combine + vectorized SoA ring post-conv + residual -> vec(out)
        let mut live: Vec<usize> = blocks.clone();
        live.push(mid);
        live.extend_from_slice(pinned);
        let out = va.alloc(&live);
        let cp = MoeCombPostArgs {
            yd_off: d.yd, inv_off: d.inv, res_off: d.vec(mid), out_off: d.vec(out),
            layer_slot: slot as u32, max_batch: self.max_batch as u32, convw_t_off: l.ffn_post_conv_t,
            seq_cs_off: self.ml.seq_cs, seq_pos_off: self.ml.seq_pos, pad0: 0, pad1: 0, pad2: 0,
        };
        self.go(e, C_COMB2, "k_moe_comb_post_b", &cp, (b_len, E.div_ceil(1024), 1), 256);
        out
    }

    // ------------------------------------------------------------------------- Packed Multi-Prompt Batched Prefill
    /// Prefill a batch of independent prompts packed together into the paged KV cache & CS pool at full tensor-core speed!
    /// Returns initialized `SeqSlot`s (each with its first predicted token already computed in `slot.next_tok`).
    pub fn prefill_packed(&mut self, prompts: &[(usize, &[u32], usize)], sample: Option<SampleCfg>) -> (Vec<SeqSlot>, usize, Duration) {
        assert!(self.prefix.is_none(), "prefill_packed is for independent prompts (no shared prefix)");
        let t0 = Instant::now();
        let total_toks: usize = prompts.iter().map(|p| p.1.len()).sum();
        let mut slots: Vec<SeqSlot> = Vec::with_capacity(prompts.len());
        let mut cs_ids = Vec::with_capacity(prompts.len());
        let (mut total_need_swa, mut total_need_full) = (0usize, 0usize);
        for &(id, toks, max_new) in prompts {
            assert!(!toks.is_empty() && toks.len() < self.max_ctx);
            let cs_slot = self.alloc_cs();
            cs_ids.push(cs_slot);
            let pf = toks.len().div_ceil(KV_BLK);
            total_need_full += pf;
            total_need_swa += pf.min(SWA_RING_BLKS);
            slots.push(SeqSlot {
                id, cs_slot, pos: 0, suf: 0, swa_pages: Vec::new(), full_pages: Vec::new(),
                generated: Vec::new(), max_new, next_tok: 0, done: false,
            });
        }
        self.zero_cs(&cs_ids);

        // Ensure KV pool pages for all admitted prompts upfront so memory budget for packed chunk is exact
        let kv_bud = self.kv_budget();
        self.pool.ensure_free(&self.eng.gpu, total_need_swa, total_need_full, kv_bud);

        // Long prompts (> EXT = 1024 tokens) would wrap the 33-page SWA ring if prefilled in-place in a single chunk;
        // prefill them individually via Engine::encode_prefill (linear SWA scratch, ~2200 tok/s) and copy into paged KV!
        let need_slice = sample.is_some_and(|s| !s.greedy());
        let mut short_indices: Vec<usize> = Vec::with_capacity(prompts.len());
        for (i, &(_, toks, _)) in prompts.iter().enumerate() {
            if toks.len() > EXT {
                self.prefill_long_into_slot(&mut slots[i], toks, sample);
            } else {
                short_indices.push(i);
            }
        }

        // Pack all short prompts (each <= 1024 tokens, p0 = 0 -> zero SWA ring wrap conflict!) into chunks <= chunk_lim
        if !short_indices.is_empty() {
            let short_toks: usize = short_indices.iter().map(|&i| prompts[i].1.len()).sum();
            let non_act = self.kv_bytes() + self.cs_bytes() + self.meta_bytes();
            let act_budget = self.max_mem_bytes.saturating_sub(non_act);
            let mut chunk_lim = self.eng.chunk.min(short_toks.max(64));
            while chunk_lim > 64 {
                let da = DecodeArena::new(chunk_lim, self.eng.n_vocab, need_slice, true);
                if da.end as usize * 4 <= act_budget { break; }
                chunk_lim /= 2;
            }

            let mut k = 0usize;
            while k < short_indices.len() {
                let mut items: Vec<(usize, usize, &[u32], bool)> = Vec::new();
                let mut t_sum = 0usize;
                while k < short_indices.len() {
                    let si = short_indices[k];
                    let p = prompts[si].1;
                    if t_sum > 0 && t_sum + p.len() > chunk_lim { break; }
                    items.push((si, 0, p, true));
                    t_sum += p.len();
                    k += 1;
                }
                self.run_packed_chunk(&mut slots, &items, t_sum, sample, need_slice);
            }
        }
        // Immediately shrink activation arena back down to decode size!
        self.shrink_arena(self.max_batch, need_slice);
        (slots, total_toks, t0.elapsed())
    }

    /// Prefill a single prompt > EXT tokens via Engine::encode_prefill and scatter its KV & CS into `slot`'s pages.
    fn prefill_long_into_slot(&mut self, slot: &mut SeqSlot, toks: &[u32], sample: Option<SampleCfg>) {
        let p_len = toks.len();
        for p in (0..p_len).step_by(KV_BLK) { slot.ensure_pages_for(&mut self.pool, p); }
        slot.ensure_pages_for(&mut self.pool, p_len - 1);

        self.eng.reset();
        self.eng.reserve(p_len + 1, false);
        self.eng.ensure_batch(p_len.min(self.eng.chunk));
        self.eng.set_tokens(0, toks);
        let mut p0 = 0;
        let mut last = None;
        while p0 < p_len {
            let t = (p_len - p0).min(self.eng.chunk);
            let cb = self.eng.gpu.cmd();
            self.eng.encode_prefill(&cb, p0, t, p0 + t == p_len, true);
            cb.commit();
            last = Some(cb);
            p0 += t;
        }
        last.unwrap().waitUntilCompleted();
        let next_tok = match sample {
            Some(sc) if !sc.greedy() => {
                let mut smp = crate::sampler::Sampler::new(sc.temp, sc.top_k, sc.top_p, self.eng.softcap, sc.seed.wrapping_add(slot.id as u64 * 0x9E37_79B9_7F4A_7C15));
                smp.sample(self.eng.logits())
            }
            _ => self.eng.token_at(p_len),
        };

        // Upload slot's SWA page table, Full page table, and cs_slot into meta[0]
        let mp = self.meta[0].contents().as_ptr() as *mut u32;
        let swa_off = 0usize;
        let full_off = SWA_RING_BLKS.div_ceil(16) * 16;
        let cs_off = full_off + slot.full_pages.len().div_ceil(16) * 16;
        unsafe {
            for (k, &pid) in slot.swa_pages.iter().enumerate() { *mp.add(swa_off + k) = pid; }
            for (k, &pid) in slot.full_pages.iter().enumerate() { *mp.add(full_off + k) = pid; }
            *mp.add(cs_off) = slot.cs_slot;
        }

        let cb = self.eng.gpu.cmd();
        let e = Enc::new(&cb);
        e.buf(&self.meta[0], 0, 7);
        // Copy live SWA window [p_len - EXT .. p_len) into slot-major paged SWA pool
        let swa_n = p_len.min(EXT);
        let swa_p0 = p_len - swa_n;
        let a_swa = CopyToPagedArgs {
            n_pos: swa_n as u32, pos0: swa_p0 as u32, src_cap: self.eng.swa_cap.get() as u32, is_swa: 1,
            pool_cap: self.pool.swa_cap as u32, pt_off: swa_off as u32, pt_mod: SWA_RING_BLKS as u32, suf0: swa_p0 as u32,
        };
        e.pso(&self.eng.gpu.pso("k_copy_kv_to_paged"));
        e.buf(&self.eng.kv_swa.borrow(), 0, 3);
        e.buf(&self.pool.swa, 0, 8);
        e.bytes(&a_swa, 5);
        e.dispatch((swa_n, NKV, N_SWA), 32);
        e.barrier();
        // Copy all Full positions [0 .. p_len) into slot-major paged Full pool
        let a_full = CopyToPagedArgs {
            n_pos: p_len as u32, pos0: 0, src_cap: self.eng.full_cap.get() as u32, is_swa: 0,
            pool_cap: self.pool.full_cap as u32, pt_off: full_off as u32, pt_mod: SWA_RING_BLKS as u32, suf0: 0,
        };
        e.buf(&self.eng.kv_full.borrow(), 0, 3);
        e.buf(&self.pool.full, 0, 8);
        e.bytes(&a_full, 5);
        e.dispatch((p_len, NKV, N_FULL), 32);
        e.barrier();
        // Broadcast Engine's AoS conv state into slot-major SoA ring CS_POOL at p_len % 3
        let a_cs = CopyCsArgs { dst_off: cs_off as u32, n: 1, max_batch: self.max_batch as u32, prefix_len: p_len as u32 };
        e.pso(&self.eng.gpu.pso("k_broadcast_cs"));
        e.buf(&self.eng.cs, 0, 4);
        e.buf(&self.cs_pool, 0, 8);
        e.bytes(&a_cs, 5);
        e.dispatch((NCH.div_ceil(256), 1, N_SLOT), 256);
        e.end();
        cb.commit();
        cb.waitUntilCompleted();

        self.eng.release_batch();
        self.eng.shrink_kv();
        slot.pos = p_len;
        slot.suf = p_len;
        slot.next_tok = next_tok;
    }

    fn run_packed_chunk(
        &mut self, slots: &mut [SeqSlot], items: &[(usize, usize, &[u32], bool)],
        t_tot: usize, sample: Option<SampleCfg>, need_slice: bool,
    ) {
        let eng = self.eng;
        let nseq = items.len();
        // Ensure paged KV capacity for all tokens in this chunk
        let (mut need_swa, mut need_full) = (0usize, 0usize);
        for &(si, p0, toks, _) in items {
            let end_p = p0 + toks.len() - 1;
            let (ns, nf) = slots[si].pages_needed_for(end_p);
            need_swa += ns; need_full += nf;
        }
        let kv_bud = self.kv_budget();
        self.pool.ensure_free(&eng.gpu, need_swa, need_full, kv_bud);
        for &(si, p0, toks, _) in items {
            for p in (p0..p0 + toks.len()).step_by(KV_BLK) { slots[si].ensure_pages_for(&mut self.pool, p); }
            slots[si].ensure_pages_for(&mut self.pool, p0 + toks.len() - 1);
        }

        // Ensure activation arena is big enough for t_tot tokens (plus packed wo_y/ffn_y!)
        self.ensure_arena(t_tot, need_slice, true);

        // Build packed token buffer & metadata buffer
        let tok_buf = eng.gpu.buffer(t_tot.max(16) * 4);
        let mut qtiles: Vec<[u32; 4]> = Vec::new();
        let mut tok_pos: Vec<u32> = Vec::with_capacity(t_tot);
        let mut tok_seq: Vec<u32> = Vec::with_capacity(t_tot);
        let mut seq_start: Vec<u32> = Vec::with_capacity(nseq);
        let mut seq_len: Vec<u32> = Vec::with_capacity(nseq);
        let mut seq_p0: Vec<u32> = Vec::with_capacity(nseq);
        let mut seq_cs: Vec<u32> = Vec::with_capacity(nseq);
        let mut last_indices: Vec<u32> = Vec::new();
        let mut finished_slots: Vec<usize> = Vec::new();

        let tp = tok_buf.contents().as_ptr() as *mut i32;
        let mut cur_t = 0u32;
        for (s_loc, &(si, p0, toks, is_last)) in items.iter().enumerate() {
            seq_start.push(cur_t);
            seq_len.push(toks.len() as u32);
            seq_p0.push(p0 as u32);
            seq_cs.push(slots[si].cs_slot);
            for (k, &tok) in toks.iter().enumerate() {
                unsafe { *tp.add(cur_t as usize + k) = tok as i32; }
                tok_pos.push((p0 + k) as u32);
                tok_seq.push(s_loc as u32);
            }
            let mut lt = 0usize;
            while lt < toks.len() {
                let nq = (toks.len() - lt).min(32);
                qtiles.push([s_loc as u32, cur_t + lt as u32, (p0 + lt) as u32, nq as u32]);
                lt += nq;
            }
            if is_last {
                last_indices.push(cur_t + toks.len() as u32 - 1);
                finished_slots.push(si);
            }
            cur_t += toks.len() as u32;
        }

        // Pack metadata into a temporary GPU buffer
        let mfb = self.max_full_blks;
        let mut mo = 0usize;
        let mut take_m = |n: usize| { let r = mo; mo += n.div_ceil(16) * 16; r };
        let o_tok_pos = take_m(t_tot);
        let o_tok_seq = take_m(t_tot);
        let o_seq_start = take_m(nseq);
        let o_seq_len = take_m(nseq);
        let o_seq_p0 = take_m(nseq);
        let o_seq_cs = take_m(nseq);
        let o_swa_pt = take_m(nseq * SWA_RING_BLKS);
        let o_full_pt = take_m(nseq * mfb);
        let o_qtiles = take_m(qtiles.len() * 4);
        let o_last = take_m(last_indices.len().max(1));
        let pmeta = eng.gpu.buffer(mo.max(16) * 4);
        let mp = pmeta.contents().as_ptr() as *mut u32;
        unsafe {
            std::ptr::copy_nonoverlapping(tok_pos.as_ptr(), mp.add(o_tok_pos), t_tot);
            std::ptr::copy_nonoverlapping(tok_seq.as_ptr(), mp.add(o_tok_seq), t_tot);
            std::ptr::copy_nonoverlapping(seq_start.as_ptr(), mp.add(o_seq_start), nseq);
            std::ptr::copy_nonoverlapping(seq_len.as_ptr(), mp.add(o_seq_len), nseq);
            std::ptr::copy_nonoverlapping(seq_p0.as_ptr(), mp.add(o_seq_p0), nseq);
            std::ptr::copy_nonoverlapping(seq_cs.as_ptr(), mp.add(o_seq_cs), nseq);
            for (s_loc, &(si, _, _, _)) in items.iter().enumerate() {
                for (k, &pid) in slots[si].swa_pages.iter().enumerate() { *mp.add(o_swa_pt + s_loc * SWA_RING_BLKS + k) = pid; }
                for (k, &pid) in slots[si].full_pages.iter().enumerate() { *mp.add(o_full_pt + s_loc * mfb + k) = pid; }
            }
            for (i, q) in qtiles.iter().enumerate() {
                for j in 0..4 { *mp.add(o_qtiles + i * 4 + j) = q[j]; }
            }
            for (i, &li) in last_indices.iter().enumerate() { *mp.add(o_last + i) = li; }
        }

        let out_tok_buf = eng.gpu.buffer(finished_slots.len().max(16) * 4);
        let cb = eng.gpu.cmd();
        let e = Enc::new(&cb);
        e.buf(&eng.w, 0, 0);
        e.buf(&eng.pb, 0, 1);
        e.buf(&self.act, 0, 2);
        e.buf(&self.pool.swa, 0, 3);
        e.buf(&self.cs_pool, 0, 4);
        e.buf(&tok_buf, 0, 6);
        e.buf(&pmeta, 0, 7);

        let d = self.da;
        let ea = EmbedArgs {
            emb_lo: eng.tok_embd.off as u32, emb_hi: (eng.tok_embd.off >> 32) as u32, row_bytes: eng.tok_embd.row_bytes,
            norm_off: eng.tok_norm, out0: d.vec(0), out1: d.vec(0), tok_index: 0, eps: eng.eps,
        };
        self.go(&e, C_EMBED, "k_embed", &ea, (t_tot, 1, 1), 256);

        let pm = PackedMetaOffs {
            tok_pos: o_tok_pos as u32, tok_seq: o_tok_seq as u32, seq_start: o_seq_start as u32, seq_len: o_seq_len as u32,
            seq_p0: o_seq_p0 as u32, seq_cs: o_seq_cs as u32, swa_pt: o_swa_pt as u32, full_pt: o_full_pt as u32,
            qtiles: o_qtiles as u32, nseq, n_qtiles: qtiles.len(),
        };
        let mut va = VecAlloc { next: 1, n: B_NVEC };
        let mut blocks: Vec<usize> = Vec::new();
        let mut cur = 0usize;
        for ip in 0..LOOP_START { cur = self.packed_layer(&e, t_tot, &pm, ip, ip, cur, &mut blocks, None, &mut va, &[]); }
        let prelude = cur;
        let prelude_blocks = blocks.clone();
        let mut bank = (0usize, 0usize);
        for pass in 0..N_PASS {
            if pass > 0 { cur = prelude; blocks = prelude_blocks.clone(); }
            let msg = if pass > 0 { Some(bank) } else { None };
            let pinned: Vec<usize> = { let mut p = vec![prelude]; if pass > 0 { p.push(bank.0); p.push(bank.1); } p };
            for ip in LOOP_START..LOOP_END {
                cur = self.packed_layer(&e, t_tot, &pm, slot_of(pass, ip), ip, cur, &mut blocks, msg, &mut va, &pinned);
            }
            bank = (blocks[prelude_blocks.len() + 1], cur);
        }
        for ip in LOOP_END..N_PHYS { cur = self.packed_layer(&e, t_tot, &pm, ip, ip, cur, &mut blocks, None, &mut va, &[]); }

        // Compute first predicted token for every prompt that finished in this chunk
        let n_fin = finished_slots.len();
        if n_fin > 0 {
            let mut s_off = [0u32; 6];
            let mut st: Vec<u32> = blocks.iter().map(|&v| d.vec(v)).collect();
            st.push(d.vec(cur));
            for (i, &v) in st.iter().enumerate() { s_off[i] = v; }
            let oma = OutMixGatherArgs {
                n_s: st.len() as u32, s: n_fin as u32, s_off,
                q_off: eng.output_res_query, norm_off: eng.output_norm, hout_off: d.fh, last_idx_off: o_last as u32,
                eps: eng.eps, pad0: 0, pad1: 0, pad2: 0,
            };
            self.go(&e, C_OMIX, "k_out_mix_gather", &oma, (n_fin, 1, 1), 32);
            e.buf(&out_tok_buf, 0, 6);
            let w = &eng.output;
            match sample {
                Some(sc) if !sc.greedy() => {
                    let mut vstart = 0usize;
                    let mut first = true;
                    while vstart < eng.n_vocab {
                        let vcount = (eng.n_vocab - vstart).min(LM_VSLICE);
                        let la = LmHeadArgs {
                            w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty,
                            n_vocab: eng.n_vocab as u32, k: w.k, xh_off: d.fh, part_off: d.slice_buf,
                            b: n_fin as u32, nvch: 0, vstart: vstart as u32, vcount: vcount as u32,
                        };
                        let tiles = vcount.div_ceil(64);
                        if n_fin > 64 { self.go(&e, C_LMHEAD, "k_lmhead_slice_sg8", &la, (n_fin.div_ceil(128), tiles, 1), 256); }
                        else { self.go(&e, C_LMHEAD, "k_lmhead_slice_sg4", &la, (1, tiles, 1), 128); }
                        let tm = TopkMergeArgs {
                            slice_off: d.slice_buf, vstart: vstart as u32, vcount: vcount as u32, topk_off: d.topk_buf,
                            top_k: sc.top_k.clamp(1, 64) as u32, first_slice: first as u32, pad0: 0, pad1: 0,
                        };
                        self.go(&e, C_ARGMAX, "k_topk_merge", &tm, (n_fin, 1, 1), 256);
                        vstart += vcount;
                        first = false;
                    }
                    e.buf(&self.rng_buf, 0, 7);
                    let sa = SampleBatchArgs {
                        topk_off: d.topk_buf, top_k: sc.top_k.clamp(1, 64) as u32, out_base: 0, rng_off: 0,
                        temp: sc.temp, top_p: sc.top_p, softcap: eng.softcap, pad: 0.0,
                    };
                    self.go(&e, C_ARGMAX, "k_sample_topk", &sa, (n_fin, 1, 1), 64);
                }
                _ => {
                    let nvch = eng.n_vocab.div_ceil(LM_VCHUNK);
                    let la = LmHeadArgs {
                        w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty,
                        n_vocab: eng.n_vocab as u32, k: w.k, xh_off: d.fh, part_off: d.am_part,
                        b: n_fin as u32, nvch: nvch as u32, vstart: 0, vcount: 0,
                    };
                    if n_fin > 64 { self.go(&e, C_LMHEAD, "k_lmhead_argmax_sg8", &la, (n_fin.div_ceil(128), nvch, 1), 256); }
                    else { self.go(&e, C_LMHEAD, "k_lmhead_argmax_sg4", &la, (1, nvch, 1), 128); }
                    let aa = ArgmaxBatchArgs { part_off: d.am_part, nvch: nvch as u32, out_base: 0, pad: 0 };
                    self.go(&e, C_ARGMAX, "k_argmax_batch", &aa, (n_fin, 1, 1), 128);
                }
            }
        }
        e.end();
        cb.commit();
        cb.waitUntilCompleted();

        for &(si, p0, toks, _) in items {
            slots[si].pos = p0 + toks.len();
            slots[si].suf = p0 + toks.len();
        }
        let otp = out_tok_buf.contents().as_ptr() as *const i32;
        for (i, &si) in finished_slots.iter().enumerate() {
            slots[si].next_tok = unsafe { *otp.add(i) as u32 };
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn packed_layer(
        &self, e: &Enc, t: usize, pm: &PackedMetaOffs, slot: usize, ip: usize, cur: usize,
        blocks: &mut Vec<usize>, msg: Option<(usize, usize)>, va: &mut VecAlloc, pinned: &[usize],
    ) -> usize {
        let eng = self.eng;
        let l = &eng.layers[ip];
        let d = self.da;
        let ctl = msg.map(|_| &eng.loops[ip - LOOP_START]);
        let es = E as u32;

        // 1. Attention inputs
        let norms: [i32; 3] = if l.swa { [l.attn_norm; 3] } else { [l.q_inp_norm, l.k_inp_norm, l.v_inp_norm] };
        let mut st: Vec<u32> = blocks.iter().map(|&v| d.vec(v)).collect();
        st.push(d.vec(cur));
        let nq = if l.attn_res_query >= 0 { l.n_attn_q } else { 1 };
        if let Some(lw) = ctl {
            let (b0, b1) = msg.unwrap();
            let mut ja = eng.mix_job_off(&st, l.attn_res_query, nq, &d.inp[..nq as usize], &[], es);
            let mut jm = eng.mix_job_off(&[d.vec(b0), d.vec(b1)], lw.msg_query as i32, 4, &d.msg, &[], es);
            if l.swa { ja.hcopy[0] = [d.cath[0] as i32, d.cath[1] as i32, d.cath[2] as i32]; }
            else { for j in 0..3 { ja.hcopy[j][0] = d.cath[j] as i32; } }
            for j in 0..4 { jm.hcopy[j][0] = (d.cath[j] + (E / 2) as u32) as i32; }
            self.mix_b(e, C_MIXA, &[ja, jm], t);
            let jobs: Vec<(usize, u32, u32, u32)> = (0..3).map(|j| (j, if l.swa { d.inp[0] } else { d.inp[j] }, d.msg[j], norms[j] as u32)).collect();
            self.ctrl_b(e, lw, &jobs, t, &[d.xh[0], d.xh[1], d.xh[2]]);
        } else {
            let nn: Vec<i32> = if l.swa { vec![l.attn_norm] } else { norms.to_vec() };
            let mut ja = eng.mix_job_off(&st, l.attn_res_query, nq, &d.inp[..nq as usize], &nn, es);
            for q in 0..nq as usize { ja.hout[q] = d.xh[q] as i32; }
            ja.nof32 = 0xF;
            self.mix_b(e, C_MIXA, &[ja], t);
        }

        // 2. QKV projections
        let xi: [usize; 3] = if ctl.is_some() || !l.swa { [0, 1, 2] } else { [0, 0, 0] };
        let mut hs = vec![
            Engine::hseg(&l.wq, d.xh[xi[0]], d.q), Engine::hseg(&l.wgate, d.xh[xi[0]], d.gate),
            Engine::hseg(&l.wrel, d.xh[xi[0]], d.rel), Engine::hseg(&l.wsink, d.xh[xi[0]], d.sink),
            Engine::hseg(&l.wk, d.xh[xi[1]], d.k), Engine::hseg(&l.wv, d.xh[xi[2]], d.v),
        ];
        hs[0].hout = 1; hs[1].hout = 1;
        if let (Some(kp), Some(vp)) = (&l.kconvp, &l.vconvp) {
            hs.push(Engine::hseg(kp, d.xh[xi[1]], d.kdyn));
            hs.push(Engine::hseg(vp, d.xh[xi[2]], d.vdyn));
        }
        self.gemm_h(e, C_QKV, &hs, t);

        // 3. Packed K/V prep into slot-major paged KV pool + ring conv state updates + packed causal attention
        let (is_swa, slot_idx) = eng.slot_kind[slot];
        let pool_cap = if is_swa { self.pool.swa_cap } else { self.pool.full_cap };
        if is_swa { e.buf(&self.pool.swa, 0, 3); } else { e.buf(&self.pool.full, 0, 3); }
        let pt_off = if is_swa { pm.swa_pt } else { pm.full_pt };
        let pt_stride = if is_swa { SWA_RING_BLKS } else { self.max_full_blks };
        let ka = PackedKvPrepArgs {
            t: t as u32, pool_cap: pool_cap as u32, slot_idx: slot_idx as u32, is_swa: is_swa as u32,
            pt_stride: pt_stride as u32, pt_mod: SWA_RING_BLKS as u32,
            tok_pos_off: pm.tok_pos, tok_seq_off: pm.tok_seq, seq_start_off: pm.seq_start, seq_p0_off: pm.seq_p0,
            seq_cs_off: pm.seq_cs, pt_off, layer_slot: slot as u32, max_batch: self.max_batch as u32,
            q_off: d.q, k_off: d.k, v_off: d.v, r_off: d.rel,
            kdyn_off: if l.swa { -1 } else { d.kdyn as i32 }, vdyn_off: if l.swa { -1 } else { d.vdyn as i32 },
            qp_off: d.qp, rp_off: d.rp, kconv_t_w: l.k_conv_t, vconv_t_w: l.v_conv_t, qclip_w: l.q_clip, relclip_w: l.rel_clip,
            log_alpha: eng.log_alpha, log_floor: eng.log_floor,
        };
        self.go(e, C_KVPREP, "k_packed_kvprep_b", &ka, (t, NKV, 1), 128);
        let suk = PackedStateArgs {
            x_off: d.k, ldx: (NKV * HD) as u32, layer_slot: slot as u32, branch: 0, max_batch: self.max_batch as u32,
            nch: (NKV * HD) as u32, seq_start_off: pm.seq_start, seq_len_off: pm.seq_len, seq_p0_off: pm.seq_p0, seq_cs_off: pm.seq_cs, pad0: 0, pad1: 0,
        };
        self.go(e, C_KVPREP, "k_packed_state_upd", &suk, (NKV * HD / 256, pm.nseq, 1), 256);
        let suv = PackedStateArgs {
            x_off: d.v, ldx: (NKV * HD) as u32, layer_slot: slot as u32, branch: 1, max_batch: self.max_batch as u32,
            nch: (NKV * HD) as u32, seq_start_off: pm.seq_start, seq_len_off: pm.seq_len, seq_p0_off: pm.seq_p0, seq_cs_off: pm.seq_cs, pad0: 0, pad1: 0,
        };
        self.go(e, C_KVPREP, "k_packed_state_upd", &suv, (NKV * HD / 256, pm.nseq, 1), 256);

        let aa = PackedAttnArgs {
            pool_cap: pool_cap as u32, slot_idx: slot_idx as u32, is_swa: is_swa as u32,
            pt_stride: pt_stride as u32, pt_mod: SWA_RING_BLKS as u32, pt_off, qtiles_off: pm.qtiles,
            qp_off: d.qp, rp_off: d.rp, sink_off: d.sink, gate_off: d.gate, out_off: d.agh, relbias_w: l.rel_bias,
            pad0: 0, pad1: 0, pad2: 0,
        };
        self.go(e, C_ATTN, "k_packed_attn_h", &aa, (pm.n_qtiles, NH, 1), 256);

        // 4. wo + packed postconv + ring state update
        self.gemm_h(e, C_WO, &[Engine::hseg(&l.wo, d.agh, d.wo_y)], t);
        let block_start = ip % BLOCK == 0;
        let mut live: Vec<usize> = blocks.clone(); live.push(cur); live.extend_from_slice(pinned);
        let mid = va.alloc(&live);
        let cv = PackedConvArgs {
            y_off: d.wo_y, ldy: es, res_off: d.vec(cur), out_off: d.vec(mid),
            layer_slot: slot as u32, branch: 2, max_batch: self.max_batch as u32, convw_t_off: l.attn_post_conv_t,
            has_res: (!block_start) as u32, seq_start_off: pm.seq_start, seq_p0_off: pm.seq_p0, tok_seq_off: pm.tok_seq, seq_cs_off: pm.seq_cs,
            t: t as u32, pad0: 0, pad1: 0,
        };
        self.go(e, C_WOPOST, "k_packed_postconv_b", &cv, (E / 256, t, 1), 256);
        let su = PackedStateArgs {
            x_off: d.wo_y, ldx: es, layer_slot: slot as u32, branch: 2, max_batch: self.max_batch as u32,
            nch: es, seq_start_off: pm.seq_start, seq_len_off: pm.seq_len, seq_p0_off: pm.seq_p0, seq_cs_off: pm.seq_cs, pad0: 0, pad1: 0,
        };
        self.go(e, C_WOPOST, "k_packed_state_upd", &su, (E / 256, pm.nseq, 1), 256);
        if block_start { blocks.push(cur); }

        // 5. FFN input
        let mut st: Vec<u32> = blocks.iter().map(|&v| d.vec(v)).collect(); st.push(d.vec(mid));
        if let Some(lw) = ctl {
            let mut jf = eng.mix_job_off(&st, l.ffn_res_query as i32, 1, &[d.fin], &[], es);
            jf.hcopy[0][0] = d.cath[3] as i32;
            self.mix_b(e, C_MIXF, &[jf], t);
            self.ctrl_b(e, lw, &[(3, d.fin, d.msg[3], l.ffn_norm)], t, &[d.fh]);
        } else {
            let mut jf = eng.mix_job_off(&st, l.ffn_res_query as i32, 1, &[d.fin], &[l.ffn_norm as i32], es);
            jf.hout[0] = d.fh as i32;
            jf.nof32 = 1;
            self.mix_b(e, C_MIXF, &[jf], t);
        }

        // 6. MoE + combine + packed postconv + ring state update
        self.moe_4class(e, l, t);
        #[repr(C)] #[derive(Clone, Copy)]
        struct MoeCombArgsLocal { yd_off: u32, inv_off: u32, sel_off: u32, out_off: u32 }
        let ca = MoeCombArgsLocal { yd_off: d.yd, inv_off: d.inv, sel_off: d.sel, out_off: d.ffn_y };
        self.go(e, C_COMB2, "k_moe_comb_h", &ca, (t, E.div_ceil(1024), 1), 256);
        let mut live: Vec<usize> = blocks.clone(); live.push(mid); live.extend_from_slice(pinned);
        let out = va.alloc(&live);
        let cv = PackedConvArgs {
            y_off: d.ffn_y, ldy: es, res_off: d.vec(mid), out_off: d.vec(out),
            layer_slot: slot as u32, branch: 3, max_batch: self.max_batch as u32, convw_t_off: l.ffn_post_conv_t,
            has_res: 1, seq_start_off: pm.seq_start, seq_p0_off: pm.seq_p0, tok_seq_off: pm.tok_seq, seq_cs_off: pm.seq_cs,
            t: t as u32, pad0: 0, pad1: 0,
        };
        self.go(e, C_FFNPOST, "k_packed_postconv_b", &cv, (E / 256, t, 1), 256);
        let su = PackedStateArgs {
            x_off: d.ffn_y, ldx: es, layer_slot: slot as u32, branch: 3, max_batch: self.max_batch as u32,
            nch: es, seq_start_off: pm.seq_start, seq_len_off: pm.seq_len, seq_p0_off: pm.seq_p0, seq_cs_off: pm.seq_cs, pad0: 0, pad1: 0,
        };
        self.go(e, C_FFNPOST, "k_packed_state_upd", &su, (E / 256, pm.nseq, 1), 256);
        out
    }

    // ------------------------------------------------------------------------- High-level Generation APIs

    /// Generate `batch_size` parallel completions sharing `prefix_toks`, with 2-step GPU pipelining!
    /// If `stop_on_eos` is false (e.g. throughput benchmarks), every stream runs for the full `max_new` steps.
    pub fn generate_shared(
        &mut self, prefix_toks: &[u32], batch_size: usize, max_new: usize,
        sample: Option<SampleCfg>, eos: u32, stop_on_eos: bool,
    ) -> (Vec<Vec<u32>>, BatchStats) {
        assert!(batch_size >= 1 && batch_size <= self.max_batch && max_new >= 1);
        let mut stats = BatchStats { peak_batch: batch_size, ..Default::default() };
        let (first_tok, pt) = self.set_prefix(prefix_toks);
        stats.prompt_tokens = prefix_toks.len();
        stats.prompt_time = pt;

        let p_len = prefix_toks.len();
        let need_slice = sample.is_some_and(|s| !s.greedy());
        self.ensure_arena(batch_size, need_slice, false);
        if let Some(sc) = sample { self.init_rng(sc.seed, batch_size); }

        // Pre-allocate pages for the entire run if they fit so pipelining NEVER stalls to grow!
        let pages_per_seq_full = max_new.div_ceil(KV_BLK);
        let pages_per_seq_swa = pages_per_seq_full.min(SWA_RING_BLKS);
        let kv_bud = self.kv_budget();
        self.pool.ensure_free(&self.eng.gpu, batch_size * pages_per_seq_swa, batch_size * pages_per_seq_full, kv_bud);

        // Initialize B sequence slots from the shared prefix
        let mut seqs: Vec<SeqSlot> = Vec::with_capacity(batch_size);
        let mut cs_slots = Vec::with_capacity(batch_size);
        let first_logits = self.prefix.as_ref().unwrap().first_logits.clone();
        for b in 0..batch_size {
            let tok0 = match sample {
                Some(sc) if !sc.greedy() => {
                    let mut smp = crate::sampler::Sampler::new(sc.temp, sc.top_k, sc.top_p, self.eng.softcap, sc.seed.wrapping_add(b as u64 * 0x9E37_79B9_7F4A_7C15));
                    smp.sample(&first_logits)
                }
                _ => first_tok,
            };
            let cs = self.alloc_cs();
            cs_slots.push(cs);
            let mut s = SeqSlot {
                id: b, cs_slot: cs, pos: p_len, suf: 0, swa_pages: Vec::new(), full_pages: Vec::new(),
                generated: vec![tok0], max_new, next_tok: tok0, done: stop_on_eos && tok0 == eos,
            };
            s.ensure_pages_for(&mut self.pool, 0);
            seqs.push(s);
        }
        self.init_cs_from_prefix(&cs_slots);

        let mut results = vec![Vec::new(); batch_size];
        let t_gen = Instant::now();
        let mut gen_tokens = batch_size; // first token already produced by prefix prefill
        let mut active: Vec<SeqSlot> = Vec::with_capacity(batch_size);
        for mut s in seqs {
            if s.done || max_new <= 1 {
                results[s.id] = std::mem::take(&mut s.generated);
                s.free_pages(&mut self.pool);
                self.free_cs(s.cs_slot);
            } else {
                active.push(s);
            }
        }

        // 2-step GPU-pipelined decode loop!
        if !active.is_empty() {
            let (g_add, steps) = self.run_pipelined_decode(&mut active, &mut results, sample, eos, stop_on_eos);
            gen_tokens += g_add;
            stats.decode_steps = steps;
        }
        for mut s in active {
            results[s.id] = std::mem::take(&mut s.generated);
            s.free_pages(&mut self.pool);
            self.free_cs(s.cs_slot);
        }
        stats.gen_tokens = gen_tokens;
        stats.gen_time = t_gen.elapsed();
        stats.kv_bytes = self.kv_bytes();
        stats.cs_bytes = self.cs_bytes();
        stats.act_bytes = self.act_bytes();
        (results, stats)
    }

    /// Generate completions for arbitrary independent prompts using packed batched prefill +
    /// 2-step GPU-pipelined paged decode + continuous batching (retiring finished slots & admitting queued prompts).
    pub fn generate_independent(
        &mut self, prompts: &[Vec<u32>], max_new: usize,
        sample: Option<SampleCfg>, eos: u32, stop_on_eos: bool,
    ) -> (Vec<Vec<u32>>, BatchStats) {
        assert!(!prompts.is_empty() && max_new >= 1);
        self.prefix = None;
        let mut stats = BatchStats::default();
        let need_slice = sample.is_some_and(|s| !s.greedy());
        if let Some(sc) = sample { self.init_rng(sc.seed, self.max_batch); }

        let mut results = vec![Vec::new(); prompts.len()];
        let mut queue: VecDeque<usize> = (0..prompts.len()).collect();
        let mut active: Vec<SeqSlot> = Vec::with_capacity(self.max_batch);

        while !queue.is_empty() || !active.is_empty() {
            let room = self.max_batch.saturating_sub(active.len());
            if room > 0 && !queue.is_empty() {
                let mut admit = Vec::new();
                while admit.len() < room {
                    if let Some(id) = queue.pop_front() { admit.push((id, prompts[id].as_slice(), max_new)); }
                    else { break; }
                }
                let (mut new_slots, ptoks, pt) = self.prefill_packed(&admit, sample);
                stats.prompt_tokens += ptoks;
                stats.prompt_time += pt;
                for mut s in new_slots.drain(..) {
                    let tok0 = s.next_tok;
                    s.generated.push(tok0);
                    stats.gen_tokens += 1;
                    if (stop_on_eos && tok0 == eos) || max_new <= 1 {
                        results[s.id] = std::mem::take(&mut s.generated);
                        s.free_pages(&mut self.pool);
                        self.free_cs(s.cs_slot);
                    } else {
                        active.push(s);
                    }
                }
            }
            if active.is_empty() { break; }
            stats.peak_batch = stats.peak_batch.max(active.len());
            self.ensure_arena(active.len(), need_slice, false);

            let t_gen = Instant::now();
            let stop_when_freed = !queue.is_empty();
            let (g_add, steps) = if stop_when_freed {
                self.run_pipelined_decode_until(&mut active, &mut results, sample, eos, stop_on_eos, (self.max_batch / 4).max(1))
            } else {
                self.run_pipelined_decode(&mut active, &mut results, sample, eos, stop_on_eos)
            };
            stats.gen_tokens += g_add;
            stats.decode_steps += steps;
            stats.gen_time += t_gen.elapsed();
        }
        stats.kv_bytes = self.kv_bytes();
        stats.cs_bytes = self.cs_bytes();
        stats.act_bytes = self.act_bytes();
        (results, stats)
    }

    fn run_pipelined_decode(
        &mut self, active: &mut Vec<SeqSlot>, results: &mut [Vec<u32>],
        sample: Option<SampleCfg>, eos: u32, stop_on_eos: bool,
    ) -> (usize, usize) {
        self.run_pipelined_decode_until(active, results, sample, eos, stop_on_eos, usize::MAX)
    }

    /// 2-step GPU-pipelined decode loop. Drains cleanly when any sequence finishes & compacts `active`,
    /// or returns early once `retire_trigger` slots have freed up (for continuous batching admission).
    fn run_pipelined_decode_until(
        &mut self, active: &mut Vec<SeqSlot>, results: &mut [Vec<u32>],
        sample: Option<SampleCfg>, eos: u32, stop_on_eos: bool, retire_trigger: usize,
    ) -> (usize, usize) {
        let mut gen_tokens = 0usize;
        let mut steps = 0usize;
        let mut retired_count = 0usize;

        while !active.is_empty() {
            if retired_count >= retire_trigger { break; }
            let b_len = active.len();
            let tp0 = self.toks[0].contents().as_ptr() as *mut i32;
            for (b, s) in active.iter().enumerate() { unsafe { *tp0.add(b) = s.next_tok as i32; } }

            let mut inflight: VecDeque<(usize, usize, CmdBuf)> = VecDeque::new();
            let mut next_slot = 0usize;
            let mut lookahead = 0usize;

            let submit = |this: &mut Self, active: &mut [SeqSlot], inflight: &mut VecDeque<(usize, usize, CmdBuf)>, next_slot: &mut usize, lookahead: &mut usize| -> bool {
                let max_pos = active.iter().map(|s| s.pos + *lookahead).max().unwrap();
                if max_pos + 1 >= this.max_ctx { return false; }
                let min_suf = active.iter().map(|s| s.suf + *lookahead).min().unwrap();
                let (mut need_s, mut need_f) = (0usize, 0usize);
                for s in active.iter() {
                    let (ns, nf) = s.pages_needed_for(s.suf + *lookahead);
                    need_s += ns; need_f += nf;
                }
                if need_s > this.pool.free_swa_pages() || need_f > this.pool.free_full_pages() {
                    for (_, _, cb) in inflight.iter() { cb.waitUntilCompleted(); }
                    let kv_bud = this.kv_budget();
                    this.pool.ensure_free(&this.eng.gpu, need_s, need_f, kv_bud);
                }
                for s in active.iter_mut() { s.ensure_pages_for(&mut this.pool, s.suf + *lookahead); }
                let m_idx = *next_slot;
                let tin = *next_slot;
                let tout = *next_slot ^ 1;
                this.upload_decode_meta(active, m_idx, *lookahead);
                let cb = this.eng.gpu.cmd();
                this.encode_decode_step(&cb, active.len(), max_pos, min_suf, m_idx, tin, tout, sample);
                cb.commit();
                inflight.push_back((m_idx, tout, cb));
                *next_slot ^= 1;
                *lookahead += 1;
                true
            };

            let max_ctx = self.max_ctx;
            let can_submit_more = |active: &[SeqSlot], lookahead: usize| -> bool {
                active.iter().all(|s| s.generated.len() + lookahead < s.max_new && s.pos + lookahead + 1 < max_ctx)
            };

            if !can_submit_more(active, 0) {
                let mut keep = Vec::new();
                for mut s in active.drain(..) {
                    if s.generated.len() >= s.max_new || s.pos + 1 >= self.max_ctx {
                        results[s.id] = std::mem::take(&mut s.generated);
                        s.free_pages(&mut self.pool);
                        self.free_cs(s.cs_slot);
                        retired_count += 1;
                    } else { keep.push(s); }
                }
                *active = keep;
                continue;
            }

            submit(self, active, &mut inflight, &mut next_slot, &mut lookahead);

            loop {
                if inflight.len() < 2 && can_submit_more(active, lookahead) {
                    submit(self, active, &mut inflight, &mut next_slot, &mut lookahead);
                }
                let (_, tout, cb) = inflight.pop_front().unwrap();
                cb.waitUntilCompleted();
                lookahead -= 1;
                steps += 1;

                let out_p = self.toks[tout].contents().as_ptr() as *const i32;
                let mut any_done = false;
                for (b, s) in active.iter_mut().enumerate() {
                    let tok = unsafe { *out_p.add(b) as u32 };
                    s.pos += 1;
                    s.suf += 1;
                    s.generated.push(tok);
                    s.next_tok = tok;
                    gen_tokens += 1;
                    if (stop_on_eos && tok == eos) || s.generated.len() >= s.max_new || s.pos + 1 >= self.max_ctx {
                        s.done = true;
                        any_done = true;
                    }
                }

                if any_done || inflight.is_empty() {
                    if let Some((_, tout2, cb2)) = inflight.pop_front() {
                        cb2.waitUntilCompleted();
                        steps += 1;
                        let out_p2 = self.toks[tout2].contents().as_ptr() as *const i32;
                        for (b, s) in active.iter_mut().enumerate() {
                            if !s.done {
                                let tok = unsafe { *out_p2.add(b) as u32 };
                                s.pos += 1;
                                s.suf += 1;
                                s.generated.push(tok);
                                s.next_tok = tok;
                                gen_tokens += 1;
                                if (stop_on_eos && tok == eos) || s.generated.len() >= s.max_new || s.pos + 1 >= self.max_ctx {
                                    s.done = true;
                                }
                            }
                        }
                    }
                    let mut keep = Vec::with_capacity(b_len);
                    for mut s in active.drain(..) {
                        if s.done {
                            results[s.id] = std::mem::take(&mut s.generated);
                            s.free_pages(&mut self.pool);
                            self.free_cs(s.cs_slot);
                            retired_count += 1;
                        } else {
                            keep.push(s);
                        }
                    }
                    *active = keep;
                    break;
                }
            }
        }
        (gen_tokens, steps)
    }

    // ------------------------------------------------------------------------- Exact logits & token steppers (for check-batch!)
    /// Advance `seqs` by ONE batched decode step using `encode_decode_step` and return the sampled/argmax token IDs.
    pub fn step_tokens(&mut self, seqs: &mut [SeqSlot], sample: Option<SampleCfg>) -> Vec<u32> {
        let b_len = seqs.len();
        let need_slice = sample.is_some_and(|s| !s.greedy());
        self.ensure_arena(b_len, need_slice, false);
        if let Some(sc) = sample { self.init_rng(sc.seed, b_len); }
        let (mut ns, mut nf) = (0usize, 0usize);
        for s in seqs.iter() { let (a, b) = s.pages_needed_for(s.suf); ns += a; nf += b; }
        let kv_bud = self.kv_budget();
        self.pool.ensure_free(&self.eng.gpu, ns, nf, kv_bud);
        for s in seqs.iter_mut() { s.ensure_pages_for(&mut self.pool, s.suf); }
        let tp = self.toks[0].contents().as_ptr() as *mut i32;
        for (b, s) in seqs.iter().enumerate() { unsafe { *tp.add(b) = s.next_tok as i32; } }
        self.upload_decode_meta(seqs, 0, 0);
        let max_pos = seqs.iter().map(|s| s.pos).max().unwrap();
        let min_suf = seqs.iter().map(|s| s.suf).min().unwrap();
        let cb = self.eng.gpu.cmd();
        self.encode_decode_step(&cb, b_len, max_pos, min_suf, 0, 0, 1, sample);
        cb.commit();
        cb.waitUntilCompleted();
        let out_p = self.toks[1].contents().as_ptr() as *const i32;
        let mut out = Vec::with_capacity(b_len);
        for (b, s) in seqs.iter_mut().enumerate() {
            let tok = unsafe { *out_p.add(b) as u32 };
            s.pos += 1; s.suf += 1; s.next_tok = tok;
            out.push(tok);
        }
        out
    }

    /// Advance `seqs` by ONE batched decode step and return exact f32 logits `[B][n_vocab]` for every sequence.
    pub fn step_logits(&mut self, seqs: &mut [SeqSlot]) -> Vec<Vec<f32>> {
        let b_len = seqs.len();
        self.ensure_arena(b_len, true, false);
        let (mut ns, mut nf) = (0usize, 0usize);
        for s in seqs.iter() { let (a, b) = s.pages_needed_for(s.suf); ns += a; nf += b; }
        let kv_bud = self.kv_budget();
        self.pool.ensure_free(&self.eng.gpu, ns, nf, kv_bud);
        for s in seqs.iter_mut() { s.ensure_pages_for(&mut self.pool, s.suf); }
        let tp = self.toks[0].contents().as_ptr() as *mut i32;
        for (b, s) in seqs.iter().enumerate() { unsafe { *tp.add(b) = s.next_tok as i32; } }
        self.upload_decode_meta(seqs, 0, 0);
        let max_pos = seqs.iter().map(|s| s.pos).max().unwrap();
        let min_suf = seqs.iter().map(|s| s.suf).min().unwrap();

        let eng = self.eng;
        let d = self.da;
        let cb = eng.gpu.cmd();
        let e = Enc::new(&cb);
        self.bind_batch(&e, 0, 0);
        let ea = EmbedArgs {
            emb_lo: eng.tok_embd.off as u32, emb_hi: (eng.tok_embd.off >> 32) as u32, row_bytes: eng.tok_embd.row_bytes,
            norm_off: eng.tok_norm, out0: d.vec(0), out1: d.vec(0), tok_index: 0, eps: eng.eps,
        };
        self.go(&e, C_EMBED, "k_embed", &ea, (b_len, 1, 1), 256);
        let mut va = VecAlloc { next: 1, n: B_NVEC };
        let mut blocks: Vec<usize> = Vec::new();
        let mut cur = 0usize;
        for ip in 0..LOOP_START { cur = self.decode_layer(&e, b_len, max_pos, min_suf, ip, ip, cur, &mut blocks, None, &mut va, &[]); }
        let prelude = cur;
        let prelude_blocks = blocks.clone();
        let mut bank = (0usize, 0usize);
        for pass in 0..N_PASS {
            if pass > 0 { cur = prelude; blocks = prelude_blocks.clone(); }
            let msg = if pass > 0 { Some(bank) } else { None };
            let pinned: Vec<usize> = { let mut p = vec![prelude]; if pass > 0 { p.push(bank.0); p.push(bank.1); } p };
            for ip in LOOP_START..LOOP_END {
                cur = self.decode_layer(&e, b_len, max_pos, min_suf, slot_of(pass, ip), ip, cur, &mut blocks, msg, &mut va, &pinned);
            }
            bank = (blocks[prelude_blocks.len() + 1], cur);
        }
        for ip in LOOP_END..N_PHYS { cur = self.decode_layer(&e, b_len, max_pos, min_suf, ip, ip, cur, &mut blocks, None, &mut va, &[]); }
        let mut st: Vec<u32> = blocks.iter().map(|&v| d.vec(v)).collect();
        st.push(d.vec(cur));
        let mut jo = eng.mix_job_off(&st, eng.output_res_query as i32, 1, &[d.fin], &[eng.output_norm as i32], E as u32);
        jo.hout[0] = d.fh as i32;
        jo.nof32 = 1;
        self.mix_b(&e, C_OMIX, &[jo], b_len);
        e.end();
        cb.commit();
        cb.waitUntilCompleted();

        let mut out = vec![vec![0f32; eng.n_vocab]; b_len];
        let w = &eng.output;
        let mut vstart = 0usize;
        while vstart < eng.n_vocab {
            let vcount = (eng.n_vocab - vstart).min(LM_VSLICE);
            let cb2 = eng.gpu.cmd();
            let e2 = Enc::new(&cb2);
            self.bind_batch(&e2, 0, 0);
            let la = LmHeadArgs {
                w_lo: w.off as u32, w_hi: (w.off >> 32) as u32, row_bytes: w.row_bytes, wtype: w.ty,
                n_vocab: eng.n_vocab as u32, k: w.k, xh_off: d.fh, part_off: d.slice_buf,
                b: b_len as u32, nvch: 0, vstart: vstart as u32, vcount: vcount as u32,
            };
            let tiles = vcount.div_ceil(64);
            if b_len > 64 { self.go(&e2, C_LMHEAD, "k_lmhead_slice_sg8", &la, (b_len.div_ceil(128), tiles, 1), 256); }
            else { self.go(&e2, C_LMHEAD, "k_lmhead_slice_sg4", &la, (1, tiles, 1), 128); }
            e2.end(); cb2.commit(); cb2.waitUntilCompleted();
            let sp = unsafe { (self.act.contents().as_ptr() as *const f32).add(d.slice_buf as usize) };
            for b in 0..b_len {
                let sl = unsafe { std::slice::from_raw_parts(sp.add(b * vcount), vcount) };
                out[b][vstart..vstart + vcount].copy_from_slice(sl);
            }
            vstart += vcount;
        }
        for s in seqs.iter_mut() { s.pos += 1; s.suf += 1; }
        out
    }

    /// Create fresh empty SeqSlots for manual step-by-step verification (check-batch).
    pub fn make_seq_slots(&mut self, n: usize, from_prefix: bool) -> Vec<SeqSlot> {
        let p_len = if from_prefix { self.prefix.as_ref().unwrap().prefix_len } else { 0 };
        let mut seqs = Vec::with_capacity(n);
        let mut cs_ids = Vec::with_capacity(n);
        for b in 0..n {
            let cs = self.alloc_cs();
            cs_ids.push(cs);
            seqs.push(SeqSlot {
                id: b, cs_slot: cs, pos: p_len, suf: 0, swa_pages: Vec::new(), full_pages: Vec::new(),
                generated: Vec::new(), max_new: self.max_ctx, next_tok: 0, done: false,
            });
        }
        if from_prefix { self.init_cs_from_prefix(&cs_ids); } else { self.zero_cs(&cs_ids); }
        seqs
    }

    // ------------------------------------------------------------------------- Isolated GPU Profiler (`prof-batch`)
    pub fn profile_decode(&mut self, b_len: usize, ctx_len: usize, shared: bool, sample: Option<SampleCfg>) {
        let need_slice = sample.is_some_and(|s| !s.greedy());
        self.ensure_arena(b_len, need_slice, false);
        if let Some(sc) = sample { self.init_rng(sc.seed, b_len); }

        let (p_len, suf_pos) = if shared {
            let p = ctx_len.max(16);
            let ptoks: Vec<u32> = (0..p).map(|i| 1000 + (i as u32 * 13) % 5000).collect();
            self.set_prefix(&ptoks);
            (p, 16usize)
        } else {
            self.prefix = None;
            (0usize, ctx_len.max(16))
        };

        let pages_full = (suf_pos + 1).div_ceil(KV_BLK);
        let pages_swa = pages_full.min(SWA_RING_BLKS);
        let kv_bud = self.kv_budget();
        self.pool.ensure_free(&self.eng.gpu, b_len * pages_swa, b_len * pages_full, kv_bud);

        let mut seqs = Vec::with_capacity(b_len);
        let mut cs_ids = Vec::with_capacity(b_len);
        let tp = self.toks[0].contents().as_ptr() as *mut i32;
        for b in 0..b_len {
            let cs = self.alloc_cs();
            cs_ids.push(cs);
            let mut s = SeqSlot {
                id: b, cs_slot: cs, pos: p_len + suf_pos, suf: suf_pos,
                swa_pages: Vec::new(), full_pages: Vec::new(),
                generated: Vec::new(), max_new: 16, next_tok: 1000 + (b as u32 * 37) % 10000, done: false,
            };
            for p in (0..=suf_pos).step_by(KV_BLK) { s.ensure_pages_for(&mut self.pool, p); }
            s.ensure_pages_for(&mut self.pool, suf_pos);
            unsafe { *tp.add(b) = s.next_tok as i32; }
            seqs.push(s);
        }
        if shared { self.init_cs_from_prefix(&cs_ids); } else { self.zero_cs(&cs_ids); }
        self.upload_decode_meta(&seqs, 0, 0);

        let max_pos = p_len + suf_pos;
        let run_once = |this: &Self, skip: u32| -> f64 {
            this.skip.set(skip);
            let cb = this.eng.gpu.cmd();
            this.encode_decode_step(&cb, b_len, max_pos, suf_pos, 0, 0, 1, sample);
            cb.commit();
            cb.waitUntilCompleted();
            (cb.GPUEndTime() - cb.GPUStartTime()) * 1e3
        };
        let med = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };

        // Warmup + measure full step with clean CS before each rep
        if shared { self.init_cs_from_prefix(&cs_ids); } else { self.zero_cs(&cs_ids); }
        run_once(self, 0);
        let base = med((0..5).map(|_| {
            if shared { self.init_cs_from_prefix(&cs_ids); } else { self.zero_cs(&cs_ids); }
            run_once(self, 0)
        }).collect());
        // Populate valid FFN/MoE scratch (without lm_head clobbering aliased d.ind/d.tl) to inspect tile distribution
        let prep_moe = |this: &Self| {
            if shared { this.init_cs_from_prefix(&cs_ids); } else { this.zero_cs(&cs_ids); }
            run_once(this, (1 << C_LMHEAD) | (1 << C_ARGMAX));
        };
        prep_moe(self);
        let ind_p = unsafe { (self.act.contents().as_ptr() as *const u32).add(self.da.ind as usize) };
        let t_cnt = unsafe { [*ind_p.add(0), *ind_p.add(4), *ind_p.add(8), *ind_p.add(12)] };
        println!("batched decode B={b_len} ctx={} ({}{}): {base:.2} ms/step -> {:.1} tok/s ({:.2} tok/s/stream)",
                 max_pos, if shared { "shared-prefix" } else { "independent" },
                 if self.eng.kv8 { ", kv8" } else { ", fp16" },
                 b_len as f64 / (base * 1e-3), 1e3 / base);
        println!("  memory: KV {:.1} MiB | CS {:.1} MiB | Act {:.1} MiB | total {:.2} / {:.2} GiB cap",
                 self.kv_bytes() as f64 / 1048576.0, self.cs_bytes() as f64 / 1048576.0, self.act_bytes() as f64 / 1048576.0,
                 self.total_bytes() as f64 / (1u64 << 30) as f64, self.max_mem_bytes as f64 / (1u64 << 30) as f64);
        println!("  last-layer MoE tiles [sg8, sg4, dir2, dir1] = {:?}", t_cnt);

        let empty = med((0..3).map(|_| run_once(self, u32::MAX)).collect());
        let mut tot = 0.0;
        for (c, name) in CLS_NAMES.iter().enumerate() {
            let cu = c as u32;
            // Re-populate valid FFN/MoE aliased scratch before indirect/indexed MoE classes!
            if matches!(cu, C_GU | C_DOWN | C_GATHER | C_COMB2) { prep_moe(self); }
            let d = med((0..3).map(|_| run_once(self, !(1u32 << cu))).collect()) - empty;
            tot += d;
            if d > 0.20 { println!("  {name:>11}: {:7.2} ms ({:5.1}%)", d, d / base * 100.0); }
        }
        println!("  sum of isolated classes: {tot:.2} ms (empty overhead: {empty:.2} ms)");
        for mut s in seqs { s.free_pages(&mut self.pool); self.free_cs(s.cs_slot); }
    }
}

struct PackedMetaOffs {
    tok_pos: u32, tok_seq: u32, seq_start: u32, seq_len: u32, seq_p0: u32,
    seq_cs: u32, swa_pt: u32, full_pt: u32, qtiles: u32,
    nseq: usize, n_qtiles: usize,
}
