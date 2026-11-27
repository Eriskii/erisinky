//! TokenDance (arXiv:2604.03143) + KVCOMM (NeurIPS 2025, arXiv:2510.12872) cross-context collective KV sharing,
//! diff-aware Master-Mirror storage, online anchor-pool offset alignment, and empirical Inkling 2 Mini probes.
//!
//! Why Inkling 2 Mini is an ideal substrate for TokenDance + KVCOMM:
//! - **Zero RoPE**: Stored (K, V) vectors are completely position-independent; relative bias (`r . B[d]`) is added
//!   dynamically at attention time for `d < EXT = 1024` and is zero for `d >= 1024`. Zero RoPE de-/re-rotation!
//! - **Hybrid 40 SWA (`EXT = 1024`) + 8 Full slots (`blk.5`, `blk.11` x3, `blk.17` x3, `blk.23`)**:
//!   Past token 1024 of a shared block `S`, SWA slots cannot attend to the private prefix at all, while the 8 Full
//!   slots carry the true prefix-induced KV offset shift — and `blk.5` (slot 5) is a natural early check layer!

use crate::engine::{Engine, EXT, HD, LOOP_END, LOOP_START, NCH, NH, NKV, NREL, N_PHYS, N_SLOT};
use crate::metal::{Buf, Enc};
use crate::tokenizer::Tokenizer;
use objc2_metal::{MTLBuffer, MTLCommandBuffer};
use std::collections::HashMap;

pub const KV_BLK: usize = 32;
pub const N_SWA_SLOTS: usize = 40;
pub const N_FULL_SLOTS: usize = 8;
/// First full-attention layer (`blk.5`, slot 5): TokenDance early check layer.
pub const CHECK_LAYER_SLOT: usize = 5;
/// Distance buckets inside shared block `S`: `[0..16), [16..128), [128..1024), [1024..inf)`.
pub const N_DIST_BUCKETS: usize = 4;
pub const DIST_BUCKET_EDGES: [usize; N_DIST_BUCKETS] = [16, 128, 1024, usize::MAX];
pub const DIST_BUCKET_NAMES: [&str; N_DIST_BUCKETS] = ["0..16", "16..128", "128..1024", "1024+"];

/// New Metal kernel names to register in `engine::KERNELS`.
pub const KV_SHARE_KERNELS: &[&str] = &[
    "k_attn_master_diff_b",
    "k_kv_diff_extract",
    "k_kvcomm_apply_offset",
];

// ============================================================================= GPU kernel argument structs

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MasterDiffAttnArgs {
    pub b_count: u32,
    pub master_pool_cap: u32,
    pub slot_idx: u32,
    pub is_swa: u32,
    pub pt_stride: u32,
    pub pt_mod: u32,
    pub seq_pos_off: u32,
    pub master_pt_off: u32,
    pub seg_pt_off: u32,
    pub q_off: u32,
    pub r_off: u32,
    pub sink_off: u32,
    pub qclip_w: u32,
    pub relclip_w: u32,
    pub relbias_w: u32,
    pub gate_off: u32,
    pub out_off: u32,
    pub part_off: u32,
    pub nsplit: u32,
    pub diff_cap: u32,
    pub diff_mode: u32,
    pub max_segs: u32,
    pub diff_map_stride: u32,
    pub diff_map_slot_stride: u32,
    pub q_prepped: u32,
    pub pad0: u32,
    pub log_alpha: f32,
    pub log_floor: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct KvDiffExtractArgs {
    pub n_tok: u32,
    pub agent_pos0: u32,
    pub master_pos0: u32,
    pub agent_cap: u32,
    pub master_cap: u32,
    pub agent_is_ring: u32,
    pub master_is_ring: u32,
    pub slot_idx: u32,
    pub diff_cap: u32,
    pub diff_mode: u32,
    pub metric_mode: u32,
    pub diff_map_base: u32,
    pub agent_k_base: u64,
    pub agent_v_base: u64,
    pub master_k_base: u64,
    pub master_v_base: u64,
    pub thresh: f32,
    pub eps: f32,
    pub write_scores: u32,
    pub pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct KvCommOffsetArgs {
    pub n_tok: u32,
    pub pos0: u32,
    pub cap: u32,
    pub is_ring: u32,
    pub n_buckets: u32,
    pub use_bucket_map: u32,
    pub pad0: u32,
    pub pad1: u32,
    pub k_base: u64,
    pub v_base: u64,
    pub ofs_base: u64,
    pub scale: f32,
    pub pad2: f32,
}

/// Mirrors `CombineBArgs` in `batch.metal` for `k_attn_combine_b` when `nsplit > 1`.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CombineBArgs {
    part_off: u32,
    nsplit: u32,
    gate_off: u32,
    out_off: u32,
}

// ============================================================================= fp16 & math utilities

fn build_h2f_lut() -> Vec<f32> {
    (0..65536u32)
        .map(|h| {
            let sg = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
            let e = ((h >> 10) & 31) as i32;
            let m = (h & 1023) as f32;
            if e == 0 {
                sg * m * 2f32.powi(-24)
            } else if e == 31 {
                0.0
            } else {
                sg * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
            }
        })
        .collect()
}

#[inline]
pub fn f32_to_f16(v: f32) -> u16 {
    if !v.is_finite() {
        return if v.is_sign_negative() { 0xFBFF } else { 0x7BFF };
    }
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if v == 0.0 || exp <= -10 {
        return sign;
    }
    if exp <= 0 {
        // Subnormal fp16
        let mant = (bits & 0x7fffff) | 0x800000;
        let shift = (14 - exp) as u32;
        let half_lsb = 1u32 << (shift - 1);
        let sub = (mant + half_lsb - 1 + ((mant >> shift) & 1)) >> shift;
        return sign | (sub as u16);
    }
    let mant = (bits >> 13) & 0x3ff;
    let rem = bits & 0x1fff;
    let mut h = ((exp as u32) << 10 | mant)
        + if rem > 0x1000 || (rem == 0x1000 && (mant & 1) == 1) { 1 } else { 0 };
    if h >= 0x7c00 {
        h = 0x7bff;
    }
    sign | h as u16
}

#[inline]
fn dist_bucket(local_pos: usize) -> usize {
    if local_pos < 16 {
        0
    } else if local_pos < 128 {
        1
    } else if local_pos < EXT {
        2
    } else {
        3
    }
}

#[inline]
fn softcap(x: f32, c: f32) -> f32 {
    if c > 0.0 { c * (x / c).tanh() } else { x }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b })
        .0
}

fn kl_div(gt: &[f32], cand: &[f32]) -> f64 {
    let lse = |v: &[f32]| -> f64 {
        let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let s: f64 = v.iter().map(|&x| ((x as f64) - m).exp()).sum();
        m + s.ln()
    };
    let (lg, lc) = (lse(gt), lse(cand));
    gt.iter()
        .zip(cand)
        .map(|(&g, &c)| {
            let log_pg = g as f64 - lg;
            let log_pc = c as f64 - lc;
            log_pg.exp() * (log_pg - log_pc)
        })
        .sum::<f64>()
        .max(0.0)
}

fn slot_label(slot: usize, is_swa: bool) -> String {
    let phys = if slot < N_PHYS {
        slot
    } else {
        LOOP_START + (slot - N_PHYS) % (LOOP_END - LOOP_START)
    };
    let pass = if slot < N_PHYS {
        0
    } else {
        1 + (slot - N_PHYS) / (LOOP_END - LOOP_START)
    };
    format!("s{:02}:blk.{:02}/p{}[{}]", slot, phys, pass, if is_swa { "SWA " } else { "FULL" })
}

#[inline]
fn eng_kv_slot_bases(eng: &Engine, slot: usize) -> (bool, usize, u64, u64, usize) {
    let (is_swa, kind_idx) = eng.slot_kind[slot];
    let cap = if is_swa { eng.swa_cap.get() } else { eng.full_cap.get() };
    let k_base = (kind_idx * 2 * NKV * cap * HD) as u64;
    let v_base = ((kind_idx * 2 + 1) * NKV * cap * HD) as u64;
    (is_swa, kind_idx, k_base, v_base, cap)
}

// ============================================================================= 1. Core Data Structures (256K-ready)

/// Content-addressed identifier for a canonical shared Master KV block.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MasterBlockId(pub u64);

/// Canonical shared KV block stored ONCE and referenced by any number of agents (zero RoPE metadata).
pub struct MasterBlock {
    pub id: MasterBlockId,
    pub token_hash: u64,
    pub tokens: Vec<u32>,
    pub len: usize,
    pub n_pages: usize,
    /// Logical page IDs assigned in a unified paged Master pool (addressed with 64-bit offsets).
    pub master_pages: Vec<u32>,
    /// Canonical Full-attention KV: `[n_full][2(K,V)][NKV][len][HD]` in fp16.
    pub full_kv: Vec<u16>,
    pub n_full: usize,
    /// Canonical SWA tail KV (last `min(len, EXT)` tokens): `[n_swa][2(K,V)][NKV][swa_tail_len][HD]` in fp16.
    pub swa_tail_kv: Vec<u16>,
    pub n_swa: usize,
    pub swa_tail_len: usize,
    /// Canonical conv-state seam captured immediately at the end of this block (`N_SLOT * NCH * 3 * 4` bytes).
    pub end_cs: Vec<u8>,
    /// Prefix length `L_0` under which this canonical Master block was computed.
    pub origin_prefix_len: usize,
}

impl MasterBlock {
    #[inline]
    pub fn full_vec(&self, fi: usize, is_v: usize, g: usize, local_pos: usize) -> &[u16] {
        let base = (((fi * 2 + is_v) * NKV + g) * self.len + local_pos) * HD;
        &self.full_kv[base..base + HD]
    }

    #[inline]
    pub fn swa_tail_vec(&self, si: usize, is_v: usize, g: usize, tail_pos: usize) -> &[u16] {
        let base = (((si * 2 + is_v) * NKV + g) * self.swa_tail_len + tail_pos) * HD;
        &self.swa_tail_kv[base..base + HD]
    }

    /// 64-bit byte offset of a page in a slot-major paged Master pool (`[n_slots][pool_cap][2][NKV][KV_BLK][HD]`).
    #[inline]
    pub fn page_offset_bytes(slot_idx: usize, pool_cap: usize, page_id: u32, kvb: usize) -> u64 {
        let slot_elems = 2u64 * (NKV as u64) * (KV_BLK as u64) * (HD as u64);
        ((slot_idx as u64) * (pool_cap as u64) + (page_id as u64)) * slot_elems * (kvb as u64)
    }

    /// Exact canonical Master storage footprint in bytes (Full KV + SWA tail KV + conv seam).
    pub fn master_bytes(&self) -> u64 {
        (self.full_kv.len() as u64) * 2
            + (self.swa_tail_kv.len() as u64) * 2
            + (self.end_cs.len() as u64)
    }
}

/// Content-addressed canonical store of Master KV blocks shared across concurrent agents.
#[derive(Default)]
pub struct MasterBlockStore {
    blocks: HashMap<MasterBlockId, MasterBlock>,
    by_hash: HashMap<u64, MasterBlockId>,
    next_page_id: u32,
}

impl MasterBlockStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Deterministic 64-bit FNV-1a content hash of a token slice.
    pub fn content_hash(tokens: &[u32]) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for &t in tokens {
            h ^= t as u64;
            h = h.wrapping_mul(0x100000001b3);
            h ^= h >> 33;
        }
        h
    }

    pub fn lookup_by_tokens(&self, tokens: &[u32]) -> Option<MasterBlockId> {
        let h = Self::content_hash(tokens);
        self.by_hash.get(&h).copied().filter(|id| {
            self.blocks.get(id).is_some_and(|b| b.tokens == tokens)
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn register(
        &mut self,
        tokens: Vec<u32>,
        n_full: usize,
        full_kv: Vec<u16>,
        n_swa: usize,
        swa_tail_len: usize,
        swa_tail_kv: Vec<u16>,
        end_cs: Vec<u8>,
        origin_prefix_len: usize,
    ) -> MasterBlockId {
        if let Some(existing) = self.lookup_by_tokens(&tokens) {
            return existing;
        }
        let token_hash = Self::content_hash(&tokens);
        let id = MasterBlockId(token_hash ^ ((self.blocks.len() as u64) << 48));
        let len = tokens.len();
        let n_pages = len.div_ceil(KV_BLK);
        let master_pages: Vec<u32> = (0..n_pages)
            .map(|_| {
                let p = self.next_page_id;
                self.next_page_id += 1;
                p
            })
            .collect();
        let blk = MasterBlock {
            id,
            token_hash,
            tokens,
            len,
            n_pages,
            master_pages,
            full_kv,
            n_full,
            swa_tail_kv,
            n_swa,
            swa_tail_len,
            end_cs,
            origin_prefix_len,
        };
        self.by_hash.insert(token_hash, id);
        self.blocks.insert(id, blk);
        id
    }

    pub fn get(&self, id: MasterBlockId) -> Option<&MasterBlock> {
        self.blocks.get(&id)
    }

    pub fn total_master_bytes(&self) -> u64 {
        self.blocks.values().map(|b| b.master_bytes()).sum()
    }
}

/// How sparse diff entries are combined with the canonical Master copy in `k_attn_master_diff_b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiffStorageMode {
    /// `k_eff = diff_k` (mode 0)
    Replacement = 0,
    /// `k_eff = master_k + diff_k` (mode 1)
    AdditiveDelta = 1,
    /// `k_eff = master_k + seg_offset_k + diff_k` (mode 2, hybrid KVCOMM + TokenDance)
    ResidualOverOffset = 2,
}

/// Reference from an agent's logical timeline to a shared Master block (64-bit offset for 256K+ contexts).
#[derive(Clone, Copy, Debug)]
pub struct MasterSegmentRef {
    pub block_id: MasterBlockId,
    pub logical_offset: u64,
    pub len: usize,
    pub seg_id: u32,
}

/// Compact sparse diff storage for one KV slot: stores ONLY the flagged high-deviation token positions.
#[derive(Clone)]
pub struct SparseDiffSlice {
    pub is_swa: bool,
    pub kind_idx: usize,
    /// Sorted local positions inside the shared block `S` that carry a sparse diff entry.
    pub positions: Vec<u32>,
    /// Packed fp16 payload: `[positions.len()][2(K,V)][NKV][HD]`.
    pub kv_data: Vec<u16>,
}

impl SparseDiffSlice {
    pub fn payload_bytes(&self) -> u64 {
        (self.kv_data.len() as u64) * 2
    }
    pub fn index_bytes(&self) -> u64 {
        (self.positions.len() as u64) * 4
    }
}

/// Per-segment / per-position-bucket KVCOMM additive offset attached to a replica.
#[derive(Clone)]
pub struct SegmentOffset {
    pub seg_id: u32,
    pub n_buckets: usize,
    /// `[n_full][n_buckets * NKV * HD]` in fp16
    pub full_dk: Vec<Vec<u16>>,
    pub full_dv: Vec<Vec<u16>>,
    /// `[n_swa][NKV * HD]` in fp16 (tail window)
    pub swa_dk: Vec<Vec<u16>>,
    pub swa_dv: Vec<Vec<u16>>,
}

impl SegmentOffset {
    pub fn bytes(&self) -> u64 {
        let f: usize = self.full_dk.iter().map(|v| v.len()).sum::<usize>()
            + self.full_dv.iter().map(|v| v.len()).sum::<usize>();
        let s: usize = self.swa_dk.iter().map(|v| v.len()).sum::<usize>()
            + self.swa_dv.iter().map(|v| v.len()).sum::<usize>();
        ((f + s) as u64) * 2
    }
}

/// TokenDance Master-Mirror replica: lightweight per-agent view referencing shared Master blocks + sparse diffs.
pub struct DiffAwareReplica {
    pub agent_id: usize,
    pub prefix_len: usize,
    pub segments: Vec<MasterSegmentRef>,
    pub mode: DiffStorageMode,
    pub diff_slices: Vec<SparseDiffSlice>,
    pub offsets: Vec<SegmentOffset>,
}

impl DiffAwareReplica {
    pub fn diff_payload_bytes(&self) -> u64 {
        self.diff_slices.iter().map(|s| s.payload_bytes()).sum()
    }

    pub fn metadata_bytes(&self) -> u64 {
        let seg_meta = (self.segments.len() as u64) * 24;
        let idx_meta: u64 = self.diff_slices.iter().map(|s| s.index_bytes()).sum();
        seg_meta + idx_meta
    }

    pub fn offset_bytes(&self) -> u64 {
        self.offsets.iter().map(|o| o.bytes()).sum()
    }

    pub fn avg_full_diff_frac(&self, block_len: usize) -> f64 {
        let mut cnt = 0usize;
        let mut n = 0usize;
        for s in &self.diff_slices {
            if !s.is_swa {
                cnt += s.positions.len();
                n += 1;
            }
        }
        if n == 0 || block_len == 0 {
            0.0
        } else {
            (cnt as f64) / ((n * block_len) as f64)
        }
    }
}

/// Exact memory compression accounting comparing naive per-agent dense KV vs Master-Mirror + sparse diffs.
#[derive(Clone, Debug)]
pub struct CompressionReport {
    pub n_agents: usize,
    pub context_len: usize,
    pub dense_bytes: u64,
    pub master_bytes: u64,
    pub sum_diff_bytes: u64,
    pub offset_bytes: u64,
    pub metadata_bytes: u64,
    pub total_shared_bytes: u64,
    pub compression_ratio: f64,
    pub avg_diff_frac: f64,
}

pub fn compression_accounting(store: &MasterBlockStore, replicas: &[DiffAwareReplica]) -> CompressionReport {
    let n_agents = replicas.len();
    let master_bytes = store.total_master_bytes();
    let mut context_len = 0usize;
    let mut dense_bytes = 0u64;
    let mut sum_diff_bytes = 0u64;
    let mut offset_bytes = 0u64;
    let mut metadata_bytes = 0u64;
    let mut frac_sum = 0.0f64;

    for r in replicas {
        let mut r_len = 0usize;
        for seg in &r.segments {
            if let Some(blk) = store.get(seg.block_id) {
                dense_bytes += blk.master_bytes();
                r_len += blk.len;
            }
        }
        context_len = context_len.max(r_len);
        sum_diff_bytes += r.diff_payload_bytes();
        offset_bytes += r.offset_bytes();
        metadata_bytes += r.metadata_bytes();
        frac_sum += r.avg_full_diff_frac(r_len);
    }

    let total_shared_bytes = master_bytes + sum_diff_bytes + offset_bytes + metadata_bytes;
    let compression_ratio = if total_shared_bytes > 0 {
        dense_bytes as f64 / total_shared_bytes as f64
    } else {
        1.0
    };
    CompressionReport {
        n_agents,
        context_len,
        dense_bytes,
        master_bytes,
        sum_diff_bytes,
        offset_bytes,
        metadata_bytes,
        total_shared_bytes,
        compression_ratio,
        avg_diff_frac: if n_agents > 0 { frac_sum / n_agents as f64 } else { 0.0 },
    }
}

/// Exact 64-bit memory extrapolation for `n_agents` sharing a `ctx_len` context (e.g. `N = 64/256` @ `256K`)
/// on Inkling 2 Mini's 40 SWA (`EXT = 1024`) + 8 Full slot fp16 KV architecture.
pub fn extrapolate_256k_report(
    n_agents: usize,
    ctx_len: usize,
    diff_frac_full: f64,
    diff_frac_swa_tail: f64,
    include_kvcomm_ofs: bool,
) -> CompressionReport {
    let bytes_per_tok_slot: u64 = 2 * (NKV as u64) * (HD as u64) * 2; // 4096 B per (slot, token)
    let full_bytes_one: u64 = (N_FULL_SLOTS as u64) * (ctx_len as u64) * bytes_per_tok_slot;
    let swa_bytes_one: u64 = (N_SWA_SLOTS as u64) * (ctx_len.min(EXT) as u64) * bytes_per_tok_slot;
    let cs_bytes_one: u64 = (N_SLOT * NCH * 3 * 4) as u64;
    let master_bytes = full_bytes_one + swa_bytes_one + cs_bytes_one;
    let dense_bytes = master_bytes * (n_agents as u64);

    // Master agent 0 needs 0 diff; remaining (n_agents - 1) agents store sparse diffs.
    let n_mirror = n_agents.saturating_sub(1) as u64;
    let diff_full_per = ((full_bytes_one as f64) * diff_frac_full).round() as u64;
    let diff_swa_per = ((swa_bytes_one as f64) * diff_frac_swa_tail).round() as u64;
    let sum_diff_bytes = n_mirror * (diff_full_per + diff_swa_per);

    // Compact position index (4 bytes per flagged token, shared across slots under check-layer screening)
    // + page table (4 bytes per 32-token page).
    let flagged_toks = ((ctx_len as f64) * diff_frac_full).round() as u64;
    let pt_bytes = (ctx_len.div_ceil(KV_BLK) as u64) * 4;
    let metadata_bytes = n_mirror * (flagged_toks * 4 + pt_bytes);

    // KVCOMM 4-bucket full offsets (8 * 4 * 4096 = 128 KiB) + SWA tail offset (40 * 4096 = 160 KiB) per mirror.
    let offset_bytes = if include_kvcomm_ofs {
        let per = ((N_FULL_SLOTS * N_DIST_BUCKETS + N_SWA_SLOTS) as u64) * bytes_per_tok_slot;
        n_mirror * per
    } else {
        0
    };

    let total_shared_bytes = master_bytes + sum_diff_bytes + offset_bytes + metadata_bytes;
    CompressionReport {
        n_agents,
        context_len: ctx_len,
        dense_bytes,
        master_bytes,
        sum_diff_bytes,
        offset_bytes,
        metadata_bytes,
        total_shared_bytes,
        compression_ratio: dense_bytes as f64 / total_shared_bytes.max(1) as f64,
        avg_diff_frac: diff_frac_full,
    }
}

// ============================================================================= 2. KVCOMM Online Anchor Pool

/// One observed prefix-induced KV offset anchor for a shared block `segment_id`.
#[derive(Clone)]
pub struct AnchorEntry {
    pub segment_id: MasterBlockId,
    pub agent_id: usize,
    pub prefix_sig: Vec<f32>,
    pub prefix_len: usize,
    /// `[n_full][N_DIST_BUCKETS * NKV * HD]` mean K delta `(agent - master)` per full slot & position bucket.
    pub layer_delta_k: Vec<Vec<f32>>,
    /// `[n_full][N_DIST_BUCKETS * NKV * HD]` mean V delta `(agent - master)` per full slot & position bucket.
    pub layer_delta_v: Vec<Vec<f32>>,
    /// `[n_swa][NKV * HD]` mean K/V delta across the SWA tail window of `S`.
    pub swa_tail_delta_k: Vec<Vec<f32>>,
    pub swa_tail_delta_v: Vec<Vec<f32>>,
}

/// Prediction returned by `KvCommAnchorPool::predict`.
pub struct KvCommEstimate {
    pub confident: bool,
    pub max_sim: f32,
    pub raw_cos_sims: Vec<(usize, f32, f32)>, // (agent_id, raw_cos_sim, softmax_weight)
    pub delta_k: Vec<Vec<f32>>,              // [n_full][N_DIST_BUCKETS * NKV * HD]
    pub delta_v: Vec<Vec<f32>>,              // [n_full][N_DIST_BUCKETS * NKV * HD]
    pub swa_delta_k: Vec<Vec<f32>>,          // [n_swa][NKV * HD]
    pub swa_delta_v: Vec<Vec<f32>>,          // [n_swa][NKV * HD]
}

/// KVCOMM (NeurIPS 2025) online anchor pool: caches observed prefix-induced KV offsets per shared block
/// and predicts aligned offsets for unseen prefixes via top-k cosine + length-compatibility softmax interpolation.
pub struct KvCommAnchorPool {
    pub entries: Vec<AnchorEntry>,
    pub top_k: usize,
    pub temperature: f32,
    pub len_alpha: f32,
    pub min_conf: f32,
}

impl KvCommAnchorPool {
    pub fn new(top_k: usize, temperature: f32, len_alpha: f32, min_conf: f32) -> Self {
        Self {
            entries: Vec::new(),
            top_k: top_k.max(1),
            temperature: temperature.max(1e-3),
            len_alpha,
            min_conf,
        }
    }

    pub fn insert(&mut self, entry: AnchorEntry) {
        self.entries.push(entry);
    }

    /// Build a compact L2-normalized prefix signature directly from the cached Full-attention KV of `P_i`
    /// (`0..prefix_len`) + log-length feature — zero extra GPU compute!
    pub fn build_prefix_sig(
        eng: &Engine,
        prefix_len: usize,
        lut: &[f32],
    ) -> Vec<f32> {
        let full_slots: Vec<usize> = (0..N_SLOT).filter(|&s| !eng.slot_kind[s].0).collect();
        let cap = eng.full_cap.get();
        let kv = eng.kv_full.borrow();
        let base = kv.contents().as_ptr() as *const u16;
        // Per full slot: mean K (128) + mean V (128) averaged across NKV heads, plus tail-16 mean K (128)
        // -> compact 8 * 384 + 2 = 3074 floats, directly capturing what S attends to in P_i!
        let mut sig = Vec::with_capacity(full_slots.len() * 3 * HD + 2);
        let tail0 = prefix_len.saturating_sub(16);
        let inv_n = 1.0f32 / (prefix_len.max(1) * NKV) as f32;
        let inv_t = 1.0f32 / ((prefix_len - tail0).max(1) * NKV) as f32;

        for &slot in &full_slots {
            let (_, _, kb, vb, _) = eng_kv_slot_bases(eng, slot);
            let mut mk = [0.0f32; HD];
            let mut mv = [0.0f32; HD];
            let mut tk = [0.0f32; HD];
            for g in 0..NKV {
                for p in 0..prefix_len {
                    let kr = unsafe {
                        std::slice::from_raw_parts(base.add(kb as usize + (g * cap + p) * HD), HD)
                    };
                    let vr = unsafe {
                        std::slice::from_raw_parts(base.add(vb as usize + (g * cap + p) * HD), HD)
                    };
                    for d in 0..HD {
                        let kf = lut[kr[d] as usize];
                        let vf = lut[vr[d] as usize];
                        mk[d] += kf;
                        mv[d] += vf;
                        if p >= tail0 {
                            tk[d] += kf;
                        }
                    }
                }
            }
            for d in 0..HD {
                sig.push(mk[d] * inv_n);
            }
            for d in 0..HD {
                sig.push(mv[d] * inv_n);
            }
            for d in 0..HD {
                sig.push(tk[d] * inv_t);
            }
        }
        // Append normalized log-length feature
        let ln_feat = ((prefix_len.max(1) as f32).ln() / 10.0) * 0.5;
        sig.push(ln_feat);
        sig.push(1.0 / (1.0 + (prefix_len as f32 / 512.0)));

        let norm = sig.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
        for x in &mut sig {
            *x /= norm;
        }
        sig
    }

    /// Observe exact bucketed `(agent - master)` offsets for an agent and package as an `AnchorEntry`.
    pub fn observe_anchor(
        segment_id: MasterBlockId,
        agent_id: usize,
        prefix_sig: Vec<f32>,
        prefix_len: usize,
        master: &MasterBlock,
        agent_full_kv: &[u16],
        agent_swa_tail_kv: &[u16],
        lut: &[f32],
    ) -> AnchorEntry {
        let len = master.len;
        let nf = master.n_full;
        let ns = master.n_swa;
        let mut bucket_cnt = [0usize; N_DIST_BUCKETS];
        for t in 0..len {
            bucket_cnt[dist_bucket(t)] += 1;
        }

        let mut layer_delta_k = vec![vec![0.0f32; N_DIST_BUCKETS * NKV * HD]; nf];
        let mut layer_delta_v = vec![vec![0.0f32; N_DIST_BUCKETS * NKV * HD]; nf];
        for fi in 0..nf {
            for g in 0..NKV {
                for t in 0..len {
                    let bkt = dist_bucket(t);
                    let m_k = master.full_vec(fi, 0, g, t);
                    let m_v = master.full_vec(fi, 1, g, t);
                    let base_k = (((fi * 2) * NKV + g) * len + t) * HD;
                    let base_v = (((fi * 2 + 1) * NKV + g) * len + t) * HD;
                    let a_k = &agent_full_kv[base_k..base_k + HD];
                    let a_v = &agent_full_kv[base_v..base_v + HD];
                    let dst = (bkt * NKV + g) * HD;
                    for d in 0..HD {
                        layer_delta_k[fi][dst + d] += lut[a_k[d] as usize] - lut[m_k[d] as usize];
                        layer_delta_v[fi][dst + d] += lut[a_v[d] as usize] - lut[m_v[d] as usize];
                    }
                }
            }
            for bkt in 0..N_DIST_BUCKETS {
                let c = bucket_cnt[bkt].max(1) as f32;
                let off = bkt * NKV * HD;
                for i in 0..NKV * HD {
                    layer_delta_k[fi][off + i] /= c;
                    layer_delta_v[fi][off + i] /= c;
                }
            }
        }

        let tail_len = master.swa_tail_len;
        let mut swa_tail_delta_k = vec![vec![0.0f32; NKV * HD]; ns];
        let mut swa_tail_delta_v = vec![vec![0.0f32; NKV * HD]; ns];
        let inv_tail = 1.0f32 / tail_len.max(1) as f32;
        for si in 0..ns {
            for g in 0..NKV {
                for t in 0..tail_len {
                    let m_k = master.swa_tail_vec(si, 0, g, t);
                    let m_v = master.swa_tail_vec(si, 1, g, t);
                    let base_k = (((si * 2) * NKV + g) * tail_len + t) * HD;
                    let base_v = (((si * 2 + 1) * NKV + g) * tail_len + t) * HD;
                    let a_k = &agent_swa_tail_kv[base_k..base_k + HD];
                    let a_v = &agent_swa_tail_kv[base_v..base_v + HD];
                    let dst = g * HD;
                    for d in 0..HD {
                        swa_tail_delta_k[si][dst + d] += lut[a_k[d] as usize] - lut[m_k[d] as usize];
                        swa_tail_delta_v[si][dst + d] += lut[a_v[d] as usize] - lut[m_v[d] as usize];
                    }
                }
            }
            for i in 0..NKV * HD {
                swa_tail_delta_k[si][i] *= inv_tail;
                swa_tail_delta_v[si][i] *= inv_tail;
            }
        }

        AnchorEntry {
            segment_id,
            agent_id,
            prefix_sig,
            prefix_len,
            layer_delta_k,
            layer_delta_v,
            swa_tail_delta_k,
            swa_tail_delta_v,
        }
    }

    /// Predict interpolated `(\hat{\Delta K}, \hat{\Delta V})` for an unseen prefix via top-k cosine + length weighting.
    pub fn predict(
        &self,
        segment_id: MasterBlockId,
        prefix_sig: &[f32],
        prefix_len: usize,
    ) -> Option<KvCommEstimate> {
        let cands: Vec<&AnchorEntry> = self
            .entries
            .iter()
            .filter(|e| e.segment_id == segment_id)
            .collect();
        if cands.is_empty() {
            return None;
        }

        let q_norm = prefix_sig.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
        let q_ln = (prefix_len.max(1) as f32).ln();

        let mut scored: Vec<(usize, f32, f32)> = cands
            .iter()
            .enumerate()
            .map(|(idx, e)| {
                let dot: f32 = prefix_sig.iter().zip(&e.prefix_sig).map(|(a, b)| a * b).sum();
                let e_norm = e.prefix_sig.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
                let cos = dot / (q_norm * e_norm);
                let len_compat = (-self.len_alpha * (q_ln - (e.prefix_len.max(1) as f32).ln()).abs()).exp();
                let sim = cos * len_compat;
                (idx, cos, sim)
            })
            .collect();

        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(self.top_k);

        let max_sim = scored[0].2;
        let max_logit = max_sim / self.temperature;
        let exps: Vec<f32> = scored
            .iter()
            .map(|&(_, _, s)| (s / self.temperature - max_logit).exp())
            .collect();
        let sum_e: f32 = exps.iter().sum::<f32>().max(1e-12);
        let weights: Vec<f32> = exps.iter().map(|&e| e / sum_e).collect();

        let nf = cands[0].layer_delta_k.len();
        let f_dim = cands[0].layer_delta_k[0].len();
        let ns = cands[0].swa_tail_delta_k.len();
        let s_dim = cands[0].swa_tail_delta_k[0].len();

        let mut delta_k = vec![vec![0.0f32; f_dim]; nf];
        let mut delta_v = vec![vec![0.0f32; f_dim]; nf];
        let mut swa_delta_k = vec![vec![0.0f32; s_dim]; ns];
        let mut swa_delta_v = vec![vec![0.0f32; s_dim]; ns];
        let mut raw_cos_sims = Vec::with_capacity(scored.len());

        for (&(idx, cos, _), &w) in scored.iter().zip(&weights) {
            let e = cands[idx];
            raw_cos_sims.push((e.agent_id, cos, w));
            for fi in 0..nf {
                for i in 0..f_dim {
                    delta_k[fi][i] += w * e.layer_delta_k[fi][i];
                    delta_v[fi][i] += w * e.layer_delta_v[fi][i];
                }
            }
            for si in 0..ns {
                for i in 0..s_dim {
                    swa_delta_k[si][i] += w * e.swa_tail_delta_k[si][i];
                    swa_delta_v[si][i] += w * e.swa_tail_delta_v[si][i];
                }
            }
        }

        Some(KvCommEstimate {
            confident: max_sim >= self.min_conf,
            max_sim,
            raw_cos_sims,
            delta_k,
            delta_v,
            swa_delta_k,
            swa_delta_v,
        })
    }
}

// ============================================================================= 3. KvCollector (TokenDance Collective Round Reuse)

/// Report produced when admitting an agent into a collective multi-agent All-Gather round.
#[derive(Clone, Debug)]
pub struct RoundAdmitReport {
    pub agent_id: usize,
    pub used_kvcomm_offset: bool,
    pub kvcomm_max_sim: f32,
    pub flagged_tokens: usize,
    pub total_tokens: usize,
    pub diff_frac: f64,
}

/// TokenDance collective round coordinator: computes Master KV once per All-Gather broadcast block `S`,
/// screens shifted tokens via the early check layer (`blk.5` / slot 5), applies KVCOMM anchor offsets,
/// and stores `1 Master + N compact sparse diffs`.
pub struct KvCollector {
    pub store: MasterBlockStore,
    pub anchor_pool: KvCommAnchorPool,
    pub replicas: Vec<DiffAwareReplica>,
    pub check_slot: usize,
}

impl KvCollector {
    pub fn new(check_slot: usize) -> Self {
        Self {
            store: MasterBlockStore::new(),
            anchor_pool: KvCommAnchorPool::new(3, 0.15, 0.25, 0.20),
            replicas: Vec::new(),
            check_slot,
        }
    }

    /// Select top-`p` fraction of token positions ranked by check-layer deviation scores.
    pub fn select_top_positions(scores: &[f32], frac: f64) -> Vec<u32> {
        let k = ((scores.len() as f64) * frac.clamp(0.0, 1.0)).round() as usize;
        if k == 0 {
            return Vec::new();
        }
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut keep: Vec<u32> = order[..k.min(scores.len())].iter().map(|&i| i as u32).collect();
        keep.sort_unstable();
        keep
    }

    /// Admit an agent into the collective round: build its `DiffAwareReplica` (referencing Master + sparse diff
    /// only at `selected_positions` + optional KVCOMM offset) and optionally record its anchor into `anchor_pool`.
    #[allow(clippy::too_many_arguments)]
    pub fn admit_agent_round(
        &mut self,
        agent_id: usize,
        block_id: MasterBlockId,
        prefix_len: usize,
        prefix_sig: Vec<f32>,
        selected_positions: &[u32],
        mode: DiffStorageMode,
        agent_full_kv: &[u16],
        agent_swa_tail_kv: &[u16],
        record_anchor: bool,
        lut: &[f32],
    ) -> RoundAdmitReport {
        let est = self.anchor_pool.predict(block_id, &prefix_sig, prefix_len);
        let (used_kvcomm, max_sim) = match &est {
            Some(e) if e.confident => (true, e.max_sim),
            Some(e) => (false, e.max_sim),
            None => (false, 0.0),
        };

        let master = self.store.get(block_id).expect("master block missing");
        let len = master.len;
        let nf = master.n_full;
        let ns = master.n_swa;
        let tail_start = len.saturating_sub(master.swa_tail_len);

        // Pack sparse diffs ONLY at `selected_positions` across Full slots
        let mut diff_slices = Vec::with_capacity(nf + ns);
        for fi in 0..nf {
            let mut kv_data = Vec::with_capacity(selected_positions.len() * 2 * NKV * HD);
            for &pos_u32 in selected_positions {
                let t = pos_u32 as usize;
                let bkt = dist_bucket(t);
                for is_v in 0..2 {
                    for g in 0..NKV {
                        let m_vec = master.full_vec(fi, is_v, g, t);
                        let base = (((fi * 2 + is_v) * NKV + g) * len + t) * HD;
                        let a_vec = &agent_full_kv[base..base + HD];
                        match mode {
                            DiffStorageMode::Replacement => kv_data.extend_from_slice(a_vec),
                            DiffStorageMode::AdditiveDelta => {
                                for d in 0..HD {
                                    kv_data.push(f32_to_f16(lut[a_vec[d] as usize] - lut[m_vec[d] as usize]));
                                }
                            }
                            DiffStorageMode::ResidualOverOffset => {
                                let ofs = est.as_ref().filter(|e| e.confident).map(|e| {
                                    if is_v == 0 { &e.delta_k[fi] } else { &e.delta_v[fi] }
                                });
                                let o_off = (bkt * NKV + g) * HD;
                                for d in 0..HD {
                                    let ov = ofs.map_or(0.0, |v| v[o_off + d]);
                                    kv_data.push(f32_to_f16(
                                        lut[a_vec[d] as usize] - lut[m_vec[d] as usize] - ov,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            diff_slices.push(SparseDiffSlice {
                is_swa: false,
                kind_idx: fi,
                positions: selected_positions.to_vec(),
                kv_data,
            });
        }

        // Pack SWA tail sparse diffs only for selected positions falling inside the live tail `[tail_start..len)`
        let swa_sel: Vec<u32> = selected_positions
            .iter()
            .copied()
            .filter(|&p| (p as usize) >= tail_start)
            .collect();
        if !swa_sel.is_empty() {
            for si in 0..ns {
                let mut kv_data = Vec::with_capacity(swa_sel.len() * 2 * NKV * HD);
                for &pos_u32 in &swa_sel {
                    let rel = (pos_u32 as usize) - tail_start;
                    for is_v in 0..2 {
                        for g in 0..NKV {
                            let m_vec = master.swa_tail_vec(si, is_v, g, rel);
                            let base = (((si * 2 + is_v) * NKV + g) * master.swa_tail_len + rel) * HD;
                            let a_vec = &agent_swa_tail_kv[base..base + HD];
                            match mode {
                                DiffStorageMode::Replacement => kv_data.extend_from_slice(a_vec),
                                DiffStorageMode::AdditiveDelta | DiffStorageMode::ResidualOverOffset => {
                                    for d in 0..HD {
                                        kv_data.push(f32_to_f16(lut[a_vec[d] as usize] - lut[m_vec[d] as usize]));
                                    }
                                }
                            }
                        }
                    }
                }
                diff_slices.push(SparseDiffSlice {
                    is_swa: true,
                    kind_idx: si,
                    positions: swa_sel.clone(),
                    kv_data,
                });
            }
        }

        let offsets = if let Some(e) = est.as_ref().filter(|e| e.confident) {
            vec![SegmentOffset {
                seg_id: 0,
                n_buckets: N_DIST_BUCKETS,
                full_dk: e.delta_k.iter().map(|v| v.iter().map(|&x| f32_to_f16(x)).collect()).collect(),
                full_dv: e.delta_v.iter().map(|v| v.iter().map(|&x| f32_to_f16(x)).collect()).collect(),
                swa_dk: e.swa_delta_k.iter().map(|v| v.iter().map(|&x| f32_to_f16(x)).collect()).collect(),
                swa_dv: e.swa_delta_v.iter().map(|v| v.iter().map(|&x| f32_to_f16(x)).collect()).collect(),
            }]
        } else {
            Vec::new()
        };

        if record_anchor {
            let entry = KvCommAnchorPool::observe_anchor(
                block_id,
                agent_id,
                prefix_sig,
                prefix_len,
                master,
                agent_full_kv,
                agent_swa_tail_kv,
                lut,
            );
            self.anchor_pool.insert(entry);
        }

        self.replicas.push(DiffAwareReplica {
            agent_id,
            prefix_len,
            segments: vec![MasterSegmentRef {
                block_id,
                logical_offset: prefix_len as u64,
                len,
                seg_id: 0,
            }],
            mode,
            diff_slices,
            offsets,
        });

        RoundAdmitReport {
            agent_id,
            used_kvcomm_offset: used_kvcomm,
            kvcomm_max_sim: max_sim,
            flagged_tokens: selected_positions.len(),
            total_tokens: len,
            diff_frac: selected_positions.len() as f64 / len.max(1) as f64,
        }
    }
}

// ============================================================================= 4. GPU Kernel Dispatch Helpers

/// Dispatch `k_kv_diff_extract` on GPU to compute per-token deviation scores and compact sparse diffs.
#[allow(clippy::too_many_arguments)]
pub fn gpu_kv_diff_extract(
    eng: &Engine,
    agent_kv: &Buf,
    master_kv: &Buf,
    diff_kv: &Buf,
    diff_map: &Buf,
    counter: &Buf,
    scores: &Buf,
    args: &KvDiffExtractArgs,
) {
    let cb = eng.gpu.cmd();
    let e = Enc::new(&cb);
    e.pso(&eng.gpu.pso("k_kv_diff_extract"));
    e.buf(agent_kv, 0, 0);
    e.buf(master_kv, 0, 1);
    e.buf(diff_kv, 0, 2);
    e.buf(diff_map, 0, 3);
    e.buf(counter, 0, 4);
    e.bytes(args, 5);
    e.buf(scores, 0, 6);
    e.dispatch((args.n_tok as usize, 1, 1), HD);
    e.end();
    cb.commit();
    cb.waitUntilCompleted();
}

/// Dispatch `k_kvcomm_apply_offset` in-place on GPU to align a staged KV slice with KVCOMM offsets.
pub fn gpu_kvcomm_apply_offset(
    eng: &Engine,
    kv_buf: &Buf,
    offset_kv: &Buf,
    bucket_map: &Buf,
    args: &KvCommOffsetArgs,
) {
    let cb = eng.gpu.cmd();
    let e = Enc::new(&cb);
    e.pso(&eng.gpu.pso("k_kvcomm_apply_offset"));
    e.buf(kv_buf, 0, 3);
    e.bytes(args, 5);
    e.buf(offset_kv, 0, 8);
    e.buf(bucket_map, 0, 9);
    e.dispatch((args.n_tok as usize, NKV, 1), HD);
    e.end();
    cb.commit();
    cb.waitUntilCompleted();
}

/// Dispatch `k_attn_master_diff_b` (and `k_attn_combine_b` if `args.nsplit > 1`) on GPU; returns GPU duration (seconds).
#[allow(clippy::too_many_arguments)]
pub fn gpu_attn_master_diff_b(
    eng: &Engine,
    act: &Buf,
    master_kv: &Buf,
    meta: &Buf,
    diff_kv: &Buf,
    diff_map: &Buf,
    offset_kv: &Buf,
    args: &MasterDiffAttnArgs,
) -> f64 {
    let cb = eng.gpu.cmd();
    let e = Enc::new(&cb);
    e.pso(&eng.gpu.pso("k_attn_master_diff_b"));
    e.buf(&eng.pb, 0, 1);
    e.buf(act, 0, 2);
    e.buf(master_kv, 0, 3);
    e.bytes(args, 5);
    e.buf(meta, 0, 7);
    e.buf(diff_kv, 0, 8);
    e.buf(diff_map, 0, 9);
    e.buf(offset_kv, 0, 10);
    e.dispatch((args.nsplit as usize, NKV, args.b_count as usize), HD);
    if args.nsplit > 1 {
        e.barrier();
        let ca = CombineBArgs {
            part_off: args.part_off,
            nsplit: args.nsplit,
            gate_off: args.gate_off,
            out_off: args.out_off,
        };
        e.pso(&eng.gpu.pso("k_attn_combine_b"));
        e.buf(act, 0, 2);
        e.bytes(&ca, 5);
        e.dispatch((NH, args.b_count as usize, 1), HD);
    }
    e.end();
    cb.commit();
    cb.waitUntilCompleted();
    cb.GPUEndTime() - cb.GPUStartTime()
}

// ============================================================================= 5. Engine KV Extraction & Patching Helpers

fn prefill_range(eng: &Engine, toks: &[u32], p0: usize, logits_end: bool) {
    eng.set_tokens(p0, toks);
    let step_max = EXT.min(eng.chunk);
    let mut off = 0usize;
    while off < toks.len() {
        let t = (toks.len() - off).min(step_max);
        let cb = eng.gpu.cmd();
        eng.encode_prefill(&cb, p0 + off, t, logits_end && (off + t == toks.len()), false);
        cb.commit();
        cb.waitUntilCompleted();
        off += t;
    }
}

/// Extract Full-attention KV for logical positions `[pos0 .. pos0 + len)` into `[n_full][2][NKV][len][HD]`.
fn extract_full_kv_range(eng: &Engine, pos0: usize, len: usize) -> Vec<u16> {
    let full_slots: Vec<usize> = (0..N_SLOT).filter(|&s| !eng.slot_kind[s].0).collect();
    let nf = full_slots.len();
    let cap = eng.full_cap.get();
    let kv = eng.kv_full.borrow();
    let src = kv.contents().as_ptr() as *const u16;
    let mut out = vec![0u16; nf * 2 * NKV * len * HD];
    for (fi, &slot) in full_slots.iter().enumerate() {
        let (_, _, kb, vb, _) = eng_kv_slot_bases(eng, slot);
        for (is_v, base) in [(0usize, kb), (1usize, vb)] {
            for g in 0..NKV {
                let s_off = base as usize + (g * cap + pos0) * HD;
                let d_off = ((fi * 2 + is_v) * NKV + g) * len * HD;
                unsafe {
                    std::ptr::copy_nonoverlapping(src.add(s_off), out.as_mut_ptr().add(d_off), len * HD);
                }
            }
        }
    }
    out
}

/// Extract SWA KV for logical positions `[pos0 .. pos0 + len)` (requires `len <= EXT`) into `[n_swa][2][NKV][len][HD]`.
fn extract_swa_kv_range(eng: &Engine, pos0: usize, len: usize) -> Vec<u16> {
    assert!(len <= EXT);
    let swa_slots: Vec<usize> = (0..N_SLOT).filter(|&s| eng.slot_kind[s].0).collect();
    let ns = swa_slots.len();
    let cap = eng.swa_cap.get();
    let kv = eng.kv_swa.borrow();
    let src = kv.contents().as_ptr() as *const u16;
    let mut out = vec![0u16; ns * 2 * NKV * len * HD];
    for (si, &slot) in swa_slots.iter().enumerate() {
        let (_, _, kb, vb, _) = eng_kv_slot_bases(eng, slot);
        for (is_v, base) in [(0usize, kb), (1usize, vb)] {
            for g in 0..NKV {
                let d_base = ((si * 2 + is_v) * NKV + g) * len * HD;
                for i in 0..len {
                    let ring_idx = (pos0 + i) % cap;
                    let s_off = base as usize + (g * cap + ring_idx) * HD;
                    unsafe {
                        std::ptr::copy_nonoverlapping(src.add(s_off), out.as_mut_ptr().add(d_base + i * HD), HD);
                    }
                }
            }
        }
    }
    out
}

/// Write Full-attention KV slice `[n_full][2][NKV][len][HD]` into `eng.kv_full` at `[pos0 .. pos0 + len)`.
fn write_full_kv_range(eng: &Engine, pos0: usize, len: usize, data: &[u16]) {
    let full_slots: Vec<usize> = (0..N_SLOT).filter(|&s| !eng.slot_kind[s].0).collect();
    let cap = eng.full_cap.get();
    let kv = eng.kv_full.borrow();
    let dst = kv.contents().as_ptr() as *mut u16;
    for (fi, &slot) in full_slots.iter().enumerate() {
        let (_, _, kb, vb, _) = eng_kv_slot_bases(eng, slot);
        for (is_v, base) in [(0usize, kb), (1usize, vb)] {
            for g in 0..NKV {
                let d_off = base as usize + (g * cap + pos0) * HD;
                let s_off = ((fi * 2 + is_v) * NKV + g) * len * HD;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr().add(s_off), dst.add(d_off), len * HD);
                }
            }
        }
    }
}

/// Write SWA KV slice `[n_swa][2][NKV][len][HD]` into `eng.kv_swa` at ring positions `(pos0 + i) % swa_cap`.
/// CRITICAL: Properly realigns ring indices when Agent `i` has prefix length `L_i != L_0`!
fn write_swa_kv_range(eng: &Engine, pos0: usize, len: usize, data: &[u16]) {
    let swa_slots: Vec<usize> = (0..N_SLOT).filter(|&s| eng.slot_kind[s].0).collect();
    let cap = eng.swa_cap.get();
    let kv = eng.kv_swa.borrow();
    let dst = kv.contents().as_ptr() as *mut u16;
    for (si, &slot) in swa_slots.iter().enumerate() {
        let (_, _, kb, vb, _) = eng_kv_slot_bases(eng, slot);
        for (is_v, base) in [(0usize, kb), (1usize, vb)] {
            for g in 0..NKV {
                let s_base = ((si * 2 + is_v) * NKV + g) * len * HD;
                for i in 0..len {
                    let ring_idx = (pos0 + i) % cap;
                    let d_off = base as usize + (g * cap + ring_idx) * HD;
                    unsafe {
                        std::ptr::copy_nonoverlapping(data.as_ptr().add(s_base + i * HD), dst.add(d_off), HD);
                    }
                }
            }
        }
    }
}

/// Overlay sparse exact-agent diff positions onto `eng.kv_full` and `eng.kv_swa` (for positions in live SWA tail).
fn overlay_sparse_diffs_into_engine(
    eng: &Engine,
    prefix_len: usize,
    s_len: usize,
    swa_tail_len: usize,
    positions: &[u32],
    agent_full_kv: &[u16],
    agent_swa_tail_kv: &[u16],
) {
    if positions.is_empty() {
        return;
    }
    let full_slots: Vec<usize> = (0..N_SLOT).filter(|&s| !eng.slot_kind[s].0).collect();
    let fcap = eng.full_cap.get();
    let fkv = eng.kv_full.borrow();
    let fdst = fkv.contents().as_ptr() as *mut u16;
    for (fi, &slot) in full_slots.iter().enumerate() {
        let (_, _, kb, vb, _) = eng_kv_slot_bases(eng, slot);
        for (is_v, base) in [(0usize, kb), (1usize, vb)] {
            for g in 0..NKV {
                let s_base = ((fi * 2 + is_v) * NKV + g) * s_len * HD;
                let d_base = base as usize + g * fcap * HD;
                for &p in positions {
                    let t = p as usize;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            agent_full_kv.as_ptr().add(s_base + t * HD),
                            fdst.add(d_base + (prefix_len + t) * HD),
                            HD,
                        );
                    }
                }
            }
        }
    }
    drop(fkv);

    let tail_start = s_len.saturating_sub(swa_tail_len);
    let swa_slots: Vec<usize> = (0..N_SLOT).filter(|&s| eng.slot_kind[s].0).collect();
    let scap = eng.swa_cap.get();
    let skv = eng.kv_swa.borrow();
    let sdst = skv.contents().as_ptr() as *mut u16;
    for (si, &slot) in swa_slots.iter().enumerate() {
        let (_, _, kb, vb, _) = eng_kv_slot_bases(eng, slot);
        for (is_v, base) in [(0usize, kb), (1usize, vb)] {
            for g in 0..NKV {
                let s_base = ((si * 2 + is_v) * NKV + g) * swa_tail_len * HD;
                let d_base = base as usize + g * scap * HD;
                for &p in positions {
                    let t = p as usize;
                    if t >= tail_start {
                        let rel = t - tail_start;
                        let ring_idx = (prefix_len + t) % scap;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                agent_swa_tail_kv.as_ptr().add(s_base + rel * HD),
                                sdst.add(d_base + ring_idx * HD),
                                HD,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Apply KVCOMM bucketed offsets directly onto `eng.kv_full` and `eng.kv_swa` on GPU via `k_kvcomm_apply_offset`.
fn apply_kvcomm_offsets_on_gpu(
    eng: &Engine,
    prefix_len: usize,
    s_len: usize,
    swa_tail_len: usize,
    delta_k: &[Vec<f32>],
    delta_v: &[Vec<f32>],
    swa_delta_k: &[Vec<f32>],
    swa_delta_v: &[Vec<f32>],
) {
    // Build per-token distance bucket map `[0..16)->0, [16..128)->1, [128..1024)->2, [1024..)->3`
    let bmap: Vec<u32> = (0..s_len).map(|t| dist_bucket(t) as u32).collect();
    let bmap_buf = eng.gpu.buffer_with(unsafe {
        std::slice::from_raw_parts(bmap.as_ptr() as *const u8, bmap.len() * 4)
    });

    // Pack all Full-slot offsets `[nf][N_DIST_BUCKETS][2][NKV][HD]` into one compact fp16 GPU buffer
    let full_slots: Vec<usize> = (0..N_SLOT).filter(|&s| !eng.slot_kind[s].0).collect();
    let nf = full_slots.len();
    let stride_slot = N_DIST_BUCKETS * 2 * NKV * HD;
    let mut full_ofs_h = vec![0u16; nf * stride_slot];
    for fi in 0..nf {
        for bkt in 0..N_DIST_BUCKETS {
            for g in 0..NKV {
                let src = (bkt * NKV + g) * HD;
                let dst_k = fi * stride_slot + (bkt * 2 * NKV + g) * HD;
                let dst_v = fi * stride_slot + (bkt * 2 * NKV + NKV + g) * HD;
                for d in 0..HD {
                    full_ofs_h[dst_k + d] = f32_to_f16(delta_k[fi][src + d]);
                    full_ofs_h[dst_v + d] = f32_to_f16(delta_v[fi][src + d]);
                }
            }
        }
    }
    let full_ofs_buf = eng.gpu.buffer_with(unsafe {
        std::slice::from_raw_parts(full_ofs_h.as_ptr() as *const u8, full_ofs_h.len() * 2)
    });

    // Pack and apply SWA tail offsets `[ns][1][2][NKV][HD]` together with Full slots in ONE GPU command buffer
    let swa_slots: Vec<usize> = (0..N_SLOT).filter(|&s| eng.slot_kind[s].0).collect();
    let ns = swa_slots.len();
    let s_stride = 2 * NKV * HD;
    let mut swa_ofs_h = vec![0u16; ns * s_stride];
    for si in 0..ns {
        for g in 0..NKV {
            let src = g * HD;
            let dst_k = si * s_stride + g * HD;
            let dst_v = si * s_stride + (NKV + g) * HD;
            for d in 0..HD {
                swa_ofs_h[dst_k + d] = f32_to_f16(swa_delta_k[si][src + d]);
                swa_ofs_h[dst_v + d] = f32_to_f16(swa_delta_v[si][src + d]);
            }
        }
    }
    let swa_ofs_buf = eng.gpu.buffer_with(unsafe {
        std::slice::from_raw_parts(swa_ofs_h.as_ptr() as *const u8, swa_ofs_h.len() * 2)
    });
    let tail_pos0 = prefix_len + s_len.saturating_sub(swa_tail_len);

    // Single concurrent Metal command buffer for all 8 Full + 40 SWA slots (disjoint slices -> zero hazard)
    let pso = eng.gpu.pso("k_kvcomm_apply_offset");
    let cb = eng.gpu.cmd();
    let e = Enc::new(&cb);
    e.pso(&pso);
    e.buf(&bmap_buf, 0, 9);

    let fkv = eng.kv_full.borrow();
    e.buf(&fkv, 0, 3);
    e.buf(&full_ofs_buf, 0, 8);
    for (fi, &slot) in full_slots.iter().enumerate() {
        let (_, _, kb, vb, cap) = eng_kv_slot_bases(eng, slot);
        let args = KvCommOffsetArgs {
            n_tok: s_len as u32,
            pos0: prefix_len as u32,
            cap: cap as u32,
            is_ring: 0,
            n_buckets: N_DIST_BUCKETS as u32,
            use_bucket_map: 1,
            pad0: 0,
            pad1: 0,
            k_base: kb,
            v_base: vb,
            ofs_base: (fi * stride_slot) as u64,
            scale: 1.0,
            pad2: 0.0,
        };
        e.bytes(&args, 5);
        e.dispatch((s_len, NKV, 1), HD);
    }

    let skv = eng.kv_swa.borrow();
    e.buf(&skv, 0, 3);
    e.buf(&swa_ofs_buf, 0, 8);
    for (si, &slot) in swa_slots.iter().enumerate() {
        let (_, _, kb, vb, cap) = eng_kv_slot_bases(eng, slot);
        let args = KvCommOffsetArgs {
            n_tok: swa_tail_len as u32,
            pos0: tail_pos0 as u32,
            cap: cap as u32,
            is_ring: 1,
            n_buckets: 1,
            use_bucket_map: 0,
            pad0: 0,
            pad1: 0,
            k_base: kb,
            v_base: vb,
            ofs_base: (si * s_stride) as u64,
            scale: 1.0,
            pad2: 0.0,
        };
        e.bytes(&args, 5);
        e.dispatch((swa_tail_len, NKV, 1), HD);
    }

    e.end();
    cb.commit();
    cb.waitUntilCompleted();
    drop(skv);
    drop(fkv);
}

fn snapshot_cs(eng: &Engine) -> Vec<u8> {
    unsafe {
        std::slice::from_raw_parts(eng.cs.contents().as_ptr() as *const u8, eng.cs.length()).to_vec()
    }
}

fn restore_cs(eng: &Engine, cs: &[u8]) {
    unsafe {
        std::ptr::copy_nonoverlapping(cs.as_ptr(), eng.cs.contents().as_ptr() as *mut u8, cs.len());
    }
}

/// Prefill `Q` on top of the current `[P_i | S]` KV+CS state and evaluate `decode_steps` teacher-forced decode steps.
fn eval_patched_decode(
    eng: &Engine,
    prefix_len: usize,
    s_len: usize,
    q_toks: &[u32],
    gt_toks: &[u32],
    gt_logits: &[Vec<f32>],
    decode_steps: usize,
) -> (f64, f64, usize) {
    let q_start = prefix_len + s_len;
    prefill_range(eng, q_toks, q_start, true);
    let mut sum_kl = 0.0f64;
    let mut worst_kl = 0.0f64;
    let mut agree = 0usize;

    for step in 0..decode_steps {
        if step > 0 {
            let pos = q_start + q_toks.len() + step - 1;
            eng.set_tokens(pos, &[gt_toks[step - 1]]);
            let cb = eng.gpu.cmd();
            eng.encode_step(&cb, pos, true, false);
            cb.commit();
            cb.waitUntilCompleted();
        }
        let lg: Vec<f32> = eng.logits().iter().map(|&x| softcap(x, eng.softcap)).collect();
        let kl = kl_div(&gt_logits[step], &lg);
        if argmax(&lg) == argmax(&gt_logits[step]) {
            agree += 1;
        }
        sum_kl += kl;
        if kl > worst_kl {
            worst_kl = kl;
        }
    }
    (sum_kl / decode_steps as f64, worst_kl, agree)
}

// ============================================================================= 6. Realistic Multi-Agent Test Corpus

struct MultiAgentCorpus {
    agent_names: [&'static str; 5],
    p_toks: Vec<Vec<u32>>,
    s_toks: Vec<u32>,
    q_toks: Vec<u32>,
}

fn build_multi_agent_corpus(tk: &Tokenizer, max_s_toks: usize) -> MultiAgentCorpus {
    let agent_names = [
        "Agent0:Planner(Master)",
        "Agent1:Eris(Kernels)",
        "Agent2:Ziz(Reliability)",
        "Agent3:CodeReviewer",
        "Agent4:SystemsArch(HeldOut)",
    ];

    // 5 genuinely distinct agent prefixes ranging ~150 to ~900 tokens
    let prefixes: [&str; 5] = [
        // P_0 (~170 tok): Planner (Master baseline)
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
<|message_system|><|content_text|>You are the Release Planner coordinating a multi-agent systems review. \
Track milestones, release freeze gates, configuration constants, and rollout checklists with strict precision. \
Always report exact identifiers and numeric parameters verbatim.<|end_message|>\
<|message_user|><|content_text|>Start sprint 14 readiness checklist and prepare to ingest the shared engineering dossier.<|end_message|>\
<|message_model|><|content_text|>Sprint 14 readiness tracker initialized. Waiting for the shared technical specification, \
fault-tolerance parameters, Metal 4 runtime ceilings, and release freeze metadata so I can verify sign-off gates.<|end_message|>\
<|message_user|><|content_text|>Here is the shared technical dossier:\n",

        // P_1 (~340 tok): Eris (Kernel & Systems Researcher)
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
<|message_system|><|content_text|>You are Eris, a GPU kernel and inference systems researcher specializing in \
Apple Silicon unified memory, Metal 4 cooperative tensor ops, simdgroup occupancy, and quantized KV-cache compression. \
You prioritize empirical measurements, exact roofline accounting, and zero-drift numerical verification over speculation.<|end_message|>\
<|message_user|><|content_text|>Quick recap before we inspect the dossier: what are our hard hardware constraints on the M5 48 GiB workstation?<|end_message|>\
<|message_model|><|content_text|>On the 48 GiB unified-memory node, Inkling 2 Mini Q4_K_M weights occupy roughly 11.4 GiB wired, \
leaving a strict working envelope for concurrent batch KV pools and activation arenas. \
Because the 40 sliding-window slots cap at EXT=1024 keys (160 MiB per stream at fp16), long-context scaling is dominated \
by the 8 full-attention slots (blk.5, blk.11 across 3 loop passes, blk.17 across 3 loop passes, and blk.23), \
which cost 32 KiB per token per stream. At 64 to 256 concurrent agents sharing long contexts, naive dense duplication \
blows past physical memory immediately, so Master-Mirror page sharing plus sparse diffs is mandatory. \
Send over the shared engineering dossier whenever you are ready.<|end_message|>\
<|message_user|><|content_text|>Here is the shared technical dossier:\n",

        // P_2 (~520 tok): Ziz (Adversarial Reliability & Security Reviewer)
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
<|message_system|><|content_text|>You are Ziz, an adversarial reliability, concurrency, and numerical-safety auditor. \
Your job is to hunt down subtle failure modes: 32-bit integer overflows at 256K context, SWA ring-index misalignment \
when prefix lengths diverge across sibling streams, fast-math NaN hazards from exp(-INFINITY) in empty splits, \
atomic counter overflow past sparse diff pool capacity, and silent data corruption across memory barriers.<|end_message|>\
<|message_user|><|content_text|>Before reading the full dossier, log your top three pre-audit invariants for shared KV reuse.<|end_message|>\
<|message_model|><|content_text|>Pre-audit invariants recorded:\n\
1. Ring-index invariance: When a canonical shared block S computed at origin prefix length L_0 is mapped onto a sibling \
stream with private prefix length L_i != L_0, every sliding-window key at local offset t must land at physical ring index \
(L_i + t) mod swa_cap, never blindly copied from (L_0 + t) mod swa_cap.\n\
2. 64-bit addressing: All paged Master KV, sparse diff payload, and offset table strides must widen to u64/ulong before \
multiplication so 256K tokens across 256 concurrent streams never wrap a 32-bit offset.\n\
3. Bounded atomic compaction: Any GPU sparse diff extractor writing via atomic_fetch_add must bounds-check against \
diff_cap and redirect overflows cleanly to a sentinel 0xFFFFFFFF without out-of-bounds device writes.<|end_message|>\
<|message_user|><|content_text|>Also note whether we need RoPE re-rotation when pasting shared KV blocks across different offsets.<|end_message|>\
<|message_model|><|content_text|>Negative — Inkling 2 Mini uses zero rotary embeddings on stored K/V vectors; \
positional structure enters strictly via dynamic relative bias r dot B[d] for distances d < 1024 and log-scale temperature \
tau on queries. Therefore stored Master K/V blocks are position-independent, and any residual divergence comes strictly \
from causal 4-tap conv seams, SWA prefix overlap within 1024 tokens, and full-attention prefix conditioning. Ready for the dossier.<|end_message|>\
<|message_user|><|content_text|>Here is the shared technical dossier:\n",

        // P_3 (~700 tok): Code Reviewer (Rust & Metal Maintainer)
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
<|message_system|><|content_text|>You are the Principal Rust & Metal Maintainer reviewing production changes to \
inkling-engine. You inspect ABI layout parity between #[repr(C)] Rust structs and Metal constant buffers, \
explicit padding before 8-byte ulong fields, threadgroup barrier uniformity, coalesced float4/half4 memory access, \
clean separation of concerns, and accurate telemetry reporting.<|end_message|>\
<|message_user|><|content_text|>Summarize our repository conventions for Metal buffer indices and batch attention kernels.<|end_message|>\
<|message_model|><|content_text|>Repository Metal binding conventions in batch.metal and engine.rs:\n\
- buffer(0): W (mmapped GGUF quantized weights, zero-copy).\n\
- buffer(1): PB (packed f32 norms, q_clip, rel_clip, rel_bias, conv weights, and bf16 small matrices).\n\
- buffer(2): A (activation arena holding half Q/gate/out, f32 R/sink, and split-K partials).\n\
- buffer(3): KV pool (slot-major paged KV [slot_in_kind][pool_cap][2(K,V)][NKV=8][KV_BLK=32][HD=128]).\n\
- buffer(4): CS pool (branch-separated ring-indexed conv state).\n\
- buffer(5): constant kernel argument struct passed via setBytes.\n\
- buffer(7): META uint32 buffer holding per-sequence positions and page tables.\n\
Extension kernels like Master-Mirror attention bind supplementary read-only buffers starting at buffer(8) \
(DIFF_KV at 8, DIFF_MAP at 9, OFFSET_KV at 10) while preserving 100% compatibility with k_attn_combine_b \
whenever nsplit > 1.<|end_message|>\
<|message_user|><|content_text|>What compiler flags and verification bars must every new attention kernel pass?<|end_message|>\
<|message_model|><|content_text|>Every Metal 4 kernel must compile cleanly under both default fp16 and #ifdef KV8 modes \
with MTLLanguageVersion::Version4_0 and MTLMathMode::Fast without redefining shared macros from kernels.metal or \
batch.metal. Numerical verification requires max_abs_diff < 1e-3 against an exact f32/f64 dense reference covering \
relative bias (both d < 1024 and d >= 1024), per-head attention sink subtraction, log-scale tau, sigmoid output gating, \
and split-K combine paths. Memory benchmarks must verify both register-level restore latency and bounded RAM footprint \
so 48 GiB hosts never hit swap pressure. Please share the dossier for our review.<|end_message|>\
<|message_user|><|content_text|>Here is the shared technical dossier:\n",

        // P_4 (~880 tok): Held-out Target — Distributed Systems Architect
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>\
<|message_system|><|content_text|>You are the Distributed Systems Architect designing large-scale multi-agent \
collective inference fleets running Inkling 2 Mini on Apple Silicon clusters. You focus on round-level All-Gather \
broadcast topology, hierarchical Master-Mirror cache coherence, online cross-context offset calibration (KVCOMM), \
diff-aware sparse storage budgets (TokenDance), p99 time-to-first-token latency SLOs, and fleet memory capacity \
planning from 64 up to 256 concurrent agents at 256K shared context.<|end_message|>\
<|message_user|><|content_text|>Outline the core architectural bottleneck when 64 to 256 agents enter a synchronized \
debate or code-review round.<|end_message|>\
<|message_model|><|content_text|>In a synchronized multi-agent round, each agent b in 0..N enters with its own private \
persona and conversation history P_b of varying length L_b, and then every agent receives the exact same broadcast \
context block S (for example, a 4K-to-256K codebase snapshot, research paper, or All-Gather peer transcript). \
Because P_0, P_1, ..., P_{N-1} diverge, classical radix-tree prefix caching achieves ZERO hit rate on S! \
That creates a double wall:\n\
1. Compute wall: prefilling S independently N times wastes (N-1)/N of prefill FLOPs — up to 98% redundant work at N=64.\n\
2. Memory wall: storing N dense copies of S's KV cache multiplies memory by N. On Inkling 2 Mini at 256K context, \
one dense copy costs 8.16 GiB (8.00 GiB across the 8 full-attention slots plus 0.16 GiB across the 40 SWA slots). \
At N=64 agents that explodes to 522 GiB; at N=256 agents it exceeds 2.08 TiB.<|end_message|>\
<|message_user|><|content_text|>How do TokenDance and KVCOMM combine to break both walls simultaneously?<|end_message|>\
<|message_model|><|content_text|>They attack orthogonal components of the prefix-induced KV shift:\n\
- KVCOMM observes that the bulk of the shift Delta_KV(P_b, S) across a shared block S is a smooth per-layer, \
per-head, per-distance-bucket offset driven by the prefix representation P_b. By caching observed offsets from just \
a handful of anchor agents in an online KvCommAnchorPool and interpolating via top-k cosine and length compatibility, \
new agents align directly to the canonical Master KV of S with 0% recomputation of S and sub-megabyte offset storage.\n\
- TokenDance observes that the remaining non-uniform residual shift is sharply concentrated on a small sparse subset \
of tokens (boundary seams, salient entities, high-attention anchors). Using Inkling's first full-attention layer \
(blk.5, slot 5) as a cheap early check layer identifies the top 5% to 10% shifted positions; storing one canonical \
Master copy plus compact sparse diffs and fusing Master + Offset + Diff restoration directly inside registers during \
decode attention cuts fleet KV memory by 12x to 20x while preserving near-zero KL divergence.<|end_message|>\
<|message_user|><|content_text|>Great framing. Now ingest the shared technical dossier and extract the exact operational parameters.<|end_message|>\
<|message_model|><|content_text|>Ready to analyze the dossier and verify all system parameters.<|end_message|>\
<|message_user|><|content_text|>Here is the shared technical dossier:\n",
    ];

    // Shared context block S (~3,400–4,100 tokens) with real Rust/Metal specs + 4 planted facts across Early/Mid/Far/Deep regions!
    let mut shared = String::from(
r#"================================================================================
PROJECT HELIOS — SHARED MULTI-AGENT ENGINEERING DOSSIER (REV 4.2)
================================================================================

SECTION 1: EXECUTIVE DEPLOYMENT PARAMETERS & CANARY TELEMETRY
--------------------------------------------------------------------------------
The Helios collective inference service deploys Inkling 2 Mini across unified-memory
workstations running Metal 4. During early boot validation, every worker validates its
canonical configuration manifest against the cluster registry before admitting prompts.
The active deployment canary codename is ORCHID-7491, routed through the primary
failover port 9473 with a strict verification checksum threshold of 0.0042 on all
replicated Master-Mirror KV pages. Any replica whose check-layer deviation checksum
exceeds the emergency ceiling triggers an asynchronous anchor refresh without stalling
sibling decode streams.

1.1 Master-Mirror Page Table Invariants
Every logical sequence maintains a compact array of 32-token page descriptors backed by
a single canonical Master KV pool. Pages belonging to private history P_b carry
`seg_id = 0xFFFFFFFF` and `diff_slot = 0xFFFFFFFF`, executing the zero-overhead fast
path in `k_attn_master_diff_b`. Pages belonging to a broadcast All-Gather block S point
directly to the shared canonical Master page IDs allocated once when S enters the
`MasterBlockStore`. Because Inkling 2 Mini omits rotary positional embeddings (RoPE) on
cached keys and values, a single canonical Master page remains valid regardless of the
logical offset L_b at which any sibling agent mounts it.

```rust
/// Compact page descriptor mounted into an agent's logical KV timeline.
#[derive(Clone, Copy, Debug)]
pub struct HeliosPageMount {
    pub master_page_id: u32,
    pub segment_bucket: u16,
    pub flags: u16,
    pub logical_token_base: u64,
}

impl HeliosPageMount {
    pub const FLAG_SHARED_MASTER: u16 = 0x0001;
    pub const FLAG_HAS_KVCOMM_OFS: u16 = 0x0002;
    pub const FLAG_HAS_SPARSE_DIFF: u16 = 0x0004;

    pub fn is_shared(&self) -> bool {
        (self.flags & Self::FLAG_SHARED_MASTER) != 0
    }
}
```

1.2 Causal Short-Convolution Seam Handling
Inkling 2 Mini applies four 4-tap depthwise causal convolutions per layer execution
(`k_conv`, `v_conv`, `attn_post_conv`, and `ffn_post_conv`). At the boundary where
shared block S begins immediately after private prefix P_b, tokens 0, 1, and 2 of S
observe the final three activation frames of P_b in their convolution taps. Across 48
layer executions, those boundary perturbations attenuate rapidly inside sliding-window
layers once token distance passes the 1024-token SWA horizon, while full-attention slots
propagate a structured low-rank prefix shift throughout S.

SECTION 2: CACHE COHERENCE, QUORUM GOVERNANCE & EVICTION POLICY
--------------------------------------------------------------------------------
To prevent memory fragmentation under sustained multi-round agent conversations, the
Helios block manager enforces deterministic tiering and replica accounting. Cluster
consensus for canonical Master block promotion requires a shard quorum of 5-of-7
healthy nodes, supervised by lead engineer Dr. Elena Vance from the Distributed Runtime
team. Whenever unified pool utilization crosses the cache eviction watermark of 83.5%,
the collector demotes cold unreferenced Master blocks in LRU order while pinning active
round broadcasts in wired Metal shared memory.

2.1 Online Anchor Pool Mechanics (KVCOMM)
When the first three to four distinct agents process a shared broadcast block S under
diverging prefixes, the runtime records lightweight per-slot, per-distance-bucket mean
offsets (Delta_K, Delta_V) indexed by each prefix's full-attention KV summary signature
and log-length feature. For subsequent agents entering the round, the anchor pool
computes top-k cosine similarity modulated by length compatibility:
    sim(q, a) = cos(sig_q, sig_a) * exp(-alpha * |ln(L_q) - ln(L_a)|)
and synthesizes an interpolated offset vector via temperature-scaled softmax weights.
Crucially, because Delta_i = KV(P_i + S) - KV(P_0 + S) = delta(P_i) - delta(P_0), any
convex combination of anchor offsets automatically cancels Master P_0's idiosyncratic
bias while steering the shared KV manifold toward the target agent's prefix regime.

```rust
pub struct AnchorTelemetry {
    pub block_hash: u64,
    pub num_anchors: usize,
    pub max_cosine_sim: f32,
    pub length_compat: f32,
    pub used_zero_recompute: bool,
    pub estimated_prefill_saved_ms: f64,
}

pub fn should_admit_zero_recompute(t: &AnchorTelemetry, min_conf: f32) -> bool {
    t.num_anchors >= 2 && (t.max_cosine_sim * t.length_compat) >= min_conf
}
```

2.2 TokenDance Early Check-Layer Selective Screening
Empirical profiling of Inkling 2 Mini's 48-slot execution graph highlights a remarkable
structural gift: slots 0 through 4 are sliding-window layers (blk.0..blk.4), and slot 5
(blk.5) is the very first full-attention layer in the network! Tokens inside S that
experience strong prefix-induced divergence in deeper full-attention slots (blk.11
passes 0/1/2, blk.17 passes 0/1/2, and blk.23) already exhibit elevated relative L2
key/value shift at slot 5. Screening at slot 5 lets the runtime identify the top 5% to
10% high-deviation tokens after executing only 6 of 48 slots (12.5% depth), avoiding
dense storage and redundant recomputation for the remaining 90% to 95% of tokens.

2.3 Memory Accounting Across 40 SWA + 8 Full Slots
Let us verify exact byte formulas for fp16 KV storage on Inkling 2 Mini:
  - Per (slot, token): 2 (K,V) * 8 KV heads * 128 head_dim * 2 bytes = 4,096 bytes (4 KiB).
  - 40 SWA slots capped at EXT = 1024 tokens: 40 * 1024 * 4 KiB = 160 MiB constant per stream.
  - 8 Full slots at context length L: 8 * L * 4 KiB = 32 KiB * L per stream.
  - Conv-state ring (48 slots * 6656 channels * 3 taps * 4 bytes f32) = 3.66 MiB per stream.
At L = 262,144 (256K context):
  - Single canonical Master copy = 8.00 GiB (Full) + 0.156 GiB (SWA) + 0.004 GiB (CS) = 8.16 GiB.
  - Naive 64-agent dense deployment = 522.2 GiB (10.9x larger than a 48 GiB workstation!).
  - Naive 256-agent dense deployment = 2,089.0 GiB (over 2 TiB!).
  - Master-Mirror + 5% sparse diffs at N=64 = 8.16 + 63 * 0.408 = 33.9 GiB (15.4x compression!).
  - Master-Mirror + KVCOMM 0% diff at N=64 = 8.16 + 0.02 = 8.18 GiB (63.8x compression!).

SECTION 3: METAL 4 KERNEL SCHEDULING, THERMAL LIMITS & FALLBACK PATHS
--------------------------------------------------------------------------------
To keep decode latency predictable under heavy parallel load, the Metal 4 command
scheduler partitions attention work into 128-thread threadgroups (4 simdgroups of 32
lanes), each processing one GQA key-value group (3 query heads per KV head) across
8 keys per outer iteration. Because all 64 concurrent agents read the exact same
16 MiB Master KV buffer per slot instead of thrashing 1.0 GiB of duplicate per-agent
pages, the Apple Silicon System-Level Cache (SLC) absorbs Master page traffic and
turns decode attention from DRAM-bandwidth-bound into L2/SLC-resident execution.

Operational hardware guardrails for the production inference fleet are locked as
follows: the certified hardware batch limit is B_MAX = 192 concurrent active sequences
per node, governed by a strict thermal throttle ceiling of 87.4°C on the SoC die sensor.
If dynamic load shedding detects diff-table saturation beyond 25% of block length on any
pathological adversarial prompt, the dispatcher automatically switches that stream onto
the fallback kernel k_sparse_fallback_v3 so latency SLOs remain bounded for healthy
peers sharing the same command buffer.

3.1 Register-Level Fused Restore Pipeline
Materializing dense per-agent KV caches from Master + Diff before attention wastes both
memory capacity and DRAM write/read traffic. Instead, `k_attn_master_diff_b` resolves
each key `p` inside the single-pass online-softmax inner loop:
  1. Look up `mpid = master_pt[bidx]` and `dslot = DIFF_MAP[b, p]`.
  2. Look up `seg_id = seg_pt[bidx]` (one uint32 per 32-token page, 100% L1-resident).
  3. Compute `k_eff` directly in float4 registers:
     - Mode 0 (Replacement): `has_diff ? diff_k : (master_k + seg_ofs_k)`
     - Mode 1 (Additive):    `has_diff ? (master_k + diff_k) : (master_k + seg_ofs_k)`
     - Mode 2 (Residual):    `has_diff ? (master_k + seg_ofs_k + diff_k) : (master_k + seg_ofs_k)`
  4. Apply Inkling's dynamic relative bias `dot(r_head, rel_bias[pos - p])` when `pos - p < 1024`.
  5. Update online softmax `(m, l, o)` with per-head attention sink subtraction.
  6. Broadcast `(mpid, boff, dslot, seg_id)` across all 32 simdgroup lanes using
     `simd_shuffle(..., q2 * 4)` so the coalesced value load incurs zero duplicate metadata reads!

```metal
// Conceptual fast-path dispatch contract for Master-Mirror decode attention:
//   Grid:        (nsplit, NKV=8, B)
//   Threadgroup: 128 threads (4 simdgroups x 32 lanes)
//   Memory:      Zero dense sibling KV allocation; O(1) Master + O(p * B) sparse diffs.
```

3.2 Failure Postmortem Archive — Incident #408 (Ring Misalignment)
During early prototyping on multi-length agent batches, an engineer copied Master SWA
ring bytes verbatim into Agent 3's SWA buffer without shifting by `(L_3 - L_0) mod 1024`.
At first glance, perplexity on short prompts looked acceptable, yet factual recall of
tokens inside the last 1024 positions collapsed completely with KL spiking above 4.5 nats.
The root cause was immediate: SWA attention addresses key `p` at `p % swa_cap`, so
unshifted rings scrambled every relative distance `d = pos - p` inside the 1024-token
window! Fixing `write_swa_kv_range` to remap each tail position `t` to `(L_i + t) % swa_cap`
dropped KL by over two orders of magnitude immediately.

3.3 Failure Postmortem Archive — Incident #419 (Fast-Math Inf Hazard)
Under `MTLMathMode::Fast`, computing `exp(m - mx)` when an uninitialized split begins at
`m = -INFINITY` or when masked tail lanes evaluate `t = -INFINITY` can propagate NaN if
compiler reassociation folds `-INFINITY - (-INFINITY)`. Explicitly guarding every online
softmax scale factor with `(m[h] == -INFINITY) ? 0.0f : exp(m[h] - mx)` and
`valid ? exp(t - mx) : 0.0f` eliminates the hazard with zero measurable cycle overhead.

SECTION 4: MULTI-AGENT COLLECTIVE PROTOCOL & VERIFICATION BENCHMARKS
--------------------------------------------------------------------------------
In a typical 4-round agentic debate over a shared codebase, agents exchange structured
critiques via an All-Gather barrier at the end of each round. Each round's consolidated
transcript becomes a new immutable `MasterBlock` identified by its 64-bit FNV-1a token
hash. Because subsequent rounds append onto the same shared blocks, `MasterBlockStore`
deduplicates recurring tool outputs, compiler diagnostics, and source files automatically.

4.1 Multi-Slot Deviation Signature on Inkling 2 Mini
Across Inkling's three recurrent loop passes (`blk.6..17` executed in pass 0, pass 1, and
pass 2 with learned loop gain/forget controllers), full-attention layers `blk.11` and
`blk.17` are visited three times each at distinct KV slots:
  - Pass 0: slots 11 (`blk.11/p0`) and 17 (`blk.17/p0`)
  - Pass 1: slots 29 (`blk.11/p1`) and 35 (`blk.17/p1`)
  - Pass 2: slots 41 (`blk.11/p2`) and 47 (`blk.17/p2`)
Combined with prelude full slot 5 (`blk.5/p0`) and epilogue full slot 23 (`blk.23/p0`),
these 8 full-attention slots capture long-range cross-context dependencies while the 40
sliding-window slots remain strongly localized. Consequently, measuring per-slot deviation
anatomy reveals a crisp bifurcation: SWA slots converge closely toward Master once token
offset inside S passes 1024, whereas deeper Full slots accumulate persistent low-rank
prefix shifts that KVCOMM anchor offsets cleanly cancel!

SECTION 5: RELEASE GOVERNANCE, TELEMETRY BUCKETS & BINARY ARTIFACT FORMAT
--------------------------------------------------------------------------------
Final production cutover for Project Helios is governed by the Release Engineering board.
The mandatory release freeze date is November 19, after which only severity-0 kernel
correctness patches may enter the branch. All continuous benchmark telemetry streams into
the canonical telemetry bucket prod-metal4-west, and every serialized Master-Mirror
checkpoint file must begin with the 32-bit magic header 0x54444B56 (ASCII "TDKV")
followed by schema version 2, a 64-bit token content hash, slot-kind descriptors, and
CRC32C payload verification trailers.

5.1 End-to-End Verification Acceptance Criteria
Before promoting any TokenDance + KVCOMM build to staging, automated CI verifies:
  1. Cross-context deviation anatomy across all 40 SWA slots and all 8 Full slots.
  2. TokenDance check-layer (`blk.5`) sparse diff sweep across 0%..25% budgets with exact
     teacher-forced decode KL and argmax agreement against full ground-truth recompute.
  3. Held-out agent generalization of `KvCommAnchorPool` trained strictly on prior agents,
     measuring standalone zero-recompute KVCOMM offset alignment and hybrid KVCOMM + 5%/10%
     check-layer sparse diffs.
  4. Bit-accurate GPU verification of `k_attn_master_diff_b` (max_abs_diff < 1e-3 across
     replacement, additive, and residual-over-offset modes, both single-pass and split-K)
     and B=64 decode latency benchmarking demonstrating real SLC reuse speedup.
"#);

    // Ensure S is robustly in the 3.4K..4.2K token sweet spot (capped cleanly by max_s_toks)
    let target_s_toks = 3550usize.min(max_s_toks);
    let mut s_toks = tk.encode(&shared, false);
    let mut log_idx = 0usize;
    while s_toks.len() < target_s_toks {
        log_idx += 1;
        let rec = format!(
            "\n[Telemetry Record #{:03}] round={} slot_group=full_8 anchor_pool_hits={} \
             slc_hit_rate={:.1}% master_pages={} sparse_diff_pct={:.1}% kl_nats={:.5} \
             status=VERIFIED_NOMINAL note=\"Zero-RoPE Master-Mirror page check passed without seam drift.\"",
            log_idx,
            (log_idx % 12) + 1,
            3 + (log_idx % 5),
            91.0 + (log_idx % 8) as f64 * 0.8,
            96 + (log_idx % 32),
            5.0 + (log_idx % 4) as f64 * 1.5,
            0.0012 + (log_idx % 7) as f64 * 0.0003,
        );
        shared.push_str(&rec);
        s_toks = tk.encode(&shared, false);
        if log_idx > 200 {
            break;
        }
    }
    if s_toks.len() > max_s_toks {
        s_toks.truncate(max_s_toks);
    }

    // Shared query Q forcing retrieval across ALL 4 regions of S (Early, Mid-SWA, Far-Full, Deep-Full) + role awareness
    let query_str = "\n<|end_message|><|message_user|><|content_text|>\
From your role's perspective and the shared Project Helios dossier above, state the exact operational parameters: \
(1) canary codename and primary failover port, \
(2) shard quorum and cache eviction watermark, \
(3) hardware batch limit and fallback kernel name, \
(4) release freeze date, telemetry bucket, and checkpoint magic header. \
Give a concise bullet list with exact values.<|end_message|><|message_model|><|content_text|>";

    let p_toks: Vec<Vec<u32>> = prefixes.iter().map(|p| tk.encode(p, true)).collect();
    let q_toks = tk.encode(query_str, true);

    MultiAgentCorpus {
        agent_names,
        p_toks,
        s_toks,
        q_toks,
    }
}

// ============================================================================= 7. Deviation Statistics Helpers

#[derive(Clone, Default)]
struct SlotDevAccum {
    // Per distance bucket: sum(diff_k^2), sum(master_k^2), sum(diff_v^2), sum(master_v^2)
    dk2: [f64; N_DIST_BUCKETS],
    mk2: [f64; N_DIST_BUCKETS],
    dv2: [f64; N_DIST_BUCKETS],
    mv2: [f64; N_DIST_BUCKETS],
}

impl SlotDevAccum {
    fn rel_kv(&self, bkt: usize) -> f64 {
        let num = (self.dk2[bkt] + self.dv2[bkt]).sqrt();
        let den = (self.mk2[bkt] + self.mv2[bkt]).sqrt().max(1e-12);
        num / den
    }
    fn rel_kv_all(&self) -> f64 {
        let num = (self.dk2.iter().sum::<f64>() + self.dv2.iter().sum::<f64>()).sqrt();
        let den = (self.mk2.iter().sum::<f64>() + self.mv2.iter().sum::<f64>()).sqrt().max(1e-12);
        num / den
    }
    fn rel_k_all(&self) -> f64 {
        let num = self.dk2.iter().sum::<f64>().sqrt();
        let den = self.mk2.iter().sum::<f64>().sqrt().max(1e-12);
        num / den
    }
    fn rel_v_all(&self) -> f64 {
        let num = self.dv2.iter().sum::<f64>().sqrt();
        let den = self.mv2.iter().sum::<f64>().sqrt().max(1e-12);
        num / den
    }
}

/// Pearson correlation between two equal-length score slices (used to validate check-layer `blk.5` predictiveness).
fn pearson_corr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let ma = a[..n].iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let mb = b[..n].iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let (mut cov, mut va, mut vb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let da = a[i] as f64 - ma;
        let db = b[i] as f64 - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    cov / (va * vb).sqrt().max(1e-12)
}

// ============================================================================= 8. Experiment 4: GPU Kernel Verify & Benchmark

fn run_experiment_4_gpu_verify_and_bench(eng: &Engine, master: &MasterBlock, lut: &[f32]) {
    println!("\n================================================================================");
    println!("EXPERIMENT 4: Fused GPU `k_attn_master_diff_b` Verification & B=64 Latency Bench");
    println!("================================================================================");

    // ---- 4A. Exact numerical verification against exact f64/f32 CPU dense reference
    let check_phys = 5usize; // blk.5 (Full attention slot 0)
    let layer = &eng.layers[check_phys];
    let pb_f32 = unsafe {
        std::slice::from_raw_parts(eng.pb.contents().as_ptr() as *const f32, eng.pb.length() / 4)
    };
    let qclip = &pb_f32[layer.q_clip as usize / 4..layer.q_clip as usize / 4 + NH * HD];
    let relclip = &pb_f32[layer.rel_clip as usize / 4..layer.rel_clip as usize / 4 + NH * NREL];
    let relbias = &pb_f32[layer.rel_bias as usize / 4..layer.rel_bias as usize / 4 + EXT * NREL];

    let b_ver = 8usize;
    let l_ver = 2048usize.min(master.len / KV_BLK * KV_BLK).max(KV_BLK * 4);
    let n_pages = l_ver / KV_BLK;
    let kv_slot_elems = 2 * NKV * KV_BLK * HD;
    let max_segs = 4usize;

    // Pack canonical Master KV for slot 0 into paged MASTER_KV `[1][n_pages][2][NKV][32][128]`
    let mut master_paged = vec![0u16; n_pages * kv_slot_elems];
    for pid in 0..n_pages {
        for is_v in 0..2 {
            for g in 0..NKV {
                let dst_blk = pid * kv_slot_elems + (is_v * NKV + g) * (KV_BLK * HD);
                for off in 0..KV_BLK {
                    let t = pid * KV_BLK + off;
                    let src = master.full_vec(0, is_v, g, t.min(master.len - 1));
                    master_paged[dst_blk + off * HD..dst_blk + (off + 1) * HD].copy_from_slice(src);
                }
            }
        }
    }
    let master_buf = eng.gpu.buffer_with(unsafe {
        std::slice::from_raw_parts(master_paged.as_ptr() as *const u8, master_paged.len() * 2)
    });

    // Deterministic RNG (xorshift64)
    let mut rng_state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next_f32 = |scale: f32| -> f32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let u = ((rng_state >> 40) as f32) / ((1u32 << 24) as f32);
        (u * 2.0 - 1.0) * scale
    };

    // Build sparse diffs (~10% positions per agent) & per-bucket KVCOMM offsets
    let diff_per_b = (l_ver / 10).max(1);
    let diff_cap = b_ver * diff_per_b;
    let mut diff_map = vec![u32::MAX; b_ver * l_ver];
    let mut diff_kv = vec![0u16; diff_cap * 2 * NKV * HD];
    let mut d_cursor = 0usize;
    for b in 0..b_ver {
        for k in 0..diff_per_b {
            let p = ((k * 10 + (b * 3 + k) % 10) % l_ver) as usize;
            if diff_map[b * l_ver + p] != u32::MAX {
                continue;
            }
            let dslot = d_cursor;
            d_cursor += 1;
            diff_map[b * l_ver + p] = dslot as u32;
            let dbase = dslot * 2 * NKV * HD;
            for i in 0..2 * NKV * HD {
                diff_kv[dbase + i] = f32_to_f16(next_f32(0.18));
            }
        }
    }
    let diff_kv_buf = eng.gpu.buffer_with(unsafe {
        std::slice::from_raw_parts(diff_kv.as_ptr() as *const u8, diff_kv.len() * 2)
    });
    let diff_map_buf = eng.gpu.buffer_with(unsafe {
        std::slice::from_raw_parts(diff_map.as_ptr() as *const u8, diff_map.len() * 4)
    });

    // KVCOMM offsets `[1][b_ver][max_segs][2][NKV][HD]`
    let mut offset_kv = vec![0u16; b_ver * max_segs * 2 * NKV * HD];
    for x in &mut offset_kv {
        *x = f32_to_f16(next_f32(0.06));
    }
    let offset_kv_buf = eng.gpu.buffer_with(unsafe {
        std::slice::from_raw_parts(offset_kv.as_ptr() as *const u8, offset_kv.len() * 2)
    });

    // META: seq_pos[b_ver], master_pt[b_ver][n_pages], seg_pt[b_ver][n_pages]
    let seq_pos_off = 0u32;
    let master_pt_off = b_ver as u32;
    let seg_pt_off = (b_ver + b_ver * n_pages) as u32;
    let mut meta = vec![0u32; b_ver + 2 * b_ver * n_pages];
    let mut seq_pos = vec![0usize; b_ver];
    for b in 0..b_ver {
        // Vary lengths slightly across agents to test non-uniform pos + tau + relative bias!
        let pos = l_ver - 1 - (b * 7);
        seq_pos[b] = pos;
        meta[b] = pos as u32;
        for pid in 0..n_pages {
            meta[master_pt_off as usize + b * n_pages + pid] = pid as u32;
            let tok0 = pid * KV_BLK;
            meta[seg_pt_off as usize + b * n_pages + pid] = dist_bucket(tok0) as u32;
        }
    }
    let meta_buf = eng.gpu.buffer_with(unsafe {
        std::slice::from_raw_parts(meta.as_ptr() as *const u8, meta.len() * 4)
    });

    // Activation buffer layout (all offsets in f32 units):
    let q_off = 0u32;
    let r_off = q_off + (b_ver * NH * HD / 2) as u32;
    let sink_off = r_off + (b_ver * NH * NREL) as u32;
    let gate_off = sink_off + (b_ver * NH) as u32;
    let out_off = gate_off + (b_ver * NH * HD / 2) as u32;
    let part_off = out_off + (b_ver * NH * HD / 2) as u32;
    let max_nsplit = 8usize;
    let total_f32 = part_off as usize + b_ver * NH * max_nsplit * (HD + 2);
    let act_buf = eng.gpu.buffer(total_f32 * 4);

    let mut q_h = vec![0u16; b_ver * NH * HD];
    let mut r_f = vec![0.0f32; b_ver * NH * NREL];
    let mut sink_f = vec![0.0f32; b_ver * NH];
    let mut gate_h = vec![0u16; b_ver * NH * HD];
    for x in &mut q_h { *x = f32_to_f16(next_f32(1.2)); }
    for x in &mut r_f { *x = next_f32(0.8); }
    for x in &mut sink_f { *x = next_f32(1.5); }
    for x in &mut gate_h { *x = f32_to_f16(next_f32(1.5)); }

    unsafe {
        let ap = act_buf.contents().as_ptr() as *mut u8;
        std::ptr::copy_nonoverlapping(q_h.as_ptr() as *const u8, ap.add(q_off as usize * 4), q_h.len() * 2);
        std::ptr::copy_nonoverlapping(r_f.as_ptr() as *const u8, ap.add(r_off as usize * 4), r_f.len() * 4);
        std::ptr::copy_nonoverlapping(sink_f.as_ptr() as *const u8, ap.add(sink_off as usize * 4), sink_f.len() * 4);
        std::ptr::copy_nonoverlapping(gate_h.as_ptr() as *const u8, ap.add(gate_off as usize * 4), gate_h.len() * 2);
    }

    // Verify all 3 diff modes x (nsplit=1 and nsplit=8)
    const QK_CLIP_VAL: f32 = 22.627416610717773;
    const REL_CLIP_VAL: f32 = 8.0;

    for &(diff_mode, mode_name) in &[
        (0u32, "Replacement"),
        (1u32, "AdditiveDelta"),
        (2u32, "ResidualOverOffset"),
    ] {
        // Compute exact CPU dense reference for this diff_mode
        let mut ref_out = vec![0.0f32; b_ver * NH * HD];
        for b in 0..b_ver {
            let pos = seq_pos[b];
            let nkeys = pos + 1;
            let tau = 1.0f32 + eng.log_alpha * (((pos + 1) as f32 / eng.log_floor).max(1.0)).ln();
            for h in 0..NH {
                let g = h / 3;
                let mut qv = [0.0f32; HD];
                let mut qn2 = 0.0f32;
                for d in 0..HD {
                    let v = lut[q_h[(b * NH + h) * HD + d] as usize] * qclip[h * HD + d];
                    qv[d] = v;
                    qn2 += v * v;
                }
                let qf = (QK_CLIP_VAL / (qn2.sqrt() + 1e-6)).clamp(0.0, 1.0) * tau * 0.08838834764831845;
                for d in 0..HD { qv[d] *= qf; }

                let mut rv = [0.0f32; NREL];
                let mut rn2 = 0.0f32;
                for i in 0..NREL {
                    let v = r_f[(b * NH + h) * NREL + i] * relclip[h * NREL + i];
                    rv[i] = v;
                    rn2 += v * v;
                }
                let rf = (REL_CLIP_VAL / (rn2.sqrt() + 1e-6)).clamp(0.0, 1.0) * tau;
                for i in 0..NREL { rv[i] *= rf; }

                let sink = sink_f[b * NH + h] as f64;
                let mut scores = vec![0.0f64; nkeys];
                let mut max_s = sink;
                for p in 0..nkeys {
                    let pid = p / KV_BLK;
                    let boff = p % KV_BLK;
                    let dslot = diff_map[b * l_ver + p];
                    let seg = dist_bucket(pid * KV_BLK);
                    let mkb = pid * kv_slot_elems + g * (KV_BLK * HD) + boff * HD;
                    let okb = ((b * max_segs + seg) * 2 * NKV + g) * HD;
                    let mut dot = 0.0f64;
                    for d in 0..HD {
                        let k_eff = if dslot != u32::MAX && diff_mode == 0 {
                            lut[diff_kv[(dslot as usize * 2 * NKV + g) * HD + d] as usize]
                        } else {
                            let mut k = lut[master_paged[mkb + d] as usize];
                            if dslot != u32::MAX {
                                k += lut[diff_kv[(dslot as usize * 2 * NKV + g) * HD + d] as usize];
                                if diff_mode == 2 {
                                    k += lut[offset_kv[okb + d] as usize];
                                }
                            } else {
                                k += lut[offset_kv[okb + d] as usize];
                            }
                            k
                        };
                        dot += (qv[d] as f64) * (k_eff as f64);
                    }
                    let dist = pos - p;
                    if dist < EXT {
                        for i in 0..NREL {
                            dot += (rv[i] as f64) * (relbias[dist * NREL + i] as f64);
                        }
                    }
                    scores[p] = dot;
                    if dot > max_s { max_s = dot; }
                }
                let mut denom = (sink - max_s).exp();
                let mut acc = [0.0f64; HD];
                for p in 0..nkeys {
                    let w = (scores[p] - max_s).exp();
                    denom += w;
                    let pid = p / KV_BLK;
                    let boff = p % KV_BLK;
                    let dslot = diff_map[b * l_ver + p];
                    let seg = dist_bucket(pid * KV_BLK);
                    let mvb = pid * kv_slot_elems + (NKV + g) * (KV_BLK * HD) + boff * HD;
                    let ovb = ((b * max_segs + seg) * 2 * NKV + NKV + g) * HD;
                    for d in 0..HD {
                        let v_eff = if dslot != u32::MAX && diff_mode == 0 {
                            lut[diff_kv[(dslot as usize * 2 * NKV + NKV + g) * HD + d] as usize]
                        } else {
                            let mut v = lut[master_paged[mvb + d] as usize];
                            if dslot != u32::MAX {
                                v += lut[diff_kv[(dslot as usize * 2 * NKV + NKV + g) * HD + d] as usize];
                                if diff_mode == 2 {
                                    v += lut[offset_kv[ovb + d] as usize];
                                }
                            } else {
                                v += lut[offset_kv[ovb + d] as usize];
                            }
                            v
                        };
                        acc[d] += w * (v_eff as f64);
                    }
                }
                for d in 0..HD {
                    let gv = lut[gate_h[(b * NH + h) * HD + d] as usize];
                    let sig = 1.0f64 / (1.0 + (-(gv as f64)).exp());
                    ref_out[(b * NH + h) * HD + d] = ((acc[d] / denom) * sig) as f32;
                }
            }
        }

        for &nsplit in &[1u32, 8u32] {
            let args = MasterDiffAttnArgs {
                b_count: b_ver as u32,
                master_pool_cap: n_pages as u32,
                slot_idx: 0,
                is_swa: 0,
                pt_stride: n_pages as u32,
                pt_mod: 0,
                seq_pos_off,
                master_pt_off,
                seg_pt_off,
                q_off,
                r_off,
                sink_off,
                qclip_w: layer.q_clip,
                relclip_w: layer.rel_clip,
                relbias_w: layer.rel_bias,
                gate_off,
                out_off,
                part_off,
                nsplit,
                diff_cap: diff_cap as u32,
                diff_mode,
                max_segs: max_segs as u32,
                diff_map_stride: l_ver as u32,
                diff_map_slot_stride: 0,
                q_prepped: 0,
                pad0: 0,
                log_alpha: eng.log_alpha,
                log_floor: eng.log_floor,
            };
            gpu_attn_master_diff_b(eng, &act_buf, &master_buf, &meta_buf, &diff_kv_buf, &diff_map_buf, &offset_kv_buf, &args);
            let gpu_out = unsafe {
                std::slice::from_raw_parts(
                    (act_buf.contents().as_ptr() as *const u16).add(out_off as usize * 2),
                    b_ver * NH * HD,
                )
            };
            let mut max_diff = 0.0f32;
            let mut sum_diff = 0.0f64;
            for (&gh, &rf) in gpu_out.iter().zip(&ref_out) {
                let d = (lut[gh as usize] - rf).abs();
                if d > max_diff { max_diff = d; }
                sum_diff += d as f64;
            }
            let mean_diff = sum_diff / ref_out.len() as f64;
            let ok = max_diff < 1e-3;
            println!(
                "  verify mode={:<18} nsplit={}: max_abs_diff={:.6} mean_abs_diff={:.6} [{}]",
                mode_name,
                nsplit,
                max_diff,
                mean_diff,
                if ok { "PASS < 1e-3" } else { "WARN" }
            );
        }
    }

    // Release verification buffers before allocating B=64 benchmark buffers
    drop(master_buf);
    drop(diff_kv_buf);
    drop(diff_map_buf);
    drop(offset_kv_buf);
    drop(meta_buf);
    drop(act_buf);

    // ---- 4B. B = 64 Latency & Memory Micro-Benchmark (L = 4096 shared master vs dense independent pages)
    let b_bench = 64usize;
    let l_bench = 4096usize;
    let n_pages_b = l_bench / KV_BLK; // 128 pages -> 16 MiB master for 1 Full slot
    let nsplit_b = 8u32;

    println!(
        "\n  B={} agents, L={} shared master block (1 Full slot benchmark, projected x8 Full slots):",
        b_bench, l_bench
    );
    println!(
        "  {:<36} | {:>10} | {:>11} | {:>11} | {:>9}",
        "Configuration", "1-Slot us", "8-Full ms", "8-Full MiB", "Speedup"
    );
    println!("  {}", "-".repeat(90));

    // Shared activation buffer for B=64 (populated with finite non-zero values)
    let touch_pages = |buf: &Buf, val: u32| {
        let n_u32 = buf.length() / 4;
        let ptr = buf.contents().as_ptr() as *mut u32;
        let mut i = 0usize;
        while i < n_u32 {
            unsafe { *ptr.add(i) = val ^ (i as u32 & 0x00ff_00ff); }
            i += 1024; // 1 store per 4 KiB page faults in physical unified-memory pages (avoids zero-page aliasing)
        }
    };
    let q_off_b = 0u32;
    let r_off_b = q_off_b + (b_bench * NH * HD / 2) as u32;
    let sink_off_b = r_off_b + (b_bench * NH * NREL) as u32;
    let gate_off_b = sink_off_b + (b_bench * NH) as u32;
    let out_off_b = gate_off_b + (b_bench * NH * HD / 2) as u32;
    let part_off_b = out_off_b + (b_bench * NH * HD / 2) as u32;
    let tot_f32_b = part_off_b as usize + b_bench * NH * (nsplit_b as usize) * (HD + 2);
    let act_b = eng.gpu.buffer(tot_f32_b * 4);
    touch_pages(&act_b, 0x3800_3800);

    // Helper to time a configuration (3 warmup + 15 timed runs, median)
    let bench_cfg = |master_kv_b: &Buf,
                     meta_b: &Buf,
                     diff_kv_b: &Buf,
                     diff_map_b: &Buf,
                     ofs_b: &Buf,
                     args: &MasterDiffAttnArgs|
     -> f64 {
        for _ in 0..3 {
            let _ = gpu_attn_master_diff_b(eng, &act_b, master_kv_b, meta_b, diff_kv_b, diff_map_b, ofs_b, args);
        }
        let mut ts = Vec::with_capacity(15);
        for _ in 0..15 {
            ts.push(gpu_attn_master_diff_b(eng, &act_b, master_kv_b, meta_b, diff_kv_b, diff_map_b, ofs_b, args));
        }
        ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ts[ts.len() / 2] * 1e6 // microseconds
    };

    // Baseline: Dense independent KV (64 agents x 128 distinct pages = 1.0 GiB for 1 slot, strictly bounded!)
    let dense_pages = b_bench * n_pages_b;
    let dense_kv_buf = eng.gpu.buffer(dense_pages * kv_slot_elems * 2);
    touch_pages(&dense_kv_buf, 0x3400_3400);
    let dummy_diff = eng.gpu.buffer(256);
    let dummy_dmap = eng.gpu.buffer(256);
    let dummy_ofs = eng.gpu.buffer(256);
    let mut dense_meta = vec![0u32; b_bench + b_bench * n_pages_b];
    for b in 0..b_bench {
        dense_meta[b] = (l_bench - 1) as u32;
        for pid in 0..n_pages_b {
            dense_meta[b_bench + b * n_pages_b + pid] = (b * n_pages_b + pid) as u32;
        }
    }
    let dense_meta_buf = eng.gpu.buffer_with(unsafe {
        std::slice::from_raw_parts(dense_meta.as_ptr() as *const u8, dense_meta.len() * 4)
    });
    let dense_args = MasterDiffAttnArgs {
        b_count: b_bench as u32,
        master_pool_cap: dense_pages as u32,
        slot_idx: 0,
        is_swa: 0,
        pt_stride: n_pages_b as u32,
        pt_mod: 0,
        seq_pos_off: 0,
        master_pt_off: b_bench as u32,
        seg_pt_off: u32::MAX,
        q_off: q_off_b,
        r_off: r_off_b,
        sink_off: sink_off_b,
        qclip_w: layer.q_clip,
        relclip_w: layer.rel_clip,
        relbias_w: layer.rel_bias,
        gate_off: gate_off_b,
        out_off: out_off_b,
        part_off: part_off_b,
        nsplit: nsplit_b,
        diff_cap: 0,
        diff_mode: 0,
        max_segs: 0,
        diff_map_stride: 0,
        diff_map_slot_stride: 0,
        q_prepped: 0,
        pad0: 0,
        log_alpha: eng.log_alpha,
        log_floor: eng.log_floor,
    };
    let dense_us = bench_cfg(&dense_kv_buf, &dense_meta_buf, &dummy_diff, &dummy_dmap, &dummy_ofs, &dense_args);
    let dense_8slot_mib = (dense_pages * kv_slot_elems * 2 * N_FULL_SLOTS) as f64 / 1048576.0;
    println!(
        "  {:<36} | {:>8.1} us | {:>8.2} ms | {:>8.1} MiB | {:>8.2}x",
        "Dense Independent KV (B=64, 100%)",
        dense_us,
        dense_us * (N_FULL_SLOTS as f64) / 1e3,
        dense_8slot_mib,
        1.0
    );
    drop(dense_kv_buf);
    drop(dense_meta_buf);

    // Master-Mirror configurations: ONE shared 16 MiB Master KV + 0%, 5%, 10% sparse diffs + KVCOMM offsets!
    let shared_master_buf = eng.gpu.buffer(n_pages_b * kv_slot_elems * 2);
    touch_pages(&shared_master_buf, 0x3400_3400);
    let seg_pt_off_b = (b_bench + b_bench * n_pages_b) as u32;
    let mut shared_meta = vec![0u32; b_bench + 2 * b_bench * n_pages_b];
    for b in 0..b_bench {
        shared_meta[b] = (l_bench - 1) as u32;
        for pid in 0..n_pages_b {
            shared_meta[b_bench + b * n_pages_b + pid] = pid as u32;
            shared_meta[seg_pt_off_b as usize + b * n_pages_b + pid] = dist_bucket(pid * KV_BLK) as u32;
        }
    }
    let shared_meta_buf = eng.gpu.buffer_with(unsafe {
        std::slice::from_raw_parts(shared_meta.as_ptr() as *const u8, shared_meta.len() * 4)
    });
    let shared_ofs_buf = eng.gpu.buffer(b_bench * max_segs * 2 * NKV * HD * 2);
    touch_pages(&shared_ofs_buf, 0x2800_2800);

    for &(pct, label) in &[
        (0usize, "Master-Mirror + KVCOMM (0% diff)"),
        (5usize, "Master-Mirror + KVCOMM + 5% diff"),
        (10usize, "Master-Mirror + KVCOMM + 10% diff"),
    ] {
        let per_b = (l_bench * pct) / 100;
        let dcap = (b_bench * per_b).max(1);
        let mut dmap = vec![u32::MAX; b_bench * l_bench];
        if per_b > 0 {
            let mut cur = 0u32;
            let step = (l_bench / per_b).max(1);
            for b in 0..b_bench {
                for k in 0..per_b {
                    let p = (k * step + (b * 5) % step) % l_bench;
                    if dmap[b * l_bench + p] == u32::MAX && (cur as usize) < dcap {
                        dmap[b * l_bench + p] = cur;
                        cur += 1;
                    }
                }
            }
        }
        let dkv_buf = eng.gpu.buffer(dcap * 2 * NKV * HD * 2);
        touch_pages(&dkv_buf, 0x3000_3000);
        let dmap_buf = eng.gpu.buffer_with(unsafe {
            std::slice::from_raw_parts(dmap.as_ptr() as *const u8, dmap.len() * 4)
        });
        let args = MasterDiffAttnArgs {
            b_count: b_bench as u32,
            master_pool_cap: n_pages_b as u32,
            slot_idx: 0,
            is_swa: 0,
            pt_stride: n_pages_b as u32,
            pt_mod: 0,
            seq_pos_off: 0,
            master_pt_off: b_bench as u32,
            seg_pt_off: seg_pt_off_b,
            q_off: q_off_b,
            r_off: r_off_b,
            sink_off: sink_off_b,
            qclip_w: layer.q_clip,
            relclip_w: layer.rel_clip,
            relbias_w: layer.rel_bias,
            gate_off: gate_off_b,
            out_off: out_off_b,
            part_off: part_off_b,
            nsplit: nsplit_b,
            diff_cap: if pct == 0 { 0 } else { dcap as u32 },
            diff_mode: 0,
            max_segs: max_segs as u32,
            diff_map_stride: if pct == 0 { 0 } else { l_bench as u32 },
            diff_map_slot_stride: 0,
            q_prepped: 0,
            pad0: 0,
            log_alpha: eng.log_alpha,
            log_floor: eng.log_floor,
        };
        let us = bench_cfg(&shared_master_buf, &shared_meta_buf, &dkv_buf, &dmap_buf, &shared_ofs_buf, &args);
        let mem_8slot_bytes = (n_pages_b * kv_slot_elems * 2
            + if pct == 0 { 0 } else { dcap * 2 * NKV * HD * 2 }
            + b_bench * max_segs * 2 * NKV * HD * 2)
            * N_FULL_SLOTS
            + if pct == 0 { 0 } else { dmap.len() * 4 };
        println!(
            "  {:<36} | {:>8.1} us | {:>8.2} ms | {:>8.1} MiB | {:>8.2}x",
            label,
            us,
            us * (N_FULL_SLOTS as f64) / 1e3,
            mem_8slot_bytes as f64 / 1048576.0,
            dense_us / us.max(1e-6)
        );
    }
}

// ============================================================================= 9. Main Empirical Probe Entry Point

/// End-to-end empirical TokenDance + KVCOMM probe and benchmark suite on Inkling 2 Mini.
///
/// Runs 4 experiments strictly sequentially with bounded memory:
/// - **Exp 1**: Cross-Context KV Deviation Anatomy across all 40 SWA + 8 Full slots & 4 distance buckets.
/// - **Exp 2**: TokenDance Diff-Aware Storage & Check-Layer (`blk.5`) Selective Recompute KL/argmax sweep + 256K scaling.
/// - **Exp 3**: KVCOMM Online Anchor-Pool Offset Alignment trained on Agents 1..3, evaluated on held-out Agent 4.
/// - **Exp 4**: Fused GPU `k_attn_master_diff_b` exact verification (`max_diff < 1e-3`) + `B=64` latency micro-benchmark.
pub fn run_kv_share_probe(eng: &Engine, tk: &Tokenizer, decode_steps: usize) {
    assert!(!eng.kv8, "kv_share probe inspects exact fp16 KV cache (run without --kv8)");
    assert!(!eng.tier_on(), "kv_share probe expects in-memory KV cache (run without --kv-disk)");

    let decode_steps = decode_steps.clamp(4, 128);
    let prev_tail = eng.tail.get();
    eng.tail.set(false);
    let lut = build_h2f_lut();

    let max_s_budget = eng.max_ctx.saturating_sub(1400 + decode_steps).max(1280);
    let corpus = build_multi_agent_corpus(tk, max_s_budget);
    let s_len = corpus.s_toks.len();
    let q_len = corpus.q_toks.len();
    let max_p = corpus.p_toks.iter().map(|p| p.len()).max().unwrap_or(0);
    let max_total = max_p + s_len + q_len + decode_steps + 32;
    assert!(
        max_total <= eng.max_ctx,
        "kv_share probe needs ctx >= {max_total} (current max_ctx={}); pass -c {}",
        eng.max_ctx,
        max_total.div_ceil(1024) * 1024
    );

    println!("================================================================================");
    println!("TOKENDANCE + KVCOMM EMPIRICAL PROBE ON INKLING 2 MINI (Zero-RoPE, 40 SWA + 8 Full)");
    println!("================================================================================");
    println!(
        "Shared block S: {} tokens | Shared query Q: {} tokens | Decode steps: {}",
        s_len, q_len, decode_steps
    );
    for i in 0..5 {
        println!(
            "  {} -> private prefix P_{}: {} tokens (total P_{}+S+Q = {})",
            corpus.agent_names[i],
            i,
            corpus.p_toks[i].len(),
            i,
            corpus.p_toks[i].len() + s_len + q_len
        );
    }

    // Pre-allocate stable KV capacities so swa_cap == EXT and full_cap >= max_total never resize mid-probe
    let chunk_sz = EXT.min(eng.chunk);
    eng.reserve(max_total, false);
    eng.ensure_batch(chunk_sz);

    let swa_slots: Vec<usize> = (0..N_SLOT).filter(|&s| eng.slot_kind[s].0).collect();
    let full_slots: Vec<usize> = (0..N_SLOT).filter(|&s| !eng.slot_kind[s].0).collect();
    let (ns, nf) = (swa_slots.len(), full_slots.len());
    let check_full_idx = full_slots
        .iter()
        .position(|&s| s == CHECK_LAYER_SLOT)
        .unwrap_or(0);

    // -------------------------------------------------------------------------
    // Step 0: Prefill Master (Agent 0: P_0 + S) ONCE and build MasterBlockStore
    // -------------------------------------------------------------------------
    let t_master = std::time::Instant::now();
    eng.reset();
    let l0 = corpus.p_toks[0].len();
    prefill_range(eng, &corpus.p_toks[0], 0, false);

    // Prefill S in <= EXT chunks to capture Master's full-span SWA history for Exp 1 anatomy
    let mut master_swa_fullspan = vec![0u16; ns * 2 * NKV * s_len * HD];
    let mut off = 0usize;
    while off < s_len {
        let t = (s_len - off).min(chunk_sz);
        prefill_range(eng, &corpus.s_toks[off..off + t], l0 + off, false);
        let chunk_swa = extract_swa_kv_range(eng, l0 + off, t);
        for si in 0..ns {
            for is_v in 0..2 {
                for g in 0..NKV {
                    let src = ((si * 2 + is_v) * NKV + g) * t * HD;
                    let dst = (((si * 2 + is_v) * NKV + g) * s_len + off) * HD;
                    master_swa_fullspan[dst..dst + t * HD].copy_from_slice(&chunk_swa[src..src + t * HD]);
                }
            }
        }
        off += t;
    }
    let master_s_prefill_ms = t_master.elapsed().as_secs_f64() * 1e3;

    let master_full_kv = extract_full_kv_range(eng, l0, s_len);
    let swa_tail_len = s_len.min(EXT);
    let master_swa_tail_kv = extract_swa_kv_range(eng, l0 + s_len - swa_tail_len, swa_tail_len);
    let master_end_cs = snapshot_cs(eng);

    // Also stage Master slot-5 KV in a compact GPU buffer for GPU `k_kv_diff_extract` verification!
    let master_check_buf = {
        let mut buf_h = vec![0u16; 2 * NKV * s_len * HD];
        let src_base = check_full_idx * 2 * NKV * s_len * HD;
        buf_h.copy_from_slice(&master_full_kv[src_base..src_base + 2 * NKV * s_len * HD]);
        eng.gpu.buffer_with(unsafe {
            std::slice::from_raw_parts(buf_h.as_ptr() as *const u8, buf_h.len() * 2)
        })
    };

    let mut collector = KvCollector::new(CHECK_LAYER_SLOT);
    let master_id = collector.store.register(
        corpus.s_toks.clone(),
        nf,
        master_full_kv,
        ns,
        swa_tail_len,
        master_swa_tail_kv,
        master_end_cs,
        l0,
    );
    // Register Agent 0 itself as zero-diff Master replica
    collector.replicas.push(DiffAwareReplica {
        agent_id: 0,
        prefix_len: l0,
        segments: vec![MasterSegmentRef {
            block_id: master_id,
            logical_offset: l0 as u64,
            len: s_len,
            seg_id: 0,
        }],
        mode: DiffStorageMode::Replacement,
        diff_slices: Vec::new(),
        offsets: Vec::new(),
    });
    println!(
        "\n[Master] Prefilled P_0+S in {:.1} ms | canonical MasterBlock {:.1} MiB (hash {:016x})",
        master_s_prefill_ms,
        collector.store.total_master_bytes() as f64 / 1048576.0,
        master_id.0
    );

    // Accumulators for Exp 1 (across Agents 1..4) and Exp 2 (across Agents 1..4)
    let mut slot_acc = vec![SlotDevAccum::default(); N_SLOT];
    let mut check_corrs = Vec::new();
    let budgets: &[f64] = &[0.0, 0.02, 0.05, 0.10, 0.15, 0.25];
    // [budget_idx] -> (sum_mean_kl, max_worst_kl, sum_agree, total_steps)
    let mut exp2_oracle = vec![(0.0f64, 0.0f64, 0usize, 0usize); budgets.len()];
    let mut exp2_check = vec![(0.0f64, 0.0f64, 0usize, 0usize); budgets.len()];
    let mut per_agent_exp2: Vec<(usize, Vec<(f64, f64, usize)>, Vec<(f64, f64, usize)>)> = Vec::new();
    let mut gpu_extract_max_err = 0.0f32;
    let mut gpu_extract_packed_ok = true;

    // Results for Experiment 3 (held-out Agent 4)
    let mut exp3_rows: Vec<(String, f64, f64, usize, f64, f64)> = Vec::new();
    let mut exp3_anchor_info = String::new();
    let mut agent4_s_prefill_ms = 0.0f64;

    // -------------------------------------------------------------------------
    // Stream Agents 1..4 one at a time (keeps peak RAM strictly < 1.1 GiB!)
    // -------------------------------------------------------------------------
    for ag in 1..=4usize {
        let li = corpus.p_toks[ag].len();
        eng.reset();
        prefill_range(eng, &corpus.p_toks[ag], 0, false);
        let p_full_kv = extract_full_kv_range(eng, 0, li);
        let prefix_sig = KvCommAnchorPool::build_prefix_sig(eng, li, &lut);

        // Prefill S in <= EXT chunks and accumulate Exp 1 SWA deviations on the fly
        let t_ag_s = std::time::Instant::now();
        let mut off = 0usize;
        while off < s_len {
            let t = (s_len - off).min(chunk_sz);
            prefill_range(eng, &corpus.s_toks[off..off + t], li + off, false);
            let chunk_swa = extract_swa_kv_range(eng, li + off, t);
            for (si, &slot) in swa_slots.iter().enumerate() {
                for g in 0..NKV {
                    for j in 0..t {
                        let bkt = dist_bucket(off + j);
                        let a_k = ((si * 2 * NKV + g) * t + j) * HD;
                        let a_v = (((si * 2 + 1) * NKV + g) * t + j) * HD;
                        let m_k = (((si * 2 * NKV + g) * s_len) + off + j) * HD;
                        let m_v = ((((si * 2 + 1) * NKV + g) * s_len) + off + j) * HD;
                        let (mut dk2, mut mk2, mut dv2, mut mv2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                        for d in 0..HD {
                            let ka = lut[chunk_swa[a_k + d] as usize] as f64;
                            let km = lut[master_swa_fullspan[m_k + d] as usize] as f64;
                            let va = lut[chunk_swa[a_v + d] as usize] as f64;
                            let vm = lut[master_swa_fullspan[m_v + d] as usize] as f64;
                            dk2 += (ka - km) * (ka - km);
                            mk2 += km * km;
                            dv2 += (va - vm) * (va - vm);
                            mv2 += vm * vm;
                        }
                        slot_acc[slot].dk2[bkt] += dk2;
                        slot_acc[slot].mk2[bkt] += mk2;
                        slot_acc[slot].dv2[bkt] += dv2;
                        slot_acc[slot].mv2[bkt] += mv2;
                    }
                }
            }
            off += t;
        }
        if ag == 4 {
            agent4_s_prefill_ms = t_ag_s.elapsed().as_secs_f64() * 1e3;
        }

        // Capture Agent's Full KV and live SWA tail for S
        let agent_full_kv = extract_full_kv_range(eng, li, s_len);
        let agent_swa_tail_kv = extract_swa_kv_range(eng, li + s_len - swa_tail_len, swa_tail_len);

        // Run GPU `k_kv_diff_extract` on check layer (slot 5) to obtain real GPU check-layer deviation scores!
        let mut gpu_check_scores = vec![0.0f32; s_len];
        {
            let (_, _, akb, avb, acap) = eng_kv_slot_bases(eng, CHECK_LAYER_SLOT);
            let dcap_test = s_len.min(512);
            let diff_kv_tmp = eng.gpu.buffer(dcap_test * 2 * NKV * HD * 2);
            let diff_map_tmp = eng.gpu.buffer(s_len * 4);
            let ctr_tmp = eng.gpu.buffer(16);
            unsafe { std::ptr::write_bytes(ctr_tmp.contents().as_ptr() as *mut u8, 0, 16); }
            let scores_buf = eng.gpu.buffer(s_len * 4);
            let ext_args = KvDiffExtractArgs {
                n_tok: s_len as u32,
                agent_pos0: li as u32,
                master_pos0: 0,
                agent_cap: acap as u32,
                master_cap: s_len as u32,
                agent_is_ring: 0,
                master_is_ring: 0,
                slot_idx: 0,
                diff_cap: dcap_test as u32,
                diff_mode: 1,
                metric_mode: 2, // max(rel_l2_k, rel_l2_v)
                diff_map_base: 0,
                agent_k_base: akb,
                agent_v_base: avb,
                master_k_base: 0,
                master_v_base: (NKV * s_len * HD) as u64,
                thresh: 0.15,
                eps: 1e-6,
                write_scores: 1,
                pad0: 0,
            };
            let fkv_ref = eng.kv_full.borrow();
            gpu_kv_diff_extract(
                eng,
                &fkv_ref,
                &master_check_buf,
                &diff_kv_tmp,
                &diff_map_tmp,
                &ctr_tmp,
                &scores_buf,
                &ext_args,
            );
            drop(fkv_ref);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    scores_buf.contents().as_ptr() as *const f32,
                    gpu_check_scores.as_mut_ptr(),
                    s_len,
                );
            }
            let c0 = unsafe { *(ctr_tmp.contents().as_ptr() as *const u32) } as usize;
            let c1 = unsafe { *((ctr_tmp.contents().as_ptr() as *const u32).add(1)) } as usize;
            let expected_above = gpu_check_scores.iter().filter(|&&x| x > 0.15).count();
            if c0 != expected_above || c1 != expected_above.saturating_sub(dcap_test) {
                gpu_extract_packed_ok = false;
            }
        }

        // Accumulate Exp 1 Full-slot deviations + compute per-token oracle and check-layer scores
        let mut oracle_tok_scores = vec![0.0f32; s_len];
        let mut cpu_check_scores = vec![0.0f32; s_len];
        let mut deeper_tok_scores = vec![0.0f32; s_len];
        {
            let master = collector.store.get(master_id).unwrap();
            for (fi, &slot) in full_slots.iter().enumerate() {
                for t in 0..s_len {
                    let bkt = dist_bucket(t);
                    let (mut dk2, mut mk2, mut dv2, mut mv2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                    for g in 0..NKV {
                        let mk = master.full_vec(fi, 0, g, t);
                        let mv = master.full_vec(fi, 1, g, t);
                        let ak_b = (((fi * 2) * NKV + g) * s_len + t) * HD;
                        let av_b = (((fi * 2 + 1) * NKV + g) * s_len + t) * HD;
                        let ak = &agent_full_kv[ak_b..ak_b + HD];
                        let av = &agent_full_kv[av_b..av_b + HD];
                        for d in 0..HD {
                            let ka = lut[ak[d] as usize] as f64;
                            let km = lut[mk[d] as usize] as f64;
                            let va = lut[av[d] as usize] as f64;
                            let vm = lut[mv[d] as usize] as f64;
                            dk2 += (ka - km) * (ka - km);
                            mk2 += km * km;
                            dv2 += (va - vm) * (va - vm);
                            mv2 += vm * vm;
                        }
                    }
                    slot_acc[slot].dk2[bkt] += dk2;
                    slot_acc[slot].mk2[bkt] += mk2;
                    slot_acc[slot].dv2[bkt] += dv2;
                    slot_acc[slot].mv2[bkt] += mv2;

                    let rk = (dk2.sqrt() / (mk2.sqrt() + 1e-6)) as f32;
                    let rv = (dv2.sqrt() / (mv2.sqrt() + 1e-6)) as f32;
                    let dev_max = rk.max(rv);
                    oracle_tok_scores[t] += dev_max / nf as f32;
                    if fi == check_full_idx {
                        cpu_check_scores[t] = dev_max;
                    } else {
                        deeper_tok_scores[t] += dev_max / (nf - 1).max(1) as f32;
                    }
                }
            }
        }
        for t in 0..s_len {
            let err = (gpu_check_scores[t] - cpu_check_scores[t]).abs();
            if err > gpu_extract_max_err {
                gpu_extract_max_err = err;
            }
        }
        check_corrs.push(pearson_corr(&gpu_check_scores, &deeper_tok_scores));

        // Ground-Truth Q prefill + greedy decode for Agent `ag` (exact teacher-forced baseline!)
        prefill_range(eng, &corpus.q_toks, li + s_len, true);
        let mut gt_toks = Vec::with_capacity(decode_steps);
        let mut gt_logits = Vec::with_capacity(decode_steps);
        for step in 0..decode_steps {
            if step > 0 {
                let pos = li + s_len + q_len + step - 1;
                eng.set_tokens(pos, &[gt_toks[step - 1]]);
                let cb = eng.gpu.cmd();
                eng.encode_step(&cb, pos, true, false);
                cb.commit();
                cb.waitUntilCompleted();
            }
            let lg: Vec<f32> = eng.logits().iter().map(|&x| softcap(x, eng.softcap)).collect();
            let tok = argmax(&lg) as u32;
            gt_toks.push(tok);
            gt_logits.push(lg);
        }

        // Helper to mount `[P_i | Master S]` + `master.end_cs` cleanly onto `eng`
        let mount_master_for_agent = |eng: &Engine, master: &MasterBlock| {
            write_full_kv_range(eng, 0, li, &p_full_kv);
            write_full_kv_range(eng, li, s_len, &master.full_kv);
            write_swa_kv_range(eng, li + s_len - swa_tail_len, swa_tail_len, &master.swa_tail_kv);
            restore_cs(eng, &master.end_cs);
        };

        // ---- Experiment 2 sweep on Agent `ag`
        let mut ag_oracle_res = Vec::with_capacity(budgets.len());
        let mut ag_check_res = Vec::with_capacity(budgets.len());
        let mut naive_res = (0.0f64, 0.0f64, 0usize);

        for (bi, &p_frac) in budgets.iter().enumerate() {
            let master = collector.store.get(master_id).unwrap();
            if bi == 0 {
                // 0% diff = pure naive Master paste (identical for oracle & check-layer)
                mount_master_for_agent(eng, master);
                let res = eval_patched_decode(eng, li, s_len, &corpus.q_toks, &gt_toks, &gt_logits, decode_steps);
                naive_res = res;
                ag_oracle_res.push(res);
                ag_check_res.push(res);
            } else {
                // (a) Oracle sparse diff top-p%
                let pos_o = KvCollector::select_top_positions(&oracle_tok_scores, p_frac);
                mount_master_for_agent(eng, master);
                overlay_sparse_diffs_into_engine(eng, li, s_len, swa_tail_len, &pos_o, &agent_full_kv, &agent_swa_tail_kv);
                let res_o = eval_patched_decode(eng, li, s_len, &corpus.q_toks, &gt_toks, &gt_logits, decode_steps);
                ag_oracle_res.push(res_o);

                // (b) TokenDance early check-layer (`blk.5` / slot 5) top-p%
                let pos_c = KvCollector::select_top_positions(&gpu_check_scores, p_frac);
                mount_master_for_agent(eng, master);
                overlay_sparse_diffs_into_engine(eng, li, s_len, swa_tail_len, &pos_c, &agent_full_kv, &agent_swa_tail_kv);
                let res_c = eval_patched_decode(eng, li, s_len, &corpus.q_toks, &gt_toks, &gt_logits, decode_steps);
                ag_check_res.push(res_c);
            }
            exp2_oracle[bi].0 += ag_oracle_res[bi].0;
            exp2_oracle[bi].1 = exp2_oracle[bi].1.max(ag_oracle_res[bi].1);
            exp2_oracle[bi].2 += ag_oracle_res[bi].2;
            exp2_oracle[bi].3 += decode_steps;

            exp2_check[bi].0 += ag_check_res[bi].0;
            exp2_check[bi].1 = exp2_check[bi].1.max(ag_check_res[bi].1);
            exp2_check[bi].2 += ag_check_res[bi].2;
            exp2_check[bi].3 += decode_steps;
        }
        per_agent_exp2.push((ag, ag_oracle_res, ag_check_res));

        // ---- Experiment 3: Agents 1, 2, 3 populate KvCommAnchorPool; Agent 4 is strictly held out!
        if ag <= 3 {
            let pos10 = KvCollector::select_top_positions(&gpu_check_scores, 0.10);
            let _rep = collector.admit_agent_round(
                ag,
                master_id,
                li,
                prefix_sig,
                &pos10,
                DiffStorageMode::ResidualOverOffset,
                &agent_full_kv,
                &agent_swa_tail_kv,
                true, // record anchor for online pool
                &lut,
            );
        } else {
            // Free full-span SWA buffer before Exp 3 & 4
            master_swa_fullspan.clear();
            master_swa_fullspan.shrink_to_fit();

            let est = collector
                .anchor_pool
                .predict(master_id, &prefix_sig, li)
                .expect("anchor pool populated by agents 1..3");
            exp3_anchor_info = format!(
                "max_sim={:.4} (confident={}) | anchor weights: {}",
                est.max_sim,
                est.confident,
                est.raw_cos_sims
                    .iter()
                    .map(|(aid, cos, w)| format!("Agent{aid}(cos={cos:.3}, w={w:.2})"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            let master = collector.store.get(master_id).unwrap();

            // 1. Full Recompute (Ground Truth)
            exp3_rows.push(("1. Full Recompute (Ground Truth)".into(), 0.0, 0.0, decode_steps, 1.0, 1.0));
            // 2. Naive Direct Master Paste (0% diff, no offset)
            exp3_rows.push(("2. Naive Direct Master Paste (0%)".into(), naive_res.0, naive_res.1, naive_res.2, 0.0, 1.0));

            // 3. KVCOMM Anchor-Interpolated Offset (0% recompute of S! Applied on GPU via k_kvcomm_apply_offset!)
            let t_ofs = std::time::Instant::now();
            mount_master_for_agent(eng, master);
            apply_kvcomm_offsets_on_gpu(
                eng,
                li,
                s_len,
                swa_tail_len,
                &est.delta_k,
                &est.delta_v,
                &est.swa_delta_k,
                &est.swa_delta_v,
            );
            let ofs_apply_ms = t_ofs.elapsed().as_secs_f64() * 1e3;
            let res_kvcomm = eval_patched_decode(eng, li, s_len, &corpus.q_toks, &gt_toks, &gt_logits, decode_steps);
            exp3_rows.push((
                "3. KVCOMM Anchor Offset (0% recompute)".into(),
                res_kvcomm.0,
                res_kvcomm.1,
                res_kvcomm.2,
                0.0,
                agent4_s_prefill_ms / ofs_apply_ms.max(0.05),
            ));

            // 4 & 5. Hybrid KVCOMM Offset + TokenDance 5% and 10% Check-Layer Sparse Diff
            for &hyb_p in &[0.05f64, 0.10f64] {
                let pos_c = KvCollector::select_top_positions(&gpu_check_scores, hyb_p);
                mount_master_for_agent(eng, master);
                apply_kvcomm_offsets_on_gpu(
                    eng,
                    li,
                    s_len,
                    swa_tail_len,
                    &est.delta_k,
                    &est.delta_v,
                    &est.swa_delta_k,
                    &est.swa_delta_v,
                );
                overlay_sparse_diffs_into_engine(
                    eng,
                    li,
                    s_len,
                    swa_tail_len,
                    &pos_c,
                    &agent_full_kv,
                    &agent_swa_tail_kv,
                );
                let res_hyb = eval_patched_decode(eng, li, s_len, &corpus.q_toks, &gt_toks, &gt_logits, decode_steps);
                // Prefill work = 6/48 check layer + hyb_p of remaining 42/48 slots
                let frac_work = (6.0 / 48.0) + hyb_p * (42.0 / 48.0);
                exp3_rows.push((
                    format!("4. Hybrid KVCOMM + TokenDance {:>2.0}%", hyb_p * 100.0),
                    res_hyb.0,
                    res_hyb.1,
                    res_hyb.2,
                    hyb_p,
                    1.0 / frac_work,
                ));
            }

            // Diagnostic ceiling: Oracle Agent-4 bucket offset (0% diff) to show how close 3-anchor KVCOMM gets!
            let oracle_a4 = KvCommAnchorPool::observe_anchor(
                master_id,
                4,
                prefix_sig.clone(),
                li,
                master,
                &agent_full_kv,
                &agent_swa_tail_kv,
                &lut,
            );
            mount_master_for_agent(eng, master);
            apply_kvcomm_offsets_on_gpu(
                eng,
                li,
                s_len,
                swa_tail_len,
                &oracle_a4.layer_delta_k,
                &oracle_a4.layer_delta_v,
                &oracle_a4.swa_tail_delta_k,
                &oracle_a4.swa_tail_delta_v,
            );
            let res_ora_ofs = eval_patched_decode(eng, li, s_len, &corpus.q_toks, &gt_toks, &gt_logits, decode_steps);
            exp3_rows.push((
                "   [Diag] Oracle Self-Bucket Offset (0%)".into(),
                res_ora_ofs.0,
                res_ora_ofs.1,
                res_ora_ofs.2,
                0.0,
                1.0,
            ));

            // Admit Agent 4 into collector (10% hybrid replica) so collector reflects the full 5-agent round!
            let pos10 = KvCollector::select_top_positions(&gpu_check_scores, 0.10);
            let _ = collector.admit_agent_round(
                4,
                master_id,
                li,
                prefix_sig,
                &pos10,
                DiffStorageMode::ResidualOverOffset,
                &agent_full_kv,
                &agent_swa_tail_kv,
                true,
                &lut,
            );
        }
    }

    // -------------------------------------------------------------------------
    // Print Experiment 1 Results: Cross-Context KV Deviation Anatomy
    // -------------------------------------------------------------------------
    println!("\n================================================================================");
    println!("EXPERIMENT 1: Cross-Context KV Deviation Anatomy on Inkling 2 Mini (Agents 1..4 vs Master)");
    println!("================================================================================");
    println!(
        "  {:<22} | {:>8} | {:>8} | {:>8} | {:>9} | {:>9} | {:>9} | {:>9}",
        "Slot (Layer/Pass/Kind)", "Rel K", "Rel V", "Rel KV", "0..16", "16..128", "128..1024", "1024+"
    );
    println!("  {}", "-".repeat(98));

    // Print all 8 Full slots explicitly + aggregate 40 SWA slots (plus sample early/mid/late SWA slots)
    let mut swa_agg = SlotDevAccum::default();
    let mut full_agg = SlotDevAccum::default();
    for slot in 0..N_SLOT {
        let is_swa = eng.slot_kind[slot].0;
        let acc = &slot_acc[slot];
        let tgt = if is_swa { &mut swa_agg } else { &mut full_agg };
        for b in 0..N_DIST_BUCKETS {
            tgt.dk2[b] += acc.dk2[b];
            tgt.mk2[b] += acc.mk2[b];
            tgt.dv2[b] += acc.dv2[b];
            tgt.mv2[b] += acc.mv2[b];
        }
        // Print all 8 Full slots + representative SWA slots across prelude, loop passes 0/1/2, and epilogue
        let show_swa = matches!(slot, 0 | 2 | 4 | 6 | 12 | 18 | 22 | 24 | 36);
        if !is_swa || show_swa {
            println!(
                "  {:<22} | {:>7.4} | {:>7.4} | {:>7.4} | {:>8.4} | {:>8.4} | {:>9.4} | {:>9.4}",
                slot_label(slot, is_swa),
                acc.rel_k_all(),
                acc.rel_v_all(),
                acc.rel_kv_all(),
                acc.rel_kv(0),
                acc.rel_kv(1),
                acc.rel_kv(2),
                acc.rel_kv(3),
            );
        }
    }
    println!("  {}", "-".repeat(98));
    println!(
        "  {:<22} | {:>7.4} | {:>7.4} | {:>7.4} | {:>8.4} | {:>8.4} | {:>9.4} | {:>9.4}",
        "ALL 40 SWA SLOTS MEAN",
        swa_agg.rel_k_all(),
        swa_agg.rel_v_all(),
        swa_agg.rel_kv_all(),
        swa_agg.rel_kv(0),
        swa_agg.rel_kv(1),
        swa_agg.rel_kv(2),
        swa_agg.rel_kv(3),
    );
    println!(
        "  {:<22} | {:>7.4} | {:>7.4} | {:>7.4} | {:>8.4} | {:>8.4} | {:>9.4} | {:>9.4}",
        "ALL  8 FULL SLOTS MEAN",
        full_agg.rel_k_all(),
        full_agg.rel_v_all(),
        full_agg.rel_kv_all(),
        full_agg.rel_kv(0),
        full_agg.rel_kv(1),
        full_agg.rel_kv(2),
        full_agg.rel_kv(3),
    );
    let mean_corr = check_corrs.iter().sum::<f64>() / check_corrs.len().max(1) as f64;
    println!(
        "\n  Check-layer (`blk.5` / slot 5) vs deeper 7 Full slots per-token Pearson r = {:.4}",
        mean_corr
    );
    println!(
        "  GPU `k_kv_diff_extract` score parity vs CPU f64: max_err = {:.2e}, atomic_count_ok = {}",
        gpu_extract_max_err, gpu_extract_packed_ok
    );

    // -------------------------------------------------------------------------
    // Print Experiment 2 Results: TokenDance Sparse Diff & Check-Layer Sweep
    // -------------------------------------------------------------------------
    println!("\n================================================================================");
    println!("EXPERIMENT 2: TokenDance Diff-Aware Storage & Check-Layer (`blk.5`) Sweep (Agents 1..4)");
    println!("================================================================================");
    println!(
        "  {:>6} | {:>10} {:>10} {:>9} | {:>10} {:>10} {:>9} | {:>10} | {:>10}",
        "Diff %",
        "Oracle KL",
        "Worst KL",
        "Argmax",
        "Check5 KL",
        "Worst KL",
        "Argmax",
        "N=64@256K",
        "N=256@256K"
    );
    println!("  {}", "-".repeat(104));
    for (bi, &p_frac) in budgets.iter().enumerate() {
        let o = exp2_oracle[bi];
        let c = exp2_check[bi];
        let r64 = extrapolate_256k_report(64, 262_144, p_frac, p_frac, false);
        let r256 = extrapolate_256k_report(256, 262_144, p_frac, p_frac, false);
        println!(
            "  {:>5.0}% | {:>10.5} {:>10.5} {:>4}/{:<4} | {:>10.5} {:>10.5} {:>4}/{:<4} | {:>5.1}x ({:.0}G) | {:>5.1}x ({:.0}G)",
            p_frac * 100.0,
            o.0 / 4.0,
            o.1,
            o.2,
            o.3,
            c.0 / 4.0,
            c.1,
            c.2,
            c.3,
            r64.compression_ratio,
            r64.total_shared_bytes as f64 / (1u64 << 30) as f64,
            r256.compression_ratio,
            r256.total_shared_bytes as f64 / (1u64 << 30) as f64,
        );
    }
    let dense_64 = extrapolate_256k_report(64, 262_144, 1.0, 1.0, false).dense_bytes as f64 / (1u64 << 30) as f64;
    let dense_256 = extrapolate_256k_report(256, 262_144, 1.0, 1.0, false).dense_bytes as f64 / (1u64 << 30) as f64;
    println!(
        "  (Dense fp16 baseline @ 256K context: N=64 -> {:.1} GiB, N=256 -> {:.1} GiB; 1 Master = 8.16 GiB)",
        dense_64, dense_256
    );
    println!("  Per-agent Check-Layer (`blk.5`) mean KL breakdown [0% / 5% / 10% / 25%]:");
    for (ag, _, c_res) in &per_agent_exp2 {
        println!(
            "    {:<28}: 0%={:.5} ({}/{}) | 5%={:.5} ({}/{}) | 10%={:.5} ({}/{}) | 25%={:.5} ({}/{})",
            corpus.agent_names[*ag],
            c_res[0].0, c_res[0].2, decode_steps,
            c_res[2].0, c_res[2].2, decode_steps,
            c_res[3].0, c_res[3].2, decode_steps,
            c_res[5].0, c_res[5].2, decode_steps,
        );
    }

    // Print live 5-agent KvCollector accounting at 10% hybrid
    let live_rep = compression_accounting(&collector.store, &collector.replicas);
    println!(
        "  Live 5-agent KvCollector (10% check-layer diff): dense {:.1} MiB -> shared {:.1} MiB \
         (master {:.1} + diff {:.1} + ofs {:.2} + meta {:.2}) = {:.2}x compression",
        live_rep.dense_bytes as f64 / 1048576.0,
        live_rep.total_shared_bytes as f64 / 1048576.0,
        live_rep.master_bytes as f64 / 1048576.0,
        live_rep.sum_diff_bytes as f64 / 1048576.0,
        live_rep.offset_bytes as f64 / 1048576.0,
        live_rep.metadata_bytes as f64 / 1048576.0,
        live_rep.compression_ratio,
    );

    // -------------------------------------------------------------------------
    // Print Experiment 3 Results: KVCOMM Online Anchor-Pool Alignment (Held-Out Agent 4)
    // -------------------------------------------------------------------------
    println!("\n================================================================================");
    println!("EXPERIMENT 3: KVCOMM Online Anchor-Pool Offset Alignment (Trained Agents 1..3 -> Held-Out Agent 4)");
    println!("================================================================================");
    println!("  Anchor lookup for {}: {}", corpus.agent_names[4], exp3_anchor_info);
    println!(
        "  {:<40} | {:>9} | {:>9} | {:>7} | {:>9} | {:>10} | {:>8}",
        "Method (Held-Out Agent 4)", "Mean KL", "Worst KL", "Argmax", "N=64@256K", "N=256@256K", "Prefill"
    );
    println!("  {}", "-".repeat(108));
    for (name, mkl, wkl, agree, p_frac, speedup) in &exp3_rows {
        let has_ofs = name.contains("KVCOMM") || name.contains("Offset");
        let r64 = extrapolate_256k_report(64, 262_144, *p_frac, *p_frac, has_ofs);
        let r256 = extrapolate_256k_report(256, 262_144, *p_frac, *p_frac, has_ofs);
        let sp_str = if *speedup >= 50.0 {
            format!("{:>6.0}x*", speedup)
        } else if *speedup > 1.01 {
            format!("{:>7.2}x", speedup)
        } else {
            "   1.00x".to_string()
        };
        println!(
            "  {:<40} | {:>9.5} | {:>9.5} | {:>3}/{:<3} | {:>8.1}x | {:>9.1}x | {}",
            name,
            mkl,
            wkl,
            agree,
            decode_steps,
            if name.starts_with("1.") { 1.0 } else { r64.compression_ratio },
            if name.starts_with("1.") { 1.0 } else { r256.compression_ratio },
            sp_str,
        );
    }
    println!("  (* 0% recompute of S: only GPU `k_kvcomm_apply_offset` runs on staged KV; Agent 4 full S prefill = {:.1} ms)", agent4_s_prefill_ms);

    // -------------------------------------------------------------------------
    // Experiment 4: Fused GPU `k_attn_master_diff_b` Verification & B=64 Micro-Benchmark
    // -------------------------------------------------------------------------
    {
        let master = collector.store.get(master_id).unwrap();
        run_experiment_4_gpu_verify_and_bench(eng, master, &lut);
    }

    eng.release_batch();
    eng.tail.set(prev_tail);
    println!("\n[kv_share] TokenDance + KVCOMM empirical probe complete.");
}
