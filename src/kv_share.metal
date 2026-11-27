// ================================ TokenDance + KVCOMM Cross-Context KV Sharing Kernels ================================
// Included AFTER batch.metal (so NH=24, NKV=8, HD=128, NREL=16, EXT=1024, KV_BLK=32, KV_SLOT_ELEMS,
// QK_CLIP, REL_CLIP, kv_t, sigm, tg_sum, tg_max, paged_kv_base are already defined).
//
// Why Inkling 2 Mini is ideal for TokenDance (arXiv:2604.03143) + KVCOMM (NeurIPS 2025, arXiv:2510.12872):
//   - Zero RoPE: stored (K, V) vectors are position-independent; relative bias r . B[d] is added dynamically for d < EXT.
//   - 40 SWA slots (EXT=1024) + 8 Full slots (blk.5, blk.11 x3, blk.17 x3, blk.23).
//   - All pool / diff / offset indices use 64-bit ulong arithmetic (256K+ context ready).
//
// Kernels:
//   1. k_attn_master_diff_b  - Fused diff-aware Master-Mirror decode attention with register-level KVCOMM offset restore
//   2. k_kv_diff_extract     - Check-layer / diff-aware storage encoder with atomic sparse diff compaction
//   3. k_kvcomm_apply_offset - In-place GPU KVCOMM per-segment / per-position-bucket offset alignment

// Compact sparse diff layout: [n_slots_kind][diff_cap][2(K,V)][NKV][HD] in half
inline ulong diff_kv_base(uint dslot, uint diff_cap, uint slot_idx, uint g, uint is_v) {
    return (((ulong) slot_idx * (ulong) diff_cap + (ulong) dslot) * (2ul * NKV) + (ulong) (is_v * NKV + g)) * (ulong) HD;
}

// Per-segment / per-bucket KVCOMM offset layout: [n_slots_kind][B][max_segs][2(K,V)][NKV][HD] in half
inline ulong offset_kv_base(uint slot_idx, uint b, uint B, uint seg_id, uint max_segs, uint g, uint is_v) {
    return ((((ulong) slot_idx * (ulong) B + (ulong) b) * (ulong) max_segs + (ulong) seg_id) * (2ul * NKV) + (ulong) (is_v * NKV + g)) * (ulong) HD;
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. k_attn_master_diff_b: Fused TokenDance Master-Mirror + Sparse Diff + KVCOMM Segment Offset Decode Attention
// ---------------------------------------------------------------------------------------------------------------------
// Buffers:
//   1  = PB        (device const uchar *): qclip, relclip, relbias tables
//   2  = A         (device float *):       Q (half), R (f32), sink (f32), gate (half), out (half), part (f32)
//   3  = MASTER_KV (device const kv_t *):  canonical shared Master KV paged pool [n_slots_kind][master_pool_cap][2][NKV][32][128]
//   5  = a         (constant MasterDiffAttnArgs &)
//   7  = META      (device const uint *):  seq_pos[B], master_pt[B][pt_stride], optional seg_pt[B][pt_stride]
//   8  = DIFF_KV   (device const half *):  compact sparse diff vectors [n_slots_kind][diff_cap][2][NKV][128]
//   9  = DIFF_MAP  (device const uint *):  (slot, b, logical_pos) -> diff slot index, or 0xFFFFFFFFu if zero-diff
//   10 = OFFSET_KV (device const half *):  per-(slot, b, seg) KVCOMM additive offset [n_slots_kind][B][max_segs][2][NKV][128]
//
// diff_mode:
//   0 = replacement:          has_diff ? diff_kv                    : master_kv + seg_offset
//   1 = additive on master:   has_diff ? master_kv + diff_kv        : master_kv + seg_offset
//   2 = residual over offset: has_diff ? master_kv + seg_ofs + diff : master_kv + seg_offset
//
// Grid: (nsplit, NKV, B), Threadgroup: 128 threads (4 simdgroups x 32 lanes).
// When nsplit == 1, writes sigmoid(gate)-gated half output directly to A + out_off.
// When nsplit > 1, writes (M, L, O[128]) partials to A + part_off compatible with k_attn_combine_b.

struct MasterDiffAttnArgs {
    uint  B, master_pool_cap, slot_idx, is_swa, pt_stride, pt_mod;
    uint  seq_pos_off, master_pt_off, seg_pt_off, q_off, r_off, sink_off;
    uint  qclip_w, relclip_w, relbias_w, gate_off, out_off, part_off;
    uint  nsplit, diff_cap, diff_mode, max_segs;
    uint  diff_map_stride, diff_map_slot_stride, q_prepped, pad0;
    float log_alpha, log_floor;
};

kernel void k_attn_master_diff_b(
    device const uchar * PB        [[buffer(1)]],
    device float *       A         [[buffer(2)]],
    device const kv_t *  MASTER_KV [[buffer(3)]],
    constant MasterDiffAttnArgs & a [[buffer(5)]],
    device const uint *  META      [[buffer(7)]],
    device const half *  DIFF_KV   [[buffer(8)]],
    device const uint *  DIFF_MAP  [[buffer(9)]],
    device const half *  OFFSET_KV [[buffer(10)]],
    uint3  tgp  [[threadgroup_position_in_grid]],
    ushort tid  [[thread_index_in_threadgroup]],
    ushort sg   [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    threadgroup float4 qs4[3][HD / 4];
    threadgroup float  rs[3][NREL];
    threadgroup float  red[4][4];

    const uint sp = tgp.x, g = tgp.y, b = tgp.z;
    const uint pos = META[a.seq_pos_off + b];
    const uint nkeys = a.is_swa ? min(pos + 1u, (uint) EXT) : (pos + 1u);
    const uint first_pos = pos + 1u - nkeys;
    const uint kps = max(32u, ((nkeys + a.nsplit - 1u) / a.nsplit + 31u) / 32u * 32u);
    const uint j0 = sp * kps, j1 = min(nkeys, j0 + kps);

    if (j0 >= nkeys) {
        if (a.nsplit > 1u) {
            device float * P = A + a.part_off;
            for (int h = 0; h < 3; h++) {
                device float * Ph = P + ((ulong) (b * NH + g * 3 + h) * a.nsplit + sp) * (HD + 2);
                if (tid == 0) { Ph[0] = -INFINITY; Ph[1] = 0.0f; }
                Ph[2 + tid] = 0.0f;
            }
        }
        return;
    }

    const float tau = a.is_swa ? 1.0f : (1.0f + a.log_alpha * log(max(1.0f, float(pos + 1u) / a.log_floor)));
    device const float * qclip = (device const float *) (PB + a.qclip_w);

    float q[3];
    for (int h = 0; h < 3; h++) {
        const uint H = g * 3 + h;
        const half qh = ((device const half *) (A + a.q_off))[(ulong) b * (NH * HD) + H * HD + tid];
        q[h] = (a.q_prepped != 0u) ? float(qh) : (float(qh) * qclip[H * HD + tid]);
    }

    float r = 0.0f;
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16, H = g * 3 + h;
        const float rv = A[a.r_off + (ulong) b * (NH * NREL) + H * NREL + i];
        r = (a.q_prepped != 0u) ? rv : (rv * ((device const float *) (PB + a.relclip_w))[H * NREL + i]);
    }

    if (a.q_prepped == 0u) {
        float rn = r * r;
        rn += simd_shuffle_xor(rn, 1); rn += simd_shuffle_xor(rn, 2);
        rn += simd_shuffle_xor(rn, 4); rn += simd_shuffle_xor(rn, 8);
        const float s0 = simd_sum(q[0] * q[0]);
        const float s1 = simd_sum(q[1] * q[1]);
        const float s2 = simd_sum(q[2] * q[2]);
        if (lane == 0) { red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int h = 0; h < 3; h++) {
            const float tot = red[0][h] + red[1][h] + red[2][h] + red[3][h];
            const float f = clamp(QK_CLIP / (sqrt(tot) + 1e-6f), 0.0f, 1.0f) * tau * 0.08838834764831845f;
            ((threadgroup float *) qs4[h])[tid] = q[h] * f;
        }
        if (tid < 48) {
            rs[tid / 16][tid % 16] = r * clamp(REL_CLIP / (sqrt(rn) + 1e-6f), 0.0f, 1.0f) * tau;
        }
    } else {
        for (int h = 0; h < 3; h++) ((threadgroup float *) qs4[h])[tid] = q[h];
        if (tid < 48) rs[tid / 16][tid % 16] = r;
    }

    float sink[3];
    for (int h = 0; h < 3; h++) sink[h] = A[a.sink_off + (ulong) b * NH + g * 3 + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uint * master_pt = META + a.master_pt_off + (ulong) b * a.pt_stride;
    const bool has_seg_pt = (a.max_segs > 0u && a.seg_pt_off != 0xFFFFFFFFu);
    const ulong dm_base = (ulong) a.slot_idx * (ulong) a.diff_map_slot_stride + (ulong) b * (ulong) a.diff_map_stride;
    device const float * relb = (device const float *) (PB + a.relbias_w);

    const ushort kk = lane / 4, pp = lane % 4;
    float m[3] = {-INFINITY, -INFINITY, -INFINITY};
    float l[3] = {0.0f, 0.0f, 0.0f};
    float4 o[3] = {0.0f, 0.0f, 0.0f};
    float4 rl[3];
    for (int h = 0; h < 3; h++) rl[h] = *(threadgroup const float4 *) (&rs[h][pp * 4]);

    for (uint jb = j0 + sg * 8; jb < j1; jb += 32) {
        const uint j = jb + kk;
        const bool valid = j < j1;
        const uint p = first_pos + min(j, j1 - 1u);
        const uint bidx = a.is_swa ? ((p / KV_BLK) % a.pt_mod) : (p / KV_BLK);
        const uint boff = p % KV_BLK;
        const uint mpid = master_pt[bidx];
        const uint dslot = (a.diff_cap > 0u && a.diff_map_stride > 0u)
            ? DIFF_MAP[dm_base + (ulong) (a.is_swa ? (p % a.diff_map_stride) : p)]
            : 0xFFFFFFFFu;
        const bool has_diff = (dslot != 0xFFFFFFFFu);
        const uint seg_id = has_seg_pt ? META[a.seg_pt_off + (ulong) b * a.pt_stride + bidx] : 0xFFFFFFFFu;
        const bool has_ofs = (seg_id != 0xFFFFFFFFu && seg_id < a.max_segs);

        // ---- Fused on-the-fly key restore in registers: Q . k_eff
        float acc[3] = {0.0f, 0.0f, 0.0f};
        if (has_diff && a.diff_mode == 0u) {
            // Replacement diff: skip MASTER_KV read completely
            const ulong dkb = diff_kv_base(dslot, a.diff_cap, a.slot_idx, g, 0);
            device const half4 * dr = (device const half4 *) (DIFF_KV + dkb + pp * 32);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(dr[c]);
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
        } else {
            const ulong mkb = paged_kv_base(mpid, a.master_pool_cap, a.slot_idx, g, 0) + (ulong) boff * HD;
#ifdef KV8
            device const kv_t * mkrow = MASTER_KV + mkb;
            const float mks = kv_vscale(mkrow);
            device const packed_char4 * mkr = (device const packed_char4 *) (mkrow + pp * 32);
#else
            device const half4 * mkr = (device const half4 *) (MASTER_KV + mkb + pp * 32);
#endif
            if (has_diff) {
                const ulong dkb = diff_kv_base(dslot, a.diff_cap, a.slot_idx, g, 0);
                device const half4 * dr = (device const half4 *) (DIFF_KV + dkb + pp * 32);
                if (a.diff_mode == 2u && has_ofs) {
                    const ulong okb = offset_kv_base(a.slot_idx, b, a.B, seg_id, a.max_segs, g, 0);
                    device const half4 * or_ = (device const half4 *) (OFFSET_KV + okb + pp * 32);
                    for (int c = 0; c < 8; c++) {
#ifdef KV8
                        const float4 kf = float4(char4(mkr[c])) * mks + float4(or_[c]) + float4(dr[c]);
#else
                        const float4 kf = float4(mkr[c]) + float4(or_[c]) + float4(dr[c]);
#endif
                        for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
                    }
                } else {
                    for (int c = 0; c < 8; c++) {
#ifdef KV8
                        const float4 kf = float4(char4(mkr[c])) * mks + float4(dr[c]);
#else
                        const float4 kf = float4(mkr[c]) + float4(dr[c]);
#endif
                        for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
                    }
                }
            } else if (has_ofs) {
                const ulong okb = offset_kv_base(a.slot_idx, b, a.B, seg_id, a.max_segs, g, 0);
                device const half4 * or_ = (device const half4 *) (OFFSET_KV + okb + pp * 32);
                for (int c = 0; c < 8; c++) {
#ifdef KV8
                    const float4 kf = float4(char4(mkr[c])) * mks + float4(or_[c]);
#else
                    const float4 kf = float4(mkr[c]) + float4(or_[c]);
#endif
                    for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
                }
            } else {
                // Zero-diff, zero-offset fast path: single coalesced Master read
                for (int c = 0; c < 8; c++) {
#ifdef KV8
                    const float4 kf = float4(char4(mkr[c])) * mks;
#else
                    const float4 kf = float4(mkr[c]);
#endif
                    for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
                }
            }
        }

        // ---- Inkling dynamic relative bias (d < EXT, zero past EXT)
        const uint d = pos - p;
        if (d < EXT) {
            const float4 bb = *(device const float4 *) (relb + d * NREL + pp * 4);
            for (int h = 0; h < 3; h++) acc[h] += dot(rl[h], bb);
        }

        // ---- Online softmax update with per-head attention sink
        float pr[3], sc_[3];
        for (int h = 0; h < 3; h++) {
            float t = acc[h];
            t += simd_shuffle_xor(t, 1); t += simd_shuffle_xor(t, 2);
            t = valid ? (t - sink[h]) : -INFINITY;
            const float mx = max(m[h], simd_max(t));
            sc_[h] = (m[h] == -INFINITY) ? 0.0f : exp(m[h] - mx);
            pr[h] = valid ? exp(t - mx) : 0.0f;
            l[h] = l[h] * sc_[h] + simd_sum(pr[h]) * 0.25f;
            m[h] = mx;
            o[h] *= sc_[h];
        }

        // ---- Fused on-the-fly value restore in registers: o += pr * v_eff
        // Broadcast key metadata from lane (q2 * 4) so zero extra DIFF_MAP / META reads occur
        for (ushort q2 = 0; q2 < 8; q2++) {
            const uint jq = jb + q2;
            if (jq >= j1) break;
            const uint q_mpid  = simd_shuffle(mpid,   (ushort)(q2 * 4));
            const uint q_boff  = simd_shuffle(boff,   (ushort)(q2 * 4));
            const uint q_dslot = simd_shuffle(dslot,  (ushort)(q2 * 4));
            const uint q_seg   = simd_shuffle(seg_id, (ushort)(q2 * 4));
            const bool q_has_diff = (q_dslot != 0xFFFFFFFFu);
            const bool q_has_ofs  = (q_seg != 0xFFFFFFFFu && q_seg < a.max_segs);

            float4 v;
            if (q_has_diff && a.diff_mode == 0u) {
                const ulong dvb = diff_kv_base(q_dslot, a.diff_cap, a.slot_idx, g, 1) + lane * 4;
                v = float4(*(device const half4 *) (DIFF_KV + dvb));
            } else {
                const ulong mvb = paged_kv_base(q_mpid, a.master_pool_cap, a.slot_idx, g, 1) + (ulong) q_boff * HD;
#ifdef KV8
                const float mvs = kv_vscale(MASTER_KV + mvb);
                v = float4(char4(*(device const packed_char4 *) (MASTER_KV + mvb + lane * 4))) * mvs;
#else
                v = float4(*(device const half4 *) (MASTER_KV + mvb + lane * 4));
#endif
                if (q_has_diff) {
                    const ulong dvb = diff_kv_base(q_dslot, a.diff_cap, a.slot_idx, g, 1) + lane * 4;
                    v += float4(*(device const half4 *) (DIFF_KV + dvb));
                    if (a.diff_mode == 2u && q_has_ofs) {
                        const ulong ovb = offset_kv_base(a.slot_idx, b, a.B, q_seg, a.max_segs, g, 1) + lane * 4;
                        v += float4(*(device const half4 *) (OFFSET_KV + ovb));
                    }
                } else if (q_has_ofs) {
                    const ulong ovb = offset_kv_base(a.slot_idx, b, a.B, q_seg, a.max_segs, g, 1) + lane * 4;
                    v += float4(*(device const half4 *) (OFFSET_KV + ovb));
                }
            }
            for (int h = 0; h < 3; h++) o[h] += simd_shuffle(pr[h], (ushort)(q2 * 4)) * v;
        }
    }

    threadgroup float  ms_[4][3], ls_[4][3];
    threadgroup float4 os_[4][3][32];
    if (lane == 0) {
        for (int h = 0; h < 3; h++) { ms_[sg][h] = m[h]; ls_[sg][h] = l[h]; }
    }
    for (int h = 0; h < 3; h++) os_[sg][h][lane] = o[h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (a.nsplit == 1u) {
        device half * out = (device half *) (A + a.out_off);
        device const half * gate = (device const half *) (A + a.gate_off);
        for (int h = 0; h < 3; h++) {
            const uint H = g * 3 + h;
            float M = 0.0f;
            for (int s2 = 0; s2 < 4; s2++) M = max(M, ms_[s2][h]);
            float L = exp(-M), O = 0.0f;
            for (int s2 = 0; s2 < 4; s2++) {
                const float w = (ms_[s2][h] == -INFINITY) ? 0.0f : exp(ms_[s2][h] - M);
                L += w * ls_[s2][h];
                O += w * ((threadgroup const float *) os_[s2][h])[tid];
            }
            const ulong oi = (ulong) b * (NH * HD) + H * HD + tid;
            out[oi] = half((O / L) * sigm(float(gate[oi])));
        }
    } else {
        for (int h = 0; h < 3; h++) {
            float M = -INFINITY;
            for (int s2 = 0; s2 < 4; s2++) M = max(M, ms_[s2][h]);
            float L = 0.0f, O = 0.0f;
            for (int s2 = 0; s2 < 4; s2++) {
                const float w = (ms_[s2][h] == -INFINITY) ? 0.0f : exp(ms_[s2][h] - M);
                L += w * ls_[s2][h];
                O += w * ((threadgroup const float *) os_[s2][h])[tid];
            }
            device float * P = A + a.part_off + ((ulong) (b * NH + g * 3 + h) * a.nsplit + sp) * (HD + 2);
            if (tid == 0) { P[0] = M; P[1] = L; }
            P[2 + tid] = O;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. k_kv_diff_extract: Check-Layer / Diff-Aware Storage Encoder
// ---------------------------------------------------------------------------------------------------------------------
// Compares an agent's KV slice against the canonical Master KV slice across all NKV=8 heads per token,
// computes relative L2 / cosine deviation, writes optional per-token scores, and atomically compacts
// ONLY tokens exceeding `thresh` into DIFF_KV + DIFF_MAP.
//
// Buffers:
//   0 = AGENT_KV  (device const kv_t *)
//   1 = MASTER_KV (device const kv_t *)
//   2 = DIFF_KV   (device half *):       [n_slots_kind][diff_cap][2][NKV][128]
//   3 = DIFF_MAP  (device uint *):       writes DIFF_MAP[diff_map_base + t] = slot or 0xFFFFFFFFu
//   4 = COUNTER   (device atomic_uint *): [0] = allocated diff slots, [1] = overflow count
//   5 = a         (constant KvDiffExtractArgs &)
//   6 = SCORES    (device float *):      optional [n_tok] per-token deviation score
//
// Grid: (n_tok, 1, 1), Threadgroup: 128 threads (tid = head dim 0..127).

struct KvDiffExtractArgs {
    uint  n_tok;
    uint  agent_pos0;
    uint  master_pos0;
    uint  agent_cap;
    uint  master_cap;
    uint  agent_is_ring;
    uint  master_is_ring;
    uint  slot_idx;
    uint  diff_cap;
    uint  diff_mode;      // 0 = replacement (agent_kv), 1 = additive delta (agent_kv - master_kv)
    uint  metric_mode;    // 0 = rel_l2(K), 1 = cosine_dist(K), 2 = max(rel_l2(K), rel_l2(V))
    uint  diff_map_base;
    ulong agent_k_base;
    ulong agent_v_base;
    ulong master_k_base;
    ulong master_v_base;
    float thresh;
    float eps;
    uint  write_scores;
    uint  pad0;
};

kernel void k_kv_diff_extract(
    device const kv_t *  AGENT_KV  [[buffer(0)]],
    device const kv_t *  MASTER_KV [[buffer(1)]],
    device half *        DIFF_KV   [[buffer(2)]],
    device uint *        DIFF_MAP  [[buffer(3)]],
    device atomic_uint * COUNTER   [[buffer(4)]],
    constant KvDiffExtractArgs & a [[buffer(5)]],
    device float *       SCORES    [[buffer(6)]],
    uint   tgp  [[threadgroup_position_in_grid]],
    ushort tid  [[thread_index_in_threadgroup]],
    ushort sg   [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    threadgroup float red[4][6];
    threadgroup uint  claimed_slot;

    const uint t = tgp;
    if (t >= a.n_tok) return;

    const uint apos = a.agent_pos0 + t;
    const uint mpos = a.master_pos0 + t;
    const uint aidx = a.agent_is_ring ? (apos % a.agent_cap) : apos;
    const uint midx = a.master_is_ring ? (mpos % a.master_cap) : mpos;

    float ka[NKV], km[NKV], va[NKV], vm[NKV];
    float dk2 = 0.0f, mk2 = 0.0f, ak2 = 0.0f, dotk = 0.0f;
    float dv2 = 0.0f, mv2 = 0.0f;

    for (uint g = 0; g < NKV; g++) {
        const ulong ak_off = a.agent_k_base  + ((ulong) g * a.agent_cap  + aidx) * HD;
        const ulong av_off = a.agent_v_base  + ((ulong) g * a.agent_cap  + aidx) * HD;
        const ulong mk_off = a.master_k_base + ((ulong) g * a.master_cap + midx) * HD;
        const ulong mv_off = a.master_v_base + ((ulong) g * a.master_cap + midx) * HD;
#ifdef KV8
        const float ka_v = float(AGENT_KV[ak_off + tid])  * kv_vscale(AGENT_KV + ak_off);
        const float va_v = float(AGENT_KV[av_off + tid])  * kv_vscale(AGENT_KV + av_off);
        const float km_v = float(MASTER_KV[mk_off + tid]) * kv_vscale(MASTER_KV + mk_off);
        const float vm_v = float(MASTER_KV[mv_off + tid]) * kv_vscale(MASTER_KV + mv_off);
#else
        const float ka_v = float(AGENT_KV[ak_off + tid]);
        const float va_v = float(AGENT_KV[av_off + tid]);
        const float km_v = float(MASTER_KV[mk_off + tid]);
        const float vm_v = float(MASTER_KV[mv_off + tid]);
#endif
        ka[g] = ka_v; km[g] = km_v;
        va[g] = va_v; vm[g] = vm_v;

        const float dk = ka_v - km_v;
        const float dv = va_v - vm_v;
        dk2  += dk * dk;
        mk2  += km_v * km_v;
        ak2  += ka_v * ka_v;
        dotk += ka_v * km_v;
        dv2  += dv * dv;
        mv2  += vm_v * vm_v;
    }

    const float s0 = simd_sum(dk2);
    const float s1 = simd_sum(mk2);
    const float s2 = simd_sum(ak2);
    const float s3 = simd_sum(dotk);
    const float s4 = simd_sum(dv2);
    const float s5 = simd_sum(mv2);
    if (lane == 0) {
        red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2;
        red[sg][3] = s3; red[sg][4] = s4; red[sg][5] = s5;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        const float tot_dk2  = red[0][0] + red[1][0] + red[2][0] + red[3][0];
        const float tot_mk2  = red[0][1] + red[1][1] + red[2][1] + red[3][1];
        const float tot_ak2  = red[0][2] + red[1][2] + red[2][2] + red[3][2];
        const float tot_dotk = red[0][3] + red[1][3] + red[2][3] + red[3][3];
        const float tot_dv2  = red[0][4] + red[1][4] + red[2][4] + red[3][4];
        const float tot_mv2  = red[0][5] + red[1][5] + red[2][5] + red[3][5];

        const float rel_k = sqrt(max(0.0f, tot_dk2)) / (sqrt(max(0.0f, tot_mk2)) + a.eps);
        const float rel_v = sqrt(max(0.0f, tot_dv2)) / (sqrt(max(0.0f, tot_mv2)) + a.eps);
        const float cos_d = 1.0f - tot_dotk / (sqrt(max(0.0f, tot_ak2 * tot_mk2)) + a.eps);

        float dev = rel_k;
        if (a.metric_mode == 1u) dev = cos_d;
        else if (a.metric_mode == 2u) dev = max(rel_k, rel_v);

        if (a.write_scores != 0u) SCORES[t] = dev;

        if (dev > a.thresh && a.diff_cap > 0u) {
            const uint slot = atomic_fetch_add_explicit(&COUNTER[0], 1u, memory_order_relaxed);
            if (slot < a.diff_cap) {
                claimed_slot = slot;
                DIFF_MAP[(ulong) a.diff_map_base + t] = slot;
            } else {
                atomic_fetch_add_explicit(&COUNTER[1], 1u, memory_order_relaxed);
                claimed_slot = 0xFFFFFFFFu;
                DIFF_MAP[(ulong) a.diff_map_base + t] = 0xFFFFFFFFu;
            }
        } else {
            claimed_slot = 0xFFFFFFFFu;
            DIFF_MAP[(ulong) a.diff_map_base + t] = 0xFFFFFFFFu;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint dslot = claimed_slot;
    if (dslot != 0xFFFFFFFFu) {
        for (uint g = 0; g < NKV; g++) {
            const ulong dkb = diff_kv_base(dslot, a.diff_cap, a.slot_idx, g, 0);
            const ulong dvb = diff_kv_base(dslot, a.diff_cap, a.slot_idx, g, 1);
            const float out_k = (a.diff_mode == 0u) ? ka[g] : (ka[g] - km[g]);
            const float out_v = (a.diff_mode == 0u) ? va[g] : (va[g] - vm[g]);
            DIFF_KV[dkb + tid] = half(out_k);
            DIFF_KV[dvb + tid] = half(out_v);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. k_kvcomm_apply_offset: In-Place GPU KVCOMM Offset Alignment for Prefill / Staged Reuse
// ---------------------------------------------------------------------------------------------------------------------
// Applies an interpolated KVCOMM (delta_k, delta_v) offset broadcast per segment or per position bucket
// directly in-place onto a staged KV slice in Engine's [NKV][cap][HD] layout.
//
// Buffers:
//   3 = KV         (device kv_t *):       modified in-place
//   5 = a          (constant KvCommOffsetArgs &)
//   8 = OFFSET_KV  (device const half *): [n_buckets][2(K,V)][NKV][128]
//   9 = BUCKET_MAP (device const uint *): optional [n_tok] bucket index (0..n_buckets-1, or >= n_buckets to skip)
//
// Grid: (n_tok, NKV, 1), Threadgroup: 128 threads (tid = head dim 0..127).

struct KvCommOffsetArgs {
    uint  n_tok;
    uint  pos0;
    uint  cap;
    uint  is_ring;
    uint  n_buckets;
    uint  use_bucket_map;
    uint  pad0, pad1;
    ulong k_base;
    ulong v_base;
    ulong ofs_base;
    float scale;
    float pad2;
};

kernel void k_kvcomm_apply_offset(
    device kv_t *       KV         [[buffer(3)]],
    constant KvCommOffsetArgs & a  [[buffer(5)]],
    device const half * OFFSET_KV  [[buffer(8)]],
    device const uint * BUCKET_MAP [[buffer(9)]],
    uint2  tgp  [[threadgroup_position_in_grid]],
    ushort tid  [[thread_index_in_threadgroup]],
    ushort sg   [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
#ifdef KV8
    threadgroup float sh[32];
#endif
    const uint t = tgp.x, g = tgp.y;
    if (t >= a.n_tok || g >= NKV) return;

    uint bkt = 0u;
    if (a.n_buckets > 1u) {
        if (a.use_bucket_map != 0u) {
            bkt = BUCKET_MAP[t];
            if (bkt >= a.n_buckets) return;
        } else {
            bkt = min(a.n_buckets - 1u, (uint) (((ulong) t * (ulong) a.n_buckets) / max(1u, a.n_tok)));
        }
    }

    const uint pos = a.pos0 + t;
    const uint idx = a.is_ring ? (pos % a.cap) : pos;
    const ulong k_off = a.k_base + ((ulong) g * a.cap + idx) * HD;
    const ulong v_off = a.v_base + ((ulong) g * a.cap + idx) * HD;
    const ulong ok_off = a.ofs_base + ((ulong) bkt * (2ul * NKV) + g) * HD;
    const ulong ov_off = a.ofs_base + ((ulong) bkt * (2ul * NKV) + NKV + g) * HD;

    const float dk = float(OFFSET_KV[ok_off + tid]) * a.scale;
    const float dv = float(OFFSET_KV[ov_off + tid]) * a.scale;

#ifdef KV8
    device kv_t * krow = KV + k_off;
    const float k_new = float(krow[tid]) * kv_vscale(krow) + dk;
    const float kmx = tg_max(fabs(k_new), sh, sg, lane, 4);
    const uint ck = kv_code(kmx);
    krow[tid] = (kv_t) (int) kv_quant(k_new, kv_scale(ck), ck, tid);

    device kv_t * vrow = KV + v_off;
    const float v_new = float(vrow[tid]) * kv_vscale(vrow) + dv;
    const float vmx = tg_max(fabs(v_new), sh, sg, lane, 4);
    const uint cv = kv_code(vmx);
    vrow[tid] = (kv_t) (int) kv_quant(v_new, kv_scale(cv), cv, tid);
#else
    KV[k_off + tid] = half(float(KV[k_off + tid]) + dk);
    KV[v_off + tid] = half(float(KV[v_off + tid]) + dv);
#endif
}
