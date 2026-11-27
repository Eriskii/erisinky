// ================================ High-Throughput Batched MoE + FP16 CS + 256K WHT Sparse KV Kernels ================================
// Included AFTER batch.metal (and prefill.metal / kernels.metal) in Engine::load_with.
// Already in scope:
//   E=2304, NH=24, NKV=8, HD=128, NREL=16, EXT=1024, NEXP=128, TOPK=8, FF=1152,
//   QK_CLIP, REL_CLIP, HB_M=64, HB_N=64, HB_K=32, KV_BLK=32, SWA_RING_BLKS=33, CS_NCH, KV_SLOT_ELEMS,
//   block_q4_K, block_q6_K, block_q8_0, block_q5_0, scale_min_k4, WHead, wstage, deq16, SGOP,
//   sigm, bf16f, tg_sum, tg_max, cs_branch_base, paged_kv_base, float_to_ord,
//   MoeGemmHArgs, WoPostArgs, MoeCombPostArgs, PagedAttnArgs, CopyCsArgs, kv_t.

#define NTC_OPT 6
#define SPARSE_NEG_INF (-1e30f)
#define SPARSE_NEG_THR (-1e29f)

// ============================================================================================================
// PART 1: High-Throughput Batched MoE Small-Tile Kernels + 6-Class Router
// ============================================================================================================

// ---- 6-class MoE router with runtime-tunable cutoffs:
//   class 0: k_moe_sg8                     (full 128-token chunks + r > 64)
//   class 1: k_moe_sg4                     (c_sg2_max < r <= 64)
//   class 2: k_moe_sg2                     (c_sg1_max < r <= c_sg2_max)
//   class 3: k_moe_sg1                     (c_dir_max < r <= c_sg1_max)
//   class 4: k_moe_dir_gu_dyn4/8 + dn_dyn4/8 (2 <= r <= c_dir_max)
//   class 5: k_moe_dir_gu1 + dn1           (r == 1)
struct Route6Args {
    uint sel_off, T, perm_off, inv_off, aw_off, ind_off;
    uint gu_y[NTC_OPT];
    uint dn_y[NTC_OPT];
    uint tiles_off[NTC_OPT];
    uint c_dir_max;   // e.g. 4 (for dyn4) or 8 (for dyn8)
    uint c_sg1_max;   // e.g. 16
    uint c_sg2_max;   // e.g. 32
    uint pad0;
};

inline uint route6_class(uint r, uint c_dir_max, uint c_sg1_max, uint c_sg2_max) {
    if (r > 64u) return 0u;
    if (r == 1u) return 5u;
    if (r <= c_dir_max) return 4u;
    if (r <= c_sg1_max) return 3u;
    if (r <= c_sg2_max) return 2u;
    return 1u;
}

kernel void k_moe_route6(device float * A [[buffer(2)]], constant Route6Args & a [[buffer(5)]],
                         ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint cnt[NEXP];
    threadgroup uint off[NEXP], toff[NTC_OPT][NEXP];
    threadgroup atomic_uint fill[NEXP];
    if (tid < NEXP) {
        atomic_store_explicit(&cnt[tid], 0u, memory_order_relaxed);
        atomic_store_explicit(&fill[tid], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = a.T * TOPK;
    for (uint i = tid; i < n; i += 1024u) {
        const uint e = (uint) A[a.sel_off + (i / TOPK) * 16u + (i % TOPK)];
        atomic_fetch_add_explicit(&cnt[e], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint s = 0u, ts[NTC_OPT] = {0u, 0u, 0u, 0u, 0u, 0u};
        for (int e = 0; e < NEXP; e++) {
            const uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
            const uint r = c % 128u;
            off[e] = s; s += c;
            for (int k = 0; k < NTC_OPT; k++) toff[k][e] = ts[k];
            ts[0] += c / 128u;
            if (r > 0u) {
                const uint cls = route6_class(r, a.c_dir_max, a.c_sg1_max, a.c_sg2_max);
                ts[cls] += 1u;
            }
        }
        device uint * ind = (device uint *) (A + a.ind_off);
        for (int c = 0; c < NTC_OPT; c++) {
            ind[c * 4 + 0] = ts[c]; ind[c * 4 + 1] = a.gu_y[c]; ind[c * 4 + 2] = 1u; ind[c * 4 + 3] = ts[c];
            ind[(NTC_OPT + c) * 4 + 0] = ts[c]; ind[(NTC_OPT + c) * 4 + 1] = a.dn_y[c]; ind[(NTC_OPT + c) * 4 + 2] = 1u; ind[(NTC_OPT + c) * 4 + 3] = ts[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint * perm = (device uint *) (A + a.perm_off);
    device uint * inv = (device uint *) (A + a.inv_off);
    for (uint i = tid; i < n; i += 1024u) {
        const uint t = i / TOPK, k = i % TOPK;
        const uint e = (uint) A[a.sel_off + t * 16u + k];
        const uint slot = off[e] + atomic_fetch_add_explicit(&fill[e], 1u, memory_order_relaxed);
        perm[slot] = t;
        inv[i] = slot;
        A[a.aw_off + slot] = A[a.sel_off + t * 16u + TOPK + k];
    }
    if (tid < NEXP) {
        const uint c = atomic_load_explicit(&cnt[tid], memory_order_relaxed);
        const uint n128 = c / 128u, r = c % 128u;
        device uint * t0 = (device uint *) (A + a.tiles_off[0]) + toff[0][tid] * 4u;
        for (uint j = 0; j < n128; j++) {
            t0[j * 4 + 0] = tid; t0[j * 4 + 1] = off[tid] + j * 128u; t0[j * 4 + 2] = 128u; t0[j * 4 + 3] = 0u;
        }
        if (r > 0u) {
            const uint cls = route6_class(r, a.c_dir_max, a.c_sg1_max, a.c_sg2_max);
            device uint * tt = (cls == 0u)
                ? (t0 + n128 * 4u)
                : ((device uint *) (A + a.tiles_off[cls]) + toff[cls][tid] * 4u);
            tt[0] = tid; tt[1] = off[tid] + n128 * 128u; tt[2] = r; tt[3] = 0u;
        }
    }
}

// ---- Zero-spill dynamic direct quantized MoE gate_up + SiTU * aw -> half acth (cnt in 1..=MAX_NT).
// Grid: (tiles, FF / 4), TG = 128 (4 simdgroups, 1 gate + 1 up row per simdgroup).
// Superblock-per-simdgroup layout: all 32 lanes read one 256-weight Q4_K block ONCE into 16 float regs
// (wg_l, wg_h, wu_l, wu_h) and dot across all `cnt` tokens from L1 — ZERO TG staging, ZERO barriers!
template <int MAX_NT>
kernel void k_moe_dir_gu_dyn(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]],
                             constant MoeGemmHArgs & a [[buffer(5)]],
                             uint2 tgq [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    if (tgq.x >= ((device const uint *) (A + a.ntiles_off))[0]) return;
    device const uint * tl = (device const uint *) (A + a.tiles_off) + tgq.x * 4u;
    const uint e = tl[0], as0 = tl[1], cnt = min(tl[2], (uint) MAX_NT);
    const uint row = tgq.y * 4u + sg;
    if (row >= FF || cnt == 0u) return;

    const ulong ebase = (ulong(a.w_hi) << 32 | a.w_lo) + (ulong) e * a.expert_bytes;
    device const block_q4_K * wg = (device const block_q4_K *) (W + ebase + (ulong) row * a.row_bytes);
    device const block_q4_K * wu = (device const block_q4_K *) (W + ebase + (ulong) (FF + row) * a.row_bytes);
    device const half * X = (device const half *) (A + a.xh_off);

    const int j = lane / 8, l0 = 4 * (lane % 8);
    const int nb = (int) (a.K / 256u);
    float sumg[MAX_NT];
    float sumu[MAX_NT];
    for (int t = 0; t < MAX_NT; t++) { sumg[t] = 0.0f; sumu[t] = 0.0f; }

    for (int ib = 0; ib < nb; ib++) {
        // Load & dequantize 8 gate weights + 8 up weights ONCE per superblock
        device const block_q4_K * bg = wg + ib;
        const uint qg = *(device const uint *) (bg->qs + 4 * lane);
        float gsc0, gm0, gsc1, gm1;
        scale_min_k4(2 * j, bg->scales, gsc0, gm0);
        scale_min_k4(2 * j + 1, bg->scales, gsc1, gm1);
        const float gd = float(bg->d), gdm = float(bg->dmin);
        const float4 wgl = fma(float4(qg & 0xFu, (qg >> 8) & 0xFu, (qg >> 16) & 0xFu, (qg >> 24) & 0xFu), gd * gsc0, -(gdm * gm0));
        const float4 wgh = fma(float4((qg >> 4) & 0xFu, (qg >> 12) & 0xFu, (qg >> 20) & 0xFu, (qg >> 28) & 0xFu), gd * gsc1, -(gdm * gm1));

        device const block_q4_K * bu = wu + ib;
        const uint qu = *(device const uint *) (bu->qs + 4 * lane);
        float usc0, um0, usc1, um1;
        scale_min_k4(2 * j, bu->scales, usc0, um0);
        scale_min_k4(2 * j + 1, bu->scales, usc1, um1);
        const float ud = float(bu->d), udm = float(bu->dmin);
        const float4 wul = fma(float4(qu & 0xFu, (qu >> 8) & 0xFu, (qu >> 16) & 0xFu, (qu >> 24) & 0xFu), ud * usc0, -(udm * um0));
        const float4 wuh = fma(float4((qu >> 4) & 0xFu, (qu >> 12) & 0xFu, (qu >> 20) & 0xFu, (qu >> 28) & 0xFu), ud * usc1, -(udm * um1));

        const uint xoff = (uint) (ib * 256 + 64 * j + l0);
        for (uint t = 0; t < cnt; t++) {
            device const half * xt = X + (ulong) (as0 + t) * a.K + xoff;
            const float4 xl = float4(*(device const half4 *) xt);
            const float4 xh = float4(*(device const half4 *) (xt + 32));
            sumg[t] += dot(wgl, xl) + dot(wgh, xh);
            sumu[t] += dot(wul, xl) + dot(wuh, xh);
        }
    }

    device half * act = (device half *) (A + a.out_off);
    for (uint t = 0; t < cnt; t++) {
        const float gt = simd_sum(sumg[t]);
        const float up = simd_sum(sumu[t]);
        if (lane == 0) {
            const float su = a.lbeta > 0.0f ? a.lbeta * tanh(up / a.lbeta) : up;
            const float o = (A[a.aw_off + as0 + t] * a.beta) * tanh(gt / a.beta) / (1.0f + exp(-gt)) * su;
            act[(ulong) (as0 + t) * FF + row] = half(o);
        }
    }
}
template [[host_name("k_moe_dir_gu_dyn4")]] kernel void k_moe_dir_gu_dyn<4>(device const uchar *, device float *, constant MoeGemmHArgs &, uint2, ushort, ushort);
template [[host_name("k_moe_dir_gu_dyn8")]] kernel void k_moe_dir_gu_dyn<8>(device const uchar *, device float *, constant MoeGemmHArgs &, uint2, ushort, ushort);

// ---- Zero-spill dynamic direct quantized MoE down (Q8_0 & Q5_0) -> half yd (cnt in 1..=MAX_NT).
// Grid: (tiles, E / 4), TG = 128 (4 simdgroups, 1 row per simdgroup).
// Unpacks each 32-weight Q8_0 or Q5_0 block ONCE into 8 char4 regs (only 8 32-bit regs!) + fp32 scale `d`,
// then dots against all `cnt` tokens from L1 — ZERO TG staging, ZERO barriers, exact fp32 scaling!
template <int MAX_NT>
kernel void k_moe_dir_dn_dyn(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]],
                             constant MoeGemmHArgs & a [[buffer(5)]],
                             uint2 tgq [[threadgroup_position_in_grid]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    if (tgq.x >= ((device const uint *) (A + a.ntiles_off))[0]) return;
    device const uint * tl = (device const uint *) (A + a.tiles_off) + tgq.x * 4u;
    const uint e = tl[0], as0 = tl[1], cnt = min(tl[2], (uint) MAX_NT);
    const uint row = tgq.y * 4u + sg;
    if (row >= a.nrows || cnt == 0u) return;

    const ulong ebase = (ulong(a.w_hi) << 32 | a.w_lo) + (ulong) e * a.expert_bytes;
    device const uchar * rowp = W + ebase + (ulong) row * a.row_bytes;
    device const half * X = (device const half *) (A + a.xh_off);
    const int nb = (int) (a.K / 32u);

    float sumf[MAX_NT];
    for (int t = 0; t < MAX_NT; t++) sumf[t] = 0.0f;

    for (int ib = lane; ib < nb; ib += 32) {
        char4 qv[8];
        float d;
        if (a.wtype == 8u) {
            device const block_q8_0 * bq = (device const block_q8_0 *) rowp + ib;
            d = float(bq->d);
            device const packed_char4 * q = (device const packed_char4 *) bq->qs;
            for (int i = 0; i < 8; i++) qv[i] = char4(q[i]);
        } else {
            device const block_q5_0 * bq = (device const block_q5_0 *) rowp + ib;
            d = float(bq->d);
            const ushort4 qa = ushort4(((device const packed_ushort4 *) bq->qs)[0]);
            const ushort4 qb = ushort4(((device const packed_ushort4 *) bq->qs)[1]);
            const uint4 vv = uint4(uint(qa.x) | (uint(qa.y) << 16), uint(qa.z) | (uint(qa.w) << 16),
                                   uint(qb.x) | (uint(qb.y) << 16), uint(qb.z) | (uint(qb.w) << 16));
            const uint hb_lo = ((device const ushort *) bq->qh)[0];
            const uint hb_hi = ((device const ushort *) bq->qh)[1];
            for (int i = 0; i < 4; i++) {
                const uint n0 = vv[i] & 0x0F0F0F0Fu;
                const uint h0 = (((hb_lo >> (4 * i)) & 0xFu) * 0x00204081u) & 0x01010101u;
                qv[i] = as_type<char4>(n0 | ((0x01010101u - h0) * 0xF0u));
                const uint n1 = (vv[i] >> 4) & 0x0F0F0F0Fu;
                const uint h1 = (((hb_hi >> (4 * i)) & 0xFu) * 0x00204081u) & 0x01010101u;
                qv[4 + i] = as_type<char4>(n1 | ((0x01010101u - h1) * 0xF0u));
            }
        }
        for (uint t = 0; t < cnt; t++) {
            device const half4 * xp = (device const half4 *) (X + (ulong) (as0 + t) * a.K + (ulong) ib * 32ul);
            float s = 0.0f;
            for (int i = 0; i < 8; i++) s += dot(float4(xp[i]), float4(qv[i]));
            sumf[t] += s * d;
        }
    }

    device half * yd = (device half *) (A + a.out_off);
    for (uint t = 0; t < cnt; t++) {
        const float y = simd_sum(sumf[t]);
        if (lane == 0) yd[(ulong) (as0 + t) * a.ldo + row] = half(y);
    }
}
template [[host_name("k_moe_dir_dn_dyn4")]] kernel void k_moe_dir_dn_dyn<4>(device const uchar *, device float *, constant MoeGemmHArgs &, uint2, ushort, ushort);
template [[host_name("k_moe_dir_dn_dyn8")]] kernel void k_moe_dir_dn_dyn<8>(device const uchar *, device float *, constant MoeGemmHArgs &, uint2, ushort, ushort);

// ---- 2-simdgroup (64-thread) grouped MoE tensor GEMM for 17..32 tokens (cuts idle SGs in half vs k_moe_sg4!).
// Grid: (tiles, FF/32 for situ=1 or E/64 for situ=0), TG = 64.
// Each of the 64 threads stages 2 chunks of 16 halfs (double-buffered sa[2][64*32]).
kernel void k_moe_sg2(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]],
                      constant MoeGemmHArgs & a [[buffer(5)]],
                      uint2 tgq [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]],
                      ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[2][HB_M * HB_K];
    const uint cb = tgq.y;
    if (tgq.x >= ((device const uint *) (A + a.ntiles_off))[0]) return;
    device const uint * tl = (device const uint *) (A + a.tiles_off) + tgq.x * 4u;
    const uint e = tl[0], as0 = tl[1], cnt = tl[2], ts0 = sg * 16u;
    const bool on = ts0 < cnt;
    const int nt = on ? min(16, (int) (cnt - ts0)) : 0;
    const uint K = a.K;

    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(HB_K, HB_M));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(HB_K, HB_M));
    auto tX = tensor((device half *) (A + a.xh_off) + (ulong) (as0 + (on ? ts0 : 0u)) * K, dextents<int32_t, 2>((int) K, nt), array<int, 2>({1, (int) K}));
    SGOP(mm);
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    for (short i = 0; i < 32; ++i) cT[i] = 0.0f;

    const ulong ebase = (ulong(a.w_hi) << 32 | a.w_lo) + (ulong) e * a.expert_bytes;
    const uint widx0 = tid * 16u, rr0 = widx0 / HB_K, hp0 = (widx0 % HB_K) / 16u;
    const uint widx1 = (tid + 64u) * 16u, rr1 = widx1 / HB_K, hp1 = (widx1 % HB_K) / 16u;
    const uint wrow0 = a.situ ? ((rr0 < 32u) ? cb * 32u + rr0 : FF + cb * 32u + (rr0 - 32u)) : cb * HB_M + rr0;
    const uint wrow1 = a.situ ? ((rr1 < 32u) ? cb * 32u + rr1 : FF + cb * 32u + (rr1 - 32u)) : cb * HB_M + rr1;
    device const uchar * wr0 = W + ebase + (ulong) wrow0 * a.row_bytes;
    device const uchar * wr1 = W + ebase + (ulong) wrow1 * a.row_bytes;

    WHead wh0, wh1;
    half4 wv0[4], wv1[4];
    wstage(wr0, a.wtype, 0u, hp0, wh0, wv0);
    wstage(wr1, a.wtype, 0u, hp1, wh1, wv1);
    for (int i = 0; i < 4; i++) {
        ((threadgroup half4 *) (sa[0] + widx0))[i] = wv0[i];
        ((threadgroup half4 *) (sa[0] + widx1))[i] = wv1[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint buf = 0u;
    for (uint k0 = 0u; k0 < K; k0 += HB_K) {
        const bool more = (k0 + HB_K) < K;
        if (more) {
            wstage(wr0, a.wtype, k0 + HB_K, hp0, wh0, wv0);
            wstage(wr1, a.wtype, k0 + HB_K, hp1, wh1, wv1);
        }
        if (on) {
            auto sx = tX.slice(k0, 0);
            if (buf == 0u) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT);
        }
        if (more) {
            for (int i = 0; i < 4; i++) {
                ((threadgroup half4 *) (sa[buf ^ 1u] + widx0))[i] = wv0[i];
                ((threadgroup half4 *) (sa[buf ^ 1u] + widx1))[i] = wv1[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1u;
    }

    if (!on) return;
    if (!a.situ) {
        device half * yd = (device half *) (A + a.out_off);
        for (short q = 0; q < 8; ++q) {
            const auto ix = cT.get_multidimensional_index(4 * q);
            const uint r = ix[0], t = ts0 + ix[1];
            if (t < cnt) *(device half4 *) (yd + (ulong) (as0 + t) * a.ldo + cb * HB_M + r) =
                half4(float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]));
        }
        return;
    }
    device half * act = (device half *) (A + a.out_off);
    for (short q = 0; q < 4; ++q) {
        const auto ix = cT.get_multidimensional_index(4 * q);
        const uint r = ix[0], t = ts0 + ix[1];
        if (t < cnt) {
            const float4 gt = float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]);
            const float4 up = float4(cT[4 * q + 16], cT[4 * q + 17], cT[4 * q + 18], cT[4 * q + 19]);
            const float4 su = a.lbeta > 0.0f ? a.lbeta * tanh(up / a.lbeta) : up;
            const float4 o = (A[a.aw_off + as0 + t] * a.beta) * tanh(gt / a.beta) / (1.0f + exp(-gt)) * su;
            *(device half4 *) (act + (ulong) (as0 + t) * FF + cb * 32u + r) = half4(o);
        }
    }
}

// ---- 1-simdgroup (32-thread) grouped MoE tensor GEMM for 1..16 tokens!
// Grid: (tiles, FF/32 for situ=1 or E/64 for situ=0), TG = 32.
// Single-buffered 4 KiB sa[64*32] (2x TG occupancy) + ZERO idle simdgroups + fast simdgroup_barrier!
kernel void k_moe_sg1(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]],
                      constant MoeGemmHArgs & a [[buffer(5)]],
                      uint2 tgq [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup half sa[HB_M * HB_K];
    const uint cb = tgq.y;
    if (tgq.x >= ((device const uint *) (A + a.ntiles_off))[0]) return;
    device const uint * tl = (device const uint *) (A + a.tiles_off) + tgq.x * 4u;
    const uint e = tl[0], as0 = tl[1], cnt = tl[2];
    if (cnt == 0u) return;
    const int nt = min(16, (int) cnt);
    const uint K = a.K;

    auto tA = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(HB_K, HB_M));
    auto tX = tensor((device half *) (A + a.xh_off) + (ulong) as0 * K, dextents<int32_t, 2>((int) K, nt), array<int, 2>({1, (int) K}));
    SGOP(mm);
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA), float>();
    for (short i = 0; i < 32; ++i) cT[i] = 0.0f;

    const ulong ebase = (ulong(a.w_hi) << 32 | a.w_lo) + (ulong) e * a.expert_bytes;
    device const uchar * wr[4];
    uint widx[4], hp[4];
    WHead wh[4];
    for (int c = 0; c < 4; c++) {
        const uint idx = (tid + (uint) c * 32u) * 16u;
        const uint rr = idx / HB_K;
        widx[c] = idx;
        hp[c] = (idx % HB_K) / 16u;
        const uint wrow = a.situ ? ((rr < 32u) ? cb * 32u + rr : FF + cb * 32u + (rr - 32u)) : cb * HB_M + rr;
        wr[c] = W + ebase + (ulong) wrow * a.row_bytes;
    }

    for (uint k0 = 0u; k0 < K; k0 += HB_K) {
        for (int c = 0; c < 4; c++) {
            half4 wv[4];
            wstage(wr[c], a.wtype, k0, hp[c], wh[c], wv);
            for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa + widx[c]))[i] = wv[i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        auto sx = tX.slice(k0, 0);
        mm.run(sx, tA, cT);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!a.situ) {
        device half * yd = (device half *) (A + a.out_off);
        for (short q = 0; q < 8; ++q) {
            const auto ix = cT.get_multidimensional_index(4 * q);
            const uint r = ix[0], t = ix[1];
            if (t < cnt) *(device half4 *) (yd + (ulong) (as0 + t) * a.ldo + cb * HB_M + r) =
                half4(float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]));
        }
        return;
    }
    device half * act = (device half *) (A + a.out_off);
    for (short q = 0; q < 4; ++q) {
        const auto ix = cT.get_multidimensional_index(4 * q);
        const uint r = ix[0], t = ix[1];
        if (t < cnt) {
            const float4 gt = float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]);
            const float4 up = float4(cT[4 * q + 16], cT[4 * q + 17], cT[4 * q + 18], cT[4 * q + 19]);
            const float4 su = a.lbeta > 0.0f ? a.lbeta * tanh(up / a.lbeta) : up;
            const float4 o = (A[a.aw_off + as0 + t] * a.beta) * tanh(gt / a.beta) / (1.0f + exp(-gt)) * su;
            *(device half4 *) (act + (ulong) (as0 + t) * FF + cb * 32u + r) = half4(o);
        }
    }
}

// ============================================================================================================
// PART 2: FP16 CS_POOL Kernels (half conv state — 2x smaller CS & 2x less CS bandwidth!)
// ============================================================================================================

template <int NSG>
kernel void k_wo_post_sg_hcs(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]],
                             device float * A [[buffer(2)]], device half * CS [[buffer(4)]],
                             constant WoPostArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                             uint2 tgq [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[2][HB_M * HB_K];
    const uint r0 = tgq.y * HB_M, t0 = (tgq.x * NSG + sg) * 16u, K = a.K, B = a.B;
    const bool on = t0 < B;
    const int nt = on ? min(16, (int) (B - t0)) : 0;
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(HB_K, HB_M));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(HB_K, HB_M));
    auto tX = tensor((device half *) (A + a.xh_off) + (ulong) (on ? t0 : 0u) * K, dextents<int32_t, 2>((int) K, nt), array<int, 2>({1, (int) K}));
    SGOP(mm);
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    const bool wact = tid < 128;
    const uint widx = tid * 16u, hp = (widx % HB_K) / 16u;
    device const uchar * wr = W + (ulong(a.w_hi) << 32 | a.w_lo) + (ulong) min(r0 + widx / HB_K, a.nrows - 1u) * a.row_bytes;
    WHead wh; half4 wv[4];
    if (wact) { wstage(wr, a.wtype, 0u, hp, wh, wv); for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[0] + widx))[i] = wv[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0u;
    for (uint k0 = 0u; k0 < K; k0 += HB_K) {
        const bool more = (k0 + HB_K) < K;
        if (more && wact) wstage(wr, a.wtype, k0 + HB_K, hp, wh, wv);
        if (on) { auto sx = tX.slice(k0, 0); if (buf == 0u) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT); }
        if (more && wact) for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[buf ^ 1u] + widx))[i] = wv[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1u;
    }
    if (!on) return;
    const ulong branch_base = cs_branch_base(a.layer_slot, 2u, a.max_batch);
    device const uint * seq_cs = META + a.seq_cs_off;
    device const uint * seq_pos = META + a.seq_pos_off;
    device const float * cw = (device const float *) (PB + a.convw_t_off);
    for (short q = 0; q < 8; ++q) {
        const auto ix = cT.get_multidimensional_index(4 * q);
        const uint r = r0 + ix[0], t = t0 + ix[1];
        if (r < a.nrows && t < B) {
            const uint tap0 = seq_pos[t] % 3u;
            const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
            const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
            device half * hb = CS + branch_base + (ulong) seq_cs[t] * (3ul * E);
            const float4 h0 = float4(*(device const half4 *) (hb + (ulong) tap0 * E + r));
            const float4 h1 = float4(*(device const half4 *) (hb + (ulong) tap1 * E + r));
            const float4 h2 = float4(*(device const half4 *) (hb + (ulong) tap2 * E + r));
            const float4 w0 = *(device const float4 *) (cw + 0 * E + r);
            const float4 w1 = *(device const float4 *) (cw + 1 * E + r);
            const float4 w2 = *(device const float4 *) (cw + 2 * E + r);
            const float4 w3 = *(device const float4 *) (cw + 3 * E + r);
            const float4 y = float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]);
            *(device half4 *) (hb + (ulong) tap0 * E + r) = half4(y);
            float4 o = (y + w0 * h0 + w1 * h1 + w2 * h2 + w3 * y) * 0.4472135954999579f;
            if (a.has_res) o += *(device const float4 *) (A + a.res_off + (ulong) t * E + r);
            *(device float4 *) (A + a.out_off + (ulong) t * E + r) = o;
        }
    }
}
template [[host_name("k_wo_post_sg8_hcs")]] kernel void k_wo_post_sg_hcs<8>(device const uchar *, device const uchar *, device float *, device half *, constant WoPostArgs &, device const uint *, uint2, ushort, ushort);
template [[host_name("k_wo_post_sg4_hcs")]] kernel void k_wo_post_sg_hcs<4>(device const uchar *, device const uchar *, device float *, device half *, constant WoPostArgs &, device const uint *, uint2, ushort, ushort);

kernel void k_moe_comb_post_hcs(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                                device half * CS [[buffer(4)]], constant MoeCombPostArgs & a [[buffer(5)]],
                                device const uint * META [[buffer(7)]],
                                uint2 tgp [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]]) {
    const uint t = tgp.x, c = (tgp.y * 256u + tid) * 4u;
    if (c >= E) return;
    device const uint * inv = (device const uint *) (A + a.inv_off) + t * TOPK;
    device const half * yd = (device const half *) (A + a.yd_off);
    float4 y = 0.0f;
    for (int k = 0; k < TOPK; k++) y += float4(*(device const half4 *) (yd + (ulong) inv[k] * E + c));
    const uint tap0 = META[a.seq_pos_off + t] % 3u;
    const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
    const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
    device half * hb = CS + cs_branch_base(a.layer_slot, 3u, a.max_batch) + (ulong) META[a.seq_cs_off + t] * (3ul * E);
    device const float * cw = (device const float *) (PB + a.convw_t_off);
    const float4 h0 = float4(*(device const half4 *) (hb + (ulong) tap0 * E + c));
    const float4 h1 = float4(*(device const half4 *) (hb + (ulong) tap1 * E + c));
    const float4 h2 = float4(*(device const half4 *) (hb + (ulong) tap2 * E + c));
    const float4 w0 = *(device const float4 *) (cw + 0 * E + c);
    const float4 w1 = *(device const float4 *) (cw + 1 * E + c);
    const float4 w2 = *(device const float4 *) (cw + 2 * E + c);
    const float4 w3 = *(device const float4 *) (cw + 3 * E + c);
    *(device half4 *) (hb + (ulong) tap0 * E + c) = half4(y);
    const float4 o = (y + w0 * h0 + w1 * h1 + w2 * h2 + w3 * y) * 0.4472135954999579f + *(device const float4 *) (A + a.res_off + (ulong) t * E + c);
    *(device float4 *) (A + a.out_off + (ulong) t * E + c) = o;
}

// Streaming B=1 Q4_K row dot product against fp16 activations in L1 (exact match to dot_q4k).
template <int NR>
inline void dot_q4k_half(device const uchar * const * w, int nb, device const half * x, ushort lane, thread float * sumf) {
    const short ix = lane / 8, it = lane % 8, iq = it / 4, ir = it % 4;
    float yl[16], yh[16];
    for (int ib = ix; ib < nb; ib += 4) {
        device const half * y4 = x + ib * 256 + 64 * iq + 8 * ir;
        float4 sumy = 0.0f;
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = float(y4[i +   0]); sumy[0] += yl[i + 0];
            yl[i + 8] = float(y4[i +  32]); sumy[1] += yl[i + 8];
            yh[i + 0] = float(y4[i + 128]); sumy[2] += yh[i + 0];
            yh[i + 8] = float(y4[i + 160]); sumy[3] += yh[i + 8];
        }
        for (short row = 0; row < NR; ++row) {
            device const block_q4_K * b = (device const block_q4_K *) w[row] + ib;
            device const ushort * sc = (device const ushort *) b->scales + iq;
            device const ushort * q1 = (device const ushort *) b->qs + 16 * iq + 4 * ir;
            device const ushort * q2 = q1 + 32;
            ushort sc16[4];
            thread const uchar * sc8 = (thread const uchar *) sc16;
            sc16[0] = sc[0] & 0x3f3f;
            sc16[1] = sc[2] & 0x3f3f;
            sc16[2] = ((sc[4] >> 0) & 0x0f0f) | ((sc[0] & 0xc0c0) >> 2);
            sc16[3] = ((sc[4] >> 4) & 0x0f0f) | ((sc[2] & 0xc0c0) >> 2);
            float4 acc1 = 0.0f, acc2 = 0.0f;
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }
            sumf[row] += float(b->d) * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                                        (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                                        (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                                        (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                         float(b->dmin) * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }
    }
}

// Streaming B=1 Q4_K wo matvec reading half agh directly + fused FP16 CS branch-2 ring post-conv + optional residual.
// Grid: (ceil(nrows / 8), 1, 1), TG = 128 (4 simdgroups x 2 rows/sg = 8 rows/TG).
kernel void k_wo_mv_hcs(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]],
                        device float * A [[buffer(2)]], device half * CS [[buffer(4)]],
                        constant WoPostArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                        uint tg [[threadgroup_position_in_grid]],
                        ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * MV_NSG + sg) * MV_NR;
    if (row0 >= a.nrows) return;
    const ulong wbase = ulong(a.w_hi) << 32 | a.w_lo;
    device const uchar * wp[MV_NR];
    for (int rr = 0; rr < MV_NR; rr++) wp[rr] = W + wbase + (ulong) min(row0 + (uint) rr, a.nrows - 1u) * a.row_bytes;
    float sumf[MV_NR] = {0.0f, 0.0f};
    dot_q4k_half<MV_NR>(wp, (int) (a.K / 256u), (device const half *) (A + a.xh_off), lane, sumf);
    const float y0 = simd_sum(sumf[0]);
    const float y1 = simd_sum(sumf[1]);
    if (lane == 0) {
        const uint tap0 = META[a.seq_pos_off] % 3u;
        const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
        const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
        device half * hb = CS + cs_branch_base(a.layer_slot, 2u, a.max_batch) + (ulong) META[a.seq_cs_off] * (3ul * E);
        device const float * cw = (device const float *) (PB + a.convw_t_off);
        const float ys[2] = { y0, y1 };
        const half2 h0_2 = *(device const half2 *) (hb + (ulong) tap0 * E + row0);
        const half2 h1_2 = *(device const half2 *) (hb + (ulong) tap1 * E + row0);
        const half2 h2_2 = *(device const half2 *) (hb + (ulong) tap2 * E + row0);
        *(device half2 *) (hb + (ulong) tap0 * E + row0) = half2(half(y0), half(y1));
        const float h0[2] = { float(h0_2.x), float(h0_2.y) };
        const float h1[2] = { float(h1_2.x), float(h1_2.y) };
        const float h2[2] = { float(h2_2.x), float(h2_2.y) };
        for (int rr = 0; rr < 2; rr++) {
            const uint r = row0 + (uint) rr;
            if (r < a.nrows) {
                const float conv = cw[0 * E + r] * h0[rr] + cw[1 * E + r] * h1[rr] + cw[2 * E + r] * h2[rr] + cw[3 * E + r] * ys[rr];
                float o = (ys[rr] + conv) * 0.4472135954999579f;
                if (a.has_res) o += A[a.res_off + r];
                A[a.out_off + r] = o;
            }
        }
    }
}

// Streaming B=1 MoE down projection (all TOPK experts + weighted sum) + fused FP16 CS branch-3 ring post-conv + residual.
// Exact clone of k_moe_down with FP16 CS_POOL ring update. Grid: (E / 4, 1, 1), TG = 64 (2 simdgroups x 2 rows/sg).
struct MoeDownHcsArgs {
    uint sel_off, act_off, dn_lo, dn_hi;
    uint dn_expert_bytes, dn_row_bytes, dn_type, out_off;
    uint res_off, convw_t_off, layer_slot, max_batch;
    uint seq_cs_off, seq_pos_off, pad0, pad1;
};

kernel void k_moe_down_hcs(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]],
                           device float * A [[buffer(2)]], device half * CS [[buffer(4)]],
                           constant MoeDownHcsArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                           uint tg [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    const uint row0 = (tg * DN_NSG + sg) * DN_NR;
    const ulong base = ulong(a.dn_hi) << 32 | a.dn_lo;
    float acc[DN_NR] = {0.0f, 0.0f};
    for (int s = 0; s < 9; s++) {
        const int f = lane + 32 * s, k = f / 36, b = f % 36;
        device const float4 * x4 = (device const float4 *) (A + a.act_off + k * FF + b * 32);
        float4 xv[8];
        for (int i = 0; i < 8; i++) xv[i] = x4[i];
        const int e = (int) A[a.sel_off + k];
        down_rows<DN_NR>(W + base + (ulong) e * a.dn_expert_bytes, row0, a.dn_row_bytes, a.dn_type, b, xv, A[a.sel_off + TOPK + k], acc);
    }
    const float y0 = simd_sum(acc[0]);
    const float y1 = simd_sum(acc[1]);
    if (lane == 0) {
        const uint tap0 = META[a.seq_pos_off] % 3u;
        const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
        const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
        device half * hb = CS + cs_branch_base(a.layer_slot, 3u, a.max_batch) + (ulong) META[a.seq_cs_off] * (3ul * E);
        device const float * cw = (device const float *) (PB + a.convw_t_off);
        const float ys[2] = { y0, y1 };
        const half2 h0_2 = *(device const half2 *) (hb + (ulong) tap0 * E + row0);
        const half2 h1_2 = *(device const half2 *) (hb + (ulong) tap1 * E + row0);
        const half2 h2_2 = *(device const half2 *) (hb + (ulong) tap2 * E + row0);
        *(device half2 *) (hb + (ulong) tap0 * E + row0) = half2(half(y0), half(y1));
        const float h0[2] = { float(h0_2.x), float(h0_2.y) };
        const float h1[2] = { float(h1_2.x), float(h1_2.y) };
        const float h2[2] = { float(h2_2.x), float(h2_2.y) };
        for (int rr = 0; rr < 2; rr++) {
            const uint r = row0 + (uint) rr;
            const float conv = cw[0 * E + r] * h0[rr] + cw[1 * E + r] * h1[rr] + cw[2 * E + r] * h2[rr] + cw[3 * E + r] * ys[rr];
            A[a.out_off + r] = A[a.res_off + r] + (ys[rr] + conv) * 0.4472135954999579f;
        }
    }
}

// Fast GPU zero of specific sequence slots in FP16 slot-major CS_POOL. Grid: (ceil(CS_NCH / 256), n, 48), TG = 256.
kernel void k_zero_cs_hcs(device half * POOL [[buffer(4)]], constant CopyCsArgs & a [[buffer(5)]],
                          device const uint * META [[buffer(7)]],
                          uint3 tgp [[threadgroup_position_in_grid]],
                          ushort tid [[thread_index_in_threadgroup]]) {
    const uint ch = tgp.x * 256u + tid;
    if (ch >= CS_NCH) return;
    const uint seq_cs = META[a.dst_off + tgp.y];
    const uint slot = tgp.z;
    uint branch, lc, dim;
    if (ch < NKV * HD) { branch = 0u; lc = ch; dim = NKV * HD; }
    else if (ch < 2u * NKV * HD) { branch = 1u; lc = ch - NKV * HD; dim = NKV * HD; }
    else if (ch < 2u * NKV * HD + E) { branch = 2u; lc = ch - 2u * NKV * HD; dim = E; }
    else { branch = 3u; lc = ch - (2u * NKV * HD + E); dim = E; }
    device half * dst = POOL + cs_branch_base(slot, branch, a.max_batch) + (ulong) seq_cs * (3ul * dim);
    dst[0ul * dim + lc] = half(0.0h);
    dst[1ul * dim + lc] = half(0.0h);
    dst[2ul * dim + lc] = half(0.0h);
}

// Broadcast Engine's AoS fp32 conv state [48][NCH][3] into FP16 slot-major SoA ring CS_POOL.
kernel void k_broadcast_cs_hcs(device const float * SRC [[buffer(4)]], device half * POOL [[buffer(8)]],
                               constant CopyCsArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                               uint3 tgp [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]]) {
    const uint ch = tgp.x * 256u + tid;
    if (ch >= CS_NCH) return;
    const uint seq_cs = META[a.dst_off + tgp.y];
    const uint slot = tgp.z;
    device const float * src = SRC + ((ulong) slot * CS_NCH + ch) * 3ul;
    const half h0 = half(src[0]), h1 = half(src[1]), h2 = half(src[2]);
    const uint tap0 = a.prefix_len % 3u;
    const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
    const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
    uint branch, lc, dim;
    if (ch < NKV * HD) { branch = 0u; lc = ch; dim = NKV * HD; }
    else if (ch < 2u * NKV * HD) { branch = 1u; lc = ch - NKV * HD; dim = NKV * HD; }
    else if (ch < 2u * NKV * HD + E) { branch = 2u; lc = ch - 2u * NKV * HD; dim = E; }
    else { branch = 3u; lc = ch - (2u * NKV * HD + E); dim = E; }
    device half * dst = POOL + cs_branch_base(slot, branch, a.max_batch) + (ulong) seq_cs * (3ul * dim);
    dst[(ulong) tap0 * dim + lc] = h0;
    dst[(ulong) tap1 * dim + lc] = h1;
    dst[(ulong) tap2 * dim + lc] = h2;
}

// Single-pass fused paged decode attention (exact mirror of k_attn_paged_b, using FP16 CS_POOL).
kernel void k_attn_paged_b_hcs(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                               device kv_t * KV [[buffer(3)]], device half * CS [[buffer(4)]],
                               constant PagedAttnArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                               uint3 tgp [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 qs4[3][HD / 4];
    threadgroup float rs[3][NREL];
    threadgroup float kc[HD], vc[HD];
    threadgroup float red[4][8];
    const uint sp = tgp.x, g = tgp.y, b = tgp.z;
    const uint pos = META[a.seq_pos_off + b];
    const uint suf = META[a.seq_suf_off + b];
    const uint nkeys = a.is_swa ? min(pos + 1u, (uint) EXT) : (pos + 1u);
    const uint first_pos = pos + 1u - nkeys;
    const uint kps = max(32u, ((nkeys + a.nsplit - 1u) / a.nsplit + 31u) / 32u * 32u);
    const uint j0 = sp * kps, j1 = min(nkeys, j0 + kps);
    const uint last_active_sp = (nkeys - 1u) / kps;
    const bool owns_kv = (sp == last_active_sp);
    device const uint * pt = META + a.pt_off + (ulong) b * a.pt_stride;

    if (j0 >= nkeys) {
        device float * P = A + a.part_off;
        for (int h = 0; h < 3; h++) {
            device float * Ph = P + ((ulong) (b * NH + g * 3 + h) * a.nsplit + sp) * (HD + 2);
            if (tid == 0) { Ph[0] = -INFINITY; Ph[1] = 0.0f; }
            Ph[2 + tid] = 0.0f;
        }
        return;
    }

    const float tau = a.is_swa ? 1.0f : (1.0f + a.log_alpha * log(max(1.0f, float(pos + 1u) / a.log_floor)));
    device const float * qclip = (device const float *) (PB + a.qclip_w);
    float q[3];
    for (int h = 0; h < 3; h++) {
        const uint H = g * 3 + h;
        const half qh = ((device const half *) (A + a.q_off))[(ulong) b * (NH * HD) + H * HD + tid];
        q[h] = float(qh) * qclip[H * HD + tid];
    }
    float kx = 0.0f, vx = 0.0f;
    if (owns_kv) {
        const uint seq_cs = META[a.seq_cs_off + b];
        const uint tap0 = pos % 3u;
        const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
        const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
        const uint c = g * HD + tid;
        const uint dim = NKV * HD;
        const float scl = a.kdyn_off >= 0 ? (1.0f / 3.0f) : 0.4472135954999579f;
        for (int which = 0; which < 2; which++) {
            device half * hb = CS + cs_branch_base(a.layer_slot, (uint) which, a.max_batch) + (ulong) seq_cs * (3ul * dim);
            device const float * w = (device const float *) (PB + (which == 0 ? a.kconv_t_w : a.vconv_t_w));
            const float x = A[(which == 0 ? a.k_off : a.v_off) + (ulong) b * dim + c];
            const float h0 = float(hb[(ulong) tap0 * dim + c]);
            const float h1 = float(hb[(ulong) tap1 * dim + c]);
            const float h2 = float(hb[(ulong) tap2 * dim + c]);
            float acc = w[0 * dim + c] * h0 + w[1 * dim + c] * h1 + w[2 * dim + c] * h2 + w[3 * dim + c] * x;
            const int dyn = which == 0 ? a.kdyn_off : a.vdyn_off;
            if (dyn >= 0) { device const float * tp = A + dyn + (ulong) b * 32 + g * 4; acc += tp[0] * x + tp[1] * h2 + tp[2] * h1 + tp[3] * h0; }
            hb[(ulong) tap0 * dim + c] = half(x);
            if (which == 0) kx = (x + acc) * scl; else vx = (x + acc) * scl;
        }
    }
    float r = 0.0f;
    if (tid < 48) {
        const uint h = tid / 16u, i = tid % 16u, H = g * 3 + h;
        r = A[a.r_off + (ulong) b * (NH * NREL) + H * NREL + i] * ((device const float *) (PB + a.relclip_w))[H * NREL + i];
    }
    float rn = r * r;
    rn += simd_shuffle_xor(rn, 1); rn += simd_shuffle_xor(rn, 2); rn += simd_shuffle_xor(rn, 4); rn += simd_shuffle_xor(rn, 8);
    {
        const float s0 = simd_sum(q[0] * q[0]), s1 = simd_sum(q[1] * q[1]), s2 = simd_sum(q[2] * q[2]);
        const float s3 = simd_sum(kx * kx), s4 = simd_sum(vx * vx);
        const float s5 = simd_max(fabs(kx)), s6 = simd_max(fabs(vx));
        if (lane == 0) { red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; red[sg][3] = s3; red[sg][4] = s4; red[sg][5] = s5; red[sg][6] = s6; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot[5];
    for (int i = 0; i < 5; i++) tot[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];
    for (int h = 0; h < 3; h++) {
        const float f = clamp(QK_CLIP / (sqrt(tot[h]) + 1e-6f), 0.0f, 1.0f) * tau * 0.08838834764831845f;
        ((threadgroup float *) qs4[h])[tid] = q[h] * f;
    }
    if (tid < 48) rs[tid / 16u][tid % 16u] = r * clamp(REL_CLIP / (sqrt(rn) + 1e-6f), 0.0f, 1.0f) * tau;
    if (owns_kv) {
        const float kf = clamp(QK_CLIP / (sqrt(tot[3]) + 1e-6f), 0.0f, 1.0f), vf = rsqrt(tot[4] / HD + 1e-6f);
        const float k = kx * kf, v = vx * vf;
        const uint bidx = a.is_swa ? ((suf / KV_BLK) % a.pt_mod) : (suf / KV_BLK);
        const uint boff = suf % KV_BLK;
        const uint pid = pt[bidx];
        const ulong kb = paged_kv_base(pid, a.pool_cap, a.slot_idx, g, 0) + (ulong) boff * HD;
        const ulong vb = paged_kv_base(pid, a.pool_cap, a.slot_idx, g, 1) + (ulong) boff * HD;
#ifdef KV8
        const uint ck = kv_code(max(max(red[0][5], red[1][5]), max(red[2][5], red[3][5])) * kf);
        const uint cv = kv_code(max(max(red[0][6], red[1][6]), max(red[2][6], red[3][6])) * vf);
        const float sk = kv_scale(ck), sv = kv_scale(cv);
        const float qk = kv_quant(k, sk, ck, tid), qv = kv_quant(v, sv, cv, tid);
        kc[tid] = qk * sk; vc[tid] = qv * sv;
        KV[kb + tid] = (kv_t) (int) qk;
        KV[vb + tid] = (kv_t) (int) qv;
#else
        kc[tid] = k; vc[tid] = v;
        KV[kb + tid] = half(k);
        KV[vb + tid] = half(v);
#endif
    }
    float sink[3];
    for (int h = 0; h < 3; h++) sink[h] = A[a.sink_off + (ulong) b * NH + g * 3 + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float * relb = (device const float *) (PB + a.relbias_w);
    const ushort kk = lane / 4, pp = lane % 4;
    float m[3] = {-INFINITY, -INFINITY, -INFINITY}, l[3] = {0, 0, 0};
    float4 o[3] = {0, 0, 0};
    float4 rl[3];
    for (int h = 0; h < 3; h++) rl[h] = *(threadgroup const float4 *) (&rs[h][pp * 4]);
    for (uint jb = j0 + sg * 8; jb < j1; jb += 32) {
        const uint j = jb + kk;
        const bool valid = j < j1;
        const uint p = first_pos + min(j, j1 - 1u);
        const uint psuf = p - a.prefix_len;
        const uint bidx = a.is_swa ? ((psuf / KV_BLK) % a.pt_mod) : (psuf / KV_BLK);
        const uint boff = psuf % KV_BLK;
        const uint pid = pt[bidx];
        float acc[3] = {0, 0, 0};
        if (p == pos) {
            for (int c = 0; c < 8; c++) {
                const float4 kf = *(threadgroup const float4 *) (&kc[pp * 32 + c * 4]);
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
        } else {
            const ulong kb = paged_kv_base(pid, a.pool_cap, a.slot_idx, g, 0) + (ulong) boff * HD;
#ifdef KV8
            device const kv_t * krow = KV + kb;
            device const packed_char4 * kr = (device const packed_char4 *) (krow + pp * 32);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(char4(kr[c]));
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
            const float ks = kv_vscale(krow);
            for (int h = 0; h < 3; h++) acc[h] *= ks;
#else
            device const half4 * kr = (device const half4 *) (KV + kb + pp * 32);
            for (int c = 0; c < 8; c++) {
                const float4 kf = float4(kr[c]);
                for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
            }
#endif
        }
        const uint d = pos - p;
        if (d < EXT) {
            const float4 bb = *(device const float4 *) (relb + d * NREL + pp * 4);
            for (int h = 0; h < 3; h++) acc[h] += dot(rl[h], bb);
        }
        float pr[3], sc_[3];
        for (int h = 0; h < 3; h++) {
            float t = acc[h];
            t += simd_shuffle_xor(t, 1); t += simd_shuffle_xor(t, 2);
            t = valid ? t - sink[h] : -INFINITY;
            const float mx = max(m[h], simd_max(t));
            sc_[h] = exp(m[h] - mx);
            pr[h] = exp(t - mx);
            l[h] = l[h] * sc_[h] + simd_sum(pr[h]) * 0.25f;
            m[h] = mx;
            o[h] *= sc_[h];
        }
#ifdef KV8
        {
            const ulong vb = paged_kv_base(pid, a.pool_cap, a.slot_idx, g, 1) + (ulong) boff * HD;
            const float vsc = (p == pos) ? 1.0f : kv_vscale(KV + vb);
            for (int h = 0; h < 3; h++) pr[h] *= vsc;
        }
#endif
        const uint suf_lo = first_pos + jb - a.prefix_len;
        const uint suf_hi = first_pos + min(jb + 7u, j1 - 1u) - a.prefix_len;
        const uint blk_lo = suf_lo / KV_BLK, blk_hi = suf_hi / KV_BLK;
        const ulong vbase_lo = paged_kv_base(pt[a.is_swa ? (blk_lo % a.pt_mod) : blk_lo], a.pool_cap, a.slot_idx, g, 1);
        const ulong vbase_hi = (blk_lo == blk_hi) ? vbase_lo : paged_kv_base(pt[a.is_swa ? (blk_hi % a.pt_mod) : blk_hi], a.pool_cap, a.slot_idx, g, 1);
        for (ushort q2 = 0; q2 < 8; q2++) {
            const uint jq = jb + q2;
            if (jq >= j1) break;
            const uint pq = first_pos + jq;
            float4 v;
            if (pq == pos) {
                v = *(threadgroup const float4 *) (&vc[lane * 4]);
            } else {
                const uint qsuf = pq - a.prefix_len;
                const ulong vb = ((qsuf / KV_BLK == blk_lo) ? vbase_lo : vbase_hi) + (ulong) (qsuf % KV_BLK) * HD + lane * 4;
#ifdef KV8
                v = float4(char4(*(device const packed_char4 *) (KV + vb)));
#else
                v = float4(*(device const half4 *) (KV + vb));
#endif
            }
            for (int h = 0; h < 3; h++) o[h] += simd_shuffle(pr[h], q2 * 4) * v;
        }
    }
    threadgroup float ms_[4][3], ls_[4][3];
    threadgroup float4 os_[4][3][32];
    if (lane == 0) for (int h = 0; h < 3; h++) { ms_[sg][h] = m[h]; ls_[sg][h] = l[h]; }
    for (int h = 0; h < 3; h++) os_[sg][h][lane] = o[h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (a.nsplit == 1) {
        device half * out = (device half *) (A + a.out_off);
        device const half * gate = (device const half *) (A + a.gate_off);
        for (int h = 0; h < 3; h++) {
            const uint H = g * 3 + h;
            float M = 0.0f;
            for (int s2 = 0; s2 < 4; s2++) M = max(M, ms_[s2][h]);
            float L = exp(-M), O = 0.0f;
            for (int s2 = 0; s2 < 4; s2++) {
                const float w = ms_[s2][h] == -INFINITY ? 0.0f : exp(ms_[s2][h] - M);
                L += w * ls_[s2][h];
                O += w * ((threadgroup const float *) os_[s2][h])[tid];
            }
            const ulong oi = (ulong) b * (NH * HD) + H * HD + tid;
            out[oi] = half(O / L * sigm(float(gate[oi])));
        }
    } else {
        for (int h = 0; h < 3; h++) {
            float M = -INFINITY;
            for (int s2 = 0; s2 < 4; s2++) M = max(M, ms_[s2][h]);
            float L = 0.0f, O = 0.0f;
            for (int s2 = 0; s2 < 4; s2++) {
                const float w = ms_[s2][h] == -INFINITY ? 0.0f : exp(ms_[s2][h] - M);
                L += w * ls_[s2][h];
                O += w * ((threadgroup const float *) os_[s2][h])[tid];
            }
            device float * P = A + a.part_off + ((ulong) (b * NH + g * 3 + h) * a.nsplit + sp) * (HD + 2);
            if (tid == 0) { P[0] = M; P[1] = L; }
            P[2 + tid] = O;
        }
    }
}

// Single-sequence chunked post-conv + optional residual using FP16 CS_POOL (for streaming 256K prefill).
// Grid: (E / 256, T, 1), TG = 256.
struct ChunkConvHcsArgs {
    uint y_off, ldy, res_off, out_off;
    uint layer_slot, branch, max_batch, convw_t_off;
    uint has_res, seq_cs, p0, T;
};

kernel void k_postconv_hcs_chunk(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                                 device const half * CS [[buffer(4)]], constant ChunkConvHcsArgs & a [[buffer(5)]],
                                 uint2 tgp [[threadgroup_position_in_grid]],
                                 ushort tid [[thread_index_in_threadgroup]]) {
    const uint t = tgp.y, c = tgp.x * 256u + tid;
    if (c >= E || t >= a.T) return;
    const uint tap0 = a.p0 % 3u, tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u), tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
    const uint taps[3] = { tap0, tap1, tap2 };
    device const half * hb = CS + cs_branch_base(a.layer_slot, a.branch, a.max_batch) + (ulong) a.seq_cs * (3ul * E);
    device const float * w = (device const float *) (PB + a.convw_t_off);
    float x[4];
    for (int j = 0; j < 4; j++) {
        const int lt = int(t) - 3 + j;
        x[j] = lt >= 0 ? A[a.y_off + (ulong) lt * a.ldy + c] : float(hb[(ulong) taps[3 + lt] * E + c]);
    }
    const float o = (x[3] + w[0 * E + c] * x[0] + w[1 * E + c] * x[1] + w[2 * E + c] * x[2] + w[3 * E + c] * x[3]) * 0.4472135954999579f;
    A[a.out_off + (ulong) t * E + c] = (a.has_res ? A[a.res_off + (ulong) t * E + c] : 0.0f) + o;
}

// Single-sequence chunked FP16 CS ring state update after a chunk of length T starting at p0.
// Grid: (ceil(nch / 256), 1, 1), TG = 256.
struct StateHcsArgs {
    uint x_off, ldx, layer_slot, branch;
    uint max_batch, nch, seq_cs, p0;
    uint T, pad0, pad1, pad2;
};

kernel void k_state_upd_hcs(device const float * A [[buffer(2)]], device half * CS [[buffer(4)]],
                            constant StateHcsArgs & a [[buffer(5)]],
                            uint tg [[threadgroup_position_in_grid]],
                            ushort tid [[thread_index_in_threadgroup]]) {
    const uint c = tg * 256u + tid;
    if (c >= a.nch) return;
    const uint dim = a.nch;
    device half * hb = CS + cs_branch_base(a.layer_slot, a.branch, a.max_batch) + (ulong) a.seq_cs * (3ul * dim);
    const uint ot0 = a.p0 % 3u, ot1 = (ot0 == 2u) ? 0u : (ot0 + 1u), ot2 = (ot1 == 2u) ? 0u : (ot1 + 1u);
    const half oldh[3] = { hb[(ulong) ot0 * dim + c], hb[(ulong) ot1 * dim + c], hb[(ulong) ot2 * dim + c] };
    half v[3];
    for (int j = 0; j < 3; j++) {
        const int idx = int(a.T) + j;
        v[j] = idx >= 3 ? half(A[a.x_off + (ulong) (idx - 3) * a.ldx + c]) : oldh[idx];
    }
    const uint nt0 = (a.p0 + a.T) % 3u, nt1 = (nt0 == 2u) ? 0u : (nt0 + 1u), nt2 = (nt1 == 2u) ? 0u : (nt1 + 1u);
    hb[(ulong) nt0 * dim + c] = v[0];
    hb[(ulong) nt1 * dim + c] = v[1];
    hb[(ulong) nt2 * dim + c] = v[2];
}

// ============================================================================================================
// PART 3: WHT-128 Rotated Ultra-Low-Bit KV (KV4 / KV3 / K3V2 / KV2) + Per-Head DC Mean
//         + WHT-2bit-Sketch Page Scoring + Tiered SWA Outer-Ring WHT-K2V2 Compression
// ============================================================================================================

#define WHT2_NSUB 8u
#define SWA_INNER_BLKS 9u
#define SWA_OUTER_BLKS 24u

// Optimal N(0,1) Lloyd-Max centroids (Max 1960):
//   2-bit thresholds: -0.9815988, 0.0, +0.9815988
//   3-bit thresholds: -1.7479275, -1.0499573, -0.5005497, 0.0, +0.5005497, +1.0499573, +1.7479275
constant float LM2_VAL[4] = {
    -1.5104176f, -0.45278004f, 0.45278004f, 1.5104176f
};
constant float LM3_VAL[8] = {
    -2.1519456f, -1.3439093f, -0.7560053f, -0.24509418f,
     0.24509418f, 0.7560053f,  1.3439093f,  2.1519456f
};

// Deterministic per-(slot, g, dim) Rademacher sign in {-1.0f, +1.0f} so Q = H_128 * D is strictly orthogonal.
inline float rademacher_sign(uint slot, uint g, uint d) {
    uint h = (slot * 2654435761u) ^ (g * 2246822519u) ^ (d * 3266489917u);
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    return (h & 1u) ? -1.0f : 1.0f;
}

// Branch-free 32-lane bitplane ballot within a simdgroup (lane i contributes bit i of the returned uint32).
inline uint sg_ballot32(uint bit, ushort lane) {
    uint w = (bit & 1u) << lane;
    w |= simd_shuffle_xor(w, 1);
    w |= simd_shuffle_xor(w, 2);
    w |= simd_shuffle_xor(w, 4);
    w |= simd_shuffle_xor(w, 8);
    w |= simd_shuffle_xor(w, 16);
    return w;
}

// Exact normalized 128-point Walsh-Hadamard transform over a 128-thread threadgroup (1 element/thread).
// Symmetric & self-inverse: H_128^T = H_128, H_128 * H_128 = I.
// Uses 5 intra-simdgroup shuffle butterflies + 1 cross-simdgroup 4x4 Hadamard pass via sh[128].
inline float wht128_tg(float x, ushort tid, ushort sg, ushort lane, threadgroup float * sh) {
    // 5 intra-simdgroup butterfly stages: strides 1, 2, 4, 8, 16
    float o;
    o = simd_shuffle_xor(x, 1);  x = (lane & 1)  ? (o - x) : (x + o);
    o = simd_shuffle_xor(x, 2);  x = (lane & 2)  ? (o - x) : (x + o);
    o = simd_shuffle_xor(x, 4);  x = (lane & 4)  ? (o - x) : (x + o);
    o = simd_shuffle_xor(x, 8);  x = (lane & 8)  ? (o - x) : (x + o);
    o = simd_shuffle_xor(x, 16); x = (lane & 16) ? (o - x) : (x + o);
    // Stages 6 & 7 (strides 32 & 64) combined in one 4x4 symmetric Hadamard across the 4 simdgroups
    sh[tid] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float a0 = sh[lane];
    const float a1 = sh[32u + lane];
    const float a2 = sh[64u + lane];
    const float a3 = sh[96u + lane];
    const float b0 = a0 + a1, b1 = a0 - a1, b2 = a2 + a3, b3 = a2 - a3;
    float r;
    if (sg == 0)      r = b0 + b2;
    else if (sg == 1) r = b1 + b3;
    else if (sg == 2) r = b0 - b2;
    else              r = b1 - b3;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r * 0.08838834764831845f; // 1 / sqrt(128)
}

// Per-vector bit-width decoding for kv_bits mode:
//   kv_bits == 4  : WHT-KV4  (K=4b [64B], V=4b [64B] + 4B scales = 132 B/pair, 3.88x vs FP16)
//   kv_bits == 3  : WHT-K3V2 (K=3b [48B], V=2b [32B] + 4B scales =  84 B/pair, 6.10x vs FP16)
//   kv_bits == 33 : WHT-KV3  (K=3b [48B], V=3b [48B] + 4B scales = 100 B/pair, 5.12x vs FP16)
//   kv_bits == 2  : WHT-KV2  (K=2b [32B], V=2b [32B] + 4B scales =  68 B/pair, 7.53x vs FP16)
inline uint kv_k_bits(uint kv_bits) {
    if (kv_bits == 4u) return 4u;
    if (kv_bits == 3u || kv_bits == 33u) return 3u;
    return 2u;
}
inline uint kv_v_bits(uint kv_bits) {
    if (kv_bits == 4u) return 4u;
    if (kv_bits == 33u) return 3u;
    return 2u;
}
inline ulong kv_words_per_vec(uint vec_bits) {
    return (vec_bits == 4u) ? 16ul : ((vec_bits == 3u) ? 12ul : 8ul);
}

// 64-bit safe vector index into separate contiguous half scales buffer:
// Layout: [slot_idx][pool_cap][2 (K,V)][NKV][page_toks] (1 half/vec = 2 B)
inline ulong comp_vec_idx(uint pid, uint pool_cap, uint slot_idx, uint g, uint is_v, uint boff, uint page_toks) {
    return ((((ulong) slot_idx * (ulong) pool_cap + (ulong) pid) * 2ul + (ulong) is_v) * (ulong) NKV + (ulong) g) * (ulong) page_toks + (ulong) boff;
}

// 64-bit safe uint32 word offset into KV_COMP supporting BOTH symmetric (KV4/KV3/KV2) and asymmetric (K3V2) pages!
// Within each (slot_idx, pid) page: K block [NKV][page_toks][k_words] followed by V block [NKV][page_toks][v_words].
// Both K and V blocks are guaranteed 16-byte aligned (since 16, 12, and 8 uint32 are all multiples of 4 uint32 = 16 B).
inline ulong comp_word_off(uint pid, uint pool_cap, uint slot_idx, uint g, uint is_v, uint boff, uint page_toks, uint kv_bits) {
    const ulong kw = kv_words_per_vec(kv_k_bits(kv_bits));
    const ulong vw = kv_words_per_vec(kv_v_bits(kv_bits));
    const ulong nkp = (ulong) NKV * (ulong) page_toks;
    const ulong page_base = ((ulong) slot_idx * (ulong) pool_cap + (ulong) pid) * nkp * (kw + vw);
    const ulong head_tok = (ulong) g * (ulong) page_toks + (ulong) boff;
    return is_v ? (page_base + nkp * kw + head_tok * vw) : (page_base + head_tok * kw);
}

// 64-bit safe vector index into tiered SWA inner ring buffer:
// Layout: [slot_idx][inner_cap][2 (K,V)][NKV][KV_BLK=32] (multiply by HD for element offset in SWA_INNER).
inline ulong swa_inner_vec_idx(uint pid, uint inner_cap, uint slot_idx, uint g, uint is_v, uint boff) {
    return ((((ulong) slot_idx * (ulong) inner_cap + (ulong) pid) * 2ul + (ulong) is_v) * (ulong) NKV + (ulong) g) * (ulong) KV_BLK + (ulong) boff;
}

// Quantize & pack one 128-dim rotated vector (1 element per thread across 128 threads) into 4-bit, 3-bit bitplane, or 2-bit Lloyd-Max.
//   4-bit: 16 uint32 words (8 consecutive 4-bit nibbles/uint32, uniform symmetric [-7..+7] + 8)
//   3-bit: 12 uint32 words (3 bitplanes: lo[0..3], mid[4..7], hi[8..11], N(0,1) Lloyd-Max LM3_VAL, RMS scale)
//   2-bit:  8 uint32 words (16 consecutive 2-bit codes/uint32, N(0,1) Lloyd-Max LM2_VAL, RMS scale)
inline void pack_wht_vec(float val, uint vec_bits, ulong word_off, ulong vidx, ushort tid, ushort sg, ushort lane,
                         threadgroup float * red4, device uint * KV_COMP, device half * KV_SCALES) {
    if (vec_bits == 4u) {
        const float am = simd_max(fabs(val));
        if (lane == 0) red4[sg] = am;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float mx = max(max(red4[0], red4[1]), max(red4[2], red4[3]));
        const float sc = max(mx / 7.0f, 1e-7f);
        const uint q = (uint) (clamp((int) rint(val / sc), -7, 7) + 8); // 1..15 (decode: float(q) - 8.0f)
        uint w = q << ((tid & 7u) * 4u);
        w |= simd_shuffle_xor(w, 1);
        w |= simd_shuffle_xor(w, 2);
        w |= simd_shuffle_xor(w, 4);
        if ((tid & 7u) == 0u) KV_COMP[word_off + (ulong) (tid >> 3)] = w;
        if (tid == 0) KV_SCALES[vidx] = half(sc);
    } else if (vec_bits == 3u) {
        // 3-bit Lloyd-Max on WHT-Gaussian coordinates, packed into 3 coalesced bitplanes (12 uint32 = 48 B exact)
        const float ss = simd_sum(val * val);
        if (lane == 0) red4[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float tot_ss = red4[0] + red4[1] + red4[2] + red4[3];
        const float rms = max(sqrt(tot_ss * (1.0f / 128.0f)), 1e-7f);
        const float u = val / rms;
        const uint q = (u >= -1.7479275f ? 1u : 0u)
                     + (u >= -1.0499573f ? 1u : 0u)
                     + (u >= -0.5005497f ? 1u : 0u)
                     + (u >=  0.0f       ? 1u : 0u)
                     + (u >=  0.5005497f ? 1u : 0u)
                     + (u >=  1.0499573f ? 1u : 0u)
                     + (u >=  1.7479275f ? 1u : 0u);
        const uint w0 = sg_ballot32(q & 1u, lane);
        const uint w1 = sg_ballot32((q >> 1) & 1u, lane);
        const uint w2 = sg_ballot32((q >> 2) & 1u, lane);
        if (lane == 0) {
            KV_COMP[word_off + (ulong) sg]       = w0;
            KV_COMP[word_off + 4ul + (ulong) sg] = w1;
            KV_COMP[word_off + 8ul + (ulong) sg] = w2;
        }
        if (tid == 0) KV_SCALES[vidx] = half(rms);
    } else {
        // 2-bit Lloyd-Max on WHT-Gaussian coordinates (8 uint32 = 32 B)
        const float ss = simd_sum(val * val);
        if (lane == 0) red4[sg] = ss;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float tot_ss = red4[0] + red4[1] + red4[2] + red4[3];
        const float rms = max(sqrt(tot_ss * (1.0f / 128.0f)), 1e-7f);
        const float u = val / rms;
        const uint q = (u >= -0.9815988f ? 1u : 0u) + (u >= 0.0f ? 1u : 0u) + (u >= 0.9815988f ? 1u : 0u);
        uint w = q << ((tid & 15u) * 2u);
        w |= simd_shuffle_xor(w, 1);
        w |= simd_shuffle_xor(w, 2);
        w |= simd_shuffle_xor(w, 4);
        w |= simd_shuffle_xor(w, 8);
        if ((tid & 15u) == 0u) KV_COMP[word_off + (ulong) (tid >> 4)] = w;
        if (tid == 0) KV_SCALES[vidx] = half(rms);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Dequantize one element `d` (0..127) from 4-bit, 3-bit bitplane, or 2-bit compressed pool.
inline float unpack_wht_elem(device const uint * KV_COMP, device const half * KV_SCALES,
                             uint vec_bits, ulong word_off, ulong vidx, ushort d) {
    const float sc = float(KV_SCALES[vidx]);
    if (vec_bits == 4u) {
        const uint w = KV_COMP[word_off + (ulong) (d >> 3)];
        const uint q = (w >> ((d & 7u) * 4u)) & 0xFu;
        return float(int(q) - 8) * sc;
    } else if (vec_bits == 3u) {
        const ulong chunk = (ulong) (d >> 5);
        const uint bit = d & 31u;
        const uint q = ((KV_COMP[word_off + chunk] >> bit) & 1u)
                     | (((KV_COMP[word_off + 4ul + chunk] >> bit) & 1u) << 1)
                     | (((KV_COMP[word_off + 8ul + chunk] >> bit) & 1u) << 2);
        return LM3_VAL[q] * sc;
    } else {
        const uint w = KV_COMP[word_off + (ulong) (d >> 4)];
        const uint q = (w >> ((d & 15u) * 2u)) & 3u;
        return LM2_VAL[q] * sc;
    }
}

// Dot 32 dims (pp*32 .. pp*32+31) of rotated compressed K against qs4[0..2][pp*8 + 0..7].
// Branch-free coalesced bitplane extraction for 3-bit K!
inline void dot_wht_k32(device const uint * KV_COMP, device const half * KV_SCALES,
                        uint k_bits, ulong word_off, ulong vidx,
                        ushort pp, threadgroup const float4 qs4[3][HD / 4], thread float acc[3]) {
    const float sc = float(KV_SCALES[vidx]);
    float lacc[3] = {0.0f, 0.0f, 0.0f};
    if (k_bits == 4u) {
        const uint4 w4 = *(device const uint4 *) (KV_COMP + word_off + (ulong) pp * 4ul);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint u = w4[i];
            const float4 k0 = float4(int4(int(u & 0xFu) - 8, int((u >> 4) & 0xFu) - 8, int((u >> 8) & 0xFu) - 8, int((u >> 12) & 0xFu) - 8));
            const float4 k1 = float4(int4(int((u >> 16) & 0xFu) - 8, int((u >> 20) & 0xFu) - 8, int((u >> 24) & 0xFu) - 8, int((u >> 28) & 0xFu) - 8));
            for (int h = 0; h < 3; h++) {
                lacc[h] += dot(qs4[h][pp * 8 + i * 2 + 0], k0) + dot(qs4[h][pp * 8 + i * 2 + 1], k1);
            }
        }
    } else if (k_bits == 3u) {
        // Adjacent lanes pp=0..3 coalesce all 3 bitplanes (lo[0..3], mid[4..7], hi[8..11]) into three 16-byte loads!
        uint b0 = KV_COMP[word_off + (ulong) pp];
        uint b1 = KV_COMP[word_off + 4ul + (ulong) pp];
        uint b2 = KV_COMP[word_off + 8ul + (ulong) pp];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const uint c0 = ((b0 >> 0) & 1u) | (((b1 >> 0) & 1u) << 1) | (((b2 >> 0) & 1u) << 2);
            const uint c1 = ((b0 >> 1) & 1u) | (((b1 >> 1) & 1u) << 1) | (((b2 >> 1) & 1u) << 2);
            const uint c2 = ((b0 >> 2) & 1u) | (((b1 >> 2) & 1u) << 1) | (((b2 >> 2) & 1u) << 2);
            const uint c3 = ((b0 >> 3) & 1u) | (((b1 >> 3) & 1u) << 1) | (((b2 >> 3) & 1u) << 2);
            b0 >>= 4; b1 >>= 4; b2 >>= 4;
            const float4 kf = float4(LM3_VAL[c0], LM3_VAL[c1], LM3_VAL[c2], LM3_VAL[c3]);
            for (int h = 0; h < 3; h++) lacc[h] += dot(qs4[h][pp * 8 + i], kf);
        }
    } else {
        const uint2 w2 = *(device const uint2 *) (KV_COMP + word_off + (ulong) pp * 2ul);
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const uint u = w2[i];
            const float4 k0 = float4(LM2_VAL[u & 3u], LM2_VAL[(u >> 2) & 3u], LM2_VAL[(u >> 4) & 3u], LM2_VAL[(u >> 6) & 3u]);
            const float4 k1 = float4(LM2_VAL[(u >> 8) & 3u], LM2_VAL[(u >> 10) & 3u], LM2_VAL[(u >> 12) & 3u], LM2_VAL[(u >> 14) & 3u]);
            const float4 k2 = float4(LM2_VAL[(u >> 16) & 3u], LM2_VAL[(u >> 18) & 3u], LM2_VAL[(u >> 20) & 3u], LM2_VAL[(u >> 22) & 3u]);
            const float4 k3 = float4(LM2_VAL[(u >> 24) & 3u], LM2_VAL[(u >> 26) & 3u], LM2_VAL[(u >> 28) & 3u], LM2_VAL[(u >> 30) & 3u]);
            for (int h = 0; h < 3; h++) {
                lacc[h] += dot(qs4[h][pp * 8 + i * 4 + 0], k0) + dot(qs4[h][pp * 8 + i * 4 + 1], k1)
                         + dot(qs4[h][pp * 8 + i * 4 + 2], k2) + dot(qs4[h][pp * 8 + i * 4 + 3], k3);
            }
        }
    }
    for (int h = 0; h < 3; h++) acc[h] += lacc[h] * sc;
}

// Dequantize 4 consecutive V dims (lane*4 .. lane*4+3) from 4-bit, 3-bit bitplane, or 2-bit compressed pool (scale included!).
inline float4 load_wht_v4(device const uint * KV_COMP, device const half * KV_SCALES,
                          uint v_bits, ulong word_off, ulong vidx, ushort lane) {
    const float sc = float(KV_SCALES[vidx]);
    if (v_bits == 4u) {
        const ushort u = ((device const ushort *) (KV_COMP + word_off))[lane];
        return float4(int4(int(u & 0xFu) - 8, int((u >> 4) & 0xFu) - 8, int((u >> 8) & 0xFu) - 8, int((u >> 12) & 0xFu) - 8)) * sc;
    } else if (v_bits == 3u) {
        const ulong chunk = (ulong) (lane >> 3);
        const uint shift = (lane & 7u) * 4u;
        const uint b0 = KV_COMP[word_off + chunk] >> shift;
        const uint b1 = KV_COMP[word_off + 4ul + chunk] >> shift;
        const uint b2 = KV_COMP[word_off + 8ul + chunk] >> shift;
        const uint c0 = ((b0 >> 0) & 1u) | (((b1 >> 0) & 1u) << 1) | (((b2 >> 0) & 1u) << 2);
        const uint c1 = ((b0 >> 1) & 1u) | (((b1 >> 1) & 1u) << 1) | (((b2 >> 1) & 1u) << 2);
        const uint c2 = ((b0 >> 2) & 1u) | (((b1 >> 2) & 1u) << 1) | (((b2 >> 2) & 1u) << 2);
        const uint c3 = ((b0 >> 3) & 1u) | (((b1 >> 3) & 1u) << 1) | (((b2 >> 3) & 1u) << 2);
        return float4(LM3_VAL[c0], LM3_VAL[c1], LM3_VAL[c2], LM3_VAL[c3]) * sc;
    } else {
        const uchar u = ((device const uchar *) (KV_COMP + word_off))[lane];
        return float4(LM2_VAL[u & 3u], LM2_VAL[(u >> 2) & 3u], LM2_VAL[(u >> 4) & 3u], LM2_VAL[(u >> 6) & 3u]) * sc;
    }
}

// ---- 0. k_compute_k_dc_mean:
// Compute per-head key DC mean [n_slots][NKV][HD] (half) from Engine's kv_full buffer over [src_pos0 .. src_pos0 + n_toks).
// Grid: (NKV, n_slots, 1), TG = 128 (1 dim per thread).
struct ComputeDcMeanArgs {
    uint src_cap, src_pos0, n_toks, n_slots;
};

kernel void k_compute_k_dc_mean(device const kv_t * SRC [[buffer(3)]], constant ComputeDcMeanArgs & a [[buffer(5)]],
                                device half * K_DC_MEAN [[buffer(10)]],
                                uint2 tgp [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]]) {
    const uint g = tgp.x, s = tgp.y;
    if (s >= a.n_slots) return;
    const uint n = max(a.n_toks, 1u);
    const ulong base = (((ulong) s * 2ul + 0ul) * NKV + g) * (ulong) a.src_cap * HD;
    float acc = 0.0f;
    for (uint i = 0u; i < a.n_toks; i++) {
        const ulong row = base + (ulong) (a.src_pos0 + i) * HD;
#ifdef KV8
        acc += float(SRC[row + tid]) * kv_vscale(SRC + row);
#else
        acc += float(SRC[row + tid]);
#endif
    }
    K_DC_MEAN[((ulong) s * NKV + g) * HD + tid] = half(acc / float(n));
}

// ---- 1. k_kvprep_wht_write:
// Fused K/V short-conv (FP16 CS_POOL), clip/norm, per-head DC key mean subtraction + sink-exact dc_bias
// (`eff_sink[h] = sink[h] - dot(q_orig[h], mu)` preserves 100% mathematical equivalence with attention sink!),
// orthogonal WHT-128(D * q/(k-mu)/v), KV4/K3V2/KV3/KV2 packing + 64-bit paged write into KV_COMP/KV_SCALES!
// Supports BOTH batched decode (is_prefill == 0: grid=(B, NKV, 1)) and single-seq chunked prefill (is_prefill == 1: grid=(T, NKV, 1)).
// TG = 128.
struct WhtKvPrepArgs {
    uint B, pool_cap, slot_idx, page_toks, kv_bits; // kv_bits = 4 (KV4), 3 (K3V2), 33 (KV3), or 2 (KV2)
    uint layer_slot, max_batch, is_prefill, p0;
    uint seq_pos_off, seq_cs_off, write_pid_off;
    uint q_off, k_off, v_off, r_off;
    int  kdyn_off, vdyn_off;
    uint kconv_t_w, vconv_t_w, qclip_w, relclip_w;
    float log_alpha, log_floor;
    uint qp_off, rp_off, sink_off, eff_sink_off;
    uint use_dc_mean, pad0, pad1, pad2;
};

kernel void k_kvprep_wht_write(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                               device uint * KV_COMP [[buffer(3)]], device half * CS [[buffer(4)]],
                               constant WhtKvPrepArgs & a [[buffer(5)]], device half * KV_SCALES [[buffer(6)]],
                               device const uint * META [[buffer(7)]],
                               device const half * K_DC_MEAN [[buffer(10)]],
                               uint2 tgp [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[128];
    threadgroup float red[4][8];
    threadgroup float red4[4];
    const uint idx = tgp.x, g = tgp.y;
    if (idx >= a.B) return;

    const uint pos = a.is_prefill ? (a.p0 + idx) : META[a.seq_pos_off + idx];
    const uint seq_cs = a.is_prefill ? META[a.seq_cs_off] : META[a.seq_cs_off + idx];
    const float tau = 1.0f + a.log_alpha * log(max(1.0f, float(pos + 1u) / a.log_floor));
    device const float * qclip = (device const float *) (PB + a.qclip_w);

    float q[3];
    for (int h = 0; h < 3; h++) {
        const uint H = g * 3u + (uint) h;
        const half qh = ((device const half *) (A + a.q_off))[(ulong) idx * (NH * HD) + H * HD + tid];
        q[h] = float(qh) * qclip[H * HD + tid];
    }

    const uint c = g * HD + tid;
    const uint dim = NKV * HD;
    const float scl = a.kdyn_off >= 0 ? (1.0f / 3.0f) : 0.4472135954999579f;
    float kx = 0.0f, vx = 0.0f;

    if (!a.is_prefill) {
        const uint tap0 = pos % 3u;
        const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
        const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
        for (int which = 0; which < 2; which++) {
            device half * hb = CS + cs_branch_base(a.layer_slot, (uint) which, a.max_batch) + (ulong) seq_cs * (3ul * dim);
            device const float * w = (device const float *) (PB + (which == 0 ? a.kconv_t_w : a.vconv_t_w));
            const float x = A[(which == 0 ? a.k_off : a.v_off) + (ulong) idx * dim + c];
            const float h0 = float(hb[(ulong) tap0 * dim + c]);
            const float h1 = float(hb[(ulong) tap1 * dim + c]);
            const float h2 = float(hb[(ulong) tap2 * dim + c]);
            float acc = w[0 * dim + c] * h0 + w[1 * dim + c] * h1 + w[2 * dim + c] * h2 + w[3 * dim + c] * x;
            const int dyn = which == 0 ? a.kdyn_off : a.vdyn_off;
            if (dyn >= 0) {
                device const float * tp = A + dyn + (ulong) idx * 32ul + g * 4u;
                acc += tp[0] * x + tp[1] * h2 + tp[2] * h1 + tp[3] * h0;
            }
            hb[(ulong) tap0 * dim + c] = half(x);
            if (which == 0) kx = (x + acc) * scl; else vx = (x + acc) * scl;
        }
    } else {
        const uint tap0 = a.p0 % 3u, tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u), tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
        const uint taps[3] = { tap0, tap1, tap2 };
        for (int which = 0; which < 2; which++) {
            const uint xo = which == 0 ? a.k_off : a.v_off;
            device const half * hb = CS + cs_branch_base(a.layer_slot, (uint) which, a.max_batch) + (ulong) seq_cs * (3ul * dim);
            device const float * w = (device const float *) (PB + (which == 0 ? a.kconv_t_w : a.vconv_t_w));
            float x[4];
            for (int j = 0; j < 4; j++) {
                const int lt = int(idx) - 3 + j;
                x[j] = lt >= 0 ? A[xo + (ulong) lt * dim + c] : float(hb[(ulong) taps[3 + lt] * dim + c]);
            }
            float acc = w[0 * dim + c] * x[0] + w[1 * dim + c] * x[1] + w[2 * dim + c] * x[2] + w[3 * dim + c] * x[3];
            const int dyn = which == 0 ? a.kdyn_off : a.vdyn_off;
            if (dyn >= 0) {
                device const float * tp = A + dyn + (ulong) idx * 32ul + g * 4u;
                acc += tp[0] * x[3] + tp[1] * x[2] + tp[2] * x[1] + tp[3] * x[0];
            }
            if (which == 0) kx = (x[3] + acc) * scl; else vx = (x[3] + acc) * scl;
        }
    }

    float r = 0.0f;
    if (tid < 48) {
        const uint h = tid / 16u, i = tid % 16u, H = g * 3u + h;
        r = A[a.r_off + (ulong) idx * (NH * NREL) + H * NREL + i] * ((device const float *) (PB + a.relclip_w))[H * NREL + i];
    }
    float rn = r * r;
    rn += simd_shuffle_xor(rn, 1); rn += simd_shuffle_xor(rn, 2); rn += simd_shuffle_xor(rn, 4); rn += simd_shuffle_xor(rn, 8);
    {
        const float s0 = simd_sum(q[0] * q[0]), s1 = simd_sum(q[1] * q[1]), s2 = simd_sum(q[2] * q[2]);
        const float s3 = simd_sum(kx * kx), s4 = simd_sum(vx * vx);
        if (lane == 0) { red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; red[sg][3] = s3; red[sg][4] = s4; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot[5];
    for (int i = 0; i < 5; i++) tot[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];

    const float mu = (a.use_dc_mean != 0u) ? float(K_DC_MEAN[((ulong) a.slot_idx * NKV + g) * HD + tid]) : 0.0f;
    const float sign = rademacher_sign(a.layer_slot, g, tid);
    device half * qp_out = (device half *) (A + a.qp_off);
    for (int h = 0; h < 3; h++) {
        const float f = clamp(QK_CLIP / (sqrt(tot[h]) + 1e-6f), 0.0f, 1.0f) * tau * 0.08838834764831845f;
        const float qf = q[h] * f;
        // Exact attention-sink adjustment for DC key mean subtraction:
        //   q^T (k - mu) - (sink - q^T mu) == q^T k - sink  (100% invariant with sink token at 0.0!)
        if (a.use_dc_mean != 0u) {
            const float ps = simd_sum(qf * mu);
            if (lane == 0) red4[sg] = ps;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                const float dc_bias = red4[0] + red4[1] + red4[2] + red4[3];
                const ulong si = (ulong) idx * NH + g * 3u + (uint) h;
                A[a.eff_sink_off + si] = A[a.sink_off + si] - dc_bias;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        } else if (a.eff_sink_off != a.sink_off && tid == 0) {
            const ulong si = (ulong) idx * NH + g * 3u + (uint) h;
            A[a.eff_sink_off + si] = A[a.sink_off + si];
        }
        const float qr = wht128_tg(sign * qf, tid, sg, lane, sh);
        qp_out[(ulong) idx * (NH * HD) + (g * 3u + (uint) h) * HD + tid] = half(qr);
    }
    if (tid < 48) {
        const uint h = tid / 16u, i = tid % 16u, H = g * 3u + h;
        A[a.rp_off + (ulong) idx * (NH * NREL) + H * NREL + i] = r * clamp(REL_CLIP / (sqrt(rn) + 1e-6f), 0.0f, 1.0f) * tau;
    }

    const float kf = clamp(QK_CLIP / (sqrt(tot[3]) + 1e-6f), 0.0f, 1.0f);
    const float vf = rsqrt(tot[4] / float(HD) + 1e-6f);
    const float k_rot = wht128_tg(sign * (kx * kf - mu), tid, sg, lane, sh);
    const float v_rot = wht128_tg(sign * (vx * vf), tid, sg, lane, sh);

    const uint pid_idx = a.is_prefill ? (pos / a.page_toks - a.p0 / a.page_toks) : idx;
    const uint pid = META[a.write_pid_off + pid_idx];
    const uint boff = pos % a.page_toks;
    const ulong kv_k = comp_vec_idx(pid, a.pool_cap, a.slot_idx, g, 0u, boff, a.page_toks);
    const ulong kv_v = comp_vec_idx(pid, a.pool_cap, a.slot_idx, g, 1u, boff, a.page_toks);
    const ulong kw_k = comp_word_off(pid, a.pool_cap, a.slot_idx, g, 0u, boff, a.page_toks, a.kv_bits);
    const ulong kw_v = comp_word_off(pid, a.pool_cap, a.slot_idx, g, 1u, boff, a.page_toks, a.kv_bits);
    pack_wht_vec(k_rot, kv_k_bits(a.kv_bits), kw_k, kv_k, tid, sg, lane, red4, KV_COMP, KV_SCALES);
    pack_wht_vec(v_rot, kv_v_bits(a.kv_bits), kw_v, kv_v, tid, sg, lane, red4, KV_COMP, KV_SCALES);
}

// ---- Helper: Compress a contiguous chunk of Engine's standard kv_t full-KV buffer directly into WHT-4/K3V2/KV3/KV2 pages
// (with optional per-head K_DC_MEAN subtraction before WHT rotation).
// Grid: (n_toks, NKV, 1), TG = 128.
struct Fp16ToWhtArgs {
    uint src_cap, src_pos0, slot_idx, layer_slot;
    uint pool_cap, page_toks, kv_bits, dst_pos0;
    uint pid_list_off, n_toks, use_dc_mean, pad1;
};

kernel void k_fp16_to_wht_compress(device const kv_t * SRC [[buffer(3)]], constant Fp16ToWhtArgs & a [[buffer(5)]],
                                   device uint * KV_COMP [[buffer(6)]], device const uint * META [[buffer(7)]],
                                   device half * KV_SCALES [[buffer(8)]],
                                   device const half * K_DC_MEAN [[buffer(10)]],
                                   uint2 tgp [[threadgroup_position_in_grid]],
                                   ushort tid [[thread_index_in_threadgroup]],
                                   ushort sg [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[128];
    threadgroup float red4[4];
    const uint i = tgp.x, g = tgp.y;
    if (i >= a.n_toks) return;

    const ulong src_k = (((ulong) a.slot_idx * 2ul + 0ul) * NKV + g) * (ulong) a.src_cap * HD + (ulong) (a.src_pos0 + i) * HD;
    const ulong src_v = (((ulong) a.slot_idx * 2ul + 1ul) * NKV + g) * (ulong) a.src_cap * HD + (ulong) (a.src_pos0 + i) * HD;
#ifdef KV8
    const float k_raw = float(SRC[src_k + tid]) * kv_vscale(SRC + src_k);
    const float v = float(SRC[src_v + tid]) * kv_vscale(SRC + src_v);
#else
    const float k_raw = float(SRC[src_k + tid]);
    const float v = float(SRC[src_v + tid]);
#endif
    const float mu = (a.use_dc_mean != 0u) ? float(K_DC_MEAN[((ulong) a.slot_idx * NKV + g) * HD + tid]) : 0.0f;
    const float sign = rademacher_sign(a.layer_slot, g, tid);
    const float k_rot = wht128_tg(sign * (k_raw - mu), tid, sg, lane, sh);
    const float v_rot = wht128_tg(sign * v, tid, sg, lane, sh);

    const uint dpos = a.dst_pos0 + i;
    const uint pid = META[a.pid_list_off + (dpos / a.page_toks - a.dst_pos0 / a.page_toks)];
    const uint boff = dpos % a.page_toks;
    const ulong kv_k = comp_vec_idx(pid, a.pool_cap, a.slot_idx, g, 0u, boff, a.page_toks);
    const ulong kv_v = comp_vec_idx(pid, a.pool_cap, a.slot_idx, g, 1u, boff, a.page_toks);
    const ulong kw_k = comp_word_off(pid, a.pool_cap, a.slot_idx, g, 0u, boff, a.page_toks, a.kv_bits);
    const ulong kw_v = comp_word_off(pid, a.pool_cap, a.slot_idx, g, 1u, boff, a.page_toks, a.kv_bits);
    pack_wht_vec(k_rot, kv_k_bits(a.kv_bits), kw_k, kv_k, tid, sg, lane, red4, KV_COMP, KV_SCALES);
    pack_wht_vec(v_rot, kv_v_bits(a.kv_bits), kw_v, kv_v, tid, sg, lane, red4, KV_COMP, KV_SCALES);
}

// ---- 2A. Legacy per-page Quest summary record (528 bytes, 16-byte aligned!) kept for Quest comparison.
struct KvPageSummary {
    char k_min[128];
    char k_max[128];
    char k_mean[128];
    char v_mean[128];
    half sc_kmin;
    half sc_kmax;
    half sc_kmean;
    half sc_vmean;
    uint count;
    uint page_id;
};

// ---- 2B. Compact WHT-2bit-sketch + WHT-4bit centroid page record (416 bytes, 16-byte aligned!):
//   - 8 sub-blocks (16-tok resolution at page_toks=128) x 32 B WHT-2bit sketch + 8 half scales = 272 B (34.0 MiB @ 256K!)
//   - Page (k_mean4, v_mean4) in WHT-4bit (64 B + 64 B + 4 B scales = 132 B; <0.0005 KL background centroid!)
//   - Total = 416 B/head/page = 52.0 MiB @ 256K (14 MiB SMALLER than Quest's 66 MiB AND 4.8x better KL!)
struct KvPageWht2 {
    uint k_sk[WHT2_NSUB][8]; // 256 B: 8 sub-blocks x 32 B WHT-2bit Lloyd-Max key sketches
    uint k_mean4[16];        //  64 B: page K mean centroid (WHT-4bit symmetric, 8 nibbles/uint32)
    uint v_mean4[16];        //  64 B: page V mean centroid (WHT-4bit symmetric, 8 nibbles/uint32)
    half sc_sk[WHT2_NSUB];   //  16 B: RMS scales for the 8 sub-block sketches
    half sc_kmean;           //   2 B: max/7 scale for k_mean4
    half sc_vmean;           //   2 B: max/7 scale for v_mean4
    uint count;              //   4 B: active token count in page (<= page_toks)
    uint page_id;            //   4 B: hot/logical page id
    uint pad0;               //   4 B: 16-byte alignment (sizeof = 416 B)
};

struct PageSumArgs {
    uint pool_cap, slot_idx, page_toks, kv_bits;
    uint n_pages, pid_list_off, spid_list_off, cnt_list_off;
    uint sum_pool_cap, pad0, pad1, pad2;
};

// Legacy Quest summarizer: Grid: (n_pages, NKV, n_slots_or_1), TG = 128 (1 dim per thread).
kernel void k_page_summarize_kv(device const uint * KV_COMP [[buffer(3)]], constant PageSumArgs & a [[buffer(5)]],
                                device const half * KV_SCALES [[buffer(6)]], device const uint * META [[buffer(7)]],
                                device KvPageSummary * SUM [[buffer(8)]],
                                uint3 tgp [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[32];
    const uint i = tgp.x, g = tgp.y, s = a.slot_idx + tgp.z;
    if (i >= a.n_pages) return;
    const uint pid = META[a.pid_list_off + i];
    const uint spid = META[a.spid_list_off + i];
    const uint cnt = (a.cnt_list_off != 0xFFFFFFFFu) ? min(META[a.cnt_list_off + i], a.page_toks) : a.page_toks;
    const uint eff_cnt = max(cnt, 1u);

    const uint kb = kv_k_bits(a.kv_bits);
    const uint vb = kv_v_bits(a.kv_bits);
    const ulong kw = kv_words_per_vec(kb);
    const ulong vw = kv_words_per_vec(vb);
    const ulong k_vbase = comp_vec_idx(pid, a.pool_cap, s, g, 0u, 0u, a.page_toks);
    const ulong v_vbase = comp_vec_idx(pid, a.pool_cap, s, g, 1u, 0u, a.page_toks);
    const ulong k_wbase = comp_word_off(pid, a.pool_cap, s, g, 0u, 0u, a.page_toks, a.kv_bits);
    const ulong v_wbase = comp_word_off(pid, a.pool_cap, s, g, 1u, 0u, a.page_toks, a.kv_bits);

    float kmin = 1e30f, kmax = -1e30f, ksum = 0.0f, vsum = 0.0f;
    for (uint t = 0u; t < eff_cnt; t++) {
        const float kv = unpack_wht_elem(KV_COMP, KV_SCALES, kb, k_wbase + (ulong) t * kw, k_vbase + (ulong) t, tid);
        const float vv = unpack_wht_elem(KV_COMP, KV_SCALES, vb, v_wbase + (ulong) t * vw, v_vbase + (ulong) t, tid);
        kmin = min(kmin, kv);
        kmax = max(kmax, kv);
        ksum += kv;
        vsum += vv;
    }
    const float inv_c = 1.0f / float(eff_cnt);
    const float kmean = ksum * inv_c;
    const float vmean = vsum * inv_c;

    const float sc_kmin  = max(tg_max(fabs(kmin),  sh, sg, lane, 4) / 127.0f, 1e-7f);
    const float sc_kmax  = max(tg_max(fabs(kmax),  sh, sg, lane, 4) / 127.0f, 1e-7f);
    const float sc_kmean = max(tg_max(fabs(kmean), sh, sg, lane, 4) / 127.0f, 1e-7f);
    const float sc_vmean = max(tg_max(fabs(vmean), sh, sg, lane, 4) / 127.0f, 1e-7f);

    device KvPageSummary * dst = SUM + ((ulong) s * (ulong) a.sum_pool_cap + (ulong) spid) * NKV + g;
    dst->k_min[tid]  = char(clamp((int) rint(kmin  / sc_kmin),  -127, 127));
    dst->k_max[tid]  = char(clamp((int) rint(kmax  / sc_kmax),  -127, 127));
    dst->k_mean[tid] = char(clamp((int) rint(kmean / sc_kmean), -127, 127));
    dst->v_mean[tid] = char(clamp((int) rint(vmean / sc_vmean), -127, 127));
    if (tid == 0) {
        dst->sc_kmin  = half(sc_kmin);
        dst->sc_kmax  = half(sc_kmax);
        dst->sc_kmean = half(sc_kmean);
        dst->sc_vmean = half(sc_vmean);
        dst->count    = eff_cnt;
        dst->page_id  = pid;
    }
}

// ---- 2C. k_page_summarize_wht2:
// Builds compact KvPageWht2 (416 B: 8 sub-block WHT-2bit key sketches + WHT-4bit k_mean/v_mean centroids).
// Grid: (n_pages, NKV, n_slots_or_1), TG = 128 (1 dim per thread).
kernel void k_page_summarize_wht2(device const uint * KV_COMP [[buffer(3)]], constant PageSumArgs & a [[buffer(5)]],
                                  device const half * KV_SCALES [[buffer(6)]], device const uint * META [[buffer(7)]],
                                  device KvPageWht2 * SUM2 [[buffer(8)]],
                                  uint3 tgp [[threadgroup_position_in_grid]],
                                  ushort tid [[thread_index_in_threadgroup]],
                                  ushort sg [[simdgroup_index_in_threadgroup]],
                                  ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[32];
    threadgroup float red4[4];
    const uint i = tgp.x, g = tgp.y, s = a.slot_idx + tgp.z;
    if (i >= a.n_pages) return;
    const uint pid = META[a.pid_list_off + i];
    const uint spid = META[a.spid_list_off + i];
    const uint cnt = (a.cnt_list_off != 0xFFFFFFFFu) ? min(META[a.cnt_list_off + i], a.page_toks) : a.page_toks;
    const uint eff_cnt = max(cnt, 1u);

    const uint kb = kv_k_bits(a.kv_bits);
    const uint vb = kv_v_bits(a.kv_bits);
    const ulong kw = kv_words_per_vec(kb);
    const ulong vw = kv_words_per_vec(vb);
    const ulong k_vbase = comp_vec_idx(pid, a.pool_cap, s, g, 0u, 0u, a.page_toks);
    const ulong v_vbase = comp_vec_idx(pid, a.pool_cap, s, g, 1u, 0u, a.page_toks);
    const ulong k_wbase = comp_word_off(pid, a.pool_cap, s, g, 0u, 0u, a.page_toks, a.kv_bits);
    const ulong v_wbase = comp_word_off(pid, a.pool_cap, s, g, 1u, 0u, a.page_toks, a.kv_bits);

    device KvPageWht2 * dst = SUM2 + ((ulong) s * (ulong) a.sum_pool_cap + (ulong) spid) * NKV + g;
    const uint sb_len = max(a.page_toks / WHT2_NSUB, 1u); // 16 for page_toks=128, 8 for page_toks=64

    float ksum = 0.0f;
    for (uint sb = 0u; sb < WHT2_NSUB; sb++) {
        const uint t0 = sb * sb_len;
        const uint t1 = min(t0 + sb_len, eff_cnt);
        if (t1 > t0) {
            float sbs = 0.0f;
            for (uint t = t0; t < t1; t++) {
                sbs += unpack_wht_elem(KV_COMP, KV_SCALES, kb, k_wbase + (ulong) t * kw, k_vbase + (ulong) t, tid);
            }
            ksum += sbs;
            const float sb_mean = sbs / float(t1 - t0);
            const float ss = simd_sum(sb_mean * sb_mean);
            if (lane == 0) red4[sg] = ss;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float tot_ss = red4[0] + red4[1] + red4[2] + red4[3];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float rms = max(sqrt(tot_ss * (1.0f / 128.0f)), 1e-7f);
            const float u = sb_mean / rms;
            const uint q = (u >= -0.9815988f ? 1u : 0u) + (u >= 0.0f ? 1u : 0u) + (u >= 0.9815988f ? 1u : 0u);
            uint w = q << ((tid & 15u) * 2u);
            w |= simd_shuffle_xor(w, 1);
            w |= simd_shuffle_xor(w, 2);
            w |= simd_shuffle_xor(w, 4);
            w |= simd_shuffle_xor(w, 8);
            if ((tid & 15u) == 0u) dst->k_sk[sb][tid >> 4] = w;
            if (tid == 0) dst->sc_sk[sb] = half(rms);
        } else {
            if ((tid & 15u) == 0u) dst->k_sk[sb][tid >> 4] = 0u;
            if (tid == 0) dst->sc_sk[sb] = half(0.0h);
        }
    }

    float vsum = 0.0f;
    for (uint t = 0u; t < eff_cnt; t++) {
        vsum += unpack_wht_elem(KV_COMP, KV_SCALES, vb, v_wbase + (ulong) t * vw, v_vbase + (ulong) t, tid);
    }
    const float inv_c = 1.0f / float(eff_cnt);
    const float kmean = ksum * inv_c;
    const float vmean = vsum * inv_c;

    const float sc_km = max(tg_max(fabs(kmean), sh, sg, lane, 4) / 7.0f, 1e-7f);
    const uint qkm = (uint) (clamp((int) rint(kmean / sc_km), -7, 7) + 8);
    uint wkm = qkm << ((tid & 7u) * 4u);
    wkm |= simd_shuffle_xor(wkm, 1);
    wkm |= simd_shuffle_xor(wkm, 2);
    wkm |= simd_shuffle_xor(wkm, 4);
    if ((tid & 7u) == 0u) dst->k_mean4[tid >> 3] = wkm;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float sc_vm = max(tg_max(fabs(vmean), sh, sg, lane, 4) / 7.0f, 1e-7f);
    const uint qvm = (uint) (clamp((int) rint(vmean / sc_vm), -7, 7) + 8);
    uint wvm = qvm << ((tid & 7u) * 4u);
    wvm |= simd_shuffle_xor(wvm, 1);
    wvm |= simd_shuffle_xor(wvm, 2);
    wvm |= simd_shuffle_xor(wvm, 4);
    if ((tid & 7u) == 0u) dst->v_mean4[tid >> 3] = wvm;

    if (tid == 0) {
        dst->sc_kmean = half(sc_km);
        dst->sc_vmean = half(sc_vm);
        dst->count    = eff_cnt;
        dst->page_id  = pid;
        dst->pad0     = 0u;
    }
}

// ---- 3A. k_score_pages_quest (legacy Quest bounding-box scorer kept for side-by-side comparison):
// Grid: (NKV, B, 1), TG = 256 (8 simdgroups).
struct QuestScoreArgs {
    uint B, slot_idx, sum_pool_cap, max_pages;
    uint top_k, topk_stride, mask_words, use_sum_pt;
    uint qp_off, sink_off;
    uint seq_nfar_off, sum_pt_off, sum_pt_stride, sum_base_off;
    uint scores_off, topk_off, topk_cnt_off, ret_mask_off, head_mask_off, pad0;
};

kernel void k_score_pages_quest(device const float * A [[buffer(2)]], constant QuestScoreArgs & a [[buffer(5)]],
                                device const uint * META [[buffer(7)]],
                                device const KvPageSummary * SUM [[buffer(8)]],
                                device uint * AUX [[buffer(9)]],
                                uint2 tgp [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 qs4[3][HD / 4];
    threadgroup atomic_uint hist[256];
    threadgroup float cv[256];
    threadgroup uint  ci[256];
    threadgroup uint  cut_bin, cut_rem, ncand;

    const uint g = tgp.x, b = tgp.y;
    if (b >= a.B) return;

    const uint bg = b * NKV + g;
    const uint nfar = min(META[a.seq_nfar_off + b], a.max_pages);
    const uint K_ret = min(a.top_k, 128u);

    device uint * rmask = AUX + a.ret_mask_off + (ulong) bg * a.mask_words;
    for (uint i = tid; i < a.mask_words; i += 256u) rmask[i] = 0u;

    const uint hbit = a.slot_idx * NKV + g;
    const bool is_ret = ((AUX[a.head_mask_off + (hbit >> 5)] >> (hbit & 31u)) & 1u) != 0u;

    if (!is_ret || nfar == 0u || K_ret == 0u) {
        if (tid == 0) AUX[a.topk_cnt_off + bg] = 0u;
        return;
    }

    device uint * topk_out = AUX + a.topk_off + (ulong) bg * a.topk_stride;
    if (nfar <= K_ret) {
        for (uint p = tid; p < nfar; p += 256u) {
            topk_out[p] = p;
            atomic_fetch_or_explicit((device atomic_uint *) (rmask + (p >> 5)), 1u << (p & 31u), memory_order_relaxed);
        }
        if (tid == 0) AUX[a.topk_cnt_off + bg] = nfar;
        return;
    }

    device const half * qp_b = (device const half *) (A + a.qp_off) + (ulong) b * (NH * HD);
    if (tid < 128) {
        for (int h = 0; h < 3; h++) {
            ((threadgroup float *) qs4[h])[tid] = float(qp_b[(g * 3u + (uint) h) * HD + tid]);
        }
    }
    float sink[3];
    for (int h = 0; h < 3; h++) sink[h] = A[a.sink_off + (ulong) b * NH + g * 3u + (uint) h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float * sc_buf = (device float *) (AUX + a.scores_off) + (ulong) bg * a.max_pages;
    device const uint * sum_pt = META + a.sum_pt_off + (ulong) b * a.sum_pt_stride;
    const uint sum_base = a.use_sum_pt ? 0u : META[a.sum_base_off + b];
    const ulong slot_sum_base = (ulong) a.slot_idx * (ulong) a.sum_pool_cap;

    for (uint p = sg; p < nfar; p += 8u) {
        const uint spid = a.use_sum_pt ? sum_pt[p] : (sum_base + p);
        device const KvPageSummary * sm = SUM + (slot_sum_base + (ulong) spid) * NKV + g;
        const float sc_mn = float(sm->sc_kmin);
        const float sc_mx = float(sm->sc_kmax);
        const float4 kmn = float4(char4(((device const packed_char4 *) sm->k_min)[lane])) * sc_mn;
        const float4 kmx = float4(char4(((device const packed_char4 *) sm->k_max)[lane])) * sc_mx;
        float best_s = SPARSE_NEG_INF;
        for (int h = 0; h < 3; h++) {
            const float4 qv = qs4[h][lane];
            const float4 ub = max(qv * kmn, qv * kmx);
            const float s = simd_sum(ub.x + ub.y + ub.z + ub.w) - sink[h];
            best_s = max(best_s, s);
        }
        if (lane == 0) sc_buf[p] = best_s;
    }
    threadgroup_barrier(mem_flags::mem_device);

    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < nfar; p += 256u) {
        const uint bin = float_to_ord(sc_buf[p]) >> 24;
        atomic_fetch_add_explicit(&hist[bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint acc = 0u, cb = 0u;
        for (int bi = 255; bi >= 0; bi--) {
            const uint c = atomic_load_explicit(&hist[bi], memory_order_relaxed);
            if (acc + c >= K_ret) { cb = (uint) bi; break; }
            acc += c;
        }
        cut_bin = cb;
        cut_rem = K_ret - acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint cb1 = cut_bin;

    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < nfar; p += 256u) {
        const uint u = float_to_ord(sc_buf[p]);
        if ((u >> 24) == cb1) atomic_fetch_add_explicit(&hist[(u >> 16) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint acc = 0u, cb2 = 0u;
        for (int bi = 255; bi >= 0; bi--) {
            const uint c = atomic_load_explicit(&hist[bi], memory_order_relaxed);
            if (acc + c >= cut_rem) { cb2 = (uint) bi; break; }
            acc += c;
        }
        cut_bin = (cb1 << 8) | cb2;
        ncand = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint thresh16 = cut_bin;
    threadgroup atomic_uint * nc = (threadgroup atomic_uint *) &ncand;

    for (uint p = tid; p < nfar; p += 256u) {
        const float v = sc_buf[p];
        if ((float_to_ord(v) >> 16) > thresh16) {
            const uint pos_c = atomic_fetch_add_explicit(nc, 1u, memory_order_relaxed);
            if (pos_c < 256u) { cv[pos_c] = v; ci[pos_c] = p; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < nfar; p += 256u) {
        const float v = sc_buf[p];
        if ((float_to_ord(v) >> 16) == thresh16) {
            const uint pos_c = atomic_fetch_add_explicit(nc, 1u, memory_order_relaxed);
            if (pos_c < 256u) { cv[pos_c] = v; ci[pos_c] = p; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = min(ncand, 256u);
    if (tid >= n) { cv[tid] = SPARSE_NEG_INF; ci[tid] = 0x7fffffffu; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2u; k <= 256u; k <<= 1) {
        for (uint j = k >> 1; j > 0u; j >>= 1) {
            const uint ixj = tid ^ j;
            if (ixj > tid) {
                const bool asc = ((tid & k) == 0u);
                const float va = cv[tid], vb = cv[ixj];
                const uint ia = ci[tid], ib = ci[ixj];
                const bool a_better = (va > vb) || (va == vb && ia < ib);
                if (asc ? !a_better : a_better) { cv[tid] = vb; ci[tid] = ib; cv[ixj] = va; ci[ixj] = ia; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint picked = min(K_ret, n);
    if (tid < picked) {
        const uint p = ci[tid];
        topk_out[tid] = p;
        atomic_fetch_or_explicit((device atomic_uint *) (rmask + (p >> 5)), 1u << (p & 31u), memory_order_relaxed);
    }
    if (tid == 0) AUX[a.topk_cnt_off + bg] = picked;
}

// ---- 3B. k_score_pages_wht2:
// Winning WHT-2bit-sketch page scorer (4.8x better KL than Quest, 34 MiB @ 256K for 8x16-tok sub-blocks!).
// Uses exact 32-lane = 8 sub-blocks (sb = lane>>2) x 4 quarters of 32 dims (pp = lane&3):
//   - All 32 lanes read the ENTIRE k_sk[8][8] (256 B) in ONE perfectly coalesced cache-line transaction!
//   - Zero inverse WHT, pure register 2-bit Lloyd-Max LUT dots against Hq in TG memory!
//   - Outputs identical topk_out / topk_cnt / ret_mask format as k_score_pages_quest!
// Grid: (NKV, B, 1), TG = 256 (8 simdgroups).
kernel void k_score_pages_wht2(device const float * A [[buffer(2)]], constant QuestScoreArgs & a [[buffer(5)]],
                               device const uint * META [[buffer(7)]],
                               device const KvPageWht2 * SUM2 [[buffer(8)]],
                               device uint * AUX [[buffer(9)]],
                               uint2 tgp [[threadgroup_position_in_grid]],
                               ushort tid [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 qs4[3][HD / 4];
    threadgroup atomic_uint hist[256];
    threadgroup float cv[256];
    threadgroup uint  ci[256];
    threadgroup uint  cut_bin, cut_rem, ncand;

    const uint g = tgp.x, b = tgp.y;
    if (b >= a.B) return;

    const uint bg = b * NKV + g;
    const uint nfar = min(META[a.seq_nfar_off + b], a.max_pages);
    const uint K_ret = min(a.top_k, 128u);

    device uint * rmask = AUX + a.ret_mask_off + (ulong) bg * a.mask_words;
    for (uint i = tid; i < a.mask_words; i += 256u) rmask[i] = 0u;

    const uint hbit = a.slot_idx * NKV + g;
    const bool is_ret = ((AUX[a.head_mask_off + (hbit >> 5)] >> (hbit & 31u)) & 1u) != 0u;

    if (!is_ret || nfar == 0u || K_ret == 0u) {
        if (tid == 0) AUX[a.topk_cnt_off + bg] = 0u;
        return;
    }

    device uint * topk_out = AUX + a.topk_off + (ulong) bg * a.topk_stride;
    if (nfar <= K_ret) {
        for (uint p = tid; p < nfar; p += 256u) {
            topk_out[p] = p;
            atomic_fetch_or_explicit((device atomic_uint *) (rmask + (p >> 5)), 1u << (p & 31u), memory_order_relaxed);
        }
        if (tid == 0) AUX[a.topk_cnt_off + bg] = nfar;
        return;
    }

    device const half * qp_b = (device const half *) (A + a.qp_off) + (ulong) b * (NH * HD);
    if (tid < 128) {
        for (int h = 0; h < 3; h++) {
            ((threadgroup float *) qs4[h])[tid] = float(qp_b[(g * 3u + (uint) h) * HD + tid]);
        }
    }
    float sink[3];
    for (int h = 0; h < 3; h++) sink[h] = A[a.sink_off + (ulong) b * NH + g * 3u + (uint) h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float * sc_buf = (device float *) (AUX + a.scores_off) + (ulong) bg * a.max_pages;
    device const uint * sum_pt = META + a.sum_pt_off + (ulong) b * a.sum_pt_stride;
    const uint sum_base = a.use_sum_pt ? 0u : META[a.sum_base_off + b];
    const ulong slot_sum_base = (ulong) a.slot_idx * (ulong) a.sum_pool_cap;

    const ushort sb = lane >> 2;  // 0..7 sub-block within page
    const ushort pp = lane & 3u;  // 0..3 quarter of 128 dims (32 dims)
    for (uint p = sg; p < nfar; p += 8u) {
        const uint spid = a.use_sum_pt ? sum_pt[p] : (sum_base + p);
        device const KvPageWht2 * sm = SUM2 + (slot_sum_base + (ulong) spid) * NKV + g;
        const float sc = float(sm->sc_sk[sb]);
        const uint2 w2 = *(device const uint2 *) (sm->k_sk[sb] + pp * 2u);
        float acc[3] = {0.0f, 0.0f, 0.0f};
        if (sc > 0.0f) {
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                const uint u = w2[i];
                const float4 k0 = float4(LM2_VAL[u & 3u], LM2_VAL[(u >> 2) & 3u], LM2_VAL[(u >> 4) & 3u], LM2_VAL[(u >> 6) & 3u]);
                const float4 k1 = float4(LM2_VAL[(u >> 8) & 3u], LM2_VAL[(u >> 10) & 3u], LM2_VAL[(u >> 12) & 3u], LM2_VAL[(u >> 14) & 3u]);
                const float4 k2 = float4(LM2_VAL[(u >> 16) & 3u], LM2_VAL[(u >> 18) & 3u], LM2_VAL[(u >> 20) & 3u], LM2_VAL[(u >> 22) & 3u]);
                const float4 k3 = float4(LM2_VAL[(u >> 24) & 3u], LM2_VAL[(u >> 26) & 3u], LM2_VAL[(u >> 28) & 3u], LM2_VAL[(u >> 30) & 3u]);
                for (int h = 0; h < 3; h++) {
                    acc[h] += dot(qs4[h][pp * 8 + i * 4 + 0], k0) + dot(qs4[h][pp * 8 + i * 4 + 1], k1)
                            + dot(qs4[h][pp * 8 + i * 4 + 2], k2) + dot(qs4[h][pp * 8 + i * 4 + 3], k3);
                }
            }
        }
        float best_s = SPARSE_NEG_INF;
        for (int h = 0; h < 3; h++) {
            float s = acc[h];
            s += simd_shuffle_xor(s, 1);
            s += simd_shuffle_xor(s, 2);
            if (sc > 0.0f) best_s = max(best_s, s * sc - sink[h]);
        }
        const float page_s = simd_max(best_s);
        if (lane == 0) sc_buf[p] = page_s;
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Stage 2: Exact Top-K_ret selection via 2-pass 8-bit radix histogram + 256-entry bitonic sort
    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < nfar; p += 256u) {
        const uint bin = float_to_ord(sc_buf[p]) >> 24;
        atomic_fetch_add_explicit(&hist[bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint acc = 0u, cb = 0u;
        for (int bi = 255; bi >= 0; bi--) {
            const uint c = atomic_load_explicit(&hist[bi], memory_order_relaxed);
            if (acc + c >= K_ret) { cb = (uint) bi; break; }
            acc += c;
        }
        cut_bin = cb;
        cut_rem = K_ret - acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint cb1 = cut_bin;

    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < nfar; p += 256u) {
        const uint u = float_to_ord(sc_buf[p]);
        if ((u >> 24) == cb1) atomic_fetch_add_explicit(&hist[(u >> 16) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint acc = 0u, cb2 = 0u;
        for (int bi = 255; bi >= 0; bi--) {
            const uint c = atomic_load_explicit(&hist[bi], memory_order_relaxed);
            if (acc + c >= cut_rem) { cb2 = (uint) bi; break; }
            acc += c;
        }
        cut_bin = (cb1 << 8) | cb2;
        ncand = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint thresh16 = cut_bin;
    threadgroup atomic_uint * nc = (threadgroup atomic_uint *) &ncand;

    for (uint p = tid; p < nfar; p += 256u) {
        const float v = sc_buf[p];
        if ((float_to_ord(v) >> 16) > thresh16) {
            const uint pos_c = atomic_fetch_add_explicit(nc, 1u, memory_order_relaxed);
            if (pos_c < 256u) { cv[pos_c] = v; ci[pos_c] = p; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint p = tid; p < nfar; p += 256u) {
        const float v = sc_buf[p];
        if ((float_to_ord(v) >> 16) == thresh16) {
            const uint pos_c = atomic_fetch_add_explicit(nc, 1u, memory_order_relaxed);
            if (pos_c < 256u) { cv[pos_c] = v; ci[pos_c] = p; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = min(ncand, 256u);
    if (tid >= n) { cv[tid] = SPARSE_NEG_INF; ci[tid] = 0x7fffffffu; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 2u; k <= 256u; k <<= 1) {
        for (uint j = k >> 1; j > 0u; j >>= 1) {
            const uint ixj = tid ^ j;
            if (ixj > tid) {
                const bool asc = ((tid & k) == 0u);
                const float va = cv[tid], vb = cv[ixj];
                const uint ia = ci[tid], ib = ci[ixj];
                const bool a_better = (va > vb) || (va == vb && ia < ib);
                if (asc ? !a_better : a_better) { cv[tid] = vb; ci[tid] = ib; cv[ixj] = va; ci[ixj] = ia; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    const uint picked = min(K_ret, n);
    if (tid < picked) {
        const uint p = ci[tid];
        topk_out[tid] = p;
        atomic_fetch_or_explicit((device atomic_uint *) (rmask + (p >> 5)), 1u << (p & 31u), memory_order_relaxed);
    }
    if (tid == 0) AUX[a.topk_cnt_off + bg] = picked;
}

// ---- 4. k_attn_sparse_wht_b:
// Full 256K+ sparse-retrieval decode attention in ONE fused online-softmax pass:
//   (1) Local window [n_far_pages * page_toks .. pos] with rel_bias,
//   (2) Top-K_ret retrieved far pages from Hot RAM KV4/K3V2/KV3/KV2 pool (seamless centroid fallback on cold miss),
//   (3) Non-retrieved far pages' (k_mean, v_mean, ln(count)) centroids from KvPageWht2 (sum_format==1) or KvPageSummary (sum_format==0),
//   (4) Exact single inverse WHT-128 + Rademacher D per head * sigmoid(gate) -> half out!
// Grid: (NKV, B, 1), TG = 128 (4 simdgroups).
struct SparseWhtAttnArgs {
    uint B, pool_cap, slot_idx, layer_slot;
    uint page_toks, kv_bits, sum_pool_cap, use_sum_pt;
    uint use_centroids, local_pt_stride, far_pt_stride, sum_pt_stride;
    uint topk_stride, mask_words;
    uint qp_off, rp_off, sink_off, gate_off, out_off, relbias_w;
    uint seq_pos_off, seq_nfar_off, local_pt_off, far_pt_off, sum_pt_off, sum_base_off;
    uint topk_off, topk_cnt_off, ret_mask_off, head_mask_off;
    uint sum_format, pad1; // sum_format: 0 = KvPageSummary (528 B), 1 = KvPageWht2 (416 B)
};

kernel void k_attn_sparse_wht_b(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                                device const uint * KV_COMP [[buffer(3)]], constant SparseWhtAttnArgs & a [[buffer(5)]],
                                device const half * KV_SCALES [[buffer(6)]], device const uint * META [[buffer(7)]],
                                device const uchar * SUM_RAW [[buffer(8)]], device const uint * AUX [[buffer(9)]],
                                uint2 tgp [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 qs4[3][HD / 4];
    threadgroup float rs[3][NREL];
    threadgroup float ms_[4][3], ls_[4][3];
    threadgroup float4 os_[4][3][32];
    threadgroup float wht_sh[128];

    const uint g = tgp.x, b = tgp.y;
    if (b >= a.B) return;

    const uint bg = b * NKV + g;
    const uint pos = META[a.seq_pos_off + b];
    const uint nfar = META[a.seq_nfar_off + b];
    const uint local_p0 = nfar * a.page_toks;
    const uint n_local = (pos >= local_p0) ? (pos + 1u - local_p0) : 0u;

    device const half * qp_b = (device const half *) (A + a.qp_off) + (ulong) b * (NH * HD);
    for (int h = 0; h < 3; h++) {
        ((threadgroup float *) qs4[h])[tid] = float(qp_b[(g * 3u + (uint) h) * HD + tid]);
    }
    if (tid < 48) {
        const uint h = tid / 16u, i = tid % 16u, H = g * 3u + h;
        rs[h][i] = A[a.rp_off + (ulong) b * (NH * NREL) + H * NREL + i];
    }
    float sink[3];
    for (int h = 0; h < 3; h++) sink[h] = A[a.sink_off + (ulong) b * NH + g * 3u + (uint) h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float * relb = (device const float *) (PB + a.relbias_w);
    device const uint * local_pt = META + a.local_pt_off + (ulong) b * a.local_pt_stride;
    device const uint * far_pt   = META + a.far_pt_off   + (ulong) b * a.far_pt_stride;
    device const uint * sum_pt   = META + a.sum_pt_off   + (ulong) b * a.sum_pt_stride;
    const uint sum_base = a.use_sum_pt ? 0u : META[a.sum_base_off + b];
    const ulong slot_sum_base = (ulong) a.slot_idx * (ulong) a.sum_pool_cap;
    device const KvPageSummary * SUM  = (device const KvPageSummary *) SUM_RAW;
    device const KvPageWht2    * SUM2 = (device const KvPageWht2 *) SUM_RAW;

    const uint kb = kv_k_bits(a.kv_bits);
    const uint vb = kv_v_bits(a.kv_bits);
    const ulong kw = kv_words_per_vec(kb);
    const ulong vw = kv_words_per_vec(vb);

    const uint hbit = a.slot_idx * NKV + g;
    const bool is_ret = ((AUX[a.head_mask_off + (hbit >> 5)] >> (hbit & 31u)) & 1u) != 0u;
    const uint topk_cnt = is_ret ? AUX[a.topk_cnt_off + bg] : 0u;
    device const uint * topk_idx = AUX + a.topk_off + (ulong) bg * a.topk_stride;
    device const uint * ret_mask = AUX + a.ret_mask_off + (ulong) bg * a.mask_words;

    const ushort kk = lane / 4, pp = lane % 4;
    float m[3] = {SPARSE_NEG_INF, SPARSE_NEG_INF, SPARSE_NEG_INF};
    float l[3] = {0.0f, 0.0f, 0.0f};
    float4 o[3] = {0.0f, 0.0f, 0.0f};
    float4 rl[3];
    for (int h = 0; h < 3; h++) rl[h] = *(threadgroup const float4 *) (&rs[h][pp * 4]);

    // ---- Pass 1: Local Window [local_p0 .. pos] with relative bias ----
    for (uint jb = sg * 8u; jb < n_local; jb += 32u) {
        const uint j = jb + kk;
        const bool valid = j < n_local;
        const uint p = local_p0 + min(j, max(n_local, 1u) - 1u);
        const uint lblk = (p - local_p0) / a.page_toks;
        const uint boff = p % a.page_toks;
        const uint pid = local_pt[lblk];
        float acc[3] = {0.0f, 0.0f, 0.0f};
        if (valid) {
            const ulong kv_k = comp_vec_idx(pid, a.pool_cap, a.slot_idx, g, 0u, boff, a.page_toks);
            const ulong kw_k = comp_word_off(pid, a.pool_cap, a.slot_idx, g, 0u, boff, a.page_toks, a.kv_bits);
            dot_wht_k32(KV_COMP, KV_SCALES, kb, kw_k, kv_k, pp, qs4, acc);
            const uint d = pos - p;
            if (d < EXT) {
                const float4 bb = *(device const float4 *) (relb + d * NREL + pp * 4u);
                for (int h = 0; h < 3; h++) acc[h] += dot(rl[h], bb);
            }
        }
        float pr[3];
        for (int h = 0; h < 3; h++) {
            float t = acc[h];
            t += simd_shuffle_xor(t, 1); t += simd_shuffle_xor(t, 2);
            t = valid ? (t - sink[h]) : SPARSE_NEG_INF;
            const float blk_mx = simd_max(t);
            if (blk_mx > SPARSE_NEG_THR) {
                const float mx = max(m[h], blk_mx);
                const float sc = (m[h] > SPARSE_NEG_THR) ? exp(m[h] - mx) : 0.0f;
                pr[h] = valid ? exp(t - mx) : 0.0f;
                l[h] = l[h] * sc + simd_sum(pr[h]) * 0.25f;
                m[h] = mx;
                o[h] *= sc;
            } else {
                pr[h] = 0.0f;
            }
        }
        for (ushort q2 = 0; q2 < 8; q2++) {
            const uint jq = jb + q2;
            if (jq >= n_local) break;
            const uint pq = local_p0 + jq;
            const uint qpid = local_pt[(pq - local_p0) / a.page_toks];
            const uint qbo = pq % a.page_toks;
            const ulong kv_v = comp_vec_idx(qpid, a.pool_cap, a.slot_idx, g, 1u, qbo, a.page_toks);
            const ulong kw_v = comp_word_off(qpid, a.pool_cap, a.slot_idx, g, 1u, qbo, a.page_toks, a.kv_bits);
            const float4 v = load_wht_v4(KV_COMP, KV_SCALES, vb, kw_v, kv_v, lane);
            for (int h = 0; h < 3; h++) o[h] += simd_shuffle(pr[h], q2 * 4) * v;
        }
    }

    // ---- Pass 2: Top-K_ret Retrieved Far Pages (token-level WHT KV4/K3V2/KV3/KV2, NO rel_bias since d >= EXT) ----
    if (is_ret && nfar > 0u && topk_cnt > 0u) {
        for (uint ri = 0u; ri < topk_cnt; ri++) {
            const uint lp = topk_idx[ri];
            const uint pid = far_pt[lp];
            if (pid == 0xFFFFFFFFu) continue; // cold page -> Pass 3 seamlessly falls back to its centroid
            const ulong k_page_vbase = comp_vec_idx(pid, a.pool_cap, a.slot_idx, g, 0u, 0u, a.page_toks);
            const ulong v_page_vbase = comp_vec_idx(pid, a.pool_cap, a.slot_idx, g, 1u, 0u, a.page_toks);
            const ulong k_page_wbase = comp_word_off(pid, a.pool_cap, a.slot_idx, g, 0u, 0u, a.page_toks, a.kv_bits);
            const ulong v_page_wbase = comp_word_off(pid, a.pool_cap, a.slot_idx, g, 1u, 0u, a.page_toks, a.kv_bits);
            for (uint jb = sg * 8u; jb < a.page_toks; jb += 32u) {
                const uint boff = jb + kk;
                const bool valid = boff < a.page_toks;
                float acc[3] = {0.0f, 0.0f, 0.0f};
                if (valid) {
                    dot_wht_k32(KV_COMP, KV_SCALES, kb, k_page_wbase + (ulong) boff * kw, k_page_vbase + (ulong) boff, pp, qs4, acc);
                }
                float pr[3];
                for (int h = 0; h < 3; h++) {
                    float t = acc[h];
                    t += simd_shuffle_xor(t, 1); t += simd_shuffle_xor(t, 2);
                    t = valid ? (t - sink[h]) : SPARSE_NEG_INF;
                    const float blk_mx = simd_max(t);
                    if (blk_mx > SPARSE_NEG_THR) {
                        const float mx = max(m[h], blk_mx);
                        const float sc = (m[h] > SPARSE_NEG_THR) ? exp(m[h] - mx) : 0.0f;
                        pr[h] = valid ? exp(t - mx) : 0.0f;
                        l[h] = l[h] * sc + simd_sum(pr[h]) * 0.25f;
                        m[h] = mx;
                        o[h] *= sc;
                    } else {
                        pr[h] = 0.0f;
                    }
                }
                for (ushort q2 = 0; q2 < 8; q2++) {
                    const uint bo = jb + q2;
                    if (bo >= a.page_toks) break;
                    const float4 v = load_wht_v4(KV_COMP, KV_SCALES, vb, v_page_wbase + (ulong) bo * vw, v_page_vbase + (ulong) bo, lane);
                    for (int h = 0; h < 3; h++) o[h] += simd_shuffle(pr[h], q2 * 4) * v;
                }
            }
        }
    }

    // ---- Pass 3: Non-Retrieved Far Pages' (k_mean, v_mean, ln(count)) Background Centroids ----
    if (a.use_centroids != 0u && nfar > 0u) {
        for (uint lb = sg * 8u; lb < nfar; lb += 32u) {
            const uint lp = lb + kk;
            bool valid = (lp < nfar);
            if (valid && is_ret) {
                const bool in_ret = ((ret_mask[lp >> 5] >> (lp & 31u)) & 1u) != 0u;
                if (in_ret && far_pt[lp] != 0xFFFFFFFFu) valid = false;
            }
            float acc[3] = {0.0f, 0.0f, 0.0f};
            float log_cnt = 0.0f;
            float sc_vm = 0.0f;
            uint spid_l = 0u;
            if (valid) {
                spid_l = a.use_sum_pt ? sum_pt[lp] : (sum_base + lp);
                if (a.sum_format != 0u) {
                    device const KvPageWht2 * sm2 = SUM2 + (slot_sum_base + (ulong) spid_l) * NKV + g;
                    const float sc_km = float(sm2->sc_kmean);
                    const uint4 w4 = *(device const uint4 *) (sm2->k_mean4 + pp * 4u);
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const uint u = w4[i];
                        const float4 k0 = float4(int4(int(u & 0xFu) - 8, int((u >> 4) & 0xFu) - 8, int((u >> 8) & 0xFu) - 8, int((u >> 12) & 0xFu) - 8)) * sc_km;
                        const float4 k1 = float4(int4(int((u >> 16) & 0xFu) - 8, int((u >> 20) & 0xFu) - 8, int((u >> 24) & 0xFu) - 8, int((u >> 28) & 0xFu) - 8)) * sc_km;
                        for (int h = 0; h < 3; h++) {
                            acc[h] += dot(qs4[h][pp * 8 + i * 2 + 0], k0) + dot(qs4[h][pp * 8 + i * 2 + 1], k1);
                        }
                    }
                    log_cnt = log(float(max(sm2->count, 1u)));
                    sc_vm = float(sm2->sc_vmean);
                } else {
                    device const KvPageSummary * sm = SUM + (slot_sum_base + (ulong) spid_l) * NKV + g;
                    const float sc_km = float(sm->sc_kmean);
                    device const packed_char4 * km = (device const packed_char4 *) (sm->k_mean + pp * 32u);
                    for (int c = 0; c < 8; c++) {
                        const float4 kf = float4(char4(km[c])) * sc_km;
                        for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
                    }
                    log_cnt = log(float(max(sm->count, 1u)));
                    sc_vm = float(sm->sc_vmean);
                }
            }
            float pr[3];
            for (int h = 0; h < 3; h++) {
                float t = acc[h];
                t += simd_shuffle_xor(t, 1); t += simd_shuffle_xor(t, 2);
                t = valid ? (t + log_cnt - sink[h]) : SPARSE_NEG_INF;
                const float blk_mx = simd_max(t);
                if (blk_mx > SPARSE_NEG_THR) {
                    const float mx = max(m[h], blk_mx);
                    const float sc = (m[h] > SPARSE_NEG_THR) ? exp(m[h] - mx) : 0.0f;
                    const float p_unscaled = valid ? exp(t - mx) : 0.0f;
                    pr[h] = p_unscaled * sc_vm;
                    l[h] = l[h] * sc + simd_sum(p_unscaled) * 0.25f;
                    m[h] = mx;
                    o[h] *= sc;
                } else {
                    pr[h] = 0.0f;
                }
            }
            const uint vflag = valid ? 1u : 0u;
            for (ushort q2 = 0; q2 < 8; q2++) {
                if (simd_shuffle(vflag, q2 * 4) == 0u) continue;
                const uint qspid = simd_shuffle(spid_l, q2 * 4);
                float4 v;
                if (a.sum_format != 0u) {
                    device const KvPageWht2 * sm2 = SUM2 + (slot_sum_base + (ulong) qspid) * NKV + g;
                    const ushort u = ((device const ushort *) sm2->v_mean4)[lane];
                    v = float4(int4(int(u & 0xFu) - 8, int((u >> 4) & 0xFu) - 8, int((u >> 8) & 0xFu) - 8, int((u >> 12) & 0xFu) - 8));
                } else {
                    device const KvPageSummary * sm = SUM + (slot_sum_base + (ulong) qspid) * NKV + g;
                    v = float4(char4(((device const packed_char4 *) sm->v_mean)[lane]));
                }
                for (int h = 0; h < 3; h++) o[h] += simd_shuffle(pr[h], q2 * 4) * v;
            }
        }
    }

    // ---- Pass 4: Cross-simdgroup merge + SINGLE Exact Inverse WHT-128 + Rademacher D + Sigmoid Gate ----
    if (lane == 0) {
        for (int h = 0; h < 3; h++) { ms_[sg][h] = m[h]; ls_[sg][h] = l[h]; }
    }
    for (int h = 0; h < 3; h++) os_[sg][h][lane] = o[h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half * out = (device half *) (A + a.out_off);
    device const half * gate = (device const half *) (A + a.gate_off);
    const float sign = rademacher_sign(a.layer_slot, g, tid);
    for (int h = 0; h < 3; h++) {
        const uint H = g * 3u + (uint) h;
        float M = 0.0f; // sink token relative logit = 0.0f
        for (int s2 = 0; s2 < 4; s2++) if (ms_[s2][h] > SPARSE_NEG_THR) M = max(M, ms_[s2][h]);
        float L = exp(-M), O = 0.0f;
        for (int s2 = 0; s2 < 4; s2++) {
            if (ms_[s2][h] > SPARSE_NEG_THR) {
                const float w = exp(ms_[s2][h] - M);
                L += w * ls_[s2][h];
                O += w * ((threadgroup const float *) os_[s2][h])[tid];
            }
        }
        const float o_rot = O / L;
        // Exact inverse transform: D * H_128 * (H_128 * D * sum p_j v_j) = sum p_j v_j
        const float o_unrot = sign * wht128_tg(o_rot, tid, sg, lane, wht_sh);
        const ulong oi = (ulong) b * (NH * HD) + H * HD + tid;
        out[oi] = half(o_unrot * sigm(float(gate[oi])));
    }
}

// ---- 5. k_attn_wht_chunk:
// Causal chunked prefill attention for a single sequence over WHT-compressed pages (KV4/K3V2/KV3/KV2).
// Supports BOTH KvPageWht2 (sum_format==1) and legacy KvPageSummary (sum_format==0) far centroids.
// Grid: (NKV, T, 1), TG = 128.
struct ChunkWhtAttnArgs {
    uint T, p0, pool_cap, slot_idx, layer_slot;
    uint page_toks, kv_bits, sum_pool_cap, use_sum_pt;
    uint qp_off, rp_off, sink_off, gate_off, out_off, relbias_w;
    uint local_p0_page, local_pt_off, sum_pt_off, sum_base, use_centroids;
    uint sum_format, pad0, pad1, pad2;
};

kernel void k_attn_wht_chunk(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                             device const uint * KV_COMP [[buffer(3)]], constant ChunkWhtAttnArgs & a [[buffer(5)]],
                             device const half * KV_SCALES [[buffer(6)]], device const uint * META [[buffer(7)]],
                             device const uchar * SUM_RAW [[buffer(8)]],
                             uint2 tgp [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]],
                             ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 qs4[3][HD / 4];
    threadgroup float rs[3][NREL];
    threadgroup float ms_[4][3], ls_[4][3];
    threadgroup float4 os_[4][3][32];
    threadgroup float wht_sh[128];

    const uint g = tgp.x, t = tgp.y;
    if (t >= a.T) return;
    const uint pos = a.p0 + t;
    const uint local_p0 = a.local_p0_page * a.page_toks;
    const uint nfar = a.local_p0_page;
    const uint n_local = (pos >= local_p0) ? (pos + 1u - local_p0) : 0u;

    device const half * qp_t = (device const half *) (A + a.qp_off) + (ulong) t * (NH * HD);
    for (int h = 0; h < 3; h++) {
        ((threadgroup float *) qs4[h])[tid] = float(qp_t[(g * 3u + (uint) h) * HD + tid]);
    }
    if (tid < 48) {
        const uint h = tid / 16u, i = tid % 16u, H = g * 3u + h;
        rs[h][i] = A[a.rp_off + (ulong) t * (NH * NREL) + H * NREL + i];
    }
    float sink[3];
    for (int h = 0; h < 3; h++) sink[h] = A[a.sink_off + (ulong) t * NH + g * 3u + (uint) h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float * relb = (device const float *) (PB + a.relbias_w);
    device const uint * local_pt = META + a.local_pt_off;
    device const uint * sum_pt   = META + a.sum_pt_off;
    const ulong slot_sum_base = (ulong) a.slot_idx * (ulong) a.sum_pool_cap;
    device const KvPageSummary * SUM  = (device const KvPageSummary *) SUM_RAW;
    device const KvPageWht2    * SUM2 = (device const KvPageWht2 *) SUM_RAW;

    const uint kb = kv_k_bits(a.kv_bits);
    const uint vb = kv_v_bits(a.kv_bits);

    const ushort kk = lane / 4, pp = lane % 4;
    float m[3] = {SPARSE_NEG_INF, SPARSE_NEG_INF, SPARSE_NEG_INF};
    float l[3] = {0.0f, 0.0f, 0.0f};
    float4 o[3] = {0.0f, 0.0f, 0.0f};
    float4 rl[3];
    for (int h = 0; h < 3; h++) rl[h] = *(threadgroup const float4 *) (&rs[h][pp * 4]);

    // Causal local window + current chunk [local_p0 .. pos]
    for (uint jb = sg * 8u; jb < n_local; jb += 32u) {
        const uint j = jb + kk;
        const bool valid = j < n_local;
        const uint p = local_p0 + min(j, max(n_local, 1u) - 1u);
        const uint lblk = (p - local_p0) / a.page_toks;
        const uint boff = p % a.page_toks;
        const uint pid = local_pt[lblk];
        float acc[3] = {0.0f, 0.0f, 0.0f};
        if (valid) {
            const ulong kv_k = comp_vec_idx(pid, a.pool_cap, a.slot_idx, g, 0u, boff, a.page_toks);
            const ulong kw_k = comp_word_off(pid, a.pool_cap, a.slot_idx, g, 0u, boff, a.page_toks, a.kv_bits);
            dot_wht_k32(KV_COMP, KV_SCALES, kb, kw_k, kv_k, pp, qs4, acc);
            const uint d = pos - p;
            if (d < EXT) {
                const float4 bb = *(device const float4 *) (relb + d * NREL + pp * 4u);
                for (int h = 0; h < 3; h++) acc[h] += dot(rl[h], bb);
            }
        }
        float pr[3];
        for (int h = 0; h < 3; h++) {
            float scv = acc[h];
            scv += simd_shuffle_xor(scv, 1); scv += simd_shuffle_xor(scv, 2);
            scv = valid ? (scv - sink[h]) : SPARSE_NEG_INF;
            const float blk_mx = simd_max(scv);
            if (blk_mx > SPARSE_NEG_THR) {
                const float mx = max(m[h], blk_mx);
                const float sc = (m[h] > SPARSE_NEG_THR) ? exp(m[h] - mx) : 0.0f;
                pr[h] = valid ? exp(scv - mx) : 0.0f;
                l[h] = l[h] * sc + simd_sum(pr[h]) * 0.25f;
                m[h] = mx;
                o[h] *= sc;
            } else {
                pr[h] = 0.0f;
            }
        }
        for (ushort q2 = 0; q2 < 8; q2++) {
            const uint jq = jb + q2;
            if (jq >= n_local) break;
            const uint pq = local_p0 + jq;
            const uint qpid = local_pt[(pq - local_p0) / a.page_toks];
            const uint qbo = pq % a.page_toks;
            const ulong kv_v = comp_vec_idx(qpid, a.pool_cap, a.slot_idx, g, 1u, qbo, a.page_toks);
            const ulong kw_v = comp_word_off(qpid, a.pool_cap, a.slot_idx, g, 1u, qbo, a.page_toks, a.kv_bits);
            const float4 v = load_wht_v4(KV_COMP, KV_SCALES, vb, kw_v, kv_v, lane);
            for (int h = 0; h < 3; h++) o[h] += simd_shuffle(pr[h], q2 * 4) * v;
        }
    }

    // Far page centroids [0 .. nfar) (Full layers only)
    if (a.use_centroids != 0u) for (uint lb = sg * 8u; lb < nfar; lb += 32u) {
        const uint lp = lb + kk;
        const bool valid = lp < nfar;
        float acc[3] = {0.0f, 0.0f, 0.0f};
        float log_cnt = 0.0f, sc_vm = 0.0f;
        uint spid_l = 0u;
        if (valid) {
            spid_l = a.use_sum_pt ? sum_pt[lp] : (a.sum_base + lp);
            if (a.sum_format != 0u) {
                device const KvPageWht2 * sm2 = SUM2 + (slot_sum_base + (ulong) spid_l) * NKV + g;
                const float sc_km = float(sm2->sc_kmean);
                const uint4 w4 = *(device const uint4 *) (sm2->k_mean4 + pp * 4u);
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const uint u = w4[i];
                    const float4 k0 = float4(int4(int(u & 0xFu) - 8, int((u >> 4) & 0xFu) - 8, int((u >> 8) & 0xFu) - 8, int((u >> 12) & 0xFu) - 8)) * sc_km;
                    const float4 k1 = float4(int4(int((u >> 16) & 0xFu) - 8, int((u >> 20) & 0xFu) - 8, int((u >> 24) & 0xFu) - 8, int((u >> 28) & 0xFu) - 8)) * sc_km;
                    for (int h = 0; h < 3; h++) {
                        acc[h] += dot(qs4[h][pp * 8 + i * 2 + 0], k0) + dot(qs4[h][pp * 8 + i * 2 + 1], k1);
                    }
                }
                log_cnt = log(float(max(sm2->count, 1u)));
                sc_vm = float(sm2->sc_vmean);
            } else {
                device const KvPageSummary * sm = SUM + (slot_sum_base + (ulong) spid_l) * NKV + g;
                const float sc_km = float(sm->sc_kmean);
                device const packed_char4 * km = (device const packed_char4 *) (sm->k_mean + pp * 32u);
                for (int c = 0; c < 8; c++) {
                    const float4 kf = float4(char4(km[c])) * sc_km;
                    for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
                }
                log_cnt = log(float(max(sm->count, 1u)));
                sc_vm = float(sm->sc_vmean);
            }
        }
        float pr[3];
        for (int h = 0; h < 3; h++) {
            float scv = acc[h];
            scv += simd_shuffle_xor(scv, 1); scv += simd_shuffle_xor(scv, 2);
            scv = valid ? (scv + log_cnt - sink[h]) : SPARSE_NEG_INF;
            const float blk_mx = simd_max(scv);
            if (blk_mx > SPARSE_NEG_THR) {
                const float mx = max(m[h], blk_mx);
                const float sc = (m[h] > SPARSE_NEG_THR) ? exp(m[h] - mx) : 0.0f;
                const float p_unscaled = valid ? exp(scv - mx) : 0.0f;
                pr[h] = p_unscaled * sc_vm;
                l[h] = l[h] * sc + simd_sum(p_unscaled) * 0.25f;
                m[h] = mx;
                o[h] *= sc;
            } else {
                pr[h] = 0.0f;
            }
        }
        const uint vflag = valid ? 1u : 0u;
        for (ushort q2 = 0; q2 < 8; q2++) {
            if (simd_shuffle(vflag, q2 * 4) == 0u) continue;
            const uint qspid = simd_shuffle(spid_l, q2 * 4);
            float4 v;
            if (a.sum_format != 0u) {
                device const KvPageWht2 * sm2 = SUM2 + (slot_sum_base + (ulong) qspid) * NKV + g;
                const ushort u = ((device const ushort *) sm2->v_mean4)[lane];
                v = float4(int4(int(u & 0xFu) - 8, int((u >> 4) & 0xFu) - 8, int((u >> 8) & 0xFu) - 8, int((u >> 12) & 0xFu) - 8));
            } else {
                device const KvPageSummary * sm = SUM + (slot_sum_base + (ulong) qspid) * NKV + g;
                v = float4(char4(((device const packed_char4 *) sm->v_mean)[lane]));
            }
            for (int h = 0; h < 3; h++) o[h] += simd_shuffle(pr[h], q2 * 4) * v;
        }
    }

    if (lane == 0) {
        for (int h = 0; h < 3; h++) { ms_[sg][h] = m[h]; ls_[sg][h] = l[h]; }
    }
    for (int h = 0; h < 3; h++) os_[sg][h][lane] = o[h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half * out = (device half *) (A + a.out_off);
    device const half * gate = (device const half *) (A + a.gate_off);
    const float sign = rademacher_sign(a.layer_slot, g, tid);
    for (int h = 0; h < 3; h++) {
        const uint H = g * 3u + (uint) h;
        float M = 0.0f;
        for (int s2 = 0; s2 < 4; s2++) if (ms_[s2][h] > SPARSE_NEG_THR) M = max(M, ms_[s2][h]);
        float L = exp(-M), O = 0.0f;
        for (int s2 = 0; s2 < 4; s2++) {
            if (ms_[s2][h] > SPARSE_NEG_THR) {
                const float w = exp(ms_[s2][h] - M);
                L += w * ls_[s2][h];
                O += w * ((threadgroup const float *) os_[s2][h])[tid];
            }
        }
        const float o_unrot = sign * wht128_tg(O / L, tid, sg, lane, wht_sh);
        const ulong oi = (ulong) t * (NH * HD) + H * HD + tid;
        out[oi] = half(o_unrot * sigm(float(gate[oi])));
    }
}

// ============================================================================================================
// PART 4: Tiered SWA Outer-Ring (256..1024) WHT-K2V2 Compression Kernels
//   - Inner ring:  9 blocks x 32 tok (257..288 newest tokens) in exact FP16 (45 MiB) or int8+half (22.9 MiB)
//   - Outer ring: 24 blocks x 32 tok (736..768 older tokens) in WHT-K2V2 (15.9 MiB; empirical KL 0.004-0.008!)
//   - Total SWA per instance: 38.8 MiB (int8 inner) / 60.9 MiB (FP16 inner), down from 165 MiB FP16!
// ============================================================================================================

// ---- 6A. k_swa_compress_outer:
// Compresses n_jobs graduating 32-token blocks from SWA_INNER into SWA_OUTER (WHT-K2V2) across n_slots SWA layers.
// Grid: (n_jobs * KV_BLK, NKV, n_slots), TG = 128.
struct SwaCompressArgs {
    uint n_jobs, inner_cap, outer_cap, slot_start;
    uint inner_fp16, src_pid_off, dst_pid_off, pad0;
};

kernel void k_swa_compress_outer(device const uchar * SWA_INNER [[buffer(3)]], constant SwaCompressArgs & a [[buffer(5)]],
                                 device const half * SWA_INNER_SC [[buffer(6)]], device const uint * META [[buffer(7)]],
                                 device uint * SWA_OUTER_COMP [[buffer(8)]], device half * SWA_OUTER_SC [[buffer(9)]],
                                 uint3 tgp [[threadgroup_position_in_grid]],
                                 ushort tid [[thread_index_in_threadgroup]],
                                 ushort sg [[simdgroup_index_in_threadgroup]],
                                 ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[128];
    threadgroup float red4[4];
    const uint job = tgp.x / KV_BLK;
    const uint boff = tgp.x % KV_BLK;
    const uint g = tgp.y;
    const uint s = a.slot_start + tgp.z;
    if (job >= a.n_jobs) return;

    const uint src_pid = META[a.src_pid_off + job];
    const uint dst_pid = META[a.dst_pid_off + job];
    const ulong in_vk = swa_inner_vec_idx(src_pid, a.inner_cap, s, g, 0u, boff);
    const ulong in_vv = swa_inner_vec_idx(src_pid, a.inner_cap, s, g, 1u, boff);
    float k, v;
    if (a.inner_fp16 != 0u) {
        device const half * sp = (device const half *) SWA_INNER;
        k = float(sp[in_vk * HD + tid]);
        v = float(sp[in_vv * HD + tid]);
    } else {
        device const char * sp = (device const char *) SWA_INNER;
        k = float(sp[in_vk * HD + tid]) * float(SWA_INNER_SC[in_vk]);
        v = float(sp[in_vv * HD + tid]) * float(SWA_INNER_SC[in_vv]);
    }

    const float sign = rademacher_sign(100u + s, g, tid);
    const float k_rot = wht128_tg(sign * k, tid, sg, lane, sh);
    const float v_rot = wht128_tg(sign * v, tid, sg, lane, sh);

    const ulong out_vk = comp_vec_idx(dst_pid, a.outer_cap, s, g, 0u, boff, KV_BLK);
    const ulong out_vv = comp_vec_idx(dst_pid, a.outer_cap, s, g, 1u, boff, KV_BLK);
    pack_wht_vec(k_rot, 2u, out_vk * 8ul, out_vk, tid, sg, lane, red4, SWA_OUTER_COMP, SWA_OUTER_SC);
    pack_wht_vec(v_rot, 2u, out_vv * 8ul, out_vv, tid, sg, lane, red4, SWA_OUTER_COMP, SWA_OUTER_SC);
}

// ---- 6B. k_swa_load_from_engine:
// Initializes a session's Tiered SWA (SWA_INNER + SWA_OUTER WHT-K2V2) directly from Engine's kv_swa ring buffer.
// Grid: (nkeys, NKV, N_SWA), TG = 128.
struct SwaLoadEngineArgs {
    uint src_cap, first_pos, nkeys, inner_p0;
    uint inner_cap, outer_cap, inner_fp16, slot_start;
    uint inner_pt_off, outer_pt_off, pad0, pad1;
};

kernel void k_swa_load_from_engine(device const kv_t * SRC [[buffer(1)]], device uchar * SWA_INNER [[buffer(3)]],
                                   constant SwaLoadEngineArgs & a [[buffer(5)]],
                                   device half * SWA_INNER_SC [[buffer(6)]], device const uint * META [[buffer(7)]],
                                   device uint * SWA_OUTER_COMP [[buffer(8)]], device half * SWA_OUTER_SC [[buffer(9)]],
                                   uint3 tgp [[threadgroup_position_in_grid]],
                                   ushort tid [[thread_index_in_threadgroup]],
                                   ushort sg [[simdgroup_index_in_threadgroup]],
                                   ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sh[128];
    threadgroup float red4[4];
    const uint i = tgp.x, g = tgp.y, s = a.slot_start + tgp.z;
    if (i >= a.nkeys) return;

    const uint p = a.first_pos + i;
    const uint src_idx = p % a.src_cap;
    const ulong src_k = (((ulong) s * 2ul + 0ul) * NKV + g) * (ulong) a.src_cap * HD + (ulong) src_idx * HD;
    const ulong src_v = (((ulong) s * 2ul + 1ul) * NKV + g) * (ulong) a.src_cap * HD + (ulong) src_idx * HD;
#ifdef KV8
    const float k = float(SRC[src_k + tid]) * kv_vscale(SRC + src_k);
    const float v = float(SRC[src_v + tid]) * kv_vscale(SRC + src_v);
#else
    const float k = float(SRC[src_k + tid]);
    const float v = float(SRC[src_v + tid]);
#endif
    const uint blk = p / KV_BLK;
    const uint boff = p % KV_BLK;

    if (p >= a.inner_p0) {
        const uint pid = META[a.inner_pt_off + (blk % SWA_INNER_BLKS)];
        const ulong ivk = swa_inner_vec_idx(pid, a.inner_cap, s, g, 0u, boff);
        const ulong ivv = swa_inner_vec_idx(pid, a.inner_cap, s, g, 1u, boff);
        if (a.inner_fp16 != 0u) {
            device half * dst = (device half *) SWA_INNER;
            dst[ivk * HD + tid] = half(k);
            dst[ivv * HD + tid] = half(v);
        } else {
            const float amk = simd_max(fabs(k));
            if (lane == 0) red4[sg] = amk;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float sck = max(max(max(red4[0], red4[1]), max(red4[2], red4[3])) / 127.0f, 1e-7f);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float amv = simd_max(fabs(v));
            if (lane == 0) red4[sg] = amv;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float scv = max(max(max(red4[0], red4[1]), max(red4[2], red4[3])) / 127.0f, 1e-7f);
            device char * dst = (device char *) SWA_INNER;
            dst[ivk * HD + tid] = char(clamp((int) rint(k / sck), -127, 127));
            dst[ivv * HD + tid] = char(clamp((int) rint(v / scv), -127, 127));
            if (tid == 0) { SWA_INNER_SC[ivk] = half(sck); SWA_INNER_SC[ivv] = half(scv); }
        }
    } else {
        const uint pid = META[a.outer_pt_off + (blk % SWA_OUTER_BLKS)];
        const float sign = rademacher_sign(100u + s, g, tid);
        const float k_rot = wht128_tg(sign * k, tid, sg, lane, sh);
        const float v_rot = wht128_tg(sign * v, tid, sg, lane, sh);
        const ulong ovk = comp_vec_idx(pid, a.outer_cap, s, g, 0u, boff, KV_BLK);
        const ulong ovv = comp_vec_idx(pid, a.outer_cap, s, g, 1u, boff, KV_BLK);
        pack_wht_vec(k_rot, 2u, ovk * 8ul, ovk, tid, sg, lane, red4, SWA_OUTER_COMP, SWA_OUTER_SC);
        pack_wht_vec(v_rot, 2u, ovv * 8ul, ovv, tid, sg, lane, red4, SWA_OUTER_COMP, SWA_OUTER_SC);
    }
}

// ---- 6C. k_attn_swa_tiered_b:
// Single-pass fused SWA decode attention over exact inner-256 ring (FP16 or int8+half) + WHT-K2V2 outer-768 ring!
// Exact rel_bias on BOTH rings, single inverse WHT-128 + Rademacher D on outer partial before exact online-softmax
// merge with inner partial + attention sink + sigmoid gate!
// Grid: (NKV, B, 1), TG = 128 (4 simdgroups).
struct TieredSwaAttnArgs {
    uint B, inner_cap, outer_cap, slot_idx;
    uint inner_fp16, layer_slot, max_batch, pad0;
    uint seq_pos_off, seq_cs_off, inner_pt_off, outer_pt_off;
    uint q_off, k_off, v_off, r_off;
    uint sink_off, gate_off, out_off, pad1;
    uint kconv_t_w, vconv_t_w, qclip_w, relclip_w;
    uint relbias_w, pad2, pad3, pad4;
};

kernel void k_attn_swa_tiered_b(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                                device uchar * SWA_INNER [[buffer(3)]], device half * CS [[buffer(4)]],
                                constant TieredSwaAttnArgs & a [[buffer(5)]],
                                device half * SWA_INNER_SC [[buffer(6)]], device const uint * META [[buffer(7)]],
                                device const uint * SWA_OUTER_COMP [[buffer(8)]],
                                device const half * SWA_OUTER_SC [[buffer(9)]],
                                uint2 tgp [[threadgroup_position_in_grid]],
                                ushort tid [[thread_index_in_threadgroup]],
                                ushort sg [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 qs4[3][HD / 4];
    threadgroup float rs[3][NREL];
    threadgroup float kc[HD], vc[HD];
    threadgroup float red[4][8];
    threadgroup float ms_[4][3], ls_[4][3];
    threadgroup float4 os_[4][3][32];
    threadgroup float o_in[3][HD];
    threadgroup float m_in_tot[3], l_in_tot[3];
    threadgroup float wht_sh[128];

    const uint g = tgp.x, b = tgp.y;
    if (b >= a.B) return;

    const uint pos = META[a.seq_pos_off + b];
    const uint seq_cs = META[a.seq_cs_off + b];
    const uint nkeys = min(pos + 1u, (uint) EXT);
    const uint first_pos = pos + 1u - nkeys;
    const uint cur_blk = pos / KV_BLK;
    const uint inner_start_blk = (cur_blk >= (SWA_INNER_BLKS - 1u)) ? (cur_blk - (SWA_INNER_BLKS - 1u)) : 0u;
    const uint inner_p0 = inner_start_blk * KV_BLK;
    const uint n_inner = pos + 1u - inner_p0;
    const uint n_outer = (inner_p0 > first_pos) ? (inner_p0 - first_pos) : 0u;

    device const uint * inner_pt = META + a.inner_pt_off + (ulong) b * SWA_INNER_BLKS;
    device const uint * outer_pt = META + a.outer_pt_off + (ulong) b * SWA_OUTER_BLKS;

    device const float * qclip = (device const float *) (PB + a.qclip_w);
    float q[3];
    for (int h = 0; h < 3; h++) {
        const uint H = g * 3u + (uint) h;
        const half qh = ((device const half *) (A + a.q_off))[(ulong) b * (NH * HD) + H * HD + tid];
        q[h] = float(qh) * qclip[H * HD + tid];
    }

    // Fused K/V short-conv with FP16 CS_POOL (SWA has static 4-tap conv, tau = 1.0)
    float kx = 0.0f, vx = 0.0f;
    {
        const uint tap0 = pos % 3u;
        const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
        const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
        const uint c = g * HD + tid;
        const uint dim = NKV * HD;
        const float scl = 0.4472135954999579f;
        for (int which = 0; which < 2; which++) {
            device half * hb = CS + cs_branch_base(a.layer_slot, (uint) which, a.max_batch) + (ulong) seq_cs * (3ul * dim);
            device const float * w = (device const float *) (PB + (which == 0 ? a.kconv_t_w : a.vconv_t_w));
            const float x = A[(which == 0 ? a.k_off : a.v_off) + (ulong) b * dim + c];
            const float h0 = float(hb[(ulong) tap0 * dim + c]);
            const float h1 = float(hb[(ulong) tap1 * dim + c]);
            const float h2 = float(hb[(ulong) tap2 * dim + c]);
            const float acc = w[0 * dim + c] * h0 + w[1 * dim + c] * h1 + w[2 * dim + c] * h2 + w[3 * dim + c] * x;
            hb[(ulong) tap0 * dim + c] = half(x);
            if (which == 0) kx = (x + acc) * scl; else vx = (x + acc) * scl;
        }
    }

    float r = 0.0f;
    if (tid < 48) {
        const uint h = tid / 16u, i = tid % 16u, H = g * 3u + h;
        r = A[a.r_off + (ulong) b * (NH * NREL) + H * NREL + i] * ((device const float *) (PB + a.relclip_w))[H * NREL + i];
    }
    float rn = r * r;
    rn += simd_shuffle_xor(rn, 1); rn += simd_shuffle_xor(rn, 2); rn += simd_shuffle_xor(rn, 4); rn += simd_shuffle_xor(rn, 8);
    {
        const float s0 = simd_sum(q[0] * q[0]), s1 = simd_sum(q[1] * q[1]), s2 = simd_sum(q[2] * q[2]);
        const float s3 = simd_sum(kx * kx), s4 = simd_sum(vx * vx);
        const float s5 = simd_max(fabs(kx)), s6 = simd_max(fabs(vx));
        if (lane == 0) { red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; red[sg][3] = s3; red[sg][4] = s4; red[sg][5] = s5; red[sg][6] = s6; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot[5];
    for (int i = 0; i < 5; i++) tot[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];

    for (int h = 0; h < 3; h++) {
        const float f = clamp(QK_CLIP / (sqrt(tot[h]) + 1e-6f), 0.0f, 1.0f) * 0.08838834764831845f;
        ((threadgroup float *) qs4[h])[tid] = q[h] * f;
    }
    if (tid < 48) rs[tid / 16u][tid % 16u] = r * clamp(REL_CLIP / (sqrt(rn) + 1e-6f), 0.0f, 1.0f);

    // Clip/norm & write current token's (K, V) into SWA_INNER ring
    {
        const float kf = clamp(QK_CLIP / (sqrt(tot[3]) + 1e-6f), 0.0f, 1.0f);
        const float vf = rsqrt(tot[4] / float(HD) + 1e-6f);
        const float k = kx * kf, v = vx * vf;
        const uint pid = inner_pt[cur_blk % SWA_INNER_BLKS];
        const uint boff = pos % KV_BLK;
        const ulong ivk = swa_inner_vec_idx(pid, a.inner_cap, a.slot_idx, g, 0u, boff);
        const ulong ivv = swa_inner_vec_idx(pid, a.inner_cap, a.slot_idx, g, 1u, boff);
        if (a.inner_fp16 != 0u) {
            kc[tid] = k; vc[tid] = v;
            device half * dst = (device half *) SWA_INNER;
            dst[ivk * HD + tid] = half(k);
            dst[ivv * HD + tid] = half(v);
        } else {
            const float amk = max(max(red[0][5], red[1][5]), max(red[2][5], red[3][5])) * kf;
            const float amv = max(max(red[0][6], red[1][6]), max(red[2][6], red[3][6])) * vf;
            const float sck = max(amk / 127.0f, 1e-7f);
            const float scv = max(amv / 127.0f, 1e-7f);
            const int qk = clamp((int) rint(k / sck), -127, 127);
            const int qv = clamp((int) rint(v / scv), -127, 127);
            kc[tid] = float(qk) * sck;
            vc[tid] = float(qv) * scv;
            device char * dst = (device char *) SWA_INNER;
            dst[ivk * HD + tid] = (char) qk;
            dst[ivv * HD + tid] = (char) qv;
            if (tid == 0) { SWA_INNER_SC[ivk] = half(sck); SWA_INNER_SC[ivv] = half(scv); }
        }
    }

    float sink[3];
    for (int h = 0; h < 3; h++) sink[h] = A[a.sink_off + (ulong) b * NH + g * 3u + (uint) h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float * relb = (device const float *) (PB + a.relbias_w);
    const ushort kk = lane / 4, pp = lane % 4;
    float4 rl[3];
    for (int h = 0; h < 3; h++) rl[h] = *(threadgroup const float4 *) (&rs[h][pp * 4]);

    // ---- Pass 1: Exact Inner Ring [inner_p0 .. pos] (257..288 newest tokens) ----
    float m[3] = {SPARSE_NEG_INF, SPARSE_NEG_INF, SPARSE_NEG_INF};
    float l[3] = {0.0f, 0.0f, 0.0f};
    float4 o[3] = {0.0f, 0.0f, 0.0f};
    for (uint jb = sg * 8u; jb < n_inner; jb += 32u) {
        const uint j = jb + kk;
        const bool valid = j < n_inner;
        const uint p = inner_p0 + min(j, n_inner - 1u);
        const uint pid = inner_pt[(p / KV_BLK) % SWA_INNER_BLKS];
        const uint boff = p % KV_BLK;
        float acc[3] = {0.0f, 0.0f, 0.0f};
        if (valid) {
            if (p == pos) {
                for (int c = 0; c < 8; c++) {
                    const float4 kf = *(threadgroup const float4 *) (&kc[pp * 32 + c * 4]);
                    for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
                }
            } else {
                const ulong ivk = swa_inner_vec_idx(pid, a.inner_cap, a.slot_idx, g, 0u, boff);
                if (a.inner_fp16 != 0u) {
                    device const half4 * kr = (device const half4 *) (((device const half *) SWA_INNER) + ivk * HD + pp * 32u);
                    for (int c = 0; c < 8; c++) {
                        const float4 kf = float4(kr[c]);
                        for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
                    }
                } else {
                    device const packed_char4 * kr = (device const packed_char4 *) (((device const char *) SWA_INNER) + ivk * HD + pp * 32u);
                    const float ks = float(SWA_INNER_SC[ivk]);
                    for (int c = 0; c < 8; c++) {
                        const float4 kf = float4(char4(kr[c]));
                        for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + c], kf);
                    }
                    for (int h = 0; h < 3; h++) acc[h] *= ks;
                }
            }
            const uint d = pos - p;
            if (d < EXT) {
                const float4 bb = *(device const float4 *) (relb + d * NREL + pp * 4u);
                for (int h = 0; h < 3; h++) acc[h] += dot(rl[h], bb);
            }
        }
        float pr[3];
        for (int h = 0; h < 3; h++) {
            float t = acc[h];
            t += simd_shuffle_xor(t, 1); t += simd_shuffle_xor(t, 2);
            t = valid ? (t - sink[h]) : SPARSE_NEG_INF;
            const float blk_mx = simd_max(t);
            if (blk_mx > SPARSE_NEG_THR) {
                const float mx = max(m[h], blk_mx);
                const float sc = (m[h] > SPARSE_NEG_THR) ? exp(m[h] - mx) : 0.0f;
                pr[h] = valid ? exp(t - mx) : 0.0f;
                l[h] = l[h] * sc + simd_sum(pr[h]) * 0.25f;
                m[h] = mx;
                o[h] *= sc;
            } else {
                pr[h] = 0.0f;
            }
        }
        if (a.inner_fp16 == 0u && valid && p != pos) {
            const ulong ivv = swa_inner_vec_idx(pid, a.inner_cap, a.slot_idx, g, 1u, boff);
            const float vsc = float(SWA_INNER_SC[ivv]);
            for (int h = 0; h < 3; h++) pr[h] *= vsc;
        }
        for (ushort q2 = 0; q2 < 8; q2++) {
            const uint jq = jb + q2;
            if (jq >= n_inner) break;
            const uint pq = inner_p0 + jq;
            float4 v;
            if (pq == pos) {
                v = *(threadgroup const float4 *) (&vc[lane * 4]);
            } else {
                const uint qpid = inner_pt[(pq / KV_BLK) % SWA_INNER_BLKS];
                const ulong ivv = swa_inner_vec_idx(qpid, a.inner_cap, a.slot_idx, g, 1u, pq % KV_BLK);
                if (a.inner_fp16 != 0u) {
                    v = float4(*(device const half4 *) (((device const half *) SWA_INNER) + ivv * HD + lane * 4u));
                } else {
                    v = float4(char4(*(device const packed_char4 *) (((device const char *) SWA_INNER) + ivv * HD + lane * 4u)));
                }
            }
            for (int h = 0; h < 3; h++) o[h] += simd_shuffle(pr[h], q2 * 4) * v;
        }
    }

    // Merge 4 simdgroups' inner partials into (m_in_tot, l_in_tot, o_in)
    if (lane == 0) {
        for (int h = 0; h < 3; h++) { ms_[sg][h] = m[h]; ls_[sg][h] = l[h]; }
    }
    for (int h = 0; h < 3; h++) os_[sg][h][lane] = o[h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int h = 0; h < 3; h++) {
        float Mi = SPARSE_NEG_INF;
        for (int s2 = 0; s2 < 4; s2++) if (ms_[s2][h] > SPARSE_NEG_THR) Mi = max(Mi, ms_[s2][h]);
        float Li = 0.0f, Oi = 0.0f;
        if (Mi > SPARSE_NEG_THR) {
            for (int s2 = 0; s2 < 4; s2++) {
                if (ms_[s2][h] > SPARSE_NEG_THR) {
                    const float w = exp(ms_[s2][h] - Mi);
                    Li += w * ls_[s2][h];
                    Oi += w * ((threadgroup const float *) os_[s2][h])[tid];
                }
            }
        }
        o_in[h][tid] = Oi;
        if (tid == 0) { m_in_tot[h] = Mi; l_in_tot[h] = Li; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half * out = (device half *) (A + a.out_off);
    device const half * gate = (device const half *) (A + a.gate_off);

    // Fast path when context <= inner ring (n_outer == 0): ZERO WHT transforms!
    if (n_outer == 0u) {
        for (int h = 0; h < 3; h++) {
            const uint H = g * 3u + (uint) h;
            const float Mi = m_in_tot[h];
            const float M = (Mi > SPARSE_NEG_THR) ? max(0.0f, Mi) : 0.0f;
            const float wi = (Mi > SPARSE_NEG_THR) ? exp(Mi - M) : 0.0f;
            const float L = exp(-M) + wi * l_in_tot[h];
            const float O = wi * o_in[h][tid];
            const ulong oi = (ulong) b * (NH * HD) + H * HD + tid;
            out[oi] = half((O / L) * sigm(float(gate[oi])));
        }
        return;
    }

    // ---- Pass 2: WHT-K2V2 Outer Ring [first_pos .. inner_p0 - 1] (up to 767 older tokens) ----
    // Rotate qs4 IN-PLACE into WHT space once per (g, b)
    const float sign = rademacher_sign(100u + a.slot_idx, g, tid);
    for (int h = 0; h < 3; h++) {
        const float qr = wht128_tg(sign * ((threadgroup const float *) qs4[h])[tid], tid, sg, lane, wht_sh);
        ((threadgroup float *) qs4[h])[tid] = qr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int h = 0; h < 3; h++) { m[h] = SPARSE_NEG_INF; l[h] = 0.0f; o[h] = 0.0f; }
    for (uint jb = sg * 8u; jb < n_outer; jb += 32u) {
        const uint j = jb + kk;
        const bool valid = j < n_outer;
        const uint p = first_pos + min(j, n_outer - 1u);
        const uint pid = outer_pt[(p / KV_BLK) % SWA_OUTER_BLKS];
        const uint boff = p % KV_BLK;
        float acc[3] = {0.0f, 0.0f, 0.0f};
        if (valid) {
            const ulong ovk = comp_vec_idx(pid, a.outer_cap, a.slot_idx, g, 0u, boff, KV_BLK);
            dot_wht_k32(SWA_OUTER_COMP, SWA_OUTER_SC, 2u, ovk * 8ul, ovk, pp, qs4, acc);
            const uint d = pos - p;
            if (d < EXT) {
                const float4 bb = *(device const float4 *) (relb + d * NREL + pp * 4u);
                for (int h = 0; h < 3; h++) acc[h] += dot(rl[h], bb);
            }
        }
        float pr[3];
        for (int h = 0; h < 3; h++) {
            float t = acc[h];
            t += simd_shuffle_xor(t, 1); t += simd_shuffle_xor(t, 2);
            t = valid ? (t - sink[h]) : SPARSE_NEG_INF;
            const float blk_mx = simd_max(t);
            if (blk_mx > SPARSE_NEG_THR) {
                const float mx = max(m[h], blk_mx);
                const float sc = (m[h] > SPARSE_NEG_THR) ? exp(m[h] - mx) : 0.0f;
                pr[h] = valid ? exp(t - mx) : 0.0f;
                l[h] = l[h] * sc + simd_sum(pr[h]) * 0.25f;
                m[h] = mx;
                o[h] *= sc;
            } else {
                pr[h] = 0.0f;
            }
        }
        for (ushort q2 = 0; q2 < 8; q2++) {
            const uint jq = jb + q2;
            if (jq >= n_outer) break;
            const uint pq = first_pos + jq;
            const uint qpid = outer_pt[(pq / KV_BLK) % SWA_OUTER_BLKS];
            const ulong ovv = comp_vec_idx(qpid, a.outer_cap, a.slot_idx, g, 1u, pq % KV_BLK, KV_BLK);
            const float4 v = load_wht_v4(SWA_OUTER_COMP, SWA_OUTER_SC, 2u, ovv * 8ul, ovv, lane);
            for (int h = 0; h < 3; h++) o[h] += simd_shuffle(pr[h], q2 * 4) * v;
        }
    }

    // Merge outer partials, apply SINGLE inverse WHT-128 + Rademacher D per head, merge with inner + sink!
    if (lane == 0) {
        for (int h = 0; h < 3; h++) { ms_[sg][h] = m[h]; ls_[sg][h] = l[h]; }
    }
    for (int h = 0; h < 3; h++) os_[sg][h][lane] = o[h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int h = 0; h < 3; h++) {
        const uint H = g * 3u + (uint) h;
        float Mo = SPARSE_NEG_INF;
        for (int s2 = 0; s2 < 4; s2++) if (ms_[s2][h] > SPARSE_NEG_THR) Mo = max(Mo, ms_[s2][h]);
        float Lo = 0.0f, Oo_rot = 0.0f;
        if (Mo > SPARSE_NEG_THR) {
            for (int s2 = 0; s2 < 4; s2++) {
                if (ms_[s2][h] > SPARSE_NEG_THR) {
                    const float w = exp(ms_[s2][h] - Mo);
                    Lo += w * ls_[s2][h];
                    Oo_rot += w * ((threadgroup const float *) os_[s2][h])[tid];
                }
            }
        }
        // Unconditional WHT inverse (zero barrier divergence!) -> unrotated outer partial
        const float Oo_unrot = sign * wht128_tg(Oo_rot, tid, sg, lane, wht_sh);
        const float Mi = m_in_tot[h];
        float M = 0.0f; // sink at relative logit 0.0
        if (Mi > SPARSE_NEG_THR) M = max(M, Mi);
        if (Mo > SPARSE_NEG_THR) M = max(M, Mo);
        const float wi = (Mi > SPARSE_NEG_THR) ? exp(Mi - M) : 0.0f;
        const float wo = (Mo > SPARSE_NEG_THR) ? exp(Mo - M) : 0.0f;
        const float L = exp(-M) + wi * l_in_tot[h] + wo * Lo;
        const float O = wi * o_in[h][tid] + wo * Oo_unrot;
        const ulong oi = (ulong) b * (NH * HD) + H * HD + tid;
        out[oi] = half((O / L) * sigm(float(gate[oi])));
    }
}
