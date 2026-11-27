// ================================ multi-sequence batched decode & paged KV kernels ================================
// Included after prefill.metal.
// Buffers for batched decode:
//   0 = W (mmapped GGUF weights), 1 = PB (f32 params + bf16 matrices + SoA [4][dim] conv weights),
//   2 = A (batched decode activation arena),
//   3 = KV_POOL (slot-major paged KV pool, 64-bit offsets, [slot_in_kind][cap_pages][2(K,V)][NKV][KV_BLK=32][HD]),
//   4 = CS_POOL (slot-major, branch-separated, ring-indexed SoA conv state:
//       per layer slot s in 0..48 (stride cs_slot_stride = max_batch * 3 * CS_NCH):
//         branch 0 (k_conv):   offset 0,                            [max_batch][3][1024]
//         branch 1 (v_conv):   offset max_batch * 3 * 1024,         [max_batch][3][1024]
//         branch 2 (wo_post):  offset max_batch * 3 * 2048,         [max_batch][3][2304]
//         branch 3 (ffn_post): offset max_batch * 3 * (2048+2304),  [max_batch][3][2304]
//       Ring indexing at position p: oldest tap = p % 3, then (p+1)%3, (p+2)%3; update writes ONLY cs[p % 3] = y!),
//   5 = kernel args (bytes), 6 = TOKS (active batch input/output token ids),
//   7 = META (uint32 metadata buffer: seq_pos, seq_suf, seq_cs, swa_pt, full_pt),
//   8 = PRE_KV (optional shared prefix contiguous KV buffer [n_slots_kind][2][NKV][pre_cap][HD]).

#define KV_BLK 32
#define SWA_RING_BLKS 33
#define CS_NCH (2 * NKV * HD + 2 * E)
#define KV_SLOT_ELEMS (2ul * NKV * KV_BLK * HD)
#define NTC_B 4

inline ulong cs_branch_base(uint slot, uint branch, uint max_batch) {
    const ulong mb = (ulong) max_batch;
    const ulong sbase = (ulong) slot * mb * 3ul * (ulong) CS_NCH;
    if (branch == 0) return sbase;
    if (branch == 1) return sbase + mb * 3ul * (ulong) (NKV * HD);
    if (branch == 2) return sbase + mb * 3ul * (ulong) (2 * NKV * HD);
    return sbase + mb * 3ul * (ulong) (2 * NKV * HD + E);
}

// Slot-major paged KV addressing: [slot_in_kind][pool_cap][2][NKV][32][128]
inline ulong paged_kv_base(uint pid, uint pool_cap, uint slot_idx, uint g, uint is_v) {
    return ((ulong) slot_idx * (ulong) pool_cap + (ulong) pid) * KV_SLOT_ELEMS + (ulong) (is_v * NKV + g) * (KV_BLK * HD);
}

// ---- 4-class MoE router for batched decode:
// class 0: 8-SG tensor (65..128 tokens)
// class 1: 4-SG tensor (3..64 tokens)
// class 2: 2-token direct quantized (r == 2, zero TG staging / barriers)
// class 3: 1-token direct quantized (r == 1, ~290 GB/s)
struct Route4Args {
    uint sel_off, T, perm_off, inv_off, aw_off, ind_off;
    uint gu_y[NTC_B];
    uint dn_y[NTC_B];
    uint tiles_off[NTC_B];
};

kernel void k_moe_route4(device float * A [[buffer(2)]], constant Route4Args & a [[buffer(5)]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint cnt[NEXP];
    threadgroup uint off[NEXP], toff[NTC_B][NEXP];
    threadgroup atomic_uint fill[NEXP];
    if (tid < NEXP) {
        atomic_store_explicit(&cnt[tid], 0, memory_order_relaxed);
        atomic_store_explicit(&fill[tid], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = a.T * TOPK;
    for (uint i = tid; i < n; i += 1024) {
        const uint e = (uint) A[a.sel_off + (i / TOPK) * 16 + (i % TOPK)];
        atomic_fetch_add_explicit(&cnt[e], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint s = 0, ts[NTC_B] = {0, 0, 0, 0};
        for (int e = 0; e < NEXP; e++) {
            const uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed), r = c % 128;
            off[e] = s; s += c;
            for (int k = 0; k < NTC_B; k++) toff[k][e] = ts[k];
            ts[0] += c / 128 + (r > 64 ? 1 : 0);
            ts[1] += (r >= 3 && r <= 64) ? 1 : 0;
            ts[2] += (r == 2) ? 1 : 0;
            ts[3] += (r == 1) ? 1 : 0;
        }
        device uint * ind = (device uint *) (A + a.ind_off);
        for (int c = 0; c < NTC_B; c++) {
            ind[c * 4 + 0] = ts[c]; ind[c * 4 + 1] = a.gu_y[c]; ind[c * 4 + 2] = 1; ind[c * 4 + 3] = ts[c];
            ind[(NTC_B + c) * 4 + 0] = ts[c]; ind[(NTC_B + c) * 4 + 1] = a.dn_y[c]; ind[(NTC_B + c) * 4 + 2] = 1; ind[(NTC_B + c) * 4 + 3] = ts[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint * perm = (device uint *) (A + a.perm_off);
    device uint * inv = (device uint *) (A + a.inv_off);
    for (uint i = tid; i < n; i += 1024) {
        const uint t = i / TOPK, k = i % TOPK;
        const uint e = (uint) A[a.sel_off + t * 16 + k];
        const uint slot = off[e] + atomic_fetch_add_explicit(&fill[e], 1, memory_order_relaxed);
        perm[slot] = t;
        inv[i] = slot;
        A[a.aw_off + slot] = A[a.sel_off + t * 16 + TOPK + k];
    }
    if (tid < NEXP) {
        const uint c = atomic_load_explicit(&cnt[tid], memory_order_relaxed), n128 = c / 128, r = c % 128;
        device uint * t8 = (device uint *) (A + a.tiles_off[0]) + toff[0][tid] * 4;
        for (uint j = 0; j < n128; j++) { t8[j * 4] = tid; t8[j * 4 + 1] = off[tid] + j * 128; t8[j * 4 + 2] = 128; t8[j * 4 + 3] = 0; }
        if (r > 0) {
            device uint * tt;
            if (r > 64) tt = t8 + n128 * 4;
            else if (r >= 3) tt = (device uint *) (A + a.tiles_off[1]) + toff[1][tid] * 4;
            else if (r == 2) tt = (device uint *) (A + a.tiles_off[2]) + toff[2][tid] * 4;
            else tt = (device uint *) (A + a.tiles_off[3]) + toff[3][tid] * 4;
            tt[0] = tid; tt[1] = off[tid] + n128 * 128; tt[2] = r; tt[3] = 0;
        }
    }
}

// ---- Direct quantized MoE gate_up + SiTU * routing_weight -> half acth for NT = 1 or 2 tokens per tile.
template <int NT>
kernel void k_moe_dir_gu(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant MoeGemmHArgs & a [[buffer(5)]],
                         uint2 tgq [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    if (tgq.x >= ((device const uint *) (A + a.ntiles_off))[0]) return;
    device const uint * tl = (device const uint *) (A + a.tiles_off) + tgq.x * 4;
    const uint e = tl[0], as0 = tl[1];
    const uint row = tgq.y * 4 + sg;
    const ulong ebase = (ulong(a.w_hi) << 32 | a.w_lo) + (ulong) e * a.expert_bytes;
    device const uchar * wp[2] = { W + ebase + (ulong) row * a.row_bytes, W + ebase + (ulong) (FF + row) * a.row_bytes };
    device const half * X = (device const half *) (A + a.xh_off);
    const short ix = lane / 8, it = lane % 8, iq = it / 4, ir = it % 4;
    const int nb = a.K / 256;
    float sumf[2][NT];
    for (int r = 0; r < 2; r++) for (int t = 0; t < NT; t++) sumf[r][t] = 0.0f;
    for (int ib = ix; ib < nb; ib += 4) {
        float yl[NT][16], yh[NT][16];
        float4 sumy[NT];
        const uint xoff = ib * 256 + 64 * iq + 8 * ir;
        for (int t = 0; t < NT; t++) {
            device const half * y4 = X + (ulong) (as0 + t) * a.K + xoff;
            float4 sy = 0.0f;
            for (short i = 0; i < 8; ++i) {
                yl[t][i + 0] = float(y4[i + 0]);   sy[0] += yl[t][i + 0];
                yl[t][i + 8] = float(y4[i + 32]);  sy[1] += yl[t][i + 8];
                yh[t][i + 0] = float(y4[i + 128]); sy[2] += yh[t][i + 0];
                yh[t][i + 8] = float(y4[i + 160]); sy[3] += yh[t][i + 8];
            }
            sumy[t] = sy;
        }
        for (short r = 0; r < 2; ++r) {
            device const block_q4_K * b = (device const block_q4_K *) wp[r] + ib;
            device const ushort * sc = (device const ushort *) b->scales + iq;
            device const ushort * q1 = (device const ushort *) b->qs + 16 * iq + 4 * ir;
            device const ushort * q2 = q1 + 32;
            ushort sc16[4]; thread const uchar * sc8 = (thread const uchar *) sc16;
            sc16[0] = sc[0] & 0x3f3f; sc16[1] = sc[2] & 0x3f3f;
            sc16[2] = ((sc[4] >> 0) & 0x0f0f) | ((sc[0] & 0xc0c0) >> 2);
            sc16[3] = ((sc[4] >> 4) & 0x0f0f) | ((sc[2] & 0xc0c0) >> 2);
            const float bd = float(b->d), bdm = float(b->dmin);
            for (int t = 0; t < NT; t++) {
                float4 acc1 = 0.0f, acc2 = 0.0f;
                for (short i = 0; i < 4; ++i) {
                    const ushort u1 = q1[i], u2 = q2[i];
                    acc1[0] += yl[t][2 * i + 0] * (u1 & 0x000F);
                    acc1[1] += yl[t][2 * i + 1] * (u1 & 0x0F00);
                    acc1[2] += yl[t][2 * i + 8] * (u1 & 0x00F0);
                    acc1[3] += yl[t][2 * i + 9] * (u1 & 0xF000);
                    acc2[0] += yh[t][2 * i + 0] * (u2 & 0x000F);
                    acc2[1] += yh[t][2 * i + 1] * (u2 & 0x0F00);
                    acc2[2] += yh[t][2 * i + 8] * (u2 & 0x00F0);
                    acc2[3] += yh[t][2 * i + 9] * (u2 & 0xF000);
                }
                sumf[r][t] += bd * ((acc1[0] + (1.f / 256.f) * acc1[1]) * sc8[0] +
                                    (acc1[2] + (1.f / 256.f) * acc1[3]) * sc8[1] * (1.f / 16.f) +
                                    (acc2[0] + (1.f / 256.f) * acc2[1]) * sc8[4] +
                                    (acc2[2] + (1.f / 256.f) * acc2[3]) * sc8[5] * (1.f / 16.f)) -
                              bdm * (sumy[t][0] * sc8[2] + sumy[t][1] * sc8[3] + sumy[t][2] * sc8[6] + sumy[t][3] * sc8[7]);
            }
        }
    }
    device half * act = (device half *) (A + a.out_off);
    for (int t = 0; t < NT; t++) {
        const float gt = simd_sum(sumf[0][t]), up = simd_sum(sumf[1][t]);
        if (lane == 0) {
            const float su = a.lbeta > 0 ? a.lbeta * tanh(up / a.lbeta) : up;
            const float o = (A[a.aw_off + as0 + t] * a.beta) * tanh(gt / a.beta) / (1.0f + exp(-gt)) * su;
            act[(ulong) (as0 + t) * FF + row] = half(o);
        }
    }
}
template [[host_name("k_moe_dir_gu1")]] kernel void k_moe_dir_gu<1>(device const uchar *, device float *, constant MoeGemmHArgs &, uint2, ushort, ushort);
template [[host_name("k_moe_dir_gu2")]] kernel void k_moe_dir_gu<2>(device const uchar *, device float *, constant MoeGemmHArgs &, uint2, ushort, ushort);

// ---- Direct quantized MoE down -> half yd for NT = 1 or 2 tokens per tile.
template <int NT>
kernel void k_moe_dir_dn(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant MoeGemmHArgs & a [[buffer(5)]],
                         uint2 tgq [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    if (tgq.x >= ((device const uint *) (A + a.ntiles_off))[0]) return;
    device const uint * tl = (device const uint *) (A + a.tiles_off) + tgq.x * 4;
    const uint e = tl[0], as0 = tl[1];
    const uint row0 = (tgq.y * 4 + sg) * 2;
    const ulong ebase = (ulong(a.w_hi) << 32 | a.w_lo) + (ulong) e * a.expert_bytes;
    device const half * X = (device const half *) (A + a.xh_off);
    const int nb = a.K / 32;
    float sumf[2][NT];
    for (int r = 0; r < 2; r++) for (int t = 0; t < NT; t++) sumf[r][t] = 0.0f;
    for (int ib = lane; ib < nb; ib += 32) {
        float4 xv[NT][8];
        for (int t = 0; t < NT; t++) {
            device const half4 * xp = (device const half4 *) (X + (ulong) (as0 + t) * a.K + ib * 32);
            for (int i = 0; i < 8; i++) xv[t][i] = float4(xp[i]);
        }
        for (int r = 0; r < 2; r++) {
            device const uchar * rowp = W + ebase + (ulong) (row0 + r) * a.row_bytes;
            if (a.wtype == 8) {
                device const block_q8_0 * bq = (device const block_q8_0 *) rowp + ib;
                const float d = float(bq->d);
                device const packed_char4 * q = (device const packed_char4 *) bq->qs;
                float4 wv[8];
                for (int i = 0; i < 8; i++) wv[i] = float4(char4(q[i])) * d;
                for (int t = 0; t < NT; t++) {
                    float s = 0;
                    for (int i = 0; i < 8; i++) s += dot(xv[t][i], wv[i]);
                    sumf[r][t] += s;
                }
            } else {
                device const block_q5_0 * bq = (device const block_q5_0 *) rowp + ib;
                const float d = float(bq->d);
                const uint qh = as_type<uint>(uchar4(*(device const packed_uchar4 *) bq->qh));
                device const packed_uchar4 * qs4 = (device const packed_uchar4 *) bq->qs;
                float4 wv[8];
                for (int i = 0; i < 4; i++) {
                    const uchar4 q = uchar4(qs4[i]);
                    const uint hl = qh >> (4 * i), hh = qh >> (4 * i + 16);
                    wv[i] = (float4(int4(uint4(q & 0xF) | ((uint4(hl, hl >> 1, hl >> 2, hl >> 3) & 1) << 4))) - 16.0f) * d;
                    wv[4 + i] = (float4(int4(uint4(q >> 4) | ((uint4(hh, hh >> 1, hh >> 2, hh >> 3) & 1) << 4))) - 16.0f) * d;
                }
                for (int t = 0; t < NT; t++) {
                    float s = 0;
                    for (int i = 0; i < 8; i++) s += dot(xv[t][i], wv[i]);
                    sumf[r][t] += s;
                }
            }
        }
    }
    device half * yd = (device half *) (A + a.out_off);
    for (int r = 0; r < 2; r++) for (int t = 0; t < NT; t++) {
        const float y = simd_sum(sumf[r][t]);
        if (lane == 0) yd[(ulong) (as0 + t) * a.ldo + row0 + r] = half(y);
    }
}
template [[host_name("k_moe_dir_dn1")]] kernel void k_moe_dir_dn<1>(device const uchar *, device float *, constant MoeGemmHArgs &, uint2, ushort, ushort);
template [[host_name("k_moe_dir_dn2")]] kernel void k_moe_dir_dn<2>(device const uchar *, device float *, constant MoeGemmHArgs &, uint2, ushort, ushort);

// ---- Fused wo GEMM + vectorized SoA ring-indexed recurrent post-conv + optional residual.
// Only ONE float4 store to CS_POOL per quad (3x fewer CS writes) + contiguous per-layer CS slab (zero TLB thrashing)!
struct WoPostArgs {
    uint w_lo, w_hi, row_bytes, wtype, nrows, K, xh_off, out_off;
    uint res_off, has_res, layer_slot, max_batch, convw_t_off, B, seq_cs_off, seq_pos_off;
};
template <int NSG>
kernel void k_wo_post_sg(device const uchar * W [[buffer(0)]], device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                         device float * CS [[buffer(4)]], constant WoPostArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                         uint2 tgq [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[2][HB_M * HB_K];
    const uint r0 = tgq.y * HB_M, t0 = (tgq.x * NSG + sg) * 16, K = a.K, B = a.B;
    const bool on = t0 < B;
    const int nt = on ? min(16, (int) (B - t0)) : 0;
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(HB_K, HB_M));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(HB_K, HB_M));
    auto tX = tensor((device half *) (A + a.xh_off) + (ulong) (on ? t0 : 0) * K, dextents<int32_t, 2>((int) K, nt), array<int, 2>({1, (int) K}));
    SGOP(mm);
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    const bool wact = tid < 128;
    const uint widx = tid * 16, hp = (widx % HB_K) / 16;
    device const uchar * wr = W + (ulong(a.w_hi) << 32 | a.w_lo) + (ulong) min(r0 + widx / HB_K, a.nrows - 1) * a.row_bytes;
    WHead wh; half4 wv[4];
    if (wact) { wstage(wr, a.wtype, 0, hp, wh, wv); for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[0] + widx))[i] = wv[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += HB_K) {
        const bool more = k0 + HB_K < K;
        if (more && wact) wstage(wr, a.wtype, k0 + HB_K, hp, wh, wv);
        if (on) { auto sx = tX.slice(k0, 0); if (buf == 0) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT); }
        if (more && wact) for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[buf ^ 1] + widx))[i] = wv[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1;
    }
    if (!on) return;
    const ulong branch_base = cs_branch_base(a.layer_slot, 2, a.max_batch);
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
            device float * hb = CS + branch_base + (ulong) seq_cs[t] * (3ul * E);
            const float4 h0 = *(device const float4 *) (hb + (ulong) tap0 * E + r);
            const float4 h1 = *(device const float4 *) (hb + (ulong) tap1 * E + r);
            const float4 h2 = *(device const float4 *) (hb + (ulong) tap2 * E + r);
            const float4 w0 = *(device const float4 *) (cw + 0 * E + r);
            const float4 w1 = *(device const float4 *) (cw + 1 * E + r);
            const float4 w2 = *(device const float4 *) (cw + 2 * E + r);
            const float4 w3 = *(device const float4 *) (cw + 3 * E + r);
            const float4 y = float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]);
            *(device float4 *) (hb + (ulong) tap0 * E + r) = y;
            float4 o = (y + w0 * h0 + w1 * h1 + w2 * h2 + w3 * y) * 0.4472135954999579f;
            if (a.has_res) o += *(device const float4 *) (A + a.res_off + (ulong) t * E + r);
            *(device float4 *) (A + a.out_off + (ulong) t * E + r) = o;
        }
    }
}
template [[host_name("k_wo_post_sg8")]] kernel void k_wo_post_sg<8>(device const uchar *, device const uchar *, device float *, device float *, constant WoPostArgs &, device const uint *, uint2, ushort, ushort);
template [[host_name("k_wo_post_sg4")]] kernel void k_wo_post_sg<4>(device const uchar *, device const uchar *, device float *, device float *, constant WoPostArgs &, device const uint *, uint2, ushort, ushort);

// ---- Fused MoE expert sum (half yd) + vectorized SoA ring-indexed recurrent post-conv + residual -> f32 out.
struct MoeCombPostArgs {
    uint yd_off, inv_off, res_off, out_off, layer_slot, max_batch, convw_t_off, seq_cs_off, seq_pos_off, pad0, pad1, pad2;
};
kernel void k_moe_comb_post_b(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device float * CS [[buffer(4)]],
                              constant MoeCombPostArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                              uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint t = tgp.x, c = (tgp.y * 256 + tid) * 4;
    if (c >= E) return;
    device const uint * inv = (device const uint *) (A + a.inv_off) + t * TOPK;
    device const half * yd = (device const half *) (A + a.yd_off);
    float4 y = 0;
    for (int k = 0; k < TOPK; k++) y += float4(*(device const half4 *) (yd + (ulong) inv[k] * E + c));
    const uint tap0 = META[a.seq_pos_off + t] % 3u;
    const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
    const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
    device float * hb = CS + cs_branch_base(a.layer_slot, 3, a.max_batch) + (ulong) META[a.seq_cs_off + t] * (3ul * E);
    device const float * cw = (device const float *) (PB + a.convw_t_off);
    const float4 h0 = *(device const float4 *) (hb + (ulong) tap0 * E + c);
    const float4 h1 = *(device const float4 *) (hb + (ulong) tap1 * E + c);
    const float4 h2 = *(device const float4 *) (hb + (ulong) tap2 * E + c);
    const float4 w0 = *(device const float4 *) (cw + 0 * E + c);
    const float4 w1 = *(device const float4 *) (cw + 1 * E + c);
    const float4 w2 = *(device const float4 *) (cw + 2 * E + c);
    const float4 w3 = *(device const float4 *) (cw + 3 * E + c);
    *(device float4 *) (hb + (ulong) tap0 * E + c) = y;
    const float4 o = (y + w0 * h0 + w1 * h1 + w2 * h2 + w3 * y) * 0.4472135954999579f + *(device const float4 *) (A + a.res_off + (ulong) t * E + c);
    *(device float4 *) (A + a.out_off + (ulong) t * E + c) = o;
}

// ============================================================================
// Batched Paged Decode Attention (Slot-Major KV + Slot-Major SoA Ring CS + Shared-Prefix Cascade)
// ============================================================================

struct PagedAttnArgs {
    uint B, pool_cap, slot_idx, is_swa, pt_stride, pt_mod;
    uint seq_pos_off, seq_suf_off, seq_cs_off, pt_off;
    uint layer_slot, max_batch, q_off, k_off, v_off, r_off, sink_off;
    int  kdyn_off, vdyn_off;
    uint kconv_t_w, vconv_t_w, qclip_w, relclip_w, relbias_w;
    float log_alpha, log_floor;
    uint gate_off, out_off, part_off, nsplit, prefix_len;
    uint qp_off, rp_off, pad0;
};

// ---- Path A: Single-pass fused paged decode attention for B independent sequences (prefix_len == 0 or SWA past prefix).
kernel void k_attn_paged_b(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device kv_t * KV [[buffer(3)]],
                           device float * CS [[buffer(4)]], constant PagedAttnArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                           uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
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
    float kx = 0, vx = 0;
    if (owns_kv) {
        const uint seq_cs = META[a.seq_cs_off + b];
        const uint tap0 = pos % 3u;
        const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
        const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
        const uint c = g * HD + tid;
        const uint dim = NKV * HD;
        const float scl = a.kdyn_off >= 0 ? (1.0f / 3.0f) : 0.4472135954999579f;
        for (int which = 0; which < 2; which++) {
            device float * hb = CS + cs_branch_base(a.layer_slot, which, a.max_batch) + (ulong) seq_cs * (3ul * dim);
            device const float * w = (device const float *) (PB + (which == 0 ? a.kconv_t_w : a.vconv_t_w));
            const float x = A[(which == 0 ? a.k_off : a.v_off) + (ulong) b * dim + c];
            const float h0 = hb[(ulong) tap0 * dim + c];
            const float h1 = hb[(ulong) tap1 * dim + c];
            const float h2 = hb[(ulong) tap2 * dim + c];
            float acc = w[0 * dim + c] * h0 + w[1 * dim + c] * h1 + w[2 * dim + c] * h2 + w[3 * dim + c] * x;
            const int dyn = which == 0 ? a.kdyn_off : a.vdyn_off;
            if (dyn >= 0) { device const float * tp = A + dyn + (ulong) b * 32 + g * 4; acc += tp[0] * x + tp[1] * h2 + tp[2] * h1 + tp[3] * h0; }
            hb[(ulong) tap0 * dim + c] = x;
            if (which == 0) kx = (x + acc) * scl; else vx = (x + acc) * scl;
        }
    }
    float r = 0;
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16, H = g * 3 + h;
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
    if (tid < 48) rs[tid / 16][tid % 16] = r * clamp(REL_CLIP / (sqrt(rn) + 1e-6f), 0.0f, 1.0f) * tau;
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
        // Hoist page lookup when all 8 keys in this block lie on the same page
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

struct CombineBArgs { uint part_off, nsplit, gate_off, out_off; };
kernel void k_attn_combine_b(device float * A [[buffer(2)]], constant CombineBArgs & a [[buffer(5)]],
                             uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint h = tgp.x, b = tgp.y;
    device const float * P = A + a.part_off + (ulong) (b * NH + h) * a.nsplit * (HD + 2);
    float M = 0.0f;
    for (uint s = 0; s < a.nsplit; s++) M = max(M, P[s * (HD + 2)]);
    float L = exp(-M), o = 0.0f;
    for (uint s = 0; s < a.nsplit; s++) {
        const float ms = P[s * (HD + 2)];
        const float w = ms == -INFINITY ? 0.0f : exp(ms - M);
        L += w * P[s * (HD + 2) + 1];
        o += w * P[s * (HD + 2) + 2 + tid];
    }
    const ulong oi = (ulong) b * (NH * HD) + h * HD + tid;
    const float gv = float(((device const half *) (A + a.gate_off))[oi]);
    ((device half *) (A + a.out_off))[oi] = half(o / L * sigm(gv));
}

// ---- Path B (Shared Prefix Cascade Attention):
// Step 1: k_kvprep_dec_b: per-seq K/V conv + ring CS update + paged write + GQA-packed Q/R.
kernel void k_kvprep_dec_b(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device kv_t * KV [[buffer(3)]],
                           device float * CS [[buffer(4)]], constant PagedAttnArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                           uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4][8];
    const uint b = tgp.x, g = tgp.y;
    const uint pos = META[a.seq_pos_off + b];
    const uint suf = META[a.seq_suf_off + b];
    const float tau = a.is_swa ? 1.0f : (1.0f + a.log_alpha * log(max(1.0f, float(pos + 1u) / a.log_floor)));
    device const float * qclip = (device const float *) (PB + a.qclip_w);
    float q[3];
    for (int h = 0; h < 3; h++) {
        const uint H = g * 3 + h;
        const half qh = ((device const half *) (A + a.q_off))[(ulong) b * (NH * HD) + H * HD + tid];
        q[h] = float(qh) * qclip[H * HD + tid];
    }
    const uint seq_cs = META[a.seq_cs_off + b];
    const uint tap0 = pos % 3u;
    const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
    const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
    const uint c = g * HD + tid;
    const uint dim = NKV * HD;
    const float scl = a.kdyn_off >= 0 ? (1.0f / 3.0f) : 0.4472135954999579f;
    float kx = 0, vx = 0;
    for (int which = 0; which < 2; which++) {
        device float * hb = CS + cs_branch_base(a.layer_slot, which, a.max_batch) + (ulong) seq_cs * (3ul * dim);
        device const float * w = (device const float *) (PB + (which == 0 ? a.kconv_t_w : a.vconv_t_w));
        const float x = A[(which == 0 ? a.k_off : a.v_off) + (ulong) b * dim + c];
        const float h0 = hb[(ulong) tap0 * dim + c];
        const float h1 = hb[(ulong) tap1 * dim + c];
        const float h2 = hb[(ulong) tap2 * dim + c];
        float acc = w[0 * dim + c] * h0 + w[1 * dim + c] * h1 + w[2 * dim + c] * h2 + w[3 * dim + c] * x;
        const int dyn = which == 0 ? a.kdyn_off : a.vdyn_off;
        if (dyn >= 0) { device const float * tp = A + dyn + (ulong) b * 32 + g * 4; acc += tp[0] * x + tp[1] * h2 + tp[2] * h1 + tp[3] * h0; }
        hb[(ulong) tap0 * dim + c] = x;
        if (which == 0) kx = (x + acc) * scl; else vx = (x + acc) * scl;
    }
    float r = 0;
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16, H = g * 3 + h;
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
    device half * qp_g = (device half *) (A + a.qp_off) + (ulong) g * (a.B * 3) * HD;
    for (int h = 0; h < 3; h++) {
        const float f = clamp(QK_CLIP / (sqrt(tot[h]) + 1e-6f), 0.0f, 1.0f) * tau * 0.08838834764831845f;
        qp_g[(ulong) (b * 3 + h) * HD + tid] = half(q[h] * f);
    }
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16;
        A[a.rp_off + ((ulong) g * (a.B * 3) + b * 3 + h) * NREL + i] = r * clamp(REL_CLIP / (sqrt(rn) + 1e-6f), 0.0f, 1.0f) * tau;
    }
    const float kf = clamp(QK_CLIP / (sqrt(tot[3]) + 1e-6f), 0.0f, 1.0f), vf = rsqrt(tot[4] / HD + 1e-6f);
    const float k = kx * kf, v = vx * vf;
    device const uint * pt = META + a.pt_off + (ulong) b * a.pt_stride;
    const uint bidx = a.is_swa ? ((suf / KV_BLK) % a.pt_mod) : (suf / KV_BLK);
    const uint boff = suf % KV_BLK;
    const uint pid = pt[bidx];
    const ulong kb = paged_kv_base(pid, a.pool_cap, a.slot_idx, g, 0) + (ulong) boff * HD;
    const ulong vb = paged_kv_base(pid, a.pool_cap, a.slot_idx, g, 1) + (ulong) boff * HD;
#ifdef KV8
    const uint ck = kv_code(max(max(red[0][5], red[1][5]), max(red[2][5], red[3][5])) * kf);
    const uint cv = kv_code(max(max(red[0][6], red[1][6]), max(red[2][6], red[3][6])) * vf);
    KV[kb + tid] = (kv_t) (int) kv_quant(k, kv_scale(ck), ck, tid);
    KV[vb + tid] = (kv_t) (int) kv_quant(v, kv_scale(cv), cv, tid);
#else
    KV[kb + tid] = half(k);
    KV[vb + tid] = half(v);
#endif
}

// Step 2: k_attn_prefix_h: M5 neural-accelerator GQA-packed FlashDecoding over shared prefix [0..pre_n).
struct PrefixAttnArgs {
    uint B, pre_n, pre_p0, pre_cap, k_base, v_base, window;
    uint qp_off, rp_off, sink_off, mlo_off, relbias_w, seq_pos_off, pad0, pad1, pad2;
};

kernel void k_attn_prefix_h(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device kv_t * PRE_KV [[buffer(8)]],
                            constant PrefixAttnArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                            uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float ss[AQ * AK];
    threadgroup float G[AQ * AG];
    threadgroup half sp[AQ * AK];
    threadgroup float sr[AQ];
    threadgroup uint qpos_s[AQ];
    threadgroup float qsink_s[AQ];
    const uint g = tgp.y, row0 = tgp.x * AQ, ntot = a.B * 3;
    const uint nq = min((uint) AQ, ntot - row0);
    const ushort tid = sg * 32 + lane;
    if (tid < AQ) {
        if (tid < nq) {
            const uint gr = row0 + tid, b_idx = gr / 3, hh = gr % 3;
            qpos_s[tid] = META[a.seq_pos_off + b_idx];
            qsink_s[tid] = A[a.sink_off + (ulong) b_idx * NH + g * 3 + hh];
        } else {
            qpos_s[tid] = META[a.seq_pos_off + (row0 / 3)];
            qsink_s[tid] = 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint min_qp = qpos_s[0], max_qp = qpos_s[0];
    for (uint j = 1; j < nq; j++) { min_qp = min(min_qp, qpos_s[j]); max_qp = max(max_qp, qpos_s[j]); }
    const bool same_pos = (min_qp == max_qp);
    const bool small_spread = (max_qp - min_qp <= 32u);

    auto tq = tensor((device half *) (A + a.qp_off) + ((ulong) g * ntot + row0) * HD, dextents<int32_t, 2>(HD, (int) nq), array<int, 2>({1, HD}));
    auto tr = tensor(A + a.rp_off + ((ulong) g * ntot + row0) * NREL, dextents<int32_t, 2>(NREL, (int) nq), array<int, 2>({1, NREL}));
    auto ts = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(ss, dextents<int32_t, 2>(AK, AQ));
    auto tg = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(G, dextents<int32_t, 2>(AG, AQ));
    auto tp = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sp, dextents<int32_t, 2>(AK, AQ));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, AK, HD, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
                              execution_simdgroups<8>> mm_qk;
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, AG, NREL, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
                              execution_simdgroups<8>> mm_g;
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, HD, AK, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<8>> mm_pv;
    device kv_t * kbase = PRE_KV + (ulong) a.k_base + (ulong) g * a.pre_cap * HD;
    device kv_t * vbase = PRE_KV + (ulong) a.v_base + (ulong) g * a.pre_cap * HD;
    auto tv0 = tensor(vbase, dextents<int32_t, 2>(HD, AK), array<int, 2>({1, HD}));
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    for (short i = 0; i < 16; ++i) co[i] = 0.0f;
    ushort cd[4], cr[4];
    for (short q = 0; q < 4; ++q) { const auto ix = co.get_multidimensional_index(4 * q); cd[q] = ix[0]; cr[q] = ix[1]; }

    float M[4], S[4];
    for (short jj = 0; jj < 4; jj++) {
        const uint j = jj * 8 + sg;
        M[jj] = qsink_s[j]; S[jj] = 2980.9579870417283f;
    }
    device const float * relb = (device const float *) (PB + a.relbias_w);
    uint kstart = 0;
    if (a.window > 0 && min_qp + 1u > a.window) {
        const uint vis_p0 = min_qp + 1u - a.window;
        kstart = (vis_p0 > a.pre_p0) ? ((vis_p0 - a.pre_p0) / AK * AK) : 0u;
    }
    for (uint kc = kstart; kc < a.pre_n; kc += AK) {
        auto tk = tensor(kbase + (ulong) kc * HD, dextents<int32_t, 2>(HD, AK), array<int, 2>({1, HD}));
        { auto cs = mm_qk.template get_destination_cooperative_tensor<decltype(tq), decltype(tk), float>(); mm_qk.run(tq, tk, cs); cs.store(ts); }
        const int kp0 = int(a.pre_p0 + kc);
        const int dlo = max(int(min_qp) - (kp0 + AK - 1), 0);
        const bool bias = (dlo < EXT);
        if (bias && small_spread) {
            auto tb = tensor((device float *) relb + dlo * NREL, dextents<int32_t, 2>(NREL, AG), array<int, 2>({1, NREL}));
            auto cg = mm_g.template get_destination_cooperative_tensor<decltype(tr), decltype(tb), float>(); mm_g.run(tr, tb, cg); cg.store(tg);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const bool full_chunk = (kc + AK <= a.pre_n);
        const bool fast = same_pos && full_chunk && (a.window == 0 || int(min_qp) - kp0 < (int) a.window) && (!bias || int(min_qp) - kp0 < EXT);
        const int D0_same = int(min_qp) - kp0;
        const int gofs = D0_same - dlo - int(lane);
#ifdef KV8
        const float ks0 = kv_vscale(kbase + (ulong) (kc + lane) * HD), ks1 = kv_vscale(kbase + (ulong) (kc + 32 + lane) * HD);
        const float vs0 = kv_vscale(vbase + (ulong) (kc + lane) * HD), vs1 = kv_vscale(vbase + (ulong) (kc + 32 + lane) * HD);
#endif
        for (short jj = 0; jj < 4; jj++) {
            const uint j = jj * 8 + sg;
#ifdef KV8
            float s0 = ss[j * AK + lane] * ks0, s1 = ss[j * AK + 32 + lane] * ks1;
#else
            float s0 = ss[j * AK + lane], s1 = ss[j * AK + 32 + lane];
#endif
            if (fast) {
                if (bias) { threadgroup const float * gp = G + j * AG + gofs; s0 += gp[0]; s1 += gp[-32]; }
            } else {
                const int qp = int(qpos_s[j]);
                const int d0 = qp - (kp0 + int(lane)), d1 = d0 - 32;
                const bool v0 = (j < nq) && (kc + lane < a.pre_n) && (d0 >= 0) && (a.window == 0 || d0 < (int) a.window);
                const bool v1 = (j < nq) && (kc + 32 + lane < a.pre_n) && (d1 >= 0) && (a.window == 0 || d1 < (int) a.window);
                if (bias) {
                    if (small_spread) {
                        if (v0 && d0 < EXT) s0 += G[j * AG + clamp(d0 - dlo, 0, AG - 1)];
                        if (v1 && d1 < EXT) s1 += G[j * AG + clamp(d1 - dlo, 0, AG - 1)];
                    } else {
                        device const float4 * r4 = (device const float4 *) (A + a.rp_off + ((ulong) g * ntot + row0 + min(j, nq - 1u)) * NREL);
                        if (v0 && d0 < EXT) {
                            device const float4 * b4 = (device const float4 *) (relb + d0 * NREL);
                            s0 += dot(r4[0], b4[0]) + dot(r4[1], b4[1]) + dot(r4[2], b4[2]) + dot(r4[3], b4[3]);
                        }
                        if (v1 && d1 < EXT) {
                            device const float4 * b4 = (device const float4 *) (relb + d1 * NREL);
                            s1 += dot(r4[0], b4[0]) + dot(r4[1], b4[1]) + dot(r4[2], b4[2]) + dot(r4[3], b4[3]);
                        }
                    }
                }
                if (!v0) s0 = -INFINITY;
                if (!v1) s1 = -INFINITY;
            }
            const float m = simd_max(max(M[jj], max(s0, s1)));
            const float ms = exp(M[jj] - m);
            M[jj] = m;
            const half p0 = half(exp(s0 - m + 8.0f)), p1 = half(exp(s1 - m + 8.0f));
#ifdef KV8
            sp[j * AK + lane] = half(float(p0) * vs0);
            sp[j * AK + 32 + lane] = half(float(p1) * vs1);
#else
            sp[j * AK + lane] = p0;
            sp[j * AK + 32 + lane] = p1;
#endif
            S[jj] = S[jj] * ms + simd_sum(float(p0) + float(p1));
            if (lane == 0) sr[j] = ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const float r0 = sr[cr[0]], r1 = sr[cr[1]], r2 = sr[cr[2]], r3 = sr[cr[3]];
            co[0] *= r0; co[1] *= r0; co[2] *= r0; co[3] *= r0;
            co[4] *= r1; co[5] *= r1; co[6] *= r1; co[7] *= r1;
            co[8] *= r2; co[9] *= r2; co[10] *= r2; co[11] *= r2;
            co[12] *= r3; co[13] *= r3; co[14] *= r3; co[15] *= r3;
        }
        auto tv = tensor(vbase + (ulong) kc * HD, dextents<int32_t, 2>(HD, AK), array<int, 2>({1, HD}));
        mm_pv.run(tp, tv, co);
    }
    const float inv_e8 = 0.00033546262790251185f;
    for (short jj = 0; jj < 4; jj++) {
        const uint j = jj * 8 + sg;
        if (lane == 0 && j < nq) {
            const uint gr = row0 + j, b_idx = gr / 3, hh = gr % 3, H = g * 3 + hh;
            device float * dst = A + a.mlo_off + (ulong) (b_idx * NH + H) * (HD + 2);
            dst[0] = M[jj] - qsink_s[j];
            dst[1] = S[jj] * inv_e8;
        }
    }
    for (short q = 0; q < 4; ++q) {
        const uint row = cr[q];
        if (row < nq) {
            const uint gr = row0 + row, b_idx = gr / 3, hh = gr % 3, H = g * 3 + hh;
            device float * dst = A + a.mlo_off + (ulong) (b_idx * NH + H) * (HD + 2) + 2 + cd[q];
            *(device float4 *) dst = float4(co[4 * q], co[4 * q + 1], co[4 * q + 2], co[4 * q + 3]) * inv_e8;
        }
    }
}

// Step 3: k_attn_suffix_paged_b: attend to private paged suffix [suf_first .. suf] and merge with (M_pre, L_pre, O_pre).
kernel void k_attn_suffix_paged_b(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device kv_t * KV [[buffer(3)]],
                                  constant PagedAttnArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                                  uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                                  ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float4 qs4[3][HD / 4];
    threadgroup float rs[3][NREL];
    const uint g = tgp.x, b = tgp.y;
    const uint pos = META[a.seq_pos_off + b];
    const uint suf = META[a.seq_suf_off + b];
    const uint vis_p0 = (a.is_swa && pos + 1u > EXT) ? (pos + 1u - EXT) : 0u;
    const uint suf_first = (vis_p0 > a.prefix_len) ? (vis_p0 - a.prefix_len) : 0u;
    const uint nsuf = suf + 1u - suf_first;
    device const half * qp_g = (device const half *) (A + a.qp_off) + (ulong) g * (a.B * 3) * HD;
    for (int h = 0; h < 3; h++) ((threadgroup float *) qs4[h])[tid] = float(qp_g[(ulong) (b * 3 + h) * HD + tid]);
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16;
        rs[h][i] = A[a.rp_off + ((ulong) g * (a.B * 3) + b * 3 + h) * NREL + i];
    }
    float sink[3];
    for (int h = 0; h < 3; h++) sink[h] = A[a.sink_off + (ulong) b * NH + g * 3 + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const uint * pt = META + a.pt_off + (ulong) b * a.pt_stride;
    device const float * relb = (device const float *) (PB + a.relbias_w);
    const ushort kk = lane / 4, pp = lane % 4;
    float m[3] = {-INFINITY, -INFINITY, -INFINITY}, l[3] = {0, 0, 0};
    float4 o[3] = {0, 0, 0};
    float4 rl[3];
    for (int h = 0; h < 3; h++) rl[h] = *(threadgroup const float4 *) (&rs[h][pp * 4]);
    for (uint jb = sg * 8; jb < nsuf; jb += 32) {
        const uint j = jb + kk;
        const bool valid = j < nsuf;
        const uint psuf = suf_first + min(j, nsuf - 1u);
        const uint p = a.prefix_len + psuf;
        const uint bidx = a.is_swa ? ((psuf / KV_BLK) % a.pt_mod) : (psuf / KV_BLK);
        const uint boff = psuf % KV_BLK;
        const uint pid = pt[bidx];
        const ulong kb = paged_kv_base(pid, a.pool_cap, a.slot_idx, g, 0) + (ulong) boff * HD;
        float acc[3] = {0, 0, 0};
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
            const float vsc = kv_vscale(KV + vb);
            for (int h = 0; h < 3; h++) pr[h] *= vsc;
        }
#endif
        const uint suf_lo = suf_first + jb;
        const uint suf_hi = suf_first + min(jb + 7u, nsuf - 1u);
        const uint blk_lo = suf_lo / KV_BLK, blk_hi = suf_hi / KV_BLK;
        const ulong vbase_lo = paged_kv_base(pt[a.is_swa ? (blk_lo % a.pt_mod) : blk_lo], a.pool_cap, a.slot_idx, g, 1);
        const ulong vbase_hi = (blk_lo == blk_hi) ? vbase_lo : paged_kv_base(pt[a.is_swa ? (blk_hi % a.pt_mod) : blk_hi], a.pool_cap, a.slot_idx, g, 1);
        for (ushort q2 = 0; q2 < 8; q2++) {
            const uint jq = jb + q2;
            if (jq >= nsuf) break;
            const uint qsuf = suf_first + jq;
            const ulong vb = ((qsuf / KV_BLK == blk_lo) ? vbase_lo : vbase_hi) + (ulong) (qsuf % KV_BLK) * HD + lane * 4;
#ifdef KV8
            const float4 v = float4(char4(*(device const packed_char4 *) (KV + vb)));
#else
            const float4 v = float4(*(device const half4 *) (KV + vb));
#endif
            for (int h = 0; h < 3; h++) o[h] += simd_shuffle(pr[h], q2 * 4) * v;
        }
    }
    threadgroup float ms_[4][3], ls_[4][3];
    threadgroup float4 os_[4][3][32];
    if (lane == 0) for (int h = 0; h < 3; h++) { ms_[sg][h] = m[h]; ls_[sg][h] = l[h]; }
    for (int h = 0; h < 3; h++) os_[sg][h][lane] = o[h];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half * out = (device half *) (A + a.out_off);
    device const half * gate = (device const half *) (A + a.gate_off);
    for (int h = 0; h < 3; h++) {
        const uint H = g * 3 + h;
        device const float * pre = A + a.part_off + (ulong) (b * NH + H) * (HD + 2);
        const float M_pre = pre[0], L_pre = pre[1], O_pre = pre[2 + tid];
        float M = M_pre;
        for (int s2 = 0; s2 < 4; s2++) M = max(M, ms_[s2][h]);
        const float wp = (M_pre == -INFINITY) ? 0.0f : exp(M_pre - M);
        float L = wp * L_pre, O = wp * O_pre;
        for (int s2 = 0; s2 < 4; s2++) {
            const float w = (ms_[s2][h] == -INFINITY) ? 0.0f : exp(ms_[s2][h] - M);
            L += w * ls_[s2][h];
            O += w * ((threadgroup const float *) os_[s2][h])[tid];
        }
        const ulong oi = (ulong) b * (NH * HD) + H * HD + tid;
        out[oi] = half(O / L * sigm(float(gate[oi])));
    }
}

// ============================================================================
// Fused lm_head + Register Argmax & Vocab-Sliced GPU Top-K/Top-P Sampler
// ============================================================================
#define LM_VSTEP 16
#define LM_VCHUNK (LM_VSTEP * HB_M)

struct LmHeadArgs {
    uint w_lo, w_hi, row_bytes, wtype, n_vocab, K, xh_off, part_off;
    uint B, nvch, vstart, vcount;
};

template <int NSG>
kernel void k_lmhead_argmax_sg(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant LmHeadArgs & a [[buffer(5)]],
                               uint2 tgq [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                               ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup half sa[2][HB_M * HB_K];
    const uint t0 = (tgq.x * NSG + sg) * 16, K = a.K, B = a.B;
    const bool on = t0 < B;
    const int nt = on ? min(16, (int) (B - t0)) : 0;
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(HB_K, HB_M));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(HB_K, HB_M));
    auto tX = tensor((device half *) (A + a.xh_off) + (ulong) (on ? t0 : 0) * K, dextents<int32_t, 2>((int) K, nt), array<int, 2>({1, (int) K}));
    SGOP(mm);
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    ushort qr[4], qt[4];
    for (short q = 0; q < 4; ++q) { const auto ix = cT.get_multidimensional_index(4 * q); qr[q] = ix[0]; qt[q] = ix[1]; }
    float bv[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    uint  bi[4] = {0x7fffffffu, 0x7fffffffu, 0x7fffffffu, 0x7fffffffu};
    const bool wact = tid < 128;
    const uint widx = tid * 16, rr = widx / HB_K, hp = (widx % HB_K) / 16;
    const ulong wbase = ulong(a.w_hi) << 32 | a.w_lo;
    const uint v0 = tgq.y * LM_VCHUNK;
    for (uint vb = 0; vb < LM_VSTEP; vb++) {
        const uint r0 = v0 + vb * HB_M;
        if (r0 >= a.n_vocab) break;
        device const uchar * wr = W + wbase + (ulong) min(r0 + rr, a.n_vocab - 1u) * a.row_bytes;
        WHead wh; half4 wv[4];
        if (wact) { wstage(wr, a.wtype, 0, hp, wh, wv); for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[0] + widx))[i] = wv[i]; }
        for (short i = 0; i < 32; ++i) cT[i] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint buf = 0;
        for (uint k0 = 0; k0 < K; k0 += HB_K) {
            const bool more = k0 + HB_K < K;
            if (more && wact) wstage(wr, a.wtype, k0 + HB_K, hp, wh, wv);
            if (on) { auto sx = tX.slice(k0, 0); if (buf == 0) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT); }
            if (more && wact) for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[buf ^ 1] + widx))[i] = wv[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            buf ^= 1;
        }
        if (on) {
            for (short q = 0; q < 4; ++q) {
                const uint r_lo = r0 + qr[q], r_hi = r_lo + 32;
                for (int e = 0; e < 4; e++) {
                    const uint idx = r_lo + e;
                    if (idx < a.n_vocab) {
                        const float v = cT[4 * q + e];
                        if (v > bv[q] || (v == bv[q] && idx < bi[q])) { bv[q] = v; bi[q] = idx; }
                    }
                    const uint idx2 = r_hi + e;
                    if (idx2 < a.n_vocab) {
                        const float v2 = cT[4 * q + 16 + e];
                        if (v2 > bv[q] || (v2 == bv[q] && idx2 < bi[q])) { bv[q] = v2; bi[q] = idx2; }
                    }
                }
            }
        }
    }
    if (!on) return;
    for (int tt = 0; tt < nt; tt++) {
        float lv = -INFINITY; uint li = 0x7fffffffu;
        for (short q = 0; q < 4; ++q) {
            if (qt[q] == tt && (bv[q] > lv || (bv[q] == lv && bi[q] < li))) { lv = bv[q]; li = bi[q]; }
        }
        const float mv = simd_max(lv);
        const uint mi = simd_min(lv == mv ? li : 0x7fffffffu);
        if (lane == 0) {
            const ulong oi = ((ulong) (t0 + tt) * a.nvch + tgq.y) * 2;
            A[a.part_off + oi] = mv;
            A[a.part_off + oi + 1] = as_type<float>(mi);
        }
    }
}
template [[host_name("k_lmhead_argmax_sg8")]] kernel void k_lmhead_argmax_sg<8>(device const uchar *, device float *, constant LmHeadArgs &, uint2, ushort, ushort, ushort);
template [[host_name("k_lmhead_argmax_sg4")]] kernel void k_lmhead_argmax_sg<4>(device const uchar *, device float *, constant LmHeadArgs &, uint2, ushort, ushort, ushort);

struct ArgmaxBatchArgs { uint part_off, nvch, out_base, pad; };
kernel void k_argmax_batch(device float * A [[buffer(2)]], constant ArgmaxBatchArgs & a [[buffer(5)]], device int * toks [[buffer(6)]],
                           uint b [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[4]; threadgroup uint si[4];
    float best = -INFINITY; uint idx = 0x7fffffffu;
    device const float * p = A + a.part_off + (ulong) b * a.nvch * 2;
    for (uint i = tid; i < a.nvch; i += 128) {
        const float v = p[2 * i]; const uint ii = as_type<uint>(p[2 * i + 1]);
        if (v > best || (v == best && ii < idx)) { best = v; idx = ii; }
    }
    const float m = simd_max(best);
    const uint c = simd_min(best == m ? idx : 0x7fffffffu);
    if (lane == 0) { sv[sg] = m; si[sg] = c; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < 4 ? sv[lane] : -INFINITY;
        uint ii = lane < 4 ? si[lane] : 0x7fffffffu;
        const float m2 = simd_max(v);
        const uint c2 = simd_min(v == m2 ? ii : 0x7fffffffu);
        if (lane == 0) toks[a.out_base + b] = (int) c2;
    }
}

template <int NSG>
kernel void k_lmhead_slice_sg(device const uchar * W [[buffer(0)]], device float * A [[buffer(2)]], constant LmHeadArgs & a [[buffer(5)]],
                              uint2 tgq [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half sa[2][HB_M * HB_K];
    const uint rloc = tgq.y * HB_M, r0 = a.vstart + rloc, t0 = (tgq.x * NSG + sg) * 16, K = a.K, B = a.B;
    const bool on = t0 < B;
    const int nt = on ? min(16, (int) (B - t0)) : 0;
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(HB_K, HB_M));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(HB_K, HB_M));
    auto tX = tensor((device half *) (A + a.xh_off) + (ulong) (on ? t0 : 0) * K, dextents<int32_t, 2>((int) K, nt), array<int, 2>({1, (int) K}));
    SGOP(mm);
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    const bool wact = tid < 128;
    const uint widx = tid * 16, hp = (widx % HB_K) / 16;
    device const uchar * wr = W + (ulong(a.w_hi) << 32 | a.w_lo) + (ulong) min(r0 + widx / HB_K, a.n_vocab - 1u) * a.row_bytes;
    WHead wh; half4 wv[4];
    if (wact) { wstage(wr, a.wtype, 0, hp, wh, wv); for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[0] + widx))[i] = wv[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint buf = 0;
    for (uint k0 = 0; k0 < K; k0 += HB_K) {
        const bool more = k0 + HB_K < K;
        if (more && wact) wstage(wr, a.wtype, k0 + HB_K, hp, wh, wv);
        if (on) { auto sx = tX.slice(k0, 0); if (buf == 0) mm.run(sx, tA0, cT); else mm.run(sx, tA1, cT); }
        if (more && wact) for (int i = 0; i < 4; i++) ((threadgroup half4 *) (sa[buf ^ 1] + widx))[i] = wv[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf ^= 1;
    }
    if (!on) return;
    const int nr = min((int) HB_M, (int) (a.vcount - rloc));
    auto tC = tensor(A + a.part_off + (ulong) t0 * a.vcount + rloc, dextents<int32_t, 2>(nr, nt), array<int, 2>({1, (int) a.vcount}));
    cT.store(tC);
}
template [[host_name("k_lmhead_slice_sg8")]] kernel void k_lmhead_slice_sg<8>(device const uchar *, device float *, constant LmHeadArgs &, uint2, ushort, ushort);
template [[host_name("k_lmhead_slice_sg4")]] kernel void k_lmhead_slice_sg<4>(device const uchar *, device float *, constant LmHeadArgs &, uint2, ushort, ushort);

struct TopkMergeArgs { uint slice_off, vstart, vcount, topk_off, top_k, first_slice, pad0, pad1; };
inline uint float_to_ord(float x) {
    const uint u = as_type<uint>(x);
    return (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);
}
kernel void k_topk_merge(device float * A [[buffer(2)]], constant TopkMergeArgs & a [[buffer(5)]],
                         uint b [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint cut_bin, cut_rem, ncand;
    threadgroup float cv[256];
    threadgroup uint  ci[256];
    const uint K = min(a.top_k, 64u);
    device const float * lg = A + a.slice_off + (ulong) b * a.vcount;

    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < a.vcount; i += 256) {
        const uint bin = float_to_ord(lg[i]) >> 24;
        atomic_fetch_add_explicit(&hist[bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint acc = 0, cb = 0;
        for (int bi = 255; bi >= 0; bi--) {
            const uint c = atomic_load_explicit(&hist[bi], memory_order_relaxed);
            if (acc + c >= K) { cb = (uint) bi; break; }
            acc += c;
        }
        cut_bin = cb; cut_rem = K - acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint cb1 = cut_bin;

    atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < a.vcount; i += 256) {
        const uint u = float_to_ord(lg[i]);
        if ((u >> 24) == cb1) atomic_fetch_add_explicit(&hist[(u >> 16) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint acc = 0, cb2 = 0;
        for (int bi = 255; bi >= 0; bi--) {
            const uint c = atomic_load_explicit(&hist[bi], memory_order_relaxed);
            if (acc + c >= cut_rem) { cb2 = (uint) bi; break; }
            acc += c;
        }
        cut_bin = (cb1 << 8) | cb2;
        ncand = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint thresh16 = cut_bin;
    threadgroup atomic_uint * nc = (threadgroup atomic_uint *) &ncand;

    device float * tk = A + a.topk_off + (ulong) b * 128;
    if (!a.first_slice && tid < K) {
        const uint p = atomic_fetch_add_explicit(nc, 1u, memory_order_relaxed);
        cv[p] = tk[tid]; ci[p] = as_type<uint>(tk[64 + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < a.vcount; i += 256) {
        const float v = lg[i];
        if ((float_to_ord(v) >> 16) > thresh16) {
            const uint p = atomic_fetch_add_explicit(nc, 1u, memory_order_relaxed);
            if (p < 256) { cv[p] = v; ci[p] = a.vstart + i; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < a.vcount; i += 256) {
        const float v = lg[i];
        if ((float_to_ord(v) >> 16) == thresh16) {
            const uint p = atomic_fetch_add_explicit(nc, 1u, memory_order_relaxed);
            if (p < 256) { cv[p] = v; ci[p] = a.vstart + i; }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n = min(ncand, 256u);
    if (tid >= n) { cv[tid] = -INFINITY; ci[tid] = 0x7fffffffu; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2; k <= 256; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint ixj = tid ^ j;
            if (ixj > tid) {
                const bool asc = ((tid & k) == 0);
                const float va = cv[tid], vb = cv[ixj];
                const uint ia = ci[tid], ib = ci[ixj];
                const bool a_better = (va > vb) || (va == vb && ia < ib);
                if (asc ? !a_better : a_better) { cv[tid] = vb; ci[tid] = ib; cv[ixj] = va; ci[ixj] = ia; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid < K) { tk[tid] = cv[tid]; tk[64 + tid] = as_type<float>(ci[tid]); }
}

struct SampleBatchArgs { uint topk_off, top_k, out_base, rng_off; float temp, top_p, softcap, pad; };
kernel void k_sample_topk(device float * A [[buffer(2)]], constant SampleBatchArgs & a [[buffer(5)]],
                          device int * toks [[buffer(6)]], device ulong * rng [[buffer(7)]],
                          uint b [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float p[64];
    threadgroup uint  id[64];
    const uint K = min(a.top_k, 64u);
    device const float * tk = A + a.topk_off + (ulong) b * 128;
    float sc = -INFINITY;
    if (tid < K) {
        float x = tk[tid];
        if (a.softcap > 0.0f) x = a.softcap * tanh(x / a.softcap);
        sc = x / a.temp;
        id[tid] = as_type<uint>(tk[64 + tid]);
    }
    p[tid] = sc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        const float mx = p[0];
        float sum = 0.0f;
        for (uint i = 0; i < K; i++) { p[i] = exp(p[i] - mx); sum += p[i]; }
        float cum = 0.0f, total = 0.0f;
        uint n = K;
        for (uint i = 0; i < K; i++) {
            const float v = p[i] / sum;
            p[i] = v; cum += v; total += v;
            if (cum >= a.top_p) { n = i + 1; break; }
        }
        ulong st = rng[a.rng_off + b] + 0x9E3779B97F4A7C15ul;
        rng[a.rng_off + b] = st;
        ulong z = st;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
        z ^= z >> 31;
        float r = float(z >> 40) / float(1u << 24) * total;
        uint pick = id[n - 1];
        for (uint i = 0; i < n; i++) { r -= p[i]; if (r <= 0.0f) { pick = id[i]; break; } }
        toks[a.out_base + b] = (int) pick;
    }
}

// ============================================================================
// Helper kernels: prefix KV linearization + prefix CS broadcast (AoS -> slot-major SoA ring!)
// ============================================================================

struct CopyPreKvArgs { uint pre_n, pre_p0, pre_cap, src_cap, is_swa, pad0, pad1, pad2; };
kernel void k_copy_prefix_kv(device const kv_t * SRC [[buffer(3)]], device kv_t * DST [[buffer(8)]], constant CopyPreKvArgs & a [[buffer(5)]],
                             uint3 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint i = tgp.x, g = tgp.y, s = tgp.z;
    const ulong dst_k = (((ulong) s * 2 + 0) * NKV + g) * a.pre_cap * HD + (ulong) i * HD + lane * 4;
    const ulong dst_v = (((ulong) s * 2 + 1) * NKV + g) * a.pre_cap * HD + (ulong) i * HD + lane * 4;
    if (i < a.pre_n) {
        const uint p = a.pre_p0 + i;
        const uint si = a.is_swa ? (p % a.src_cap) : p;
        const ulong src_k = (((ulong) s * 2 + 0) * NKV + g) * a.src_cap * HD + (ulong) si * HD + lane * 4;
        const ulong src_v = (((ulong) s * 2 + 1) * NKV + g) * a.src_cap * HD + (ulong) si * HD + lane * 4;
        *(device kvw_t *) (DST + dst_k) = *(device const kvw_t *) (SRC + src_k);
        *(device kvw_t *) (DST + dst_v) = *(device const kvw_t *) (SRC + src_v);
    } else {
        *(device kvw_t *) (DST + dst_k) = (kvw_t) 0;
        *(device kvw_t *) (DST + dst_v) = (kvw_t) 0;
    }
}

// Unused legacy placeholder kept in KERNELS list if needed
struct CopyToPagedArgs { uint n_pos, pos0, src_cap, is_swa, pool_cap, pt_off, pt_mod, suf0; };
kernel void k_copy_kv_to_paged(device const kv_t * SRC [[buffer(3)]], device kv_t * POOL [[buffer(8)]],
                               constant CopyToPagedArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                               uint3 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint i = tgp.x, g = tgp.y, s = tgp.z;
    const uint p = a.pos0 + i, suf = a.suf0 + i;
    const uint si = a.is_swa ? (p % a.src_cap) : p;
    const ulong src_k = (((ulong) s * 2 + 0) * NKV + g) * a.src_cap * HD + (ulong) si * HD + lane * 4;
    const ulong src_v = (((ulong) s * 2 + 1) * NKV + g) * a.src_cap * HD + (ulong) si * HD + lane * 4;
    const uint bidx = a.is_swa ? ((suf / KV_BLK) % a.pt_mod) : (suf / KV_BLK);
    const uint pid = META[a.pt_off + bidx];
    const uint boff = suf % KV_BLK;
    const ulong dst_k = paged_kv_base(pid, a.pool_cap, s, g, 0) + (ulong) boff * HD + lane * 4;
    const ulong dst_v = paged_kv_base(pid, a.pool_cap, s, g, 1) + (ulong) boff * HD + lane * 4;
    *(device kvw_t *) (POOL + dst_k) = *(device const kvw_t *) (SRC + src_k);
    *(device kvw_t *) (POOL + dst_v) = *(device const kvw_t *) (SRC + src_v);
}

// Broadcast Engine's AoS prefix conv state [48][NCH][3] into slot-major SoA ring CS_POOL at ring offset prefix_len % 3!
// Grid: (ceil(CS_NCH / 256), n, 48), 256 threads.
struct CopyCsArgs { uint dst_off, n, max_batch, prefix_len; };
kernel void k_broadcast_cs(device const float * SRC [[buffer(4)]], device float * POOL [[buffer(8)]],
                           constant CopyCsArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                           uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint ch = tgp.x * 256 + tid;
    if (ch >= CS_NCH) return;
    const uint seq_cs = META[a.dst_off + tgp.y];
    const uint slot = tgp.z;
    device const float * src = SRC + ((ulong) slot * CS_NCH + ch) * 3ul;
    const float h0 = src[0], h1 = src[1], h2 = src[2];
    const uint tap0 = a.prefix_len % 3u;
    const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
    const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
    uint branch, lc, dim;
    if (ch < NKV * HD) { branch = 0; lc = ch; dim = NKV * HD; }
    else if (ch < 2 * NKV * HD) { branch = 1; lc = ch - NKV * HD; dim = NKV * HD; }
    else if (ch < 2 * NKV * HD + E) { branch = 2; lc = ch - 2 * NKV * HD; dim = E; }
    else { branch = 3; lc = ch - (2 * NKV * HD + E); dim = E; }
    device float * dst = POOL + cs_branch_base(slot, branch, a.max_batch) + (ulong) seq_cs * (3ul * dim);
    dst[(ulong) tap0 * dim + lc] = h0;
    dst[(ulong) tap1 * dim + lc] = h1;
    dst[(ulong) tap2 * dim + lc] = h2;
}

// Fast GPU zero of specific sequence slots in slot-major CS_POOL. Grid: (ceil(CS_NCH / 256), n, 48), 256 threads.
kernel void k_zero_cs_slots(device float * POOL [[buffer(4)]], constant CopyCsArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                            uint3 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint ch = tgp.x * 256 + tid;
    if (ch >= CS_NCH) return;
    const uint seq_cs = META[a.dst_off + tgp.y];
    const uint slot = tgp.z;
    uint branch, lc, dim;
    if (ch < NKV * HD) { branch = 0; lc = ch; dim = NKV * HD; }
    else if (ch < 2 * NKV * HD) { branch = 1; lc = ch - NKV * HD; dim = NKV * HD; }
    else if (ch < 2 * NKV * HD + E) { branch = 2; lc = ch - 2 * NKV * HD; dim = E; }
    else { branch = 3; lc = ch - (2 * NKV * HD + E); dim = E; }
    device float * dst = POOL + cs_branch_base(slot, branch, a.max_batch) + (ulong) seq_cs * (3ul * dim);
    dst[0 * dim + lc] = 0.0f;
    dst[1 * dim + lc] = 0.0f;
    dst[2 * dim + lc] = 0.0f;
}

// ============================================================================
// Packed Multi-Prompt Batched Prefill Kernels (using slot-major KV + slot-major SoA ring CS!)
// ============================================================================
struct PackedConvArgs {
    uint y_off, ldy, res_off, out_off, layer_slot, branch, max_batch, convw_t_off;
    uint has_res, seq_start_off, seq_p0_off, tok_seq_off, seq_cs_off, T, pad0, pad1;
};
kernel void k_packed_postconv_b(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device float * CS [[buffer(4)]],
                                constant PackedConvArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                                uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint t = tgp.y, c = tgp.x * 256 + tid;
    const uint s = META[a.tok_seq_off + t];
    const int lpos = int(t - META[a.seq_start_off + s]);
    const uint p0 = META[a.seq_p0_off + s];
    const uint tap0 = p0 % 3u;
    const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
    const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
    device const float * hb = CS + cs_branch_base(a.layer_slot, a.branch, a.max_batch) + (ulong) META[a.seq_cs_off + s] * (3ul * E);
    device const float * w = (device const float *) (PB + a.convw_t_off);
    const uint taps[3] = { tap0, tap1, tap2 };
    float x[4];
    for (int j = 0; j < 4; j++) {
        const int lt = lpos - 3 + j;
        x[j] = lt >= 0 ? A[a.y_off + (ulong) (int(t) - 3 + j) * a.ldy + c] : hb[(ulong) taps[3 + lt] * E + c];
    }
    const float o = (x[3] + w[0 * E + c] * x[0] + w[1 * E + c] * x[1] + w[2 * E + c] * x[2] + w[3 * E + c] * x[3]) * 0.4472135954999579f;
    A[a.out_off + (ulong) t * E + c] = (a.has_res ? A[a.res_off + (ulong) t * E + c] : 0.0f) + o;
}

struct PackedStateArgs {
    uint x_off, ldx, layer_slot, branch, max_batch, nch;
    uint seq_start_off, seq_len_off, seq_p0_off, seq_cs_off, pad0, pad1;
};
kernel void k_packed_state_upd(device float * A [[buffer(2)]], device float * CS [[buffer(4)]],
                               constant PackedStateArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                               uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint c = tgp.x * 256 + tid, s = tgp.y;
    if (c >= a.nch) return;
    const uint t0 = META[a.seq_start_off + s], len = META[a.seq_len_off + s], p0 = META[a.seq_p0_off + s];
    const uint dim = a.nch;
    device float * hb = CS + cs_branch_base(a.layer_slot, a.branch, a.max_batch) + (ulong) META[a.seq_cs_off + s] * (3ul * dim);
    const uint ot0 = p0 % 3u, ot1 = (ot0 == 2u) ? 0u : (ot0 + 1u), ot2 = (ot1 == 2u) ? 0u : (ot1 + 1u);
    const float oldh[3] = { hb[(ulong) ot0 * dim + c], hb[(ulong) ot1 * dim + c], hb[(ulong) ot2 * dim + c] };
    float v[3];
    for (int j = 0; j < 3; j++) {
        const int idx = int(len) + j;
        v[j] = idx >= 3 ? A[a.x_off + (ulong) (t0 + idx - 3) * a.ldx + c] : oldh[idx];
    }
    const uint nt0 = (p0 + len) % 3u, nt1 = (nt0 == 2u) ? 0u : (nt0 + 1u), nt2 = (nt1 == 2u) ? 0u : (nt1 + 1u);
    hb[(ulong) nt0 * dim + c] = v[0];
    hb[(ulong) nt1 * dim + c] = v[1];
    hb[(ulong) nt2 * dim + c] = v[2];
}

struct PackedKvPrepArgs {
    uint T, pool_cap, slot_idx, is_swa, pt_stride, pt_mod;
    uint tok_pos_off, tok_seq_off, seq_start_off, seq_p0_off, seq_cs_off, pt_off;
    uint layer_slot, max_batch;
    uint q_off, k_off, v_off, r_off; int kdyn_off, vdyn_off;
    uint qp_off, rp_off, kconv_t_w, vconv_t_w, qclip_w, relclip_w;
    float log_alpha, log_floor;
};
kernel void k_packed_kvprep_b(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device kv_t * KV [[buffer(3)]],
                              device float * CS [[buffer(4)]], constant PackedKvPrepArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                              uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[4][8];
    const uint t = tgp.x, g = tgp.y;
    const uint pos = META[a.tok_pos_off + t];
    const uint s = META[a.tok_seq_off + t];
    const int lpos = int(t - META[a.seq_start_off + s]);
    const uint p0 = META[a.seq_p0_off + s];
    const uint tap0 = p0 % 3u, tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u), tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
    const uint taps[3] = { tap0, tap1, tap2 };
    const float tau = a.is_swa ? 1.0f : (1.0f + a.log_alpha * log(max(1.0f, float(pos + 1u) / a.log_floor)));
    device const float * qclip = (device const float *) (PB + a.qclip_w);
    float q[3];
    for (int h = 0; h < 3; h++) {
        const uint H = g * 3 + h;
        const half qh = ((device const half *) (A + a.q_off))[(ulong) t * (NH * HD) + H * HD + tid];
        q[h] = float(qh) * qclip[H * HD + tid];
    }
    const uint seq_cs = META[a.seq_cs_off + s];
    const uint c = g * HD + tid;
    const uint dim = NKV * HD;
    float kv[2];
    const float scl = a.kdyn_off >= 0 ? (1.0f / 3.0f) : 0.4472135954999579f;
    for (int which = 0; which < 2; which++) {
        const uint xo = which == 0 ? a.k_off : a.v_off;
        device const float * hb = CS + cs_branch_base(a.layer_slot, which, a.max_batch) + (ulong) seq_cs * (3ul * dim);
        device const float * w = (device const float *) (PB + (which == 0 ? a.kconv_t_w : a.vconv_t_w));
        float x[4];
        for (int j = 0; j < 4; j++) {
            const int lt = lpos - 3 + j;
            x[j] = lt >= 0 ? A[xo + (ulong) (int(t) - 3 + j) * dim + c] : hb[(ulong) taps[3 + lt] * dim + c];
        }
        float acc = w[0 * dim + c] * x[0] + w[1 * dim + c] * x[1] + w[2 * dim + c] * x[2] + w[3 * dim + c] * x[3];
        const int dyn = which == 0 ? a.kdyn_off : a.vdyn_off;
        if (dyn >= 0) { device const float * tp = A + dyn + (ulong) t * 32 + g * 4; acc += tp[0] * x[3] + tp[1] * x[2] + tp[2] * x[1] + tp[3] * x[0]; }
        kv[which] = (x[3] + acc) * scl;
    }
    float r = 0;
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16, H = g * 3 + h;
        r = A[a.r_off + (ulong) t * (NH * NREL) + H * NREL + i] * ((device const float *) (PB + a.relclip_w))[H * NREL + i];
    }
    float rn = r * r;
    rn += simd_shuffle_xor(rn, 1); rn += simd_shuffle_xor(rn, 2); rn += simd_shuffle_xor(rn, 4); rn += simd_shuffle_xor(rn, 8);
    {
        const float s0 = simd_sum(q[0] * q[0]), s1 = simd_sum(q[1] * q[1]), s2 = simd_sum(q[2] * q[2]);
        const float s3 = simd_sum(kv[0] * kv[0]), s4 = simd_sum(kv[1] * kv[1]);
        const float s5 = simd_max(fabs(kv[0])), s6 = simd_max(fabs(kv[1]));
        if (lane == 0) { red[sg][0] = s0; red[sg][1] = s1; red[sg][2] = s2; red[sg][3] = s3; red[sg][4] = s4; red[sg][5] = s5; red[sg][6] = s6; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot[5];
    for (int i = 0; i < 5; i++) tot[i] = red[0][i] + red[1][i] + red[2][i] + red[3][i];
    for (int h = 0; h < 3; h++) {
        const float f = clamp(QK_CLIP / (sqrt(tot[h]) + 1e-6f), 0.0f, 1.0f) * tau * 0.08838834764831845f;
        ((device half *) (A + a.qp_off))[(ulong) t * (NH * HD) + (g * 3 + h) * HD + tid] = half(q[h] * f);
    }
    if (tid < 48) A[a.rp_off + (ulong) t * (NH * NREL) + (g * 3 + tid / 16) * NREL + tid % 16] = r * clamp(REL_CLIP / (sqrt(rn) + 1e-6f), 0.0f, 1.0f) * tau;
    const float kf = clamp(QK_CLIP / (sqrt(tot[3]) + 1e-6f), 0.0f, 1.0f), vf = rsqrt(tot[4] / HD + 1e-6f);
    const float k = kv[0] * kf, v = kv[1] * vf;
    device const uint * pt = META + a.pt_off + (ulong) s * a.pt_stride;
    const uint bidx = a.is_swa ? ((pos / KV_BLK) % a.pt_mod) : (pos / KV_BLK);
    const uint boff = pos % KV_BLK;
    const uint pid = pt[bidx];
    const ulong kb = paged_kv_base(pid, a.pool_cap, a.slot_idx, g, 0) + (ulong) boff * HD;
    const ulong vb = paged_kv_base(pid, a.pool_cap, a.slot_idx, g, 1) + (ulong) boff * HD;
#ifdef KV8
    const uint ck = kv_code(max(max(red[0][5], red[1][5]), max(red[2][5], red[3][5])) * kf);
    const uint cv = kv_code(max(max(red[0][6], red[1][6]), max(red[2][6], red[3][6])) * vf);
    KV[kb + tid] = (kv_t) (int) kv_quant(k, kv_scale(ck), ck, tid);
    KV[vb + tid] = (kv_t) (int) kv_quant(v, kv_scale(cv), cv, tid);
#else
    KV[kb + tid] = half(k);
    KV[vb + tid] = half(v);
#endif
}

#define AKP 32
#define AGP 64
struct PackedAttnArgs {
    uint pool_cap, slot_idx, is_swa, pt_stride, pt_mod;
    uint pt_off, qtiles_off, qp_off, rp_off, sink_off, gate_off, out_off, relbias_w, pad0, pad1, pad2;
};
kernel void k_packed_attn_h(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]], device kv_t * KV [[buffer(3)]],
                            constant PackedAttnArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                            uint2 tgp [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float ss[AQ * AKP];
    threadgroup float G[AQ * AGP];
    threadgroup half sp[AQ * AKP];
    threadgroup float sr[AQ];
    const uint h = tgp.y, g = h / 3;
    device const uint * qt = META + a.qtiles_off + tgp.x * 4;
    const uint s = qt[0], t0 = qt[1], pfirst = qt[2], nq = qt[3];
    const uint plast = pfirst + nq - 1u;
    const uint window = a.is_swa ? (uint) EXT : 0u;
    auto tq = tensor((device half *) (A + a.qp_off) + (ulong) t0 * (NH * HD) + h * HD, dextents<int32_t, 2>(HD, (int) nq), array<int, 2>({1, NH * HD}));
    auto tr = tensor(A + a.rp_off + (ulong) t0 * (NH * NREL) + h * NREL, dextents<int32_t, 2>(NREL, (int) nq), array<int, 2>({1, NH * NREL}));
    auto ts = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(ss, dextents<int32_t, 2>(AKP, AQ));
    auto tg = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(G, dextents<int32_t, 2>(AGP, AQ));
    auto tp = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sp, dextents<int32_t, 2>(AKP, AQ));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, AKP, HD, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
                              execution_simdgroups<8>> mm_qk;
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, AGP, NREL, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
                              execution_simdgroups<8>> mm_g;
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(AQ, HD, AKP, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
                              execution_simdgroups<8>> mm_pv;
    device const uint * pt = META + a.pt_off + (ulong) s * a.pt_stride;
    auto tv0 = tensor(KV, dextents<int32_t, 2>(HD, AKP), array<int, 2>({1, HD}));
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    for (short i = 0; i < 16; ++i) co[i] = 0.0f;
    ushort cd[4], cr[4];
    for (short q = 0; q < 4; ++q) { const auto ix = co.get_multidimensional_index(4 * q); cd[q] = ix[0]; cr[q] = ix[1]; }
    float M[4], S[4];
    for (short jj = 0; jj < 4; jj++) {
        const uint j = jj * 8 + sg;
        M[jj] = j < nq ? A[a.sink_off + (ulong) (t0 + j) * NH + h] : 0.0f;
        S[jj] = 2980.9579870417283f;
    }
    device const float * relb = (device const float *) (PB + a.relbias_w);
    uint kstart = (window > 0 && pfirst + 1u > window) ? (pfirst + 1u - window) : 0u;
    kstart = kstart / AKP * AKP;
    for (uint kc = kstart; kc <= plast; kc += AKP) {
        const uint bidx = a.is_swa ? ((kc / KV_BLK) % a.pt_mod) : (kc / KV_BLK);
        const uint pid = pt[bidx];
        device kv_t * kblk = KV + paged_kv_base(pid, a.pool_cap, a.slot_idx, g, 0);
        device kv_t * vblk = KV + paged_kv_base(pid, a.pool_cap, a.slot_idx, g, 1);
        auto tk = tensor(kblk, dextents<int32_t, 2>(HD, AKP), array<int, 2>({1, HD}));
        { auto cs = mm_qk.template get_destination_cooperative_tensor<decltype(tq), decltype(tk), float>(); mm_qk.run(tq, tk, cs); cs.store(ts); }
        const int D0 = int(pfirst) - int(kc);
        const int dlo = max(D0 - (AKP - 1), 0);
        const bool bias = (dlo < EXT);
        if (bias) {
            auto tb = tensor((device float *) relb + dlo * NREL, dextents<int32_t, 2>(NREL, AGP), array<int, 2>({1, NREL}));
            auto cg = mm_g.template get_destination_cooperative_tensor<decltype(tr), decltype(tb), float>(); mm_g.run(tr, tb, cg); cg.store(tg);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#ifdef KV8
        const float ks0 = kv_vscale(kblk + (ulong) lane * HD);
        const float vs0 = kv_vscale(vblk + (ulong) lane * HD);
#endif
        for (short jj = 0; jj < 4; jj++) {
            const uint j = jj * 8 + sg;
#ifdef KV8
            float s0 = ss[j * AKP + lane] * ks0;
#else
            float s0 = ss[j * AKP + lane];
#endif
            const int d0 = D0 + int(j) - int(lane);
            if (bias && d0 >= 0 && d0 < EXT) s0 += G[j * AGP + clamp(d0 - dlo, 0, AGP - 1)];
            if (!(j < nq && d0 >= 0 && (window == 0 || d0 < (int) window))) s0 = -INFINITY;
            const float m = simd_max(max(M[jj], s0));
            const float ms = exp(M[jj] - m);
            M[jj] = m;
            const half p0 = half(exp(s0 - m + 8.0f));
#ifdef KV8
            sp[j * AKP + lane] = half(float(p0) * vs0);
#else
            sp[j * AKP + lane] = p0;
#endif
            S[jj] = S[jj] * ms + simd_sum(float(p0));
            if (lane == 0) sr[j] = ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const float r0 = sr[cr[0]], r1 = sr[cr[1]], r2 = sr[cr[2]], r3 = sr[cr[3]];
            co[0] *= r0; co[1] *= r0; co[2] *= r0; co[3] *= r0;
            co[4] *= r1; co[5] *= r1; co[6] *= r1; co[7] *= r1;
            co[8] *= r2; co[9] *= r2; co[10] *= r2; co[11] *= r2;
            co[12] *= r3; co[13] *= r3; co[14] *= r3; co[15] *= r3;
        }
        auto tv = tensor(vblk, dextents<int32_t, 2>(HD, AKP), array<int, 2>({1, HD}));
        mm_pv.run(tp, tv, co);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short jj = 0; jj < 4; jj++) { if (lane == 0) sr[jj * 8 + sg] = 1.0f / S[jj]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device half * out = (device half *) (A + a.out_off);
    for (short q = 0; q < 4; ++q) {
        const uint row = cr[q];
        if (row < nq) {
            const ulong o = (ulong) (t0 + row) * (NH * HD) + h * HD + cd[q];
            const float4 gt = float4(*(device const half4 *) ((device const half *) (A + a.gate_off) + o));
            const float4 v = float4(co[4 * q], co[4 * q + 1], co[4 * q + 2], co[4 * q + 3]) * sr[row] / (1.0f + exp(-gt));
            *(device half4 *) (out + o) = half4(v);
        }
    }
}

struct OutMixGatherArgs {
    uint nS, S, s_off[6];
    uint q_off, norm_off, hout_off, last_idx_off;
    float eps; uint pad0, pad1, pad2;
};
kernel void k_out_mix_gather(device const uchar * PB [[buffer(1)]], device float * A [[buffer(2)]],
                             constant OutMixGatherArgs & a [[buffer(5)]], device const uint * META [[buffer(7)]],
                             uint s [[threadgroup_position_in_grid]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint tok = META[a.last_idx_off + s];
    const ulong toff = (ulong) tok * E;
    const uint nS = a.nS;
    device const float4 * Q = (device const float4 *) (PB + a.q_off);
    float G[6][6], D[6];
    for (uint i = 0; i < 6; i++) { D[i] = 0; for (uint j = 0; j < 6; j++) G[i][j] = 0; }
    for (uint k = lane; k < E / 4; k += 32) {
        float4 v[6];
        for (uint i = 0; i < nS; i++) v[i] = ((device const float4 *) (A + a.s_off[i] + toff))[k];
        const float4 qv = Q[k];
        for (uint i = 0; i < nS; i++) {
            for (uint j = i; j < nS; j++) G[i][j] += dot(v[i], v[j]);
            D[i] += dot(v[i], qv);
        }
    }
    for (uint i = 0; i < nS; i++) {
        for (uint j = i; j < nS; j++) { G[i][j] = simd_sum(G[i][j]); G[j][i] = G[i][j]; }
        D[i] = simd_sum(D[i]);
    }
    float sc[6], mx = -INFINITY;
    for (uint i = 0; i < nS; i++) { sc[i] = rsqrt(G[i][i] / E + 1e-6f) * D[i] * (1.0f / 48.0f); mx = max(mx, sc[i]); }
    float den = 0;
    for (uint i = 0; i < nS; i++) { sc[i] = exp(sc[i] - mx); den += sc[i]; }
    float al[6];
    for (uint i = 0; i < nS; i++) al[i] = sc[i] / den * rsqrt(G[i][i] / E + 1e-6f);
    float nn = 0;
    for (uint i = 0; i < nS; i++) for (uint j = 0; j < nS; j++) nn += al[i] * al[j] * G[i][j];
    const float rw = rsqrt(nn / E + a.eps);
    device const float4 * nw = (device const float4 *) (PB + a.norm_off);
    device half4 * out = (device half4 *) ((device half *) (A + a.hout_off) + (ulong) s * E);
    for (uint k = lane; k < E / 4; k += 32) {
        float4 o = 0;
        for (uint i = 0; i < nS; i++) o += al[i] * ((device const float4 *) (A + a.s_off[i] + toff))[k];
        out[k] = half4(o * rw * nw[k]);
    }
}
