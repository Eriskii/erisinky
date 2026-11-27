// ================================ Single-Stream Dual-Persona Option B LoRA kernels ================================
// Included after multistream.metal.
// Buffers:
//   0 = W (mmapped GGUF), 1 = PB (live patched params), 2 = A (activation arena), 3 = KV, 4 = CS,
//   5 = primary args, 6 = toks, 7 = CTR, 8 = secondary LoRA args,
//   10 = AD0 (Adapter 0 FP16 LoRA buffer), 11 = AD1 (Adapter 1 FP16 LoRA buffer).

inline ushort f32_to_bf16(float v) {
    uint u = as_type<uint>(v);
    uint r = u + (0x7FFFu + ((u >> 16) & 1u));
    return ushort(r >> 16);
}

// ---- 1. Merge dual linear LoRA into a BF16 matrix in PB (router & loop controllers)
// PB_live[o, k] = bf16(bf16f(PB_base[o, k]) + scale0 * (B0[o] . A0[:, k]) + scale1 * (B1[o] . A1[:, k]))
// Grid: (in_dim.div_ceil(256), out_dim, 1), TG = 256.
struct MergePbArgs {
    uint pb_off;
    uint a0_off, b0_off;
    uint a1_off, b1_off;
    uint out_dim, in_dim;
    uint r0, r1;
    float scale0, scale1;
    uint pad0;
};

kernel void k_plora_merge_pb(
    device const uchar * PB_BASE [[buffer(0)]],
    device uchar * PB_LIVE [[buffer(1)]],
    device const uchar * AD0 [[buffer(10)]],
    device const uchar * AD1 [[buffer(11)]],
    constant MergePbArgs & a [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]
) {
    threadgroup float b0_sh[32];
    threadgroup float b1_sh[32];

    const uint o = tgp.y;
    if (o >= a.out_dim) return;

    const bool use0 = (a.scale0 != 0.0f) && (a.r0 > 0u) && (a.a0_off != 0xFFFFFFFFu);
    const bool use1 = (a.scale1 != 0.0f) && (a.r1 > 0u) && (a.a1_off != 0xFFFFFFFFu);
    if (use0 && tid < a.r0) {
        device const half * b0 = (device const half *) (AD0 + a.b0_off) + (ulong) o * a.r0;
        b0_sh[tid] = float(b0[tid]);
    }
    if (use1 && tid < a.r1) {
        device const half * b1 = (device const half *) (AD1 + a.b1_off) + (ulong) o * a.r1;
        b1_sh[tid] = float(b1[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k = tgp.x * 256u + tid;
    if (k >= a.in_dim) return;

    device const ushort * src = (device const ushort *) (PB_BASE + a.pb_off);
    device ushort * dst = (device ushort *) (PB_LIVE + a.pb_off);
    float v = bf16f(src[(ulong) o * a.in_dim + k]);

    if (use0) {
        device const half * a0 = (device const half *) (AD0 + a.a0_off);
        float d = 0.0f;
        for (uint r = 0; r < a.r0; r++) {
            d += b0_sh[r] * float(a0[(ulong) r * a.in_dim + k]);
        }
        v += d * a.scale0;
    }
    if (use1) {
        device const half * a1 = (device const half *) (AD1 + a.a1_off);
        float d = 0.0f;
        for (uint r = 0; r < a.r1; r++) {
            d += b1_sh[r] * float(a1[(ulong) r * a.in_dim + k]);
        }
        v += d * a.scale1;
    }
    dst[(ulong) o * a.in_dim + k] = f32_to_bf16(v);
}

// ---- 2. Unified Dual Linear LoRA (works for both T=1 decode and T=1..4096 prefill!)
// Up to 8 segments in one dispatch (e.g. wq, wgate, wrel, wsink, wk, wv, or single wo).
// tmp layout in A: per segment `s`, token `t`: tmp0 at S.tmp_off + t * 32 + [0..r0), tmp1 at + [16..16+r1).
struct PLoraSeg {
    uint a0_off, b0_off;
    uint a1_off, b1_off;
    uint x_off, y_off;
    uint in_dim, out_dim;
    uint r0, r1;
    uint tmp_off;
    uint in_half, out_half;
    float scale0, scale1;
    uint pad0;
};

struct PLoraArgs {
    uint nseg, T, max_out_blocks, pad0;
    PLoraSeg seg[8];
};

// Stage A: compute tmp0[t, r] = scale0 * (A0[r] . x[t]), tmp1[t, r] = scale1 * (A1[r] . x[t]).
// Grid: (T, nseg, 1), TG = 256 (8 simdgroups). Stages x[t] (in_dim <= 3072) once into threadgroup memory.
kernel void k_plora_lin_a(
    device float * A [[buffer(2)]],
    device const uchar * AD0 [[buffer(10)]],
    device const uchar * AD1 [[buffer(11)]],
    constant PLoraArgs & a [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    threadgroup float xs[3072];
    const uint t = tgp.x, s = tgp.y;
    if (t >= a.T || s >= a.nseg) return;
    constant PLoraSeg & S = a.seg[s];

    if (S.in_half != 0u) {
        device const half * x = (device const half *) (A + S.x_off) + (ulong) t * S.in_dim;
        for (uint k = tid; k < S.in_dim; k += 256u) xs[k] = float(x[k]);
    } else {
        device const float * x = A + S.x_off + (ulong) t * S.in_dim;
        for (uint k = tid; k < S.in_dim; k += 256u) xs[k] = x[k];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float * dst = A + S.tmp_off + (ulong) t * 32u;
    const bool use0 = (S.scale0 != 0.0f) && (S.r0 > 0u) && (S.a0_off != 0xFFFFFFFFu);
    const bool use1 = (S.scale1 != 0.0f) && (S.r1 > 0u) && (S.a1_off != 0xFFFFFFFFu);

    if (use0) {
        device const half * a0 = (device const half *) (AD0 + S.a0_off);
        for (uint r = sg; r < S.r0; r += 8u) {
            device const half * row = a0 + (ulong) r * S.in_dim;
            float acc = 0.0f;
            for (uint k = lane * 4u; k < S.in_dim; k += 128u) {
                const float4 xv = *(threadgroup const float4 *) (xs + k);
                const float4 wv = float4(*(device const half4 *) (row + k));
                acc += dot(xv, wv);
            }
            acc = simd_sum(acc);
            if (lane == 0) dst[r] = acc * S.scale0;
        }
    } else if (tid < S.r0) {
        dst[tid] = 0.0f;
    }

    if (use1) {
        device const half * a1 = (device const half *) (AD1 + S.a1_off);
        for (uint r = sg; r < S.r1; r += 8u) {
            device const half * row = a1 + (ulong) r * S.in_dim;
            float acc = 0.0f;
            for (uint k = lane * 4u; k < S.in_dim; k += 128u) {
                const float4 xv = *(threadgroup const float4 *) (xs + k);
                const float4 wv = float4(*(device const half4 *) (row + k));
                acc += dot(xv, wv);
            }
            acc = simd_sum(acc);
            if (lane == 0) dst[16u + r] = acc * S.scale1;
        }
    } else if (tid < S.r1) {
        dst[16u + tid] = 0.0f;
    }
}

// Stage B: y[t, o] += B0[o] . tmp0[t] + B1[o] . tmp1[t]
// Grid: (max_out_blocks, T, nseg), where max_out_blocks = max(out_dim).div_ceil(256), TG = 256.
kernel void k_plora_lin_b(
    device float * A [[buffer(2)]],
    device const uchar * AD0 [[buffer(10)]],
    device const uchar * AD1 [[buffer(11)]],
    constant PLoraArgs & a [[buffer(5)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]
) {
    threadgroup float t0_sh[16];
    threadgroup float t1_sh[16];

    const uint t = tgp.y, s = tgp.z;
    if (t >= a.T || s >= a.nseg) return;
    constant PLoraSeg & S = a.seg[s];
    const uint o = tgp.x * 256u + tid;

    device const float * src = A + S.tmp_off + (ulong) t * 32u;
    const bool use0 = (S.scale0 != 0.0f) && (S.r0 > 0u) && (S.b0_off != 0xFFFFFFFFu);
    const bool use1 = (S.scale1 != 0.0f) && (S.r1 > 0u) && (S.b1_off != 0xFFFFFFFFu);
    if (use0 && tid < S.r0) t0_sh[tid] = src[tid];
    if (use1 && tid < S.r1) t1_sh[tid] = src[16u + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (o >= S.out_dim) return;
    float acc = 0.0f;
    if (use0) {
        device const half * bw = (device const half *) (AD0 + S.b0_off) + (ulong) o * S.r0;
        for (uint r = 0; r < S.r0; r++) acc += float(bw[r]) * t0_sh[r];
    }
    if (use1) {
        device const half * bw = (device const half *) (AD1 + S.b1_off) + (ulong) o * S.r1;
        for (uint r = 0; r < S.r1; r++) acc += float(bw[r]) * t1_sh[r];
    }
    if (S.out_half != 0u) {
        device half * y = (device half *) (A + S.y_off) + (ulong) t * S.out_dim;
        y[o] = half(float(y[o]) + acc);
    } else {
        A[S.y_off + (ulong) t * S.out_dim + o] += acc;
    }
}

// ---- 3. Decode post-conv + residual (used after wo + wo_lora in single-token decode)
// Grid: (E / 256, 1, 1), TG = 256.
struct PostConvDecArgs {
    uint y_off, out_off, cs_off, convw_off;
    int res_off;
    uint pad0, pad1, pad2;
};

kernel void k_postconv_dec(
    device const uchar * PB [[buffer(1)]],
    device float * A [[buffer(2)]],
    device float * CS [[buffer(4)]],
    constant PostConvDecArgs & a [[buffer(5)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]
) {
    const uint row = tg * 256u + tid;
    if (row >= E) return;
    const float y = A[a.y_off + row];
    device float * h = CS + a.cs_off + row * 3u;
    device const float * w = (device const float *) (PB + a.convw_off) + row * 4u;
    const float h0 = h[0], h1 = h[1], h2 = h[2];
    const float conv = w[0] * h0 + w[1] * h1 + w[2] * h2 + w[3] * y;
    h[0] = h1; h[1] = h2; h[2] = y;
    float o = (y + conv) * 0.4472135954999579f;
    if (a.res_off >= 0) o += A[a.res_off + row];
    A[a.out_off + row] = o;
}

// ---- 4. Decode Per-Expert MoE LoRA (T=1)
struct PLoraMoeAArgs {
    uint a0_off, a1_off;
    uint a0_exp_bytes, a1_exp_bytes;
    uint r0, r1;
    uint sel_off, x_off;
    uint x_stride, in_dim;
    uint tmp_off, mul_weight;
    float scale0, scale1;
    uint pad0, pad1;
};

// Stage A for decode MoE: Grid: (TOPK, 1, 1), TG = 256 (8 simdgroups).
// Computes tmp[k, 0..8] = scale0 * (mul_w ? w_k : 1) * (A0[e_k] . x_k) and tmp[k, 8..16] for Adapter 1.
kernel void k_plora_moe_a_dec(
    device float * A [[buffer(2)]],
    device const uchar * AD0 [[buffer(10)]],
    device const uchar * AD1 [[buffer(11)]],
    constant PLoraMoeAArgs & a [[buffer(5)]],
    uint k [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    threadgroup float xs[2304];
    if (k >= TOPK) return;
    const uint e = (uint) A[a.sel_off + k];
    const float wk = (a.mul_weight != 0u) ? A[a.sel_off + TOPK + k] : 1.0f;
    device const float * x = A + a.x_off + (ulong) k * a.x_stride;
    for (uint i = tid; i < a.in_dim; i += 256u) xs[i] = x[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float * dst = A + a.tmp_off + (ulong) k * 16u;
    const bool use0 = (a.scale0 != 0.0f) && (a.r0 > 0u) && (a.a0_off != 0xFFFFFFFFu);
    const bool use1 = (a.scale1 != 0.0f) && (a.r1 > 0u) && (a.a1_off != 0xFFFFFFFFu);

    if (use0 && sg < a.r0) {
        device const half * row = (device const half *) (AD0 + a.a0_off + (ulong) e * a.a0_exp_bytes) + (ulong) sg * a.in_dim;
        float acc = 0.0f;
        for (uint i = lane * 4u; i < a.in_dim; i += 128u) {
            acc += dot(*(threadgroup const float4 *) (xs + i), float4(*(device const half4 *) (row + i)));
        }
        acc = simd_sum(acc);
        if (lane == 0) dst[sg] = acc * (a.scale0 * wk);
    } else if (tid < 8u) {
        dst[tid] = 0.0f;
    }

    if (use1 && sg < a.r1) {
        device const half * row = (device const half *) (AD1 + a.a1_off + (ulong) e * a.a1_exp_bytes) + (ulong) sg * a.in_dim;
        float acc = 0.0f;
        for (uint i = lane * 4u; i < a.in_dim; i += 128u) {
            acc += dot(*(threadgroup const float4 *) (xs + i), float4(*(device const half4 *) (row + i)));
        }
        acc = simd_sum(acc);
        if (lane == 0) dst[8u + sg] = acc * (a.scale1 * wk);
    } else if (tid < 8u) {
        dst[8u + tid] = 0.0f;
    }
}

// Secondary LoRA args bound at buffer(8) for fused k_moe_gu_lora, k_moe_down_lora, k_moe_sg8_lora, k_moe_sg4_lora
struct PLoraMoeBArgs {
    uint b0_off, b1_off;
    uint b0_exp_bytes, b1_exp_bytes;
    uint r0, r1;
    uint tmp_off;
    uint flags; // bit 0: has0, bit 1: has1
};

// Decode gate/up + fused dual per-expert LoRA + SiTU. Grid: (FF / GU_NSG, TOPK), TG = 128.
kernel void k_moe_gu_lora(
    device const uchar * W [[buffer(0)]],
    device const uchar * PB [[buffer(1)]],
    device float * A [[buffer(2)]],
    constant MoeArgs & a [[buffer(5)]],
    constant PLoraMoeBArgs & la [[buffer(8)]],
    device const uchar * AD0 [[buffer(10)]],
    device const uchar * AD1 [[buffer(11)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint k = tgp.y;
    const int e = (int) A[a.sel_off + k];
    const uint i = tgp.x * GU_NSG + sg;
    const ulong base = (ulong(a.gu_hi) << 32 | a.gu_lo) + (ulong) e * a.gu_expert_bytes;
    device const uchar * wp[2] = { W + base + (ulong) i * a.gu_row_bytes, W + base + (ulong) (FF + i) * a.gu_row_bytes };
    float sumf[2] = {0, 0};
    dot_q4k<2>(wp, E / 256, A + a.x_off, lane, sumf);
    float gt = simd_sum(sumf[0]), up = simd_sum(sumf[1]);
    if (lane == 0) {
        if ((la.flags & 1u) != 0u) {
            device const half * be = (device const half *) (AD0 + la.b0_off + (ulong) e * la.b0_exp_bytes);
            device const half * bg = be + (ulong) i * la.r0;
            device const half * bu = be + (ulong) (FF + i) * la.r0;
            device const float * t0 = A + la.tmp_off + (ulong) k * 16u;
            for (uint r = 0; r < la.r0; r++) {
                const float tv = t0[r];
                gt += float(bg[r]) * tv;
                up += float(bu[r]) * tv;
            }
        }
        if ((la.flags & 2u) != 0u) {
            device const half * be = (device const half *) (AD1 + la.b1_off + (ulong) e * la.b1_exp_bytes);
            device const half * bg = be + (ulong) i * la.r1;
            device const half * bu = be + (ulong) (FF + i) * la.r1;
            device const float * t1 = A + la.tmp_off + (ulong) k * 16u + 8u;
            for (uint r = 0; r < la.r1; r++) {
                const float tv = t1[r];
                gt += float(bg[r]) * tv;
                up += float(bu[r]) * tv;
            }
        }
        const float act = a.beta * tanh(gt / a.beta) * sigm(gt) * (a.lbeta > 0 ? a.lbeta * tanh(up / a.lbeta) : up);
        A[a.act_off + k * FF + i] = act;
    }
}

// Decode down + fused dual per-expert LoRA + post-conv + residual. Grid: (E / 4, 1, 1), TG = 64.
kernel void k_moe_down_lora(
    device const uchar * W [[buffer(0)]],
    device const uchar * PB [[buffer(1)]],
    device float * A [[buffer(2)]],
    device float * CS [[buffer(4)]],
    constant MoeArgs & a [[buffer(5)]],
    constant PLoraMoeBArgs & la [[buffer(8)]],
    device const uchar * AD0 [[buffer(10)]],
    device const uchar * AD1 [[buffer(11)]],
    uint tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    const uint row0 = (tg * DN_NSG + sg) * DN_NR;
    const ulong base = ulong(a.dn_hi) << 32 | a.dn_lo;
    float acc[DN_NR] = {0, 0};
    for (int s = 0; s < 9; s++) {
        const int f = lane + 32 * s, k = f / 36, b = f % 36;
        device const float4 * x4 = (device const float4 *) (A + a.act_off + k * FF + b * 32);
        float4 xv[8];
        for (int i = 0; i < 8; i++) xv[i] = x4[i];
        const int e = (int) A[a.sel_off + k];
        down_rows<DN_NR>(W + base + (ulong) e * a.dn_expert_bytes, row0, a.dn_row_bytes, a.dn_type, b, xv, A[a.sel_off + TOPK + k], acc);
    }
    for (int rr = 0; rr < DN_NR; rr++) {
        float y = simd_sum(acc[rr]);
        if (lane == 0) {
            const uint row = row0 + rr;
            if ((la.flags & 3u) != 0u) {
                for (uint k = 0; k < TOPK; k++) {
                    const uint ek = (uint) A[a.sel_off + k];
                    if ((la.flags & 1u) != 0u) {
                        device const half * bd = (device const half *) (AD0 + la.b0_off + (ulong) ek * la.b0_exp_bytes) + (ulong) row * la.r0;
                        device const float * t0 = A + la.tmp_off + (ulong) k * 16u;
                        for (uint r = 0; r < la.r0; r++) y += float(bd[r]) * t0[r];
                    }
                    if ((la.flags & 2u) != 0u) {
                        device const half * bd = (device const half *) (AD1 + la.b1_off + (ulong) ek * la.b1_exp_bytes) + (ulong) row * la.r1;
                        device const float * t1 = A + la.tmp_off + (ulong) k * 16u + 8u;
                        for (uint r = 0; r < la.r1; r++) y += float(bd[r]) * t1[r];
                    }
                }
            }
            device float * h = CS + a.cs_off + row * 3u;
            device const float * w = (device const float *) (PB + a.convw_off) + row * 4u;
            const float h0 = h[0], h1 = h[1], h2 = h[2];
            const float conv = w[0] * h0 + w[1] * h1 + w[2] * h2 + w[3] * y;
            h[0] = h1; h[1] = h2; h[2] = y;
            A[a.out_off + row] = A[a.res_off + row] + (y + conv) * 0.4472135954999579f;
        }
    }
}

// ---- 5. Prefill Per-Expert MoE LoRA (T > 1, fp16 hgemm path)
struct PLoraMoePfAArgs {
    uint a0_off, a1_off;
    uint a0_exp_bytes, a1_exp_bytes;
    uint r0, r1;
    uint xh_off, sel_off, inv_off, tmp_off;
    uint T, in_dim;
    float scale0, scale1;
    uint pad0, pad1;
};

// Stage A13 for prefill: Grid (T, 1, 1), TG = 256 (8 simdgroups).
// Loads token t's half input xh[t] once into TG memory, computes t13[slot, 0..16] for all TOPK=8 experts of token t.
kernel void k_plora_moe_a13_pf(
    device float * A [[buffer(2)]],
    device const uchar * AD0 [[buffer(10)]],
    device const uchar * AD1 [[buffer(11)]],
    constant PLoraMoePfAArgs & a [[buffer(5)]],
    uint t [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    threadgroup float xs[2304];
    if (t >= a.T) return;
    device const half * x = (device const half *) (A + a.xh_off) + (ulong) t * a.in_dim;
    for (uint i = tid; i < a.in_dim; i += 256u) xs[i] = float(x[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const uint * inv = (device const uint *) (A + a.inv_off) + (ulong) t * TOPK;
    const bool use0 = (a.scale0 != 0.0f) && (a.r0 > 0u) && (a.a0_off != 0xFFFFFFFFu);
    const bool use1 = (a.scale1 != 0.0f) && (a.r1 > 0u) && (a.a1_off != 0xFFFFFFFFu);

    for (uint k = 0; k < TOPK; k++) {
        const uint e = (uint) A[a.sel_off + (ulong) t * 16u + k];
        const uint slot = inv[k];
        device float * dst = A + a.tmp_off + (ulong) slot * 16u;
        if (use0 && sg < a.r0) {
            device const half * row = (device const half *) (AD0 + a.a0_off + (ulong) e * a.a0_exp_bytes) + (ulong) sg * a.in_dim;
            float acc = 0.0f;
            for (uint i = lane * 4u; i < a.in_dim; i += 128u) {
                acc += dot(*(threadgroup const float4 *) (xs + i), float4(*(device const half4 *) (row + i)));
            }
            acc = simd_sum(acc);
            if (lane == 0) dst[sg] = acc * a.scale0;
        } else if (tid < 8u) {
            dst[tid] = 0.0f;
        }
        if (use1 && sg < a.r1) {
            device const half * row = (device const half *) (AD1 + a.a1_off + (ulong) e * a.a1_exp_bytes) + (ulong) sg * a.in_dim;
            float acc = 0.0f;
            for (uint i = lane * 4u; i < a.in_dim; i += 128u) {
                acc += dot(*(threadgroup const float4 *) (xs + i), float4(*(device const half4 *) (row + i)));
            }
            acc = simd_sum(acc);
            if (lane == 0) dst[8u + sg] = acc * a.scale1;
        } else if (tid < 8u) {
            dst[8u + tid] = 0.0f;
        }
    }
}

// Prefill grouped MoE gate/up + fused dual per-expert LoRA in registers before SiTU * aw.
template <int NSG>
kernel void k_moe_sg_lora(
    device const uchar * W [[buffer(0)]],
    device float * A [[buffer(2)]],
    constant MoeGemmHArgs & a [[buffer(5)]],
    constant PLoraMoeBArgs & la [[buffer(8)]],
    device const uchar * AD0 [[buffer(10)]],
    device const uchar * AD1 [[buffer(11)]],
    uint2 tgq [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]
) {
    threadgroup half sa[2][HB_M * HB_K];
    const uint cb = tgq.y;
    if (tgq.x >= ((device const uint *) (A + a.ntiles_off))[0]) return;
    device const uint * tl = (device const uint *) (A + a.tiles_off) + tgq.x * 4;
    const uint e = tl[0], as0 = tl[1], cnt = tl[2], ts0 = sg * 16;
    const bool on = ts0 < cnt;
    const int nt = on ? min(16, (int) (cnt - ts0)) : 0;
    const uint K = a.K;
    auto tA0 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[0], dextents<int32_t, 2>(HB_K, HB_M));
    auto tA1 = tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline>(sa[1], dextents<int32_t, 2>(HB_K, HB_M));
    auto tX = tensor((device half *) (A + a.xh_off) + (ulong) (as0 + (on ? ts0 : 0)) * K, dextents<int32_t, 2>((int) K, nt), array<int, 2>({1, (int) K}));
    SGOP(mm);
    auto cT = mm.template get_destination_cooperative_tensor<decltype(tX), decltype(tA0), float>();
    const bool wact = tid < 128;
    const uint widx = tid * 16, rr = widx / HB_K, hp = (widx % HB_K) / 16;
    const uint wrow = (rr < 32) ? cb * 32 + rr : FF + cb * 32 + (rr - 32);
    device const uchar * wr = W + (ulong(a.w_hi) << 32 | a.w_lo) + (ulong) e * a.expert_bytes + (ulong) wrow * a.row_bytes;
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
    device half * act = (device half *) (A + a.out_off);
    device const half * b0e = ((la.flags & 1u) != 0u) ? (device const half *) (AD0 + la.b0_off + (ulong) e * la.b0_exp_bytes) : nullptr;
    device const half * b1e = ((la.flags & 2u) != 0u) ? (device const half *) (AD1 + la.b1_off + (ulong) e * la.b1_exp_bytes) : nullptr;
    for (short q = 0; q < 4; ++q) {
        const auto ix = cT.get_multidimensional_index(4 * q);
        const uint r = ix[0], t = ts0 + ix[1];
        if (t < cnt) {
            float4 gt = float4(cT[4 * q], cT[4 * q + 1], cT[4 * q + 2], cT[4 * q + 3]);
            float4 up = float4(cT[4 * q + 16], cT[4 * q + 17], cT[4 * q + 18], cT[4 * q + 19]);
            const uint rg = cb * 32u + r;
            const uint ru = FF + rg;
            if (b0e != nullptr) {
                device const float * t0 = A + la.tmp_off + (ulong) (as0 + t) * 16u;
                for (uint j = 0; j < la.r0; j++) {
                    const float tv = t0[j];
                    gt.x += float(b0e[(ulong) (rg + 0u) * la.r0 + j]) * tv;
                    gt.y += float(b0e[(ulong) (rg + 1u) * la.r0 + j]) * tv;
                    gt.z += float(b0e[(ulong) (rg + 2u) * la.r0 + j]) * tv;
                    gt.w += float(b0e[(ulong) (rg + 3u) * la.r0 + j]) * tv;
                    up.x += float(b0e[(ulong) (ru + 0u) * la.r0 + j]) * tv;
                    up.y += float(b0e[(ulong) (ru + 1u) * la.r0 + j]) * tv;
                    up.z += float(b0e[(ulong) (ru + 2u) * la.r0 + j]) * tv;
                    up.w += float(b0e[(ulong) (ru + 3u) * la.r0 + j]) * tv;
                }
            }
            if (b1e != nullptr) {
                device const float * t1 = A + la.tmp_off + (ulong) (as0 + t) * 16u + 8u;
                for (uint j = 0; j < la.r1; j++) {
                    const float tv = t1[j];
                    gt.x += float(b1e[(ulong) (rg + 0u) * la.r1 + j]) * tv;
                    gt.y += float(b1e[(ulong) (rg + 1u) * la.r1 + j]) * tv;
                    gt.z += float(b1e[(ulong) (rg + 2u) * la.r1 + j]) * tv;
                    gt.w += float(b1e[(ulong) (rg + 3u) * la.r1 + j]) * tv;
                    up.x += float(b1e[(ulong) (ru + 0u) * la.r1 + j]) * tv;
                    up.y += float(b1e[(ulong) (ru + 1u) * la.r1 + j]) * tv;
                    up.z += float(b1e[(ulong) (ru + 2u) * la.r1 + j]) * tv;
                    up.w += float(b1e[(ulong) (ru + 3u) * la.r1 + j]) * tv;
                }
            }
            const float4 su = a.lbeta > 0 ? a.lbeta * tanh(up / a.lbeta) : up;
            const float4 o = (A[a.aw_off + as0 + t] * a.beta) * tanh(gt / a.beta) / (1.0f + exp(-gt)) * su;
            *(device half4 *) (act + (ulong) (as0 + t) * FF + cb * 32 + r) = half4(o);
        }
    }
}
template [[host_name("k_moe_sg8_lora")]] kernel void k_moe_sg_lora<8>(device const uchar *, device float *, constant MoeGemmHArgs &, constant PLoraMoeBArgs &, device const uchar *, device const uchar *, uint2, ushort, ushort);
template [[host_name("k_moe_sg4_lora")]] kernel void k_moe_sg_lora<4>(device const uchar *, device float *, constant MoeGemmHArgs &, constant PLoraMoeBArgs &, device const uchar *, device const uchar *, uint2, ushort, ushort);

// Stage A2 for prefill: Grid (T, TOPK, 1), TG = 256 (8 simdgroups).
// Reads acth[slot] (already multiplied by routing weight aw!) and writes t2[slot, 0..16].
kernel void k_plora_moe_a2_pf(
    device float * A [[buffer(2)]],
    device const uchar * AD0 [[buffer(10)]],
    device const uchar * AD1 [[buffer(11)]],
    constant PLoraMoePfAArgs & a [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    threadgroup float xs[1152];
    const uint t = tgp.x, k = tgp.y;
    if (t >= a.T || k >= TOPK) return;
    const uint e = (uint) A[a.sel_off + (ulong) t * 16u + k];
    const uint slot = ((device const uint *) (A + a.inv_off))[(ulong) t * TOPK + k];

    device const half * x = (device const half *) (A + a.xh_off) + (ulong) slot * FF;
    for (uint i = tid; i < FF; i += 256u) xs[i] = float(x[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float * dst = A + a.tmp_off + (ulong) slot * 16u;
    const bool use0 = (a.scale0 != 0.0f) && (a.r0 > 0u) && (a.a0_off != 0xFFFFFFFFu);
    const bool use1 = (a.scale1 != 0.0f) && (a.r1 > 0u) && (a.a1_off != 0xFFFFFFFFu);

    if (use0 && sg < a.r0) {
        device const half * row = (device const half *) (AD0 + a.a0_off + (ulong) e * a.a0_exp_bytes) + (ulong) sg * FF;
        float acc = 0.0f;
        for (uint i = lane * 4u; i < FF; i += 128u) {
            acc += dot(*(threadgroup const float4 *) (xs + i), float4(*(device const half4 *) (row + i)));
        }
        acc = simd_sum(acc);
        if (lane == 0) dst[sg] = acc * a.scale0;
    } else if (tid < 8u) {
        dst[tid] = 0.0f;
    }

    if (use1 && sg < a.r1) {
        device const half * row = (device const half *) (AD1 + a.a1_off + (ulong) e * a.a1_exp_bytes) + (ulong) sg * FF;
        float acc = 0.0f;
        for (uint i = lane * 4u; i < FF; i += 128u) {
            acc += dot(*(threadgroup const float4 *) (xs + i), float4(*(device const half4 *) (row + i)));
        }
        acc = simd_sum(acc);
        if (lane == 0) dst[8u + sg] = acc * a.scale1;
    } else if (tid < 8u) {
        dst[8u + tid] = 0.0f;
    }
}

// Stage B2 for prefill: adds sum_{k=0..7} (B2_0[e_k] . t2_0[slot_k] + B2_1[e_k] . t2_1[slot_k]) into ffn_y[t] (f32).
// Grid: (T, E / 256, 1), TG = 256.
struct PLoraMoePfB2Args {
    uint b0_off, b1_off;
    uint b0_exp_bytes, b1_exp_bytes;
    uint r0, r1;
    uint sel_off, inv_off, tmp_off, y_off;
    uint T, flags; // bit 0: has0, bit 1: has1
};

kernel void k_plora_moe_b2_pf(
    device float * A [[buffer(2)]],
    device const uchar * AD0 [[buffer(10)]],
    device const uchar * AD1 [[buffer(11)]],
    constant PLoraMoePfB2Args & a [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]
) {
    threadgroup uint tg_e[TOPK];
    threadgroup float tg_t2[TOPK][16];

    const uint t = tgp.x;
    if (t >= a.T) return;
    if (tid < TOPK) {
        tg_e[tid] = (uint) A[a.sel_off + (ulong) t * 16u + tid];
    }
    if (tid < TOPK * 16u) {
        const uint k = tid / 16u, r = tid % 16u;
        const uint slot = ((device const uint *) (A + a.inv_off))[(ulong) t * TOPK + k];
        tg_t2[k][r] = A[a.tmp_off + (ulong) slot * 16u + r];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint c = tgp.y * 256u + tid;
    if (c >= E) return;

    float acc = 0.0f;
    for (uint k = 0; k < TOPK; k++) {
        const uint ek = tg_e[k];
        if ((a.flags & 1u) != 0u) {
            device const half * bd = (device const half *) (AD0 + a.b0_off + (ulong) ek * a.b0_exp_bytes) + (ulong) c * a.r0;
            for (uint r = 0; r < a.r0; r++) acc += float(bd[r]) * tg_t2[k][r];
        }
        if ((a.flags & 2u) != 0u) {
            device const half * bd = (device const half *) (AD1 + a.b1_off + (ulong) ek * a.b1_exp_bytes) + (ulong) c * a.r1;
            for (uint r = 0; r < a.r1; r++) acc += float(bd[r]) * tg_t2[k][8u + r];
        }
    }
    A[a.y_off + (ulong) t * E + c] += acc;
}
