// ================================ 7-channel Multi-Stream + Pass-Specific LoRA kernels ================================
// Included after bench_kernels.metal.
// Uses existing constants (E, NH, NKV, HD, NREL, EXT, QK_CLIP, REL_CLIP, CS_NCH) and helpers
// (block_q4_K, scale_min_k4, tg_sum, bf16f, sigm, cs_branch_base).

// ---- 1. Multi-Stream Embedding: Q4_K embed + optional silence/tool/peer delta + RMSNorm + channel_embedding
struct MsEmbedArgs {
    uint emb_lo, emb_hi, row_bytes, norm_off;
    uint out_off, silence_id, tool_id, peer_id;
    uint ch_emb_off;     // byte offset in AD (f32 [7][E])
    uint sil_emb_off;    // byte offset in AD (f32 [E]) or 0xFFFFFFFF
    uint tool_emb_off;   // byte offset in AD (f32 [E]) or 0xFFFFFFFF
    uint peer_emb_off;   // byte offset in AD (f32 [E]) or 0xFFFFFFFF
    float eps;
    uint pad0, pad1, pad2;
};

kernel void k_ms_embed(
    device const uchar * W [[buffer(0)]],
    device const uchar * PB [[buffer(1)]],
    device float * A [[buffer(2)]],
    device const uchar * AD [[buffer(3)]],
    constant MsEmbedArgs & a [[buffer(5)]],
    device const int * toks [[buffer(6)]],
    device const uint * chs [[buffer(7)]],
    uint m [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    threadgroup float sh[32];
    const int t = toks[m];
    const uint ch = chs[m];
    const ulong off = (ulong(a.emb_hi) << 32 | a.emb_lo) + (ulong) t * a.row_bytes;
    device const block_q4_K * row = (device const block_q4_K *) (W + off);

    const bool add_sil = (a.sil_emb_off != 0xFFFFFFFFu) && (uint(t) == a.silence_id);
    const bool add_tool = (a.tool_emb_off != 0xFFFFFFFFu) && (uint(t) == a.tool_id);
    const bool add_peer = (a.peer_emb_off != 0xFFFFFFFFu) && (uint(t) == a.peer_id);
    device const float * sil_d = add_sil ? (device const float *) (AD + a.sil_emb_off) : nullptr;
    device const float * tool_d = add_tool ? (device const float *) (AD + a.tool_emb_off) : nullptr;
    device const float * peer_d = add_peer ? (device const float *) (AD + a.peer_emb_off) : nullptr;

    float v[9];
    float ss = 0.0f;
    for (int k = 0; k < 9; k++) {
        const int j = tid, pair = j / 64, l = j % 64;
        device const block_q4_K & b = row[k];
        const int sub = pair * 2 + (l >= 32 ? 1 : 0);
        const uchar q = b.qs[pair * 32 + (l & 31)];
        const int qv = l >= 32 ? (q >> 4) : (q & 0xF);
        float sc, mn;
        scale_min_k4(sub, b.scales, sc, mn);
        float x = float(b.d) * sc * qv - float(b.dmin) * mn;
        const int i = tid + k * 256;
        if (add_sil) x += sil_d[i];
        if (add_tool) x += tool_d[i];
        if (add_peer) x += peer_d[i];
        v[k] = x;
        ss += x * x;
    }
    ss = tg_sum(ss, sh, sg, lane, 8);
    const float r = rsqrt(ss / E + a.eps);
    device const float * w = (device const float *) (PB + a.norm_off);
    device const float * ce = (a.ch_emb_off != 0xFFFFFFFFu)
        ? ((device const float *) (AD + a.ch_emb_off) + (ulong) ch * E)
        : nullptr;
    for (int k = 0; k < 9; k++) {
        const int i = tid + k * 256;
        float o = v[k] * r * w[i];
        if (ce != nullptr) o += ce[i];
        A[a.out_off + m * E + i] = o;
    }
}

// ---- 2. Batched Pass-Specific LoRA add: y += scale * B @ (A @ x) for M <= 8 tokens, up to 8 segments
struct MsLoraSeg {
    uint a_off, b_off;       // byte offsets in AD (fp16: A[rank][in_dim], B[out_dim][rank])
    uint x_off, y_off;       // float-word offsets in A
    uint in_dim, out_dim, rank, tmp_off; // tmp_off: float-word offset in A for [M][rank]
    uint in_half, out_half, pad0;
    float scale;
};

struct MsLoraArgs {
    uint nseg, M, max_rank, max_out_blocks;
    MsLoraSeg seg[8];
};

// Grid: (max_rank, M, nseg), TG = 256
kernel void k_ms_lora_a(
    device float * A [[buffer(2)]],
    device const uchar * AD [[buffer(3)]],
    constant MsLoraArgs & a [[buffer(5)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    threadgroup float sh[32];
    const uint ri = tgp.x, m = tgp.y, s = tgp.z;
    if (s >= a.nseg || m >= a.M) return;
    constant MsLoraSeg & S = a.seg[s];
    if (ri >= S.rank) return;

    device const half * aw = (device const half *) (AD + S.a_off) + (ulong) ri * S.in_dim;
    float acc = 0.0f;
    if (S.in_half != 0) {
        device const half * x = (device const half *) (A + S.x_off) + (ulong) m * S.in_dim;
        for (uint k = tid; k < S.in_dim; k += 256) {
            acc += float(aw[k]) * float(x[k]);
        }
    } else {
        device const float * x = A + S.x_off + (ulong) m * S.in_dim;
        for (uint k = tid; k < S.in_dim; k += 256) {
            acc += float(aw[k]) * x[k];
        }
    }
    acc = tg_sum(acc, sh, sg, lane, 8);
    if (tid == 0) {
        A[S.tmp_off + m * S.rank + ri] = acc * S.scale;
    }
}

// Grid: (max_out_blocks, M, nseg), TG = 256
kernel void k_ms_lora_b(
    device float * A [[buffer(2)]],
    device const uchar * AD [[buffer(3)]],
    constant MsLoraArgs & a [[buffer(5)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]
) {
    threadgroup float tmp[64];
    const uint m = tgp.y, s = tgp.z;
    if (s >= a.nseg || m >= a.M) return;
    constant MsLoraSeg & S = a.seg[s];
    if (tid < S.rank) {
        tmp[tid] = A[S.tmp_off + m * S.rank + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint o = tgp.x * 256 + tid;
    if (o >= S.out_dim) return;

    device const half * bw = (device const half *) (AD + S.b_off) + (ulong) o * S.rank;
    float acc = 0.0f;
    for (uint r = 0; r < S.rank; r++) {
        acc += float(bw[r]) * tmp[r];
    }
    if (S.out_half != 0) {
        device half * y = (device half *) (A + S.y_off) + (ulong) m * S.out_dim;
        y[o] = half(float(y[o]) + acc);
    } else {
        A[S.y_off + (ulong) m * S.out_dim + o] += acc;
    }
}

// ---- 3. Exact Loop Controller Gain/Forget + LoRA: raw = u * (1 - tanh(F(g))) + tanh(G(g)) * m
struct MsCtrlGfJob {
    uint gain_w, forget_w;         // byte offsets in PB (bf16 [E][64])
    uint gain_b_off, forget_b_off; // byte offsets in AD (fp16 [E][rank])
    uint gh_off;                   // float-word offset in A (half [M][64])
    uint tmp_gain_off, tmp_forget_off; // float-word offsets in A (f32 [M][rank])
    uint u_off, m_off, raw_off;    // float-word offsets in A (f32 [M][E])
    uint rank, pad0;
};

struct MsCtrlGfArgs {
    uint M, njobs, pad0, pad1;
    MsCtrlGfJob job[4];
};

// Grid: (E / 256, M, njobs), TG = 256
kernel void k_ms_ctrl_gf(
    device const uchar * PB [[buffer(1)]],
    device float * A [[buffer(2)]],
    device const uchar * AD [[buffer(3)]],
    constant MsCtrlGfArgs & a [[buffer(5)]],
    uint3 tgp [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]
) {
    threadgroup float tg_g[64];
    threadgroup float tg_gn[64];
    threadgroup float tg_fg[64];

    const uint m = tgp.y, j = tgp.z;
    if (m >= a.M || j >= a.njobs) return;
    constant MsCtrlGfJob & J = a.job[j];
    const uint o = tgp.x * 256 + tid;

    if (tid < 64) {
        tg_g[tid] = float(((device const half *) (A + J.gh_off))[(ulong) m * 64 + tid]);
    }
    if (tid < J.rank) {
        tg_gn[tid] = A[J.tmp_gain_off + m * J.rank + tid];
        tg_fg[tid] = A[J.tmp_forget_off + m * J.rank + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (o >= E) return;

    device const ushort * gw = (device const ushort *) (PB + J.gain_w) + (ulong) o * 64;
    device const ushort * fw = (device const ushort *) (PB + J.forget_w) + (ulong) o * 64;
    float gn = 0.0f, fg = 0.0f;
    for (int k = 0; k < 64; k++) {
        const float gk = tg_g[k];
        gn += bf16f(gw[k]) * gk;
        fg += bf16f(fw[k]) * gk;
    }
    if (J.rank > 0) {
        device const half * gb = (device const half *) (AD + J.gain_b_off) + (ulong) o * J.rank;
        device const half * fb = (device const half *) (AD + J.forget_b_off) + (ulong) o * J.rank;
        for (uint r = 0; r < J.rank; r++) {
            gn += float(gb[r]) * tg_gn[r];
            fg += float(fb[r]) * tg_fg[r];
        }
    }
    const ulong idx = (ulong) m * E + o;
    const float u = A[J.u_off + idx];
    const float mv = A[J.m_off + idx];
    A[J.raw_off + idx] = u * (1.0f - tanh(fg)) + tanh(gn) * mv;
}

// ---- 4. Multi-Stream Column-Mode K/V Conv + Tick-Causal Shared-KV Attention + Conditional Non-Silent Append
// KV layout (linear fp16): [slot_idx][2 (K,V)][NKV][kv_cap][HD]
// META layout: seq_cs[M] at meta[seq_cs_off], seq_pos[M] at meta[seq_pos_off], kv_write_idx[M] at meta[kv_write_off]
struct MsAttnArgs {
    uint M, kv_cap, slot_idx, is_swa;
    uint cur_tick, n_kv, j_start, n_vis_past;
    uint layer_slot, max_channels; // max_channels = 7
    uint seq_cs_off, seq_pos_off, kv_write_off;
    uint q_off, k_off, v_off, r_off, sink_off;
    int  kdyn_off, vdyn_off;
    uint kconv_t_w, vconv_t_w, qclip_w, relclip_w, relbias_w;
    float log_alpha, log_floor;
    uint gate_off, out_off, pad0, pad1, pad2;
};

// Grid: (NKV, M, 1), TG = 128
kernel void k_ms_kv_attn(
    device const uchar * PB [[buffer(1)]],
    device float * A [[buffer(2)]],
    device half * KV [[buffer(3)]],
    device float * CS [[buffer(4)]],
    constant MsAttnArgs & a [[buffer(5)]],
    device const uint * KV_TICKS [[buffer(6)]],
    device const uint * META [[buffer(7)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    threadgroup float4 qs4[3][HD / 4];
    threadgroup float rs[3][NREL];
    threadgroup float kc[HD], vc[HD];
    threadgroup float red[4][8];

    const uint g = tgp.x, m = tgp.y;
    const float tau = a.is_swa
        ? 1.0f
        : (1.0f + a.log_alpha * log(max(1.0f, float(a.n_vis_past + 1u) / a.log_floor)));

    device const float * qclip = (device const float *) (PB + a.qclip_w);
    float q[3];
    for (int h = 0; h < 3; h++) {
        const uint H = g * 3 + h;
        const half qh = ((device const half *) (A + a.q_off))[(ulong) m * (NH * HD) + H * HD + tid];
        q[h] = float(qh) * qclip[H * HD + tid];
    }

    // Column-mode K & V width-4 causal short conv (updates ONLY this token's channel state!)
    const uint ch_cs = META[a.seq_cs_off + m];
    const uint ch_pos = META[a.seq_pos_off + m];
    const uint tap0 = ch_pos % 3u;
    const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
    const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
    const uint c = g * HD + tid;
    const uint dim = NKV * HD;
    const float scl = a.kdyn_off >= 0 ? (1.0f / 3.0f) : 0.4472135954999579f;
    float kx = 0.0f, vx = 0.0f;
    for (int which = 0; which < 2; which++) {
        device float * hb = CS + cs_branch_base(a.layer_slot, which, a.max_channels) + (ulong) ch_cs * (3ul * dim);
        device const float * w = (device const float *) (PB + (which == 0 ? a.kconv_t_w : a.vconv_t_w));
        const float x = A[(which == 0 ? a.k_off : a.v_off) + (ulong) m * dim + c];
        const float h0 = hb[(ulong) tap0 * dim + c];
        const float h1 = hb[(ulong) tap1 * dim + c];
        const float h2 = hb[(ulong) tap2 * dim + c];
        float acc = w[0 * dim + c] * h0 + w[1 * dim + c] * h1 + w[2 * dim + c] * h2 + w[3 * dim + c] * x;
        const int dyn = (which == 0) ? a.kdyn_off : a.vdyn_off;
        if (dyn >= 0) {
            device const float * tp = A + dyn + (ulong) m * 32 + g * 4;
            acc += tp[0] * x + tp[1] * h2 + tp[2] * h1 + tp[3] * h0;
        }
        hb[(ulong) tap0 * dim + c] = x;
        if (which == 0) kx = (x + acc) * scl; else vx = (x + acc) * scl;
    }

    // Rel projection + L2 clip
    float r = 0.0f;
    if (tid < 48) {
        const uint h = tid / 16, i = tid % 16, H = g * 3 + h;
        r = A[a.r_off + (ulong) m * (NH * NREL) + H * NREL + i] * ((device const float *) (PB + a.relclip_w))[H * NREL + i];
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
    for (int h = 0; h < 3; h++) {
        const float f = clamp(QK_CLIP / (sqrt(tot[h]) + 1e-6f), 0.0f, 1.0f) * tau * 0.08838834764831845f;
        ((threadgroup float *) qs4[h])[tid] = q[h] * f;
    }
    if (tid < 48) {
        rs[tid / 16][tid % 16] = r * clamp(REL_CLIP / (sqrt(rn) + 1e-6f), 0.0f, 1.0f) * tau;
    }

    const float kf = clamp(QK_CLIP / (sqrt(tot[3]) + 1e-6f), 0.0f, 1.0f);
    const float vf = rsqrt(tot[4] / HD + 1e-6f);
    const float kv_k = kx * kf, kv_v = vx * vf;
    kc[tid] = kv_k;
    vc[tid] = kv_v;

    // Linear KV base for this (slot_idx, g): [slot_idx][2][NKV][kv_cap][HD]
    const ulong slot_k_base = ((ulong) a.slot_idx * 2ul * NKV + (ulong) g) * (ulong) a.kv_cap * HD;
    const ulong slot_v_base = ((ulong) a.slot_idx * 2ul * NKV + (ulong) (NKV + g)) * (ulong) a.kv_cap * HD;

    // Conditionally append non-silent K/V at dst >= n_kv (zero race with past keys [0..n_kv)!)
    const uint write_idx = META[a.kv_write_off + m];
    if (write_idx != 0xFFFFFFFFu) {
        KV[slot_k_base + (ulong) write_idx * HD + tid] = half(kv_k);
        KV[slot_v_base + (ulong) write_idx * HD + tid] = half(kv_v);
    }

    float sink[3];
    for (int h = 0; h < 3; h++) sink[h] = A[a.sink_off + (ulong) m * NH + g * 3 + h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Block-causal multi-stream attention: past visible keys [j_start..n_kv) + self at j == n_kv
    device const float * relb = (device const float *) (PB + a.relbias_w);
    const ushort kk = lane / 4, pp = lane % 4;
    float mx_run[3] = {-INFINITY, -INFINITY, -INFINITY};
    float l_run[3] = {0.0f, 0.0f, 0.0f};
    float4 o_run[3] = {0.0f, 0.0f, 0.0f};
    float4 rl[3];
    for (int h = 0; h < 3; h++) rl[h] = *(threadgroup const float4 *) (&rs[h][pp * 4]);

    const uint j_end = a.n_kv + 1u;
    for (uint jb = a.j_start + sg * 8u; jb < j_end; jb += 32u) {
        const uint j = jb + kk;
        const bool in_range = j < j_end;
        const bool is_self = (j == a.n_kv);
        uint d = 0u;
        bool valid = in_range;
        if (in_range && !is_self) {
            const uint tk = KV_TICKS[j];
            d = a.cur_tick - tk;
            if (a.is_swa && d >= EXT) valid = false;
        }

        float acc[3] = {0.0f, 0.0f, 0.0f};
        if (valid) {
            if (is_self) {
                for (int i = 0; i < 8; i++) {
                    const float4 kf4 = *(threadgroup const float4 *) (&kc[pp * 32 + i * 4]);
                    for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + i], kf4);
                }
            } else {
                device const half4 * kr = (device const half4 *) (KV + slot_k_base + (ulong) j * HD + pp * 32);
                for (int i = 0; i < 8; i++) {
                    const float4 kf4 = float4(kr[i]);
                    for (int h = 0; h < 3; h++) acc[h] += dot(qs4[h][pp * 8 + i], kf4);
                }
            }
            if (d < EXT) {
                const float4 bb = *(device const float4 *) (relb + d * NREL + pp * 4);
                for (int h = 0; h < 3; h++) acc[h] += dot(rl[h], bb);
            }
        }

        float pr[3];
        for (int h = 0; h < 3; h++) {
            float t = acc[h];
            t += simd_shuffle_xor(t, 1); t += simd_shuffle_xor(t, 2);
            t = valid ? (t - sink[h]) : -INFINITY;
            const float smx = simd_max(t);
            const float nmx = max(mx_run[h], smx);
            if (nmx == -INFINITY) {
                pr[h] = 0.0f;
            } else {
                const float sc = (mx_run[h] == -INFINITY) ? 0.0f : exp(mx_run[h] - nmx);
                pr[h] = valid ? exp(t - nmx) : 0.0f;
                l_run[h] = l_run[h] * sc + simd_sum(pr[h]) * 0.25f;
                mx_run[h] = nmx;
                o_run[h] *= sc;
            }
        }

        for (ushort q2 = 0; q2 < 8; q2++) {
            const uint jq = jb + q2;
            if (jq >= j_end) break;
            const bool q_self = (jq == a.n_kv);
            if (!q_self && a.is_swa && (a.cur_tick - KV_TICKS[jq]) >= EXT) continue;
            float4 v4;
            if (q_self) {
                v4 = *(threadgroup const float4 *) (&vc[lane * 4]);
            } else {
                v4 = float4(*(device const half4 *) (KV + slot_v_base + (ulong) jq * HD + lane * 4));
            }
            for (int h = 0; h < 3; h++) o_run[h] += simd_shuffle(pr[h], q2 * 4) * v4;
        }
    }

    threadgroup float ms_[4][3], ls_[4][3];
    threadgroup float4 os_[4][3][32];
    if (lane == 0) {
        for (int h = 0; h < 3; h++) { ms_[sg][h] = mx_run[h]; ls_[sg][h] = l_run[h]; }
    }
    for (int h = 0; h < 3; h++) os_[sg][h][lane] = o_run[h];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device half * out = (device half *) (A + a.out_off);
    device const half * gate = (device const half *) (A + a.gate_off);
    for (int h = 0; h < 3; h++) {
        const uint H = g * 3 + h;
        float M_val = 0.0f; // sink sits at shifted score 0.0!
        for (int s2 = 0; s2 < 4; s2++) M_val = max(M_val, ms_[s2][h]);
        float L_val = exp(-M_val), O_val = 0.0f;
        for (int s2 = 0; s2 < 4; s2++) {
            const float w = (ms_[s2][h] == -INFINITY) ? 0.0f : exp(ms_[s2][h] - M_val);
            L_val += w * ls_[s2][h];
            O_val += w * ((threadgroup const float *) os_[s2][h])[tid];
        }
        const ulong oi = (ulong) m * (NH * HD) + H * HD + tid;
        out[oi] = half(O_val / L_val * sigm(float(gate[oi])));
    }
}

// ---- 5. Column-mode attention post-conv + optional residual (runs on f32 wo_y after wo GEMM + wo LoRA)
struct MsWoPostArgs {
    uint wo_y_off, res_off, out_off, has_res;
    uint layer_slot, max_channels, convw_t_off;
    uint seq_cs_off, seq_pos_off, pad0, pad1, pad2;
};

// Grid: (M, E.div_ceil(1024), 1), TG = 256
kernel void k_ms_wo_post(
    device const uchar * PB [[buffer(1)]],
    device float * A [[buffer(2)]],
    device float * CS [[buffer(4)]],
    constant MsWoPostArgs & a [[buffer(5)]],
    device const uint * META [[buffer(7)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]
) {
    const uint m = tgp.x, c = (tgp.y * 256 + tid) * 4;
    if (c >= E) return;
    const float4 y = *(device const float4 *) (A + a.wo_y_off + (ulong) m * E + c);
    const uint tap0 = META[a.seq_pos_off + m] % 3u;
    const uint tap1 = (tap0 == 2u) ? 0u : (tap0 + 1u);
    const uint tap2 = (tap1 == 2u) ? 0u : (tap1 + 1u);
    device float * hb = CS + cs_branch_base(a.layer_slot, 2, a.max_channels) + (ulong) META[a.seq_cs_off + m] * (3ul * E);
    device const float * cw = (device const float *) (PB + a.convw_t_off);
    const float4 h0 = *(device const float4 *) (hb + (ulong) tap0 * E + c);
    const float4 h1 = *(device const float4 *) (hb + (ulong) tap1 * E + c);
    const float4 h2 = *(device const float4 *) (hb + (ulong) tap2 * E + c);
    const float4 w0 = *(device const float4 *) (cw + 0 * E + c);
    const float4 w1 = *(device const float4 *) (cw + 1 * E + c);
    const float4 w2 = *(device const float4 *) (cw + 2 * E + c);
    const float4 w3 = *(device const float4 *) (cw + 3 * E + c);
    *(device float4 *) (hb + (ulong) tap0 * E + c) = y;
    float4 o = (y + w0 * h0 + w1 * h1 + w2 * h2 + w3 * y) * 0.4472135954999579f;
    if (a.has_res != 0) {
        o += *(device const float4 *) (A + a.res_off + (ulong) m * E + c);
    }
    *(device float4 *) (A + a.out_off + (ulong) m * E + c) = o;
}

// ---- 6. Output channel embedding + optional per-channel gated bottleneck head + gather for lm_head
struct MsOutPrepArgs {
    uint fin_off, out_f32_off, out_half_off, out_ch_emb_off;
    uint m_start, n_out, head_rank, pad0;
    uint head_w1_off, head_b1_off, head_w2_off, pad1;
};

// Grid: (n_out, 1, 1), TG = 256
kernel void k_ms_out_prep(
    device float * A [[buffer(2)]],
    device const uchar * AD [[buffer(3)]],
    constant MsOutPrepArgs & a [[buffer(5)]],
    device const uint * chs [[buffer(7)]],
    uint out_i [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]
) {
    if (out_i >= a.n_out) return;
    threadgroup float sh[32];
    threadgroup float z_sh[128];

    const uint m = a.m_start + out_i;
    const uint ch = chs[m];
    device const float * oce = (a.out_ch_emb_off != 0xFFFFFFFFu)
        ? ((device const float *) (AD + a.out_ch_emb_off) + (ulong) ch * E)
        : nullptr;
    device half * dst_h = (device half *) (A + a.out_half_off) + (ulong) out_i * E;
    device float * dst_f = A + a.out_f32_off + (ulong) out_i * E;
    device const float * src = A + a.fin_off + (ulong) m * E;

    float x[9];
    for (int k = 0; k < 9; k++) {
        x[k] = src[tid + k * 256];
    }

    const bool has_head = (a.head_rank > 0u) && (a.head_rank <= 128u)
        && (a.head_w1_off != 0xFFFFFFFFu) && (a.head_b1_off != 0xFFFFFFFFu) && (a.head_w2_off != 0xFFFFFFFFu)
        && (ch >= 3u) && (ch < 7u);
    if (has_head) {
        const uint slot = ch - 3u;
        const uint R = a.head_rank;
        device const float * w1 = (device const float *) (AD + a.head_w1_off);
        device const float * b1 = (device const float *) (AD + a.head_b1_off);
        for (uint r = 0; r < R; r++) {
            float acc = 0.0f;
            device const float * w1_row = w1 + ((ulong) slot * R + r) * E;
            for (int k = 0; k < 9; k++) {
                acc += x[k] * w1_row[tid + k * 256];
            }
            float dot = tg_sum(acc, sh, sg, lane, 8);
            if (tid == 0) {
                float u = dot + b1[(ulong) slot * R + r];
                z_sh[r] = u * sigm(u);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device const float * w2_base = has_head
        ? ((device const float *) (AD + a.head_w2_off) + (ulong) (ch - 3u) * E * a.head_rank)
        : nullptr;

    for (int k = 0; k < 9; k++) {
        const int i = tid + k * 256;
        float val = x[k];
        if (oce != nullptr) val += oce[i];
        if (has_head) {
            device const float * w2_row = w2_base + (ulong) i * a.head_rank;
            float delta = 0.0f;
            for (uint r = 0; r < a.head_rank; r++) {
                delta += w2_row[r] * z_sh[r];
            }
            val += delta;
        }
        dst_f[i] = val;
        dst_h[i] = half(val);
    }
}
