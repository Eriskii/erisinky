//! Rigorous end-to-end empirical 256K-KV compression & expert-routing probe suite for Inkling 2 Mini.
//!
//! Probes implemented:
//!   1. `heads`     : Per-slot & per-head DuoAttention retrieval map across all 8 full slots & 64 (slot, kv_head) pairs
//!   2. `rotquant`  : Rotated ultra-low-bit quantization (WHT_128 + Rademacher signs + exact Lloyd-Max Gaussian + 1-bit QJL)
//!   3. `centroid`  : No-RoPE weighted centroid merging (`(k_c, v_c, ln w_c)`) with exact E2E GPU KL & head-level attention metrics
//!   4. `sparse`    : Two-tier sparse top-K page retrieval (Quest min/max, centroid+radius, 2-bit WHT sketch) + background centroid
//!   5. `crosspass` : Cross-pass loop predictability (`blk.11` & `blk.17` pass 0->1->2) & centered/whitened SVD spectra
//!   6. `zipf`      : Real expert activation Zipf skew across all 24 physical layers & simulated batch working sets
//!
//! Memory safety: all temporary CPU buffers are bounded well under ~3.2 GiB even at 32K context.

use crate::engine::{
    slot_of, Engine, Prune, EXT, HD, LOOP_END, LOOP_START, NEXP, NH, NKV, N_PASS, N_PHYS, N_SLOT,
    TOPK,
};
use crate::tokenizer::Tokenizer;
use objc2_metal::{MTLBuffer, MTLCommandBuffer};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const CTX_256K: usize = 262_144;
const N_THREADS: usize = 8;

// =============================================================================
// Public entry point
// =============================================================================

/// Run the empirical 256K-KV & expert-routing probe suite.
///
/// `subtest` accepts `"all"`, `"heads"`, `"rotquant"`, `"centroid"`, `"sparse"`, `"crosspass"`, `"zipf"`,
/// or a comma-separated list of any of them (e.g. `"heads,rotquant"`).
///
/// If `prompt_text` is empty or starts with `"needle:"` (e.g. `"needle:8192"` or `"needle:16384"`),
/// a non-repetitive long needle-in-a-haystack prompt with planted early facts (`pos < 1000`) is synthesized.
pub fn run_kv_probe(
    eng: &Engine,
    tk: &Tokenizer,
    prompt_text: &str,
    decode_steps: usize,
    subtest: &str,
) {
    assert!(
        !eng.kv8 && !eng.tier_on(),
        "kv_probe requires exact FP16 in-memory KV cache (disable --kv8 and --kv-disk)"
    );

    let flags = ProbeFlags::parse(subtest);
    let d = decode_steps.clamp(4, 128);

    println!("================================================================================");
    println!("  INKLING-2-MINI EMPIRICAL 256K-KV & ROUTING PROBE SUITE");
    println!("================================================================================");

    // 1. Prepare / tokenize prompt
    let owned_prompt;
    let effective_prompt: &str = if prompt_text.is_empty() || prompt_text.starts_with("needle:") {
        let target_toks = if let Some(rest) = prompt_text.strip_prefix("needle:") {
            rest.trim().parse::<usize>().unwrap_or(8192)
        } else {
            8192
        };
        let max_safe = eng.max_ctx.saturating_sub(d + 64).max(256);
        owned_prompt = build_needle_prompt(tk, target_toks.min(max_safe));
        &owned_prompt
    } else {
        prompt_text
    };

    let mut pt = tk.encode(effective_prompt, true);
    let max_prompt = eng.max_ctx.saturating_sub(d + 16);
    if pt.len() > max_prompt {
        // Keep prefix (preserves early planted needles) + suffix (preserves final question)
        let keep_head = (max_prompt * 3) / 4;
        let keep_tail = max_prompt - keep_head;
        let mut trimmed = Vec::with_capacity(max_prompt);
        trimmed.extend_from_slice(&pt[..keep_head]);
        trimmed.extend_from_slice(&pt[pt.len() - keep_tail..]);
        println!(
            "[probe] prompt truncated {} -> {} tokens to fit max_ctx={} (kept head {} + tail {})",
            pt.len(),
            trimmed.len(),
            eng.max_ctx,
            keep_head,
            keep_tail
        );
        pt = trimmed;
    }

    let n = pt.len();
    let r0 = n.saturating_sub(EXT);
    let full_slots = collect_full_slots(eng);
    let n_full = full_slots.len();
    let n_swa = N_SLOT - n_full;
    assert!(n >= 16, "prompt too short ({n} tokens)");

    let full_256k_mib = (n_full as f64) * (CTX_256K as f64) * (2 * NKV * HD * 2) as f64 / 1048576.0;
    let swa_256k_mib = (n_swa as f64) * (EXT as f64) * (2 * NKV * HD * 2) as f64 / 1048576.0;
    println!(
        "Prompt: {n} tokens (far keys r0={r0}, local={}) | Decode steps: {d} | Full slots: {n_full}, SWA slots: {n_swa}",
        n.min(EXT)
    );
    println!(
        "256K (262,144 tok) FP16 baseline: full={full_256k_mib:.1} MiB ({:.3} GiB) + SWA={swa_256k_mib:.1} MiB = {:.3} GiB/seq",
        full_256k_mib / 1024.0,
        (full_256k_mib + swa_256k_mib) / 1024.0
    );
    println!(
        "Subtests enabled: heads={} rotquant={} centroid={} sparse={} crosspass={} zipf={}",
        flags.heads, flags.rotquant, flags.centroid, flags.sparse, flags.crosspass, flags.zipf
    );

    // 2. Prefill prompt once (with qcap if needed by any selected subtest)
    let need_qcap = flags.heads || flags.centroid || flags.sparse || flags.rotquant;
    if need_qcap {
        assert!(
            eng.hgemm,
            "qcap capture requires the fp16 attention prefill path (eng.hgemm == true)"
        );
    }

    let prev_tail = eng.tail.get();
    eng.tail.set(false);
    *eng.prune.borrow_mut() = None;
    *eng.qcap.borrow_mut() = None;
    eng.reset();
    eng.reserve(n + d + 1, false);
    eng.ensure_batch(n.min(eng.chunk).max(d));
    if need_qcap {
        eng.qcap_alloc(n + d);
    }
    eng.set_tokens(0, &pt);

    let t_pf = Instant::now();
    let mut p0 = 0usize;
    while p0 < n {
        let t = (n - p0).min(eng.chunk);
        let cb = eng.gpu.cmd();
        eng.encode_prefill(&cb, p0, t, p0 + t == n, false);
        cb.commit();
        cb.waitUntilCompleted();
        p0 += t;
    }
    let pf_sec = t_pf.elapsed().as_secs_f64();
    let first_tok = argmax_f32(eng.logits()) as u32;
    println!(
        "Prefill: {n} tokens in {pf_sec:.2}s ({:.0} tok/s)",
        n as f64 / pf_sec.max(1e-9)
    );

    // 3. Snapshot post-prefill state (cs, kv_swa) and clean CPU copy of kv_full[0..n]
    let snap = eng.snapshot();
    let kv_backup = backup_kv_full(eng, &full_slots, n);

    // 4. Exact FP16 reference decode
    let (ref_toks, ref_logits, ref_routes, ref_ms) = run_reference_decode(eng, &snap, n, first_tok, d);
    let check_eval = eval_decode(eng, &snap, n, &ref_toks, &ref_logits, d);
    println!(
        "Reference FP16 decode: {d} steps @ {ref_ms:.2} ms/step | rerun check: argmax {}/{} worst KL {:.2e}",
        check_eval.agree, d, check_eval.worst_kl
    );
    let preview = decode_tokens_preview(tk, &ref_toks, 40);
    println!("Reference greedy output ({d} tok): {preview:?}");

    // 5. Capture exact decode queries qcap[n..n+d] via one tiny prefill chunk
    let has_decode_qcap = if need_qcap && d <= eng.chunk {
        eng.restore(&snap);
        eng.set_tokens(n, &ref_toks);
        let cb = eng.gpu.cmd();
        eng.encode_prefill(&cb, n, d, false, false);
        cb.commit();
        cb.waitUntilCompleted();
        eng.restore(&snap);
        true
    } else {
        false
    };
    eng.release_batch();

    let lut = build_h2f_lut();

    // 6. Execute probes in OOM-optimal order (qcap users first, then free qcap!)
    if flags.heads {
        probe_heads(
            eng,
            &full_slots,
            &snap,
            &kv_backup,
            n,
            &ref_toks,
            &ref_logits,
            has_decode_qcap,
            &lut,
        );
        restore_kv_full_all(eng, &full_slots, &kv_backup, n);
        *eng.prune.borrow_mut() = None;
    }

    if flags.centroid {
        probe_centroid(
            eng,
            &full_slots,
            &snap,
            &kv_backup,
            n,
            &ref_toks,
            &ref_logits,
            has_decode_qcap,
            &lut,
        );
        restore_kv_full_all(eng, &full_slots, &kv_backup, n);
        *eng.prune.borrow_mut() = None;
    }

    if flags.sparse {
        probe_sparse(
            eng,
            &full_slots,
            &snap,
            &kv_backup,
            n,
            &ref_toks,
            &ref_logits,
            has_decode_qcap,
            &lut,
        );
        restore_kv_full_all(eng, &full_slots, &kv_backup, n);
        *eng.prune.borrow_mut() = None;
    }

    if flags.rotquant {
        probe_rotquant(
            eng,
            &full_slots,
            &snap,
            &kv_backup,
            n,
            &ref_toks,
            &ref_logits,
            has_decode_qcap,
            &lut,
        );
        restore_kv_full_all(eng, &full_slots, &kv_backup, n);
        eng.restore(&snap);
    }

    // Free qcap before crosspass & zipf to keep peak RAM minimal
    *eng.qcap.borrow_mut() = None;

    if flags.crosspass {
        probe_crosspass(&full_slots, &kv_backup, n, &lut);
    }

    // Drop kv_backup before zipf (zipf only needs eng + pt + ref_routes)
    drop(kv_backup);

    if flags.zipf {
        probe_zipf(eng, &pt, &ref_routes, n);
    }

    *eng.prune.borrow_mut() = None;
    eng.tail.set(prev_tail);
    println!("\n================================================================================");
    println!("  KV PROBE SUITE COMPLETE");
    println!("================================================================================");
}

// =============================================================================
// Subtest flags & slot metadata
// =============================================================================

struct ProbeFlags {
    heads: bool,
    rotquant: bool,
    centroid: bool,
    sparse: bool,
    crosspass: bool,
    zipf: bool,
}

impl ProbeFlags {
    fn parse(s: &str) -> Self {
        let mut f = Self {
            heads: false,
            rotquant: false,
            centroid: false,
            sparse: false,
            crosspass: false,
            zipf: false,
        };
        for tok in s.split(',').map(|x| x.trim().to_ascii_lowercase()) {
            match tok.as_str() {
                "all" | "" => {
                    f.heads = true;
                    f.rotquant = true;
                    f.centroid = true;
                    f.sparse = true;
                    f.crosspass = true;
                    f.zipf = true;
                }
                "heads" => f.heads = true,
                "rotquant" => f.rotquant = true,
                "centroid" => f.centroid = true,
                "sparse" => f.sparse = true,
                "crosspass" => f.crosspass = true,
                "zipf" => f.zipf = true,
                other => {
                    eprintln!(
                        "[kv_probe] unknown subtest '{other}' (expected all|heads|rotquant|centroid|sparse|crosspass|zipf)"
                    );
                }
            }
        }
        f
    }
}

#[derive(Clone)]
struct FullSlotInfo {
    slot: usize,
    fi: usize,
    phys: usize,
    pass: usize,
    label: String,
}

fn slot_to_phys_pass(slot: usize) -> (usize, usize) {
    if slot < N_PHYS {
        (slot, 0)
    } else {
        let loop_len = LOOP_END - LOOP_START;
        let rel = slot - N_PHYS;
        let pass = 1 + rel / loop_len;
        let phys = LOOP_START + rel % loop_len;
        (phys, pass)
    }
}

fn exec_to_phys_pass(exec: usize) -> (usize, usize, usize) {
    let loop_len = LOOP_END - LOOP_START;
    let loop_total = N_PASS * loop_len;
    if exec < LOOP_START {
        (exec, 0, exec)
    } else if exec < LOOP_START + loop_total {
        let rel = exec - LOOP_START;
        let pass = rel / loop_len;
        let phys = LOOP_START + rel % loop_len;
        (phys, pass, slot_of(pass, phys))
    } else {
        let phys = LOOP_END + (exec - LOOP_START - loop_total);
        (phys, 0, phys)
    }
}

fn collect_full_slots(eng: &Engine) -> Vec<FullSlotInfo> {
    let mut out = Vec::new();
    for slot in 0..N_SLOT {
        let (is_swa, kind_idx) = eng.slot_kind[slot];
        if !is_swa {
            let (phys, pass) = slot_to_phys_pass(slot);
            let tag = if phys < LOOP_START {
                "pre".to_string()
            } else if phys >= LOOP_END {
                "coda".to_string()
            } else {
                format!("p{pass}")
            };
            out.push(FullSlotInfo {
                slot,
                fi: kind_idx,
                phys,
                pass,
                label: format!("blk.{phys:02}({tag})"),
            });
        }
    }
    out.sort_by_key(|s| s.fi);
    out
}

// =============================================================================
// FP16 / math / RNG / linear algebra primitives
// =============================================================================

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
fn f32_to_f16(v: f32) -> u16 {
    if !v.is_finite() {
        return if v.is_nan() {
            0
        } else if v.is_sign_negative() {
            0xfbff
        } else {
            0x7bff
        };
    }
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if v == 0.0 {
        return sign;
    }
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = (bits & 0x7fffff) | 0x800000;
        let shift = (14 - exp) as u32;
        let sub = mant >> shift;
        let rem = mant & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        let round = if rem > half || (rem == half && (sub & 1) == 1) {
            1
        } else {
            0
        };
        return sign | ((sub + round) as u16);
    }
    let mant = (bits >> 13) & 0x3ff;
    let rem = bits & 0x1fff;
    let mut h = ((exp as u32) << 10 | mant)
        + if rem > 0x1000 || (rem == 0x1000 && (mant & 1) == 1) {
            1
        } else {
            0
        };
    if h >= 0x7c00 {
        h = 0x7bff;
    }
    sign | (h as u16)
}

#[inline]
fn softcap_f32(x: f32, c: f32) -> f32 {
    if c > 0.0 {
        c * (x / c).tanh()
    } else {
        x
    }
}

#[inline]
fn argmax_f32(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| {
            if x > b.1 {
                (i, x)
            } else {
                b
            }
        })
        .0
}

fn logit_kl_f64(ref_lg: &[f32], test_lg: &[f32]) -> f64 {
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
        let pr = log_pr.exp();
        kl += pr * (log_pr - log_pt);
    }
    kl.max(0.0)
}

#[inline]
fn dot128(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    for c in (0..HD).step_by(8) {
        for u in 0..8 {
            acc[u] += a[c + u] * b[c + u];
        }
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3]) + (acc[4] + acc[5]) + (acc[6] + acc[7])
}

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Exact normalized 128-point Fast Walsh-Hadamard Transform (`H * x`, where `H^T H = I`).
#[inline]
fn wht_128(x: &mut [f32; HD]) {
    let mut h = 1usize;
    while h < HD {
        let step = h * 2;
        let mut i = 0;
        while i < HD {
            for j in i..i + h {
                let u = x[j];
                let v = x[j + h];
                x[j] = u + v;
                x[j + h] = u - v;
            }
            i += step;
        }
        h = step;
    }
    const INV_SQRT_128: f32 = 0.08838834764831845;
    for v in x.iter_mut() {
        *v *= INV_SQRT_128;
    }
}

fn rademacher_signs(seed: u64) -> [f32; HD] {
    let mut st = seed ^ 0xA5A5_5A5A_F0F0_0F0Fu64;
    let mut s = [1.0f32; HD];
    for chunk in s.chunks_mut(64) {
        let bits = splitmix64(&mut st);
        for (i, out) in chunk.iter_mut().enumerate() {
            *out = if (bits >> i) & 1 == 0 { 1.0 } else { -1.0 };
        }
    }
    s
}

#[inline]
fn rotate_fwd(x: &[f32; HD], signs: &[f32; HD]) -> [f32; HD] {
    let mut y = [0.0f32; HD];
    for c in 0..HD {
        y[c] = x[c] * signs[c];
    }
    wht_128(&mut y);
    y
}

#[inline]
fn rotate_inv(y: &[f32; HD], signs: &[f32; HD]) -> [f32; HD] {
    let mut x = *y;
    wht_128(&mut x);
    for c in 0..HD {
        x[c] *= signs[c];
    }
    x
}

/// High-precision pure-Rust `erf(x)` accurate to ~1e-15 across all real `x`.
fn erf_f64(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    if ax == 0.0 {
        return 0.0;
    }
    if ax > 6.0 {
        return sign;
    }
    const TWO_OVER_SQRT_PI: f64 = 1.1283791670955126;
    if ax <= 3.5 {
        // Positive-term series: erf(x) = (2/sqrt(pi)) * exp(-x^2) * sum_{k>=0} (2^k x^{2k+1}) / (2k+1)!!
        let x2 = 2.0 * ax * ax;
        let mut term = ax;
        let mut sum = ax;
        for k in 1..80 {
            term *= x2 / (2 * k + 1) as f64;
            sum += term;
            if term <= sum * 1e-16 {
                break;
            }
        }
        sign * (TWO_OVER_SQRT_PI * (-ax * ax).exp() * sum).min(1.0)
    } else {
        // Lentz continued fraction for erfc(ax) = (exp(-ax^2)/sqrt(pi)) * 1 / (ax + (1/2)/(ax + 1/(ax + (3/2)/...)))
        let mut c = 1e300f64;
        let mut d = 1.0 / ax;
        let mut h = d;
        for i in 1..60 {
            let a = 0.5 * i as f64;
            d = ax + a * d;
            if d.abs() < 1e-300 {
                d = 1e-300;
            }
            c = ax + a / c;
            if c.abs() < 1e-300 {
                c = 1e-300;
            }
            d = 1.0 / d;
            let delta = c * d;
            h *= delta;
            if (delta - 1.0).abs() < 1e-15 {
                break;
            }
        }
        let erfc = (-ax * ax).exp() * (0.5 * TWO_OVER_SQRT_PI) * h;
        sign * (1.0 - erfc)
    }
}

#[inline]
fn norm_pdf(z: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.3989422804014327;
    INV_SQRT_2PI * (-0.5 * z * z).exp()
}

#[inline]
fn norm_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf_f64(z * std::f64::consts::FRAC_1_SQRT_2))
}

/// Exact Lloyd-Max optimal centroids (sorted ascending, length `2^bits`) for `N(0, 1)`.
fn lloyd_max_gaussian(bits: u32) -> Vec<f32> {
    assert!((1..=4).contains(&bits));
    let k = 1usize << bits;
    let half = k / 2;
    // Initialize positive centroids evenly in [0.1, 3.0]
    let mut pos: Vec<f64> = (0..half)
        .map(|i| 0.25 + 2.5 * (i as f64) / (half.max(1) as f64))
        .collect();
    for _ in 0..300 {
        let mut edges = Vec::with_capacity(half + 1);
        edges.push(0.0f64);
        for i in 0..half - 1 {
            edges.push(0.5 * (pos[i] + pos[i + 1]));
        }
        edges.push(f64::INFINITY);
        let mut max_delta = 0.0f64;
        for i in 0..half {
            let a = edges[i];
            let b = edges[i + 1];
            let pa = norm_pdf(a);
            let pb = if b.is_finite() { norm_pdf(b) } else { 0.0 };
            let cdf_a = norm_cdf(a);
            let cdf_b = if b.is_finite() { norm_cdf(b) } else { 1.0 };
            let mass = (cdf_b - cdf_a).max(1e-15);
            let new_c = (pa - pb) / mass;
            max_delta = max_delta.max((new_c - pos[i]).abs());
            pos[i] = new_c;
        }
        if max_delta < 1e-12 {
            break;
        }
    }
    let mut all = Vec::with_capacity(k);
    for &c in pos.iter().rev() {
        all.push((-c) as f32);
    }
    for &c in pos.iter() {
        all.push(c as f32);
    }
    all
}

/// Quantize scalar `z` to nearest level in sorted centroid table `levels`, returning `(idx, value)`.
#[inline]
fn quant_nearest(z: f32, levels: &[f32]) -> (usize, f32) {
    let mut best_i = 0usize;
    let mut best_d = (z - levels[0]).abs();
    for (i, &c) in levels.iter().enumerate().skip(1) {
        let d = (z - c).abs();
        if d < best_d {
            best_d = d;
            best_i = i;
        }
    }
    (best_i, levels[best_i])
}

/// Cholesky decomposition `A = L L^T` in-place (lower triangle) and solve `A X = B` for `nrhs` columns.
/// `a` is `n x n` row-major, `b` is `n x nrhs` row-major (overwritten with solution `X`).
fn cholesky_solve_multi(a_orig: &[f64], b: &mut [f64], n: usize, nrhs: usize) -> bool {
    let trace: f64 = (0..n).map(|i| a_orig[i * n + i]).sum::<f64>().max(1e-12);
    let mut ridge = 0.0f64;
    let mut l = vec![0.0f64; n * n];
    let mut ok = false;
    for attempt in 0..6 {
        l.copy_from_slice(a_orig);
        if ridge > 0.0 {
            for i in 0..n {
                l[i * n + i] += ridge;
            }
        }
        let mut failed = false;
        for i in 0..n {
            for j in 0..=i {
                let mut sum = l[i * n + j];
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    if sum <= 1e-18 || !sum.is_finite() {
                        failed = true;
                        break;
                    }
                    l[i * n + j] = sum.sqrt();
                } else {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
            if failed {
                break;
            }
        }
        if !failed {
            ok = true;
            break;
        }
        ridge = (trace / n as f64) * 10f64.powi(-6 + attempt);
    }
    if !ok {
        return false;
    }
    // Forward substitution L Y = B
    for i in 0..n {
        let diag = l[i * n + i];
        for r in 0..nrhs {
            let mut sum = b[i * nrhs + r];
            for k in 0..i {
                sum -= l[i * n + k] * b[k * nrhs + r];
            }
            b[i * nrhs + r] = sum / diag;
        }
    }
    // Backward substitution L^T X = Y
    for i in (0..n).rev() {
        let diag = l[i * n + i];
        for r in 0..nrhs {
            let mut sum = b[i * nrhs + r];
            for k in i + 1..n {
                sum -= l[k * n + i] * b[k * nrhs + r];
            }
            b[i * nrhs + r] = sum / diag;
        }
    }
    true
}

/// Cyclic Jacobi eigenvalues of a symmetric `n x n` matrix (sorted descending, clamped `>= 0`).
fn jacobi_eigenvalues_psd(mat: &[f64], n: usize) -> Vec<f64> {
    let mut a = mat.to_vec();
    let diag_scale = (0..n)
        .map(|i| a[i * n + i].abs())
        .fold(0.0f64, f64::max)
        .max(1e-18);
    let tol = diag_scale * 1e-11;
    for _sweep in 0..20 {
        let mut max_off = 0.0f64;
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq.abs() > max_off {
                    max_off = apq.abs();
                }
                if apq.abs() <= tol {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let tau = (aqq - app) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                a[p * n + p] = app - t * apq;
                a[q * n + q] = aqq + t * apq;
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
                for r in 0..n {
                    if r == p || r == q {
                        continue;
                    }
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    let nrp = c * arp - s * arq;
                    let nrq = s * arp + c * arq;
                    a[r * n + p] = nrp;
                    a[p * n + r] = nrp;
                    a[r * n + q] = nrq;
                    a[q * n + r] = nrq;
                }
            }
        }
        if max_off <= tol {
            break;
        }
    }
    let mut eigs: Vec<f64> = (0..n).map(|i| a[i * n + i].max(0.0)).collect();
    eigs.sort_by(|x, y| y.total_cmp(x));
    eigs
}

fn spearman_corr(xs: &[f64], ys: &[f64]) -> f64 {
    let m = xs.len();
    if m < 2 {
        return 0.0;
    }
    let ranks = |v: &[f64]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..m).collect();
        idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
        let mut r = vec![0.0f64; m];
        for (pos, &i) in idx.iter().enumerate() {
            r[i] = pos as f64;
        }
        r
    };
    let rx = ranks(xs);
    let ry = ranks(ys);
    let mx = rx.iter().sum::<f64>() / m as f64;
    let my = ry.iter().sum::<f64>() / m as f64;
    let mut num = 0.0;
    let mut dx2 = 0.0;
    let mut dy2 = 0.0;
    for i in 0..m {
        let a = rx[i] - mx;
        let b = ry[i] - my;
        num += a * b;
        dx2 += a * a;
        dy2 += b * b;
    }
    if dx2 * dy2 <= 0.0 {
        0.0
    } else {
        num / (dx2 * dy2).sqrt()
    }
}

// =============================================================================
// KV backup / restore / Engine decode helpers
// =============================================================================

/// Clean CPU backup of `kv_full` positions `0..n`, layout `[fi][kv][g][pos][HD]` in `u16`.
struct KvFullBackup {
    n: usize,
    data: Vec<u16>,
}

impl KvFullBackup {
    #[inline]
    fn row(&self, fi: usize, kv: usize, g: usize, pos: usize) -> &[u16] {
        let off = ((((fi * 2 + kv) * NKV + g) * self.n) + pos) * HD;
        &self.data[off..off + HD]
    }
}

fn backup_kv_full(eng: &Engine, full_slots: &[FullSlotInfo], n: usize) -> KvFullBackup {
    let n_full = full_slots.len();
    let mut data = vec![0u16; n_full * 2 * NKV * n * HD];
    let kv_buf = eng.kv_full.borrow();
    let src_base = kv_buf.contents().as_ptr() as *const u16;
    for s in full_slots {
        let (kb, vb, cap) = eng.kv_slot(s.slot);
        let cap = cap as usize;
        for (kv, base_off) in [(0usize, kb as usize), (1usize, vb as usize)] {
            for g in 0..NKV {
                let dst_off = (((s.fi * 2 + kv) * NKV + g) * n) * HD;
                let src_off = base_off + g * cap * HD;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_base.add(src_off),
                        data.as_mut_ptr().add(dst_off),
                        n * HD,
                    );
                }
            }
        }
    }
    KvFullBackup { n, data }
}

fn restore_kv_full_all(eng: &Engine, full_slots: &[FullSlotInfo], bak: &KvFullBackup, n: usize) {
    let kv_buf = eng.kv_full.borrow();
    let dst_base = kv_buf.contents().as_ptr() as *mut u16;
    for s in full_slots {
        let (kb, vb, cap) = eng.kv_slot(s.slot);
        let cap = cap as usize;
        for (kv, base_off) in [(0usize, kb as usize), (1usize, vb as usize)] {
            for g in 0..NKV {
                let src_off = (((s.fi * 2 + kv) * NKV + g) * bak.n) * HD;
                let dst_off = base_off + g * cap * HD;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bak.data.as_ptr().add(src_off),
                        dst_base.add(dst_off),
                        n * HD,
                    );
                }
            }
        }
    }
}

fn restore_kv_full_head(
    eng: &Engine,
    slot_info: &FullSlotInfo,
    bak: &KvFullBackup,
    g: usize,
    rows: usize,
) {
    let (kb, vb, cap) = eng.kv_slot(slot_info.slot);
    let cap = cap as usize;
    let kv_buf = eng.kv_full.borrow();
    let dst_base = kv_buf.contents().as_ptr() as *mut u16;
    for (kv, base_off) in [(0usize, kb as usize), (1usize, vb as usize)] {
        let src_off = (((slot_info.fi * 2 + kv) * NKV + g) * bak.n) * HD;
        let dst_off = base_off + g * cap * HD;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bak.data.as_ptr().add(src_off),
                dst_base.add(dst_off),
                rows * HD,
            );
        }
    }
}

#[derive(Clone, Copy, Default)]
struct DecodeEval {
    mean_kl: f64,
    worst_kl: f64,
    agree: usize,
}

fn run_reference_decode(
    eng: &Engine,
    snap: &(Vec<u8>, Vec<u8>),
    n: usize,
    first_tok: u32,
    d: usize,
) -> (Vec<u32>, Vec<Vec<f32>>, Vec<Vec<[u8; TOPK]>>, f64) {
    eng.restore(snap);
    let mut tok = first_tok;
    let mut toks = Vec::with_capacity(d);
    let mut logits = Vec::with_capacity(d);
    let mut routes = Vec::with_capacity(d);
    let t0 = Instant::now();
    for i in 0..d {
        eng.set_tokens(n + i, &[tok]);
        let cb = eng.gpu.cmd();
        eng.encode_step(&cb, n + i, true, false);
        cb.commit();
        cb.waitUntilCompleted();
        let lg: Vec<f32> = eng
            .logits()
            .iter()
            .map(|&x| softcap_f32(x, eng.softcap))
            .collect();
        toks.push(tok);
        tok = argmax_f32(&lg) as u32;
        logits.push(lg);
        routes.push(eng.routes());
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / d.max(1) as f64;
    (toks, logits, routes, ms)
}

fn eval_decode(
    eng: &Engine,
    snap: &(Vec<u8>, Vec<u8>),
    n: usize,
    ref_toks: &[u32],
    ref_logits: &[Vec<f32>],
    steps: usize,
) -> DecodeEval {
    eng.restore(snap);
    eval_decode_no_restore(eng, n, ref_toks, ref_logits, steps)
}

fn eval_decode_no_restore(
    eng: &Engine,
    n: usize,
    ref_toks: &[u32],
    ref_logits: &[Vec<f32>],
    steps: usize,
) -> DecodeEval {
    let steps = steps.min(ref_toks.len());
    let mut agree = 0usize;
    let mut worst_kl = 0.0f64;
    let mut sum_kl = 0.0f64;
    let mut lg = vec![0.0f32; eng.n_vocab];
    for i in 0..steps {
        eng.set_tokens(n + i, &[ref_toks[i]]);
        let cb = eng.gpu.cmd();
        eng.encode_step(&cb, n + i, true, false);
        cb.commit();
        cb.waitUntilCompleted();
        for (dst, &raw) in lg.iter_mut().zip(eng.logits().iter()) {
            *dst = softcap_f32(raw, eng.softcap);
        }
        let kl = logit_kl_f64(&ref_logits[i], &lg);
        if argmax_f32(&lg) == argmax_f32(&ref_logits[i]) {
            agree += 1;
        }
        worst_kl = worst_kl.max(kl);
        sum_kl += kl;
    }
    DecodeEval {
        mean_kl: sum_kl / steps.max(1) as f64,
        worst_kl,
        agree,
    }
}

fn decode_tokens_preview(tk: &Tokenizer, toks: &[u32], max_toks: usize) -> String {
    let mut bytes = Vec::new();
    for &t in toks.iter().take(max_toks) {
        bytes.extend_from_slice(&tk.decode_bytes(t));
    }
    String::from_utf8_lossy(&bytes)
        .replace('\n', "\\n")
        .chars()
        .take(160)
        .collect()
}

/// Read prepared query `q[HD]` and `sink` from `qcap` for full slot `fi`, token `pos`, query head `h`.
#[inline]
fn read_qcap_query(
    qcap_ptr: usize,
    ncap: usize,
    fi: usize,
    pos: usize,
    h: usize,
    lut: &[f32],
    out_q: &mut [f32; HD],
) -> f32 {
    let per = ncap * (NH * HD / 2 + NH) * 4;
    let q_sl = unsafe {
        std::slice::from_raw_parts(
            ((qcap_ptr + fi * per) as *const u16).add((pos * NH + h) * HD),
            HD,
        )
    };
    for c in 0..HD {
        out_q[c] = lut[q_sl[c] as usize];
    }
    unsafe { *((qcap_ptr + fi * per + ncap * NH * HD * 2) as *const f32).add(pos * NH + h) }
}

// =============================================================================
// Synthetic long Needle-in-a-Haystack prompt builder (only used if requested)
// =============================================================================

fn build_needle_prompt(tk: &Tokenizer, target_toks: usize) -> String {
    let topics = [
        "cryogenic telemetry routing",
        "oceanographic sonar bathymetry",
        "orbital debris conjunction analysis",
        "distributed ledger consensus epochs",
        "synchrotron beamline monochromator alignment",
        "geothermal wellbore casing metallurgy",
        "subsea optical repeater gain equalization",
        "atmospheric lidar aerosol inversion",
        "automated rail interlocking verification",
        "photolithography reticle phase-shift synthesis",
        "hydroelectric turbine cavitation diagnostics",
        "radio interferometry fringe calibration",
        "high-voltage DC converter harmonic damping",
        "microfluidic droplet impedance spectroscopy",
        "seismic array beamforming deconvolution",
        "autonomous glider thermal soaring heuristics",
    ];
    let sites = [
        "Reykjavik", "Tromso", "Valparaiso", "Dunedin", "Sapporo", "Halifax",
        "Antofagasta", "Brest", "Nagasaki", "Hobart", "Bergen", "Ushuaia",
    ];
    let components = [
        "capacitor bank", "FPGA crossbar", "quartz oscillator", "cryocooler manifold",
        "optical circulator", "titanium bulkhead", "differential amplifier", "bus arbiter",
        "ceramic insulator", "piezo actuator", "waveguide coupler", "pressure transducer",
    ];

    let mut body = String::with_capacity(target_toks * 5);
    body.push_str("You are reviewing an archival engineering logbook. Read all entries carefully and answer the final question using exact details from the logbook.\n\n");

    let mut st = 0x1234_5678_9ABC_DEF0u64;
    let mut entry_idx = 0usize;
    let mut planted = [false; 3];
    let prefix_hdr = "<|message_system|><|content_text|>Thinking effort level: 0.1<|end_message|><|message_user|><|content_text|>";
    let suffix_q = "\n\nQuestion: According to the early logbook entries, what is (1) the Vault-9 override passkey, (2) the Helios-4 sensor baseline in millikelvin, and (3) the name of the chief acoustic architect? Answer concisely with those three exact values.<|end_message|>";

    let mut cur_toks = tk.encode(prefix_hdr, true).len()
        + tk.encode(&body, false).len()
        + tk.encode(suffix_q, true).len();
    let mut chunk_buf = String::with_capacity(4096);

    while cur_toks < target_toks {
        chunk_buf.clear();
        for _ in 0..16 {
            let cur_chars = body.len() + chunk_buf.len();
            if !planted[0] && cur_chars > 500 {
                chunk_buf.push_str("Entry #104 (CRITICAL RECORD): The Vault-9 override passkey assigned for quarterly cold-start recovery is COBALT-7942-ZEPHYR. Keep this exact string archived.\n\n");
                planted[0] = true;
            } else if !planted[1] && cur_chars > 1700 {
                chunk_buf.push_str("Entry #119 (CRITICAL CALIBRATION): The Helios-4 cryogenic sensor baseline was certified at exactly 418.73 millikelvin after vacuum bakeout.\n\n");
                planted[1] = true;
            } else if !planted[2] && cur_chars > 2900 {
                chunk_buf.push_str("Entry #131 (PERSONNEL LOG): Responsibility for the entire low-frequency transducer array was transferred to chief acoustic architect Dr. Mireille Vask.\n\n");
                planted[2] = true;
            }

            let r0 = splitmix64(&mut st) as usize;
            let r1 = splitmix64(&mut st) as usize;
            let r2 = splitmix64(&mut st) as usize;
            let topic = topics[r0 % topics.len()];
            let site = sites[r1 % sites.len()];
            let comp = components[r2 % components.len()];
            let batch_no = 1000 + (r0 % 8999);
            let temp_c = -40 + (r1 % 110) as i32;
            let drift = (r2 % 950) as f64 / 100.0;
            let year = 2016 + (entry_idx % 10);
            let day = 1 + (r0 % 360);
            chunk_buf.push_str(&format!(
                "Log {entry_idx:04} [{site}, {year}-D{day:03}]: Routine audit of {topic} on unit B-{batch_no} noted nominal {comp} impedance at {temp_c} C with spectral drift {drift:.2} ppm/hr; maintenance window cleared without incident.\n"
            ));
            entry_idx += 1;
        }
        cur_toks += tk.encode(&chunk_buf, false).len();
        body.push_str(&chunk_buf);
    }
    format!("{prefix_hdr}{body}{suffix_q}")
}

// =============================================================================
// PROBE 1: Per-Slot & Per-Head DuoAttention Retrieval Map
// =============================================================================

/// Solve for a single anti-query key vector `k_dead` (fp16) for `(fi, g)` such that
/// `dot(q, k_dead) << -50` for all queries in `queries` (`3 * d` queries in `R^128`),
/// so `exp(dot(q, k_dead)) == 0.0` in fp32 (exact drop/eviction of far keys without modifying Metal kernels).
fn build_anti_query_key(queries: &[[f32; HD]], lut: &[f32]) -> ([u16; HD], f32) {
    let m = queries.len();
    assert!(m > 0);
    // 1. Try minimum-norm Gram solve: (Q Q^T + ridge I) alpha = -1, u = Q^T alpha
    let mut u = [0.0f64; HD];
    let mut solved = false;
    if m <= 120 {
        let mut gram = vec![0.0f64; m * m];
        for i in 0..m {
            for j in i..m {
                let mut s = 0.0f64;
                for c in 0..HD {
                    s += queries[i][c] as f64 * queries[j][c] as f64;
                }
                gram[i * m + j] = s;
                gram[j * m + i] = s;
            }
        }
        let tr = (0..m).map(|i| gram[i * m + i]).sum::<f64>().max(1e-9);
        for i in 0..m {
            gram[i * m + i] += 1e-4 * tr / m as f64;
        }
        let mut rhs = vec![-1.0f64; m];
        if cholesky_solve_multi(&gram, &mut rhs, m, 1) {
            for i in 0..m {
                for c in 0..HD {
                    u[c] += rhs[i] * queries[i][c] as f64;
                }
            }
            solved = true;
        }
    }
    // 2. Fallback or refinement if `m > 120` or margin weak: smooth-max gradient descent on unit sphere
    let worst_dot = |v: &[f64; HD]| -> f64 {
        queries
            .iter()
            .map(|q| (0..HD).map(|c| q[c] as f64 * v[c]).sum::<f64>())
            .fold(f64::NEG_INFINITY, f64::max)
    };
    if !solved || worst_dot(&u) > -0.2 {
        // Initialize with negative mean of unit queries
        u = [0.0f64; HD];
        for q in queries {
            let nrm = (0..HD).map(|c| (q[c] as f64).powi(2)).sum::<f64>().sqrt().max(1e-9);
            for c in 0..HD {
                u[c] -= q[c] as f64 / nrm;
            }
        }
        let nrm = u.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
        for c in 0..HD {
            u[c] /= nrm;
        }
        for step in 0..250 {
            let beta = 4.0 + 0.25 * step as f64;
            let dots: Vec<f64> = queries
                .iter()
                .map(|q| (0..HD).map(|c| q[c] as f64 * u[c]).sum::<f64>())
                .collect();
            let mx = dots.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut wsum = 0.0f64;
            let mut grad = [0.0f64; HD];
            for (q, &dt) in queries.iter().zip(&dots) {
                let w = (beta * (dt - mx)).exp();
                wsum += w;
                for c in 0..HD {
                    grad[c] += w * q[c] as f64;
                }
            }
            let lr = 0.15 / (1.0 + 0.02 * step as f64);
            for c in 0..HD {
                u[c] -= lr * grad[c] / wsum.max(1e-18);
            }
            let nrm = u.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-9);
            for c in 0..HD {
                u[c] /= nrm;
            }
        }
    }

    let wd = worst_dot(&u);
    let max_abs = u.iter().map(|x| x.abs()).fold(0.0f64, f64::max).max(1e-12);
    let scale = if wd < -1e-4 {
        (160.0 / (-wd)).min(8192.0 / max_abs)
    } else {
        512.0 / max_abs
    };
    let mut out = [0u16; HD];
    for c in 0..HD {
        out[c] = f32_to_f16((u[c] * scale) as f32);
    }
    let mut actual_worst = f32::NEG_INFINITY;
    let out_f32: [f32; HD] = std::array::from_fn(|c| lut[out[c] as usize]);
    for q in queries {
        let dt = dot128(q, &out_f32);
        if dt > actual_worst {
            actual_worst = dt;
        }
    }
    (out, actual_worst)
}

fn fill_head_far_const(
    eng: &Engine,
    slot_info: &FullSlotInfo,
    g: usize,
    r0: usize,
    k_vec: &[u16; HD],
    v_vec: &[u16; HD],
) {
    let (kb, vb, cap) = eng.kv_slot(slot_info.slot);
    let cap = cap as usize;
    let kv_buf = eng.kv_full.borrow();
    let base = kv_buf.contents().as_ptr() as *mut u16;
    let k_dst = unsafe { base.add(kb as usize + g * cap * HD) };
    let v_dst = unsafe { base.add(vb as usize + g * cap * HD) };
    for pos in 0..r0 {
        unsafe {
            std::ptr::copy_nonoverlapping(k_vec.as_ptr(), k_dst.add(pos * HD), HD);
            std::ptr::copy_nonoverlapping(v_vec.as_ptr(), v_dst.add(pos * HD), HD);
        }
    }
}

fn compute_head_far_mean(bak: &KvFullBackup, fi: usize, g: usize, r0: usize, lut: &[f32]) -> ([u16; HD], [u16; HD]) {
    let mut km = [0.0f64; HD];
    let mut vm = [0.0f64; HD];
    for pos in 0..r0 {
        let kr = bak.row(fi, 0, g, pos);
        let vr = bak.row(fi, 1, g, pos);
        for c in 0..HD {
            km[c] += lut[kr[c] as usize] as f64;
            vm[c] += lut[vr[c] as usize] as f64;
        }
    }
    let inv = 1.0 / r0.max(1) as f64;
    let mut k_out = [0u16; HD];
    let mut v_out = [0u16; HD];
    for c in 0..HD {
        k_out[c] = f32_to_f16((km[c] * inv) as f32);
        v_out[c] = f32_to_f16((vm[c] * inv) as f32);
    }
    (k_out, v_out)
}

#[allow(clippy::too_many_arguments)]
fn probe_heads(
    eng: &Engine,
    full_slots: &[FullSlotInfo],
    snap: &(Vec<u8>, Vec<u8>),
    bak: &KvFullBackup,
    n: usize,
    ref_toks: &[u32],
    ref_logits: &[Vec<f32>],
    has_decode_qcap: bool,
    lut: &[f32],
) {
    let d = ref_toks.len();
    let r0 = n.saturating_sub(EXT);
    println!("\n--------------------------------------------------------------------------------");
    println!("  PROBE 1 [heads]: Per-Slot & Per-Head DuoAttention Retrieval Map");
    println!("--------------------------------------------------------------------------------");
    if r0 < 64 {
        println!("  [skip] need prompt > EXT+64 ({}) tokens for far-key map (got n={n})", EXT + 64);
        return;
    }

    let n_full = full_slots.len();
    println!("Full-attention slot map ({n_full} slots, {} KV heads total):", n_full * NKV);
    for s in full_slots {
        println!(
            "  fi={}  slot={:2}  layer={}  (phys={}, pass={})",
            s.fi, s.slot, s.label, s.phys, s.pass
        );
    }

    // Build per-head anti-query mask vectors (for exact zero-far-key masking) & per-head 1-centroid means
    let (qcap_ptr, ncap) = {
        let qc = eng.qcap.borrow();
        let (buf, cap) = qc.as_ref().expect("qcap missing in probe_heads");
        (buf.contents().as_ptr() as usize, *cap)
    };
    let q_start = if has_decode_qcap { n } else { n.saturating_sub(d.min(32)) };
    let q_end = if has_decode_qcap { n + d } else { n };

    let mut k_dead = vec![vec![[0u16; HD]; NKV]; n_full];
    let mut k_mean = vec![vec![[0u16; HD]; NKV]; n_full];
    let mut v_mean = vec![vec![[0u16; HD]; NKV]; n_full];
    let mut worst_mask_dot = f32::NEG_INFINITY;
    let zero_v = [0u16; HD];

    for s in full_slots {
        for g in 0..NKV {
            let mut qs = Vec::with_capacity((q_end - q_start) * 3);
            for pos in q_start..q_end {
                for hh in 0..3 {
                    let mut q = [0.0f32; HD];
                    read_qcap_query(qcap_ptr, ncap, s.fi, pos, g * 3 + hh, lut, &mut q);
                    qs.push(q);
                }
            }
            let (kd, wd) = build_anti_query_key(&qs, lut);
            worst_mask_dot = worst_mask_dot.max(wd);
            k_dead[s.fi][g] = kd;
            let (km, vm) = compute_head_far_mean(bak, s.fi, g, r0, lut);
            k_mean[s.fi][g] = km;
            v_mean[s.fi][g] = vm;
        }
    }

    // Calibration: verify k_dead(all 64) == hardware Prune(rows=1, use_sum=false)
    let bag_dummy = eng.gpu.buffer(n_full * 2 * NKV * HD * 2);
    *eng.prune.borrow_mut() = Some(Prune {
        bag: bag_dummy,
        rows: 1,
        r0,
        lastbias: 0.0,
        use_sum: false,
    });
    let ev_prune_zero = eval_decode(eng, snap, n, ref_toks, ref_logits, d);
    *eng.prune.borrow_mut() = None;

    for s in full_slots {
        for g in 0..NKV {
            fill_head_far_const(eng, s, g, r0, &k_dead[s.fi][g], &zero_v);
        }
    }
    let ev_kdead_all = eval_decode(eng, snap, n, ref_toks, ref_logits, d);

    // All 64 heads -> 1 summary centroid (exact bias ln(r0))
    for s in full_slots {
        for g in 0..NKV {
            fill_head_far_const(eng, s, g, r0, &k_mean[s.fi][g], &v_mean[s.fi][g]);
        }
    }
    let ev_cent_all = eval_decode(eng, snap, n, ref_toks, ref_logits, d);
    restore_kv_full_all(eng, full_slots, bak, n);

    println!("\n[1A] All-64-Head Collapse Baselines & Mask Calibration (max anti-query dot={worst_mask_dot:.1}):");
    println!(
        "  Prune(0 far keys, hardware) : mean KL {:9.5}  worst KL {:9.5}  argmax {:2}/{d}",
        ev_prune_zero.mean_kl, ev_prune_zero.worst_kl, ev_prune_zero.agree
    );
    println!(
        "  k_dead mask (all 64 heads)  : mean KL {:9.5}  worst KL {:9.5}  argmax {:2}/{d}  (delta KL {:.2e})",
        ev_kdead_all.mean_kl,
        ev_kdead_all.worst_kl,
        ev_kdead_all.agree,
        (ev_prune_zero.mean_kl - ev_kdead_all.mean_kl).abs()
    );
    println!(
        "  1-centroid  (all 64 heads)  : mean KL {:9.5}  worst KL {:9.5}  argmax {:2}/{d}",
        ev_cent_all.mean_kl, ev_cent_all.worst_kl, ev_cent_all.agree
    );

    // 1B. Per-slot ablation (all 8 full slots individually, full d steps)
    println!("\n[1B] Per-Slot Far-Key Ablation (each full slot individually, {d} decode steps):");
    println!(
        "  {:>2}  {:>12} | {:>10} {:>10} {:>7} | {:>10} {:>10} {:>7}",
        "fi", "layer(pass)", "mask KL", "mask max", "agree", "1cent KL", "1cent max", "agree"
    );
    for s in full_slots {
        for g in 0..NKV {
            fill_head_far_const(eng, s, g, r0, &k_dead[s.fi][g], &zero_v);
        }
        let ev_m = eval_decode(eng, snap, n, ref_toks, ref_logits, d);
        for g in 0..NKV {
            fill_head_far_const(eng, s, g, r0, &k_mean[s.fi][g], &v_mean[s.fi][g]);
        }
        let ev_c = eval_decode(eng, snap, n, ref_toks, ref_logits, d);
        for g in 0..NKV {
            restore_kv_full_head(eng, s, bak, g, r0);
        }
        println!(
            "  {:2}  {:>12} | {:10.5} {:10.5} {:3}/{:<3} | {:10.5} {:10.5} {:3}/{:<3}",
            s.fi,
            s.label,
            ev_m.mean_kl,
            ev_m.worst_kl,
            ev_m.agree,
            d,
            ev_c.mean_kl,
            ev_c.worst_kl,
            ev_c.agree,
            d
        );
    }

    // 1C. Fast attention-based retrieval scores for all 64 (full_slot, kv_head) pairs (parallelized)
    let mut eval_qpos: Vec<usize> = Vec::new();
    let tail_q = 32usize.min(r0);
    for i in 0..tail_q {
        eval_qpos.push(n - 1 - i * (r0.min(1024)) / tail_q.max(1));
    }
    if has_decode_qcap {
        for i in 0..d {
            eval_qpos.push(n + i);
        }
    }
    eval_qpos.sort_unstable();
    eval_qpos.dedup();

    #[derive(Clone, Default)]
    struct HeadStat {
        fi: usize,
        g: usize,
        far_mass_mean: f64,
        far_mass_max: f64,
        far_peak_w: f64,
        far_norm_ent: f64,
        single_kl_mean: f64,
        single_kl_worst: f64,
        single_agree: usize,
    }

    let n_heads = n_full * NKV;
    let mut stats = vec![HeadStat::default(); n_heads];
    for (idx, st) in stats.iter_mut().enumerate() {
        st.fi = idx / NKV;
        st.g = idx % NKV;
    }

    let next_idx = AtomicUsize::new(0);
    let stats_ptr = stats.as_mut_ptr() as usize;
    let eval_qpos_ref = &eval_qpos;
    std::thread::scope(|sc| {
        for _ in 0..N_THREADS {
            sc.spawn(|| loop {
                let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= n_heads {
                    break;
                }
                let fi = idx / NKV;
                let g = idx % NKV;
                let mut keys = vec![0.0f32; n * HD];
                for j in 0..n {
                    let row = bak.row(fi, 0, g, j);
                    let dst = &mut keys[j * HD..(j + 1) * HD];
                    for c in 0..HD {
                        dst[c] = lut[row[c] as usize];
                    }
                }
                let mut scores = vec![0.0f32; n];
                let mut q = [0.0f32; HD];
                let mut sum_far_mass = 0.0f64;
                let mut max_far_mass = 0.0f64;
                let mut max_far_peak = 0.0f64;
                let mut sum_norm_ent = 0.0f64;
                let mut cnt = 0usize;

                for &pos in eval_qpos_ref {
                    let k_end = (pos + 1).min(n);
                    let far_end = pos.saturating_sub(EXT - 1).min(n);
                    if far_end == 0 {
                        continue;
                    }
                    let ln_far = (far_end as f64).ln().max(1.0);
                    for hh in 0..3 {
                        let sink = read_qcap_query(qcap_ptr, ncap, fi, pos, g * 3 + hh, lut, &mut q);
                        let mut mx = sink;
                        for j in 0..k_end {
                            let s = dot128(&q, &keys[j * HD..(j + 1) * HD]);
                            scores[j] = s;
                            if s > mx {
                                mx = s;
                            }
                        }
                        let mut z = (sink - mx).exp() as f64;
                        for j in 0..k_end {
                            z += (scores[j] - mx).exp() as f64;
                        }
                        let inv_z = 1.0 / z.max(1e-30);
                        let mut far_mass = 0.0f64;
                        let mut far_peak = 0.0f64;
                        for j in 0..far_end {
                            let p = (scores[j] - mx).exp() as f64 * inv_z;
                            far_mass += p;
                            if p > far_peak {
                                far_peak = p;
                            }
                        }
                        let mut ent = 0.0f64;
                        if far_mass > 1e-12 {
                            for j in 0..far_end {
                                let p = (scores[j] - mx).exp() as f64 * inv_z / far_mass;
                                if p > 1e-15 {
                                    ent -= p * p.ln();
                                }
                            }
                        } else {
                            ent = ln_far;
                        }
                        sum_far_mass += far_mass;
                        max_far_mass = max_far_mass.max(far_mass);
                        max_far_peak = max_far_peak.max(far_peak);
                        sum_norm_ent += ent / ln_far;
                        cnt += 1;
                    }
                }
                let out = unsafe { &mut *(stats_ptr as *mut HeadStat).add(idx) };
                if cnt > 0 {
                    out.far_mass_mean = sum_far_mass / cnt as f64;
                    out.far_mass_max = max_far_mass;
                    out.far_peak_w = max_far_peak;
                    out.far_norm_ent = sum_norm_ent / cnt as f64;
                }
            });
        }
    });

    // 1D. Exact end-to-end single-head KL sweep (all 64 heads, d_head = min(d, 8) steps)
    let d_head = d.min(8);
    let t_h = Instant::now();
    for s in full_slots {
        for g in 0..NKV {
            fill_head_far_const(eng, s, g, r0, &k_dead[s.fi][g], &zero_v);
            let ev = eval_decode(eng, snap, n, ref_toks, ref_logits, d_head);
            restore_kv_full_head(eng, s, bak, g, r0);
            let st = &mut stats[s.fi * NKV + g];
            st.single_kl_mean = ev.mean_kl;
            st.single_kl_worst = ev.worst_kl;
            st.single_agree = ev.agree;
        }
    }
    let h_sec = t_h.elapsed().as_secs_f64();

    // Rank correlation between fast attention heuristics and exact single-head KL
    let kl_vec: Vec<f64> = stats.iter().map(|s| s.single_kl_mean).collect();
    let fm_vec: Vec<f64> = stats.iter().map(|s| s.far_mass_mean).collect();
    let pk_vec: Vec<f64> = stats.iter().map(|s| s.far_peak_w).collect();
    let combo_vec: Vec<f64> = stats
        .iter()
        .map(|s| s.far_mass_mean * (1.0 - s.far_norm_ent).max(0.05) + 0.5 * s.far_peak_w)
        .collect();
    println!(
        "\n[1C] All-{n_heads}-Head Exact Single-Head KL Sweep ({d_head} steps/head in {h_sec:.1}s):"
    );
    println!(
        "  Spearman corr vs exact single-head KL: far_mass={:.3}, far_peak={:.3}, mass*(1-H)+peak={:.3}",
        spearman_corr(&fm_vec, &kl_vec),
        spearman_corr(&pk_vec, &kl_vec),
        spearman_corr(&combo_vec, &kl_vec)
    );

    // Print 8x8 heatmap of single-head mean KL
    println!("\n  8x8 Single-Head Mask KL Heatmap (rows=full slots, cols=KV heads 0..7):");
    print!("  {:>12} |", "slot");
    for g in 0..NKV {
        print!("     g={g} ");
    }
    println!();
    for s in full_slots {
        print!("  {:>12} |", s.label);
        for g in 0..NKV {
            print!(" {:7.4}", stats[s.fi * NKV + g].single_kl_mean);
        }
        println!();
    }

    // Ranked table (all 64 heads sorted by exact single-head KL descending)
    let mut ranked: Vec<usize> = (0..n_heads).collect();
    ranked.sort_by(|&a, &b| stats[b].single_kl_mean.total_cmp(&stats[a].single_kl_mean));

    println!(
        "\n  Ranked 64 Heads (top 24 + bottom 8 shown; all 64 used in sweeps):"
    );
    println!(
        "  {:>4}  {:>12} {:>2} | {:>8} {:>8} {:>7} {:>6} | {:>9} {:>9} {:>5}",
        "rank", "layer(pass)", "g", "far_mass", "max_mass", "far_pk", "H/lnN", "1h KL", "1h max", "agr"
    );
    for (rank, &idx) in ranked.iter().enumerate() {
        if rank >= 24 && rank < n_heads.saturating_sub(8) {
            if rank == 24 {
                println!("  ... ({} middle heads omitted) ...", n_heads - 32);
            }
            continue;
        }
        let st = &stats[idx];
        println!(
            "  {:4}  {:>12} {:2} | {:7.2}% {:7.2}% {:6.3} {:6.3} | {:9.5} {:9.5} {:2}/{:<2}",
            rank + 1,
            full_slots[st.fi].label,
            st.g,
            st.far_mass_mean * 100.0,
            st.far_mass_max * 100.0,
            st.far_peak_w,
            st.far_norm_ent,
            st.single_kl_mean,
            st.single_kl_worst,
            st.single_agree,
            d_head
        );
    }

    // 1E. Ranked DuoAttention subset sweeps at full `d` steps
    println!("\n[1D] Ranked DuoAttention Head-Budget Sweeps ({d} decode steps, exact 256K GiB):");
    println!(
        "  {:>6} {:>5} {:>10} | {:>9} {:>8} {:>5} | {:>9} {:>9} {:>6}",
        "keep%", "heads", "collapsed", "256K full", "256K tot", "compr", "mean KL", "worst KL", "agree"
    );
    let swa_mib = ((N_SLOT - n_full) * EXT * 2 * NKV * HD * 2) as f64 / 1048576.0;
    let base_full_mib = (n_full * CTX_256K * 2 * NKV * HD * 2) as f64 / 1048576.0;

    for &frac in &[1.0f64, 0.75, 0.50, 0.375, 0.25, 0.125] {
        let keep_k = ((n_heads as f64) * frac).round() as usize;
        let mut keep_mask = vec![false; n_heads];
        for &idx in ranked.iter().take(keep_k) {
            keep_mask[idx] = true;
        }
        let full_256k_mib = (keep_k as f64 / n_heads as f64) * base_full_mib
            + ((n_heads - keep_k) as f64 / n_heads as f64)
                * ((n_full * (EXT + 1) * 2 * NKV * HD * 2) as f64 / 1048576.0);
        let tot_256k_gib = (full_256k_mib + swa_mib) / 1024.0;
        let compr = base_full_mib / full_256k_mib.max(1e-6);

        if keep_k == n_heads {
            println!(
                "  {:5.1}% {:2}/{:<2} {:>10} | {:7.1} Mi {:6.3} Gi {:4.2}x | {:9.5} {:9.5} {:2}/{:<2}",
                100.0, keep_k, n_heads, "exact", base_full_mib, (base_full_mib + swa_mib) / 1024.0, 1.0, 0.0, 0.0, d, d
            );
            continue;
        }

        for (mode_name, use_centroid) in [("local1024", false), ("loc+1cent", true)] {
            restore_kv_full_all(eng, full_slots, bak, n);
            for idx in 0..n_heads {
                if !keep_mask[idx] {
                    let fi = idx / NKV;
                    let g = idx % NKV;
                    if use_centroid {
                        fill_head_far_const(eng, &full_slots[fi], g, r0, &k_mean[fi][g], &v_mean[fi][g]);
                    } else {
                        fill_head_far_const(eng, &full_slots[fi], g, r0, &k_dead[fi][g], &zero_v);
                    }
                }
            }
            let ev = eval_decode(eng, snap, n, ref_toks, ref_logits, d);
            println!(
                "  {:5.1}% {:2}/{:<2} {:>10} | {:7.1} Mi {:6.3} Gi {:4.2}x | {:9.5} {:9.5} {:2}/{:<2}",
                frac * 100.0,
                keep_k,
                n_heads,
                mode_name,
                full_256k_mib,
                tot_256k_gib,
                compr,
                ev.mean_kl,
                ev.worst_kl,
                ev.agree,
                d
            );
        }
    }

    // Print exact 64-head retrieval bitmaps for 50% and 25%
    println!("\n[1E] Exact 64-Head DuoAttention Retrieval Bitmaps ('#'=full 256K, '.'=local 1024):");
    for &(label, keep_k) in &[("Top-50% (32/64)", n_heads / 2), ("Top-25% (16/64)", n_heads / 4)] {
        let mut keep_mask = vec![false; n_heads];
        for &idx in ranked.iter().take(keep_k) {
            keep_mask[idx] = true;
        }
        let mut hex_bits = 0u64;
        for (i, &b) in keep_mask.iter().enumerate() {
            if b && i < 64 {
                hex_bits |= 1u64 << i;
            }
        }
        println!("  {label} bitmap = 0x{hex_bits:016x}:");
        for s in full_slots {
            let mut row_str = String::with_capacity(NKV);
            let mut cnt = 0;
            for g in 0..NKV {
                if keep_mask[s.fi * NKV + g] {
                    row_str.push('#');
                    cnt += 1;
                } else {
                    row_str.push('.');
                }
            }
            println!("    {:>12}: [{row_str}] ({cnt}/8 full)", s.label);
        }
    }
}

// =============================================================================
// PROBE 2: Rotated Ultra-Low-Bit Quantization (WHT_128 + Lloyd-Max + 1-bit QJL)
// =============================================================================

#[derive(Clone, Copy)]
enum QuantMode {
    Exact,
    UniformNoRot(u32),
    WhtUniform(u32),
    WhtLloydMax(u32),
    /// `(lm_bits)` Lloyd-Max + 1-bit QJL residual correction (total `lm_bits + 1` bits/val + 2 fp16 scales)
    WhtLmPlusQjl(u32),
}

impl QuantMode {
    fn eff_bits_per_val(self) -> f64 {
        match self {
            QuantMode::Exact => 16.0,
            QuantMode::UniformNoRot(b) | QuantMode::WhtUniform(b) | QuantMode::WhtLloydMax(b) => {
                b as f64 + 16.0 / HD as f64 // + 1 fp16 scale per 128-dim vector
            }
            QuantMode::WhtLmPlusQjl(b) => (b + 1) as f64 + 32.0 / HD as f64, // + 2 fp16 scales (sigma + residual norm)
        }
    }
}

struct LmTables {
    c: [Vec<f32>; 5], // index 1..=4
}

impl LmTables {
    fn new() -> Self {
        Self {
            c: [
                Vec::new(),
                lloyd_max_gaussian(1),
                lloyd_max_gaussian(2),
                lloyd_max_gaussian(3),
                lloyd_max_gaussian(4),
            ],
        }
    }
}

fn quantize_vec_uniform(v: &[f32; HD], bits: u32) -> [f32; HD] {
    let mx = v.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
    if mx <= 1e-12 {
        return *v;
    }
    // Mid-riser uniform grid using all 2^bits levels across [-mx, +mx]
    let n_steps = ((1u32 << bits) - 1) as f32;
    let inv_mx = 1.0 / mx;
    let mut out = [0.0f32; HD];
    for c in 0..HD {
        let u = (v[c] * inv_mx * 0.5 + 0.5).clamp(0.0, 1.0);
        let q = (u * n_steps).round() / n_steps;
        out[c] = (q * 2.0 - 1.0) * mx;
    }
    out
}

fn quantize_vec(
    x: &[f32; HD],
    mode: QuantMode,
    signs1: &[f32; HD],
    signs2: &[f32; HD],
    lm: &LmTables,
) -> [f32; HD] {
    match mode {
        QuantMode::Exact => *x,
        QuantMode::UniformNoRot(b) => quantize_vec_uniform(x, b),
        QuantMode::WhtUniform(b) => {
            let y = rotate_fwd(x, signs1);
            let yq = quantize_vec_uniform(&y, b);
            rotate_inv(&yq, signs1)
        }
        QuantMode::WhtLloydMax(b) => {
            let y = rotate_fwd(x, signs1);
            let rms = (y.iter().map(|&v| v * v).sum::<f32>() / HD as f32).sqrt();
            if rms <= 1e-12 {
                return *x;
            }
            let inv = 1.0 / rms;
            let levels = &lm.c[b as usize];
            let mut yq = [0.0f32; HD];
            for c in 0..HD {
                yq[c] = quant_nearest(y[c] * inv, levels).1 * rms;
            }
            rotate_inv(&yq, signs1)
        }
        QuantMode::WhtLmPlusQjl(b) => {
            let y = rotate_fwd(x, signs1);
            let rms = (y.iter().map(|&v| v * v).sum::<f32>() / HD as f32).sqrt();
            if rms <= 1e-12 {
                return *x;
            }
            let inv = 1.0 / rms;
            let levels = &lm.c[b as usize];
            let mut yq = [0.0f32; HD];
            let mut r = [0.0f32; HD];
            let mut r2 = 0.0f32;
            for c in 0..HD {
                let qv = quant_nearest(y[c] * inv, levels).1 * rms;
                yq[c] = qv;
                let diff = y[c] - qv;
                r[c] = diff;
                r2 += diff * diff;
            }
            let rho = r2.sqrt();
            if rho > 1e-12 {
                // QJL 1-bit unbiased residual estimator:
                // r_hat = sqrt(pi / (2 * HD)) * rho * Q2^T * sign(Q2 * r)
                let z = rotate_fwd(&r, signs2);
                const QJL_COEFF: f32 = 0.11077836568159501; // sqrt(pi / 256)
                let scl = QJL_COEFF * rho;
                let mut z_sgn = [0.0f32; HD];
                for c in 0..HD {
                    z_sgn[c] = if z[c] >= 0.0 { scl } else { -scl };
                }
                let r_hat = rotate_inv(&z_sgn, signs2);
                for c in 0..HD {
                    yq[c] += r_hat[c];
                }
            }
            rotate_inv(&yq, signs1)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn apply_full_quant_range(
    eng: &Engine,
    full_slots: &[FullSlotInfo],
    bak: &KvFullBackup,
    pos_from: usize,
    pos_to: usize,
    k_mode: QuantMode,
    v_mode: QuantMode,
    lm: &LmTables,
    lut: &[f32],
) {
    if pos_from >= pos_to {
        return;
    }
    let n_full = full_slots.len();
    let cap = eng.full_cap.get();
    let kv_buf = eng.kv_full.borrow();
    let dst_base = kv_buf.contents().as_ptr() as usize;
    let bases: Vec<(usize, usize)> = full_slots
        .iter()
        .map(|s| {
            let (kb, vb, _) = eng.kv_slot(s.slot);
            (kb as usize, vb as usize)
        })
        .collect();

    let n_tasks = n_full * NKV;
    let next = AtomicUsize::new(0);
    let bases_ref = &bases;
    std::thread::scope(|sc| {
        for _ in 0..N_THREADS {
            sc.spawn(|| loop {
                let task = next.fetch_add(1, Ordering::Relaxed);
                if task >= n_tasks {
                    break;
                }
                let fi = task / NKV;
                let g = task % NKV;
                let (kb, vb) = bases_ref[fi];
                for (kv, base_off, mode) in [(0usize, kb, k_mode), (1usize, vb, v_mode)] {
                    let s1 = rademacher_signs(((fi as u64) << 16) ^ ((kv as u64) << 8) ^ g as u64 ^ 0x1111);
                    let s2 = rademacher_signs(((fi as u64) << 16) ^ ((kv as u64) << 8) ^ g as u64 ^ 0x7777);
                    let dst_head = unsafe { (dst_base as *mut u16).add(base_off + g * cap * HD) };
                    for pos in pos_from..pos_to {
                        let src = bak.row(fi, kv, g, pos);
                        if matches!(mode, QuantMode::Exact) {
                            unsafe {
                                std::ptr::copy_nonoverlapping(src.as_ptr(), dst_head.add(pos * HD), HD);
                            }
                            continue;
                        }
                        let mut x = [0.0f32; HD];
                        for c in 0..HD {
                            x[c] = lut[src[c] as usize];
                        }
                        let xq = quantize_vec(&x, mode, &s1, &s2, lm);
                        let out = unsafe { std::slice::from_raw_parts_mut(dst_head.add(pos * HD), HD) };
                        for c in 0..HD {
                            out[c] = f32_to_f16(xq[c]);
                        }
                    }
                }
            });
        }
    });
}

fn apply_swa_outer_quant(
    eng: &Engine,
    n: usize,
    keep_recent: usize,
    k_mode: QuantMode,
    v_mode: QuantMode,
    lm: &LmTables,
    lut: &[f32],
) {
    let start_pos = n.saturating_sub(EXT);
    let end_pos = n.saturating_sub(keep_recent);
    if start_pos >= end_pos {
        return;
    }
    let swa_slots: Vec<usize> = (0..N_SLOT).filter(|&s| eng.slot_kind[s].0).collect();
    let cap = eng.swa_cap.get();
    let kv_buf = eng.kv_swa.borrow();
    let base_ptr = kv_buf.contents().as_ptr() as usize;
    let slot_bases: Vec<(usize, usize, usize)> = swa_slots
        .iter()
        .map(|&s| {
            let (kb, vb, _) = eng.kv_slot(s);
            (s, kb as usize, vb as usize)
        })
        .collect();

    let next = AtomicUsize::new(0);
    let sb_ref = &slot_bases;
    std::thread::scope(|sc| {
        for _ in 0..N_THREADS {
            sc.spawn(|| loop {
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= sb_ref.len() {
                    break;
                }
                let (slot, kb, vb) = sb_ref[idx];
                for g in 0..NKV {
                    for (kv, off, mode) in [(0usize, kb, k_mode), (1usize, vb, v_mode)] {
                        if matches!(mode, QuantMode::Exact) {
                            continue;
                        }
                        let s1 = rademacher_signs(
                            0x5A00_0000u64 ^ ((slot as u64) << 16) ^ ((kv as u64) << 8) ^ g as u64,
                        );
                        let s2 = rademacher_signs(
                            0xA500_0000u64 ^ ((slot as u64) << 16) ^ ((kv as u64) << 8) ^ g as u64,
                        );
                        for pos in start_pos..end_pos {
                            let ring_i = pos % cap;
                            let row = unsafe {
                                std::slice::from_raw_parts_mut(
                                    (base_ptr as *mut u16).add(off + (g * cap + ring_i) * HD),
                                    HD,
                                )
                            };
                            let mut x = [0.0f32; HD];
                            for c in 0..HD {
                                x[c] = lut[row[c] as usize];
                            }
                            let xq = quantize_vec(&x, mode, &s1, &s2, lm);
                            for c in 0..HD {
                                row[c] = f32_to_f16(xq[c]);
                            }
                        }
                    }
                }
            });
        }
    });
}

#[allow(clippy::too_many_arguments)]
fn probe_rotquant(
    eng: &Engine,
    full_slots: &[FullSlotInfo],
    snap: &(Vec<u8>, Vec<u8>),
    bak: &KvFullBackup,
    n: usize,
    ref_toks: &[u32],
    ref_logits: &[Vec<f32>],
    has_decode_qcap: bool,
    lut: &[f32],
) {
    let d = ref_toks.len();
    let r0 = n.saturating_sub(EXT);
    let n_full = full_slots.len();
    let n_swa = N_SLOT - n_full;
    let lm = LmTables::new();

    println!("\n--------------------------------------------------------------------------------");
    println!("  PROBE 2 [rotquant]: WHT_128 + Rademacher + Lloyd-Max Gaussian + 1-bit QJL");
    println!("--------------------------------------------------------------------------------");
    println!(
        "  Gaussian N(0,1) Lloyd-Max centroids:\n    1-bit: {:?}\n    2-bit: {:?}\n    3-bit (pos): {:?}\n    4-bit (pos): {:?}",
        lm.c[1],
        lm.c[2],
        &lm.c[3][4..],
        &lm.c[4][8..]
    );

    // 2A. Direct QJL inner-product accuracy micro-evaluation on real (q, k) pairs
    if let Some((qbuf, ncap)) = eng.qcap.borrow().as_ref() {
        let qp = qbuf.contents().as_ptr() as usize;
        let qpos = if has_decode_qcap { n } else { n - 1 };
        let stride = (n / 256).max(1);
        let q_modes = [
            ("LM-1bit (1.125b)", QuantMode::WhtLloydMax(1)),
            ("LM-2bit (2.125b)", QuantMode::WhtLloydMax(2)),
            ("LM-1b+QJL1 (2.25b)", QuantMode::WhtLmPlusQjl(1)),
            ("LM-3bit (3.125b)", QuantMode::WhtLloydMax(3)),
            ("LM-2b+QJL1 (3.25b)", QuantMode::WhtLmPlusQjl(2)),
            ("LM-4bit (4.125b)", QuantMode::WhtLloydMax(4)),
        ];
        println!("\n[2A] Key Inner-Product Unbiasedness & RMSE on Real (q, k) Pairs (pos={qpos}, all 64 heads):");
        println!("  {:>20} | {:>10} {:>10} {:>10}", "quantizer", "mean bias", "dot RMSE", "rel RMSE");
        for &(name, mode) in &q_modes {
            let mut sum_err = 0.0f64;
            let mut sum_sq = 0.0f64;
            let mut sum_ref2 = 0.0f64;
            let mut cnt = 0usize;
            for fi in 0..n_full {
                for g in 0..NKV {
                    let s1 = rademacher_signs(((fi as u64) << 16) ^ g as u64 ^ 0x1111);
                    let s2 = rademacher_signs(((fi as u64) << 16) ^ g as u64 ^ 0x7777);
                    let mut qs = [[0.0f32; HD]; 3];
                    for hh in 0..3 {
                        read_qcap_query(qp, *ncap, fi, qpos, g * 3 + hh, lut, &mut qs[hh]);
                    }
                    for pos in (0..n).step_by(stride) {
                        let row = bak.row(fi, 0, g, pos);
                        let mut k = [0.0f32; HD];
                        for c in 0..HD {
                            k[c] = lut[row[c] as usize];
                        }
                        let kq = quantize_vec(&k, mode, &s1, &s2, &lm);
                        for hh in 0..3 {
                            let r_dot = dot128(&qs[hh], &k) as f64;
                            let q_dot = dot128(&qs[hh], &kq) as f64;
                            let err = q_dot - r_dot;
                            sum_err += err;
                            sum_sq += err * err;
                            sum_ref2 += r_dot * r_dot;
                            cnt += 1;
                        }
                    }
                }
            }
            let bias = sum_err / cnt.max(1) as f64;
            let rmse = (sum_sq / cnt.max(1) as f64).sqrt();
            let rel = (sum_sq / sum_ref2.max(1e-18)).sqrt();
            println!("  {:>20} | {:+10.5} {:10.5} {:9.2}%", name, bias, rmse, rel * 100.0);
        }
    }

    // 2B. End-to-end Engine decode sweeps on kv_full
    let full_256k_mib_for = |far_k: QuantMode, far_v: QuantMode, loc_k: QuantMode, loc_v: QuantMode| -> (f64, f64) {
        let far_b = 0.5 * (far_k.eff_bits_per_val() + far_v.eff_bits_per_val());
        let loc_b = 0.5 * (loc_k.eff_bits_per_val() + loc_v.eff_bits_per_val());
        let far_toks = (CTX_256K - EXT) as f64;
        let loc_toks = EXT as f64;
        let elems_per_tok = (n_full * 2 * NKV * HD) as f64;
        let bytes = elems_per_tok * (far_toks * far_b + loc_toks * loc_b) / 8.0;
        let eff_b = (far_toks * far_b + loc_toks * loc_b) / CTX_256K as f64;
        (eff_b, bytes / 1048576.0)
    };

    let full_configs: Vec<(&str, QuantMode, QuantMode, QuantMode, QuantMode)> = vec![
        ("K4V4 uniform (no rot)", QuantMode::UniformNoRot(4), QuantMode::UniformNoRot(4), QuantMode::UniformNoRot(4), QuantMode::UniformNoRot(4)),
        ("K4V4 WHT-uniform", QuantMode::WhtUniform(4), QuantMode::WhtUniform(4), QuantMode::WhtUniform(4), QuantMode::WhtUniform(4)),
        ("K4V4 WHT-LloydMax", QuantMode::WhtLloydMax(4), QuantMode::WhtLloydMax(4), QuantMode::WhtLloydMax(4), QuantMode::WhtLloydMax(4)),
        ("K4V2 WHT-LloydMax", QuantMode::WhtLloydMax(4), QuantMode::WhtLloydMax(2), QuantMode::WhtLloydMax(4), QuantMode::WhtLloydMax(2)),
        ("K3V3 WHT-LloydMax", QuantMode::WhtLloydMax(3), QuantMode::WhtLloydMax(3), QuantMode::WhtLloydMax(3), QuantMode::WhtLloydMax(3)),
        ("K2+QJL1 / V3 WHT-LM", QuantMode::WhtLmPlusQjl(2), QuantMode::WhtLloydMax(3), QuantMode::WhtLmPlusQjl(2), QuantMode::WhtLloydMax(3)),
        ("K3V2 WHT-LloydMax", QuantMode::WhtLloydMax(3), QuantMode::WhtLloydMax(2), QuantMode::WhtLloydMax(3), QuantMode::WhtLloydMax(2)),
        ("K2V2 WHT-LloydMax", QuantMode::WhtLloydMax(2), QuantMode::WhtLloydMax(2), QuantMode::WhtLloydMax(2), QuantMode::WhtLloydMax(2)),
        ("K1+QJL1 / V2 WHT-LM", QuantMode::WhtLmPlusQjl(1), QuantMode::WhtLloydMax(2), QuantMode::WhtLmPlusQjl(1), QuantMode::WhtLloydMax(2)),
        ("Tiered: far K2V2 + loc K4V4", QuantMode::WhtLloydMax(2), QuantMode::WhtLloydMax(2), QuantMode::WhtLloydMax(4), QuantMode::WhtLloydMax(4)),
        ("Tiered: far K2V2 + loc FP16", QuantMode::WhtLloydMax(2), QuantMode::WhtLloydMax(2), QuantMode::Exact, QuantMode::Exact),
        ("Tiered: far K3V2 + loc FP16", QuantMode::WhtLloydMax(3), QuantMode::WhtLloydMax(2), QuantMode::Exact, QuantMode::Exact),
    ];

    println!("\n[2B] Full-Attention KV Quantization End-to-End Decode ({d} steps, SWA exact FP16):");
    println!(
        "  {:>28} | {:>6} {:>10} | {:>9} {:>9} {:>6}",
        "configuration", "b/val", "256K full", "mean KL", "worst KL", "agree"
    );
    for &(name, fk, fv, lk, lv) in &full_configs {
        apply_full_quant_range(eng, full_slots, bak, 0, r0, fk, fv, &lm, lut);
        apply_full_quant_range(eng, full_slots, bak, r0, n, lk, lv, &lm, lut);
        let ev = eval_decode(eng, snap, n, ref_toks, ref_logits, d);
        let (eff_b, mib_256k) = full_256k_mib_for(fk, fv, lk, lv);
        println!(
            "  {:>28} | {:6.3} {:7.1} Mi | {:9.5} {:9.5} {:2}/{:<2}",
            name, eff_b, mib_256k, ev.mean_kl, ev.worst_kl, ev.agree, d
        );
    }

    // 2C. SWA outer ring (256..1024) quantization while keeping newest 256 and kv_full exact FP16
    restore_kv_full_all(eng, full_slots, bak, n);
    println!("\n[2C] SWA Outer Ring (256..1024) Quantization (newest 256 & full-attn exact FP16):");
    if n <= 256 {
        println!("  [skip] prompt <= 256 tokens (no SWA outer ring)");
    } else {
        println!(
            "  {:>28} | {:>6} {:>10} | {:>9} {:>9} {:>6}",
            "SWA outer config", "b/val", "SWA total", "mean KL", "worst KL", "agree"
        );
        let swa_configs = [
            ("SWA outer WHT-LM K4V4", QuantMode::WhtLloydMax(4), QuantMode::WhtLloydMax(4)),
            ("SWA outer WHT-LM K3V3", QuantMode::WhtLloydMax(3), QuantMode::WhtLloydMax(3)),
            ("SWA outer WHT-LM K3V2", QuantMode::WhtLloydMax(3), QuantMode::WhtLloydMax(2)),
            ("SWA outer WHT-LM K2V2", QuantMode::WhtLloydMax(2), QuantMode::WhtLloydMax(2)),
        ];
        for &(name, km, vm) in &swa_configs {
            eng.restore(snap);
            apply_swa_outer_quant(eng, n, 256, km, vm, &lm, lut);
            let ev = eval_decode_no_restore(eng, n, ref_toks, ref_logits, d);
            let outer_b = 0.5 * (km.eff_bits_per_val() + vm.eff_bits_per_val());
            let hot_mib = (n_swa * 256 * 2 * NKV * HD * 2) as f64 / 1048576.0;
            let outer_mib = (n_swa * 768 * 2 * NKV * HD) as f64 * (outer_b / 8.0) / 1048576.0;
            let eff_b = (256.0 * 16.0 + 768.0 * outer_b) / 1024.0;
            println!(
                "  {:>28} | {:6.3} {:7.1} Mi | {:9.5} {:9.5} {:2}/{:<2}",
                name,
                eff_b,
                hot_mib + outer_mib,
                ev.mean_kl,
                ev.worst_kl,
                ev.agree,
                d
            );
        }
        eng.restore(snap);
    }
}

// =============================================================================
// PROBE 3A: No-RoPE Weighted Centroid Merging (`(k_c, v_c, ln w_c)`)
// =============================================================================

/// Fast k-means on a subset of indices `indices` of `keys` (`[N][HD]`).
/// Returns `k` clusters as `Vec<Vec<usize>>` (original positions).
fn fast_kmeans_subset(keys: &[f32], indices: &[usize], k: usize, iters: usize, seed: u64) -> Vec<Vec<usize>> {
    let m = indices.len();
    if m == 0 {
        return Vec::new();
    }
    let k = k.clamp(1, m);
    if k == 1 {
        return vec![indices.to_vec()];
    }
    if k == m {
        return indices.iter().map(|&idx| vec![idx]).collect();
    }

    // Strided + deterministic spread initialization (avoids O(m*k) k-means++ overhead on large m)
    let mut st = seed;
    let mut cents = vec![0.0f32; k * HD];
    let step = (m as f64) / (k as f64);
    for c in 0..k {
        let j = ((c as f64 * step) as usize + (splitmix64(&mut st) as usize % (step.ceil() as usize).max(1))) % m;
        let idx = indices[j];
        cents[c * HD..(c + 1) * HD].copy_from_slice(&keys[idx * HD..(idx + 1) * HD]);
    }

    let mut assign = vec![0usize; m];
    let mut counts = vec![0usize; k];
    let mut sums = vec![0.0f32; k * HD];

    for _it in 0..iters {
        counts.fill(0);
        sums.fill(0.0);
        for (pos_i, &idx) in indices.iter().enumerate() {
            let kr = &keys[idx * HD..(idx + 1) * HD];
            let mut best_c = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let cr = &cents[c * HD..(c + 1) * HD];
                let mut dist = 0.0f32;
                for d in 0..HD {
                    let df = kr[d] - cr[d];
                    dist += df * df;
                }
                if dist < best_d {
                    best_d = dist;
                    best_c = c;
                }
            }
            assign[pos_i] = best_c;
            counts[best_c] += 1;
            let sr = &mut sums[best_c * HD..(best_c + 1) * HD];
            for d in 0..HD {
                sr[d] += kr[d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                let cr = &mut cents[c * HD..(c + 1) * HD];
                let sr = &sums[c * HD..(c + 1) * HD];
                for d in 0..HD {
                    cr[d] = sr[d] * inv;
                }
            }
        }
    }

    let mut clusters = vec![Vec::new(); k];
    for (pos_i, &idx) in indices.iter().enumerate() {
        clusters[assign[pos_i]].push(idx);
    }
    clusters.retain(|cl| !cl.is_empty());
    clusters
}

/// 2-level divisive k-means clustering of `0..r0` keys into `target_m` weighted centroids,
/// preserving top `outlier_frac` of budget as unmerged isolated outlier keys (`||k - k_mean||_2`).
fn cluster_far_keys_2level(
    keys: &[f32],
    r0: usize,
    target_m: usize,
    outlier_frac: f64,
    seed: u64,
) -> Vec<Vec<usize>> {
    let target_m = target_m.clamp(1, r0);
    if target_m >= r0 {
        return (0..r0).map(|j| vec![j]).collect();
    }

    // 1. Identify isolated outlier keys purely from key geometry (causal, zero query leakage)
    let mut k_mean = [0.0f32; HD];
    for j in 0..r0 {
        let kr = &keys[j * HD..(j + 1) * HD];
        for c in 0..HD {
            k_mean[c] += kr[c];
        }
    }
    for c in 0..HD {
        k_mean[c] /= r0 as f32;
    }

    let n_out = ((target_m as f64) * outlier_frac).round() as usize;
    let n_out = n_out.min(target_m.saturating_sub(1));
    let mut order: Vec<(usize, f32)> = (0..r0)
        .map(|j| {
            let kr = &keys[j * HD..(j + 1) * HD];
            let mut d2 = 0.0f32;
            for c in 0..HD {
                let df = kr[c] - k_mean[c];
                d2 += df * df;
            }
            (j, d2)
        })
        .collect();
    if n_out > 0 {
        order.select_nth_unstable_by(n_out, |a, b| b.1.total_cmp(&a.1));
    }

    let mut result: Vec<Vec<usize>> = Vec::with_capacity(target_m);
    for &(j, _) in order.iter().take(n_out) {
        result.push(vec![j]);
    }
    let rest: Vec<usize> = order.iter().skip(n_out).map(|&(j, _)| j).collect();
    let m_clust = (target_m - n_out).max(1);

    if m_clust <= 48 || rest.len() <= 256 {
        result.extend(fast_kmeans_subset(keys, &rest, m_clust, 5, seed));
        return result;
    }

    // 2. Coarse partition into ~sqrt(m_clust) or 48 buckets, then fine local k-means per bucket
    let n_coarse = (m_clust as f64).sqrt().round().clamp(16.0, 64.0) as usize;
    let coarse = fast_kmeans_subset(keys, &rest, n_coarse, 4, seed);
    let total_rest = rest.len().max(1) as f64;
    let mut alloc: Vec<usize> = coarse
        .iter()
        .map(|b| (((b.len() as f64 / total_rest) * m_clust as f64).round() as usize).clamp(1, b.len()))
        .collect();
    // Adjust sum(alloc) to match m_clust closely
    let mut cur_sum: usize = alloc.iter().sum();
    while cur_sum > m_clust {
        if let Some((bi, _)) = alloc.iter().enumerate().filter(|&(_, &a)| a > 1).max_by_key(|&(_, &a)| a) {
            alloc[bi] -= 1;
            cur_sum -= 1;
        } else {
            break;
        }
    }
    while cur_sum < m_clust {
        if let Some((bi, _)) = coarse
            .iter()
            .enumerate()
            .filter(|&(i, b)| alloc[i] < b.len())
            .max_by_key(|&(i, b)| b.len() / alloc[i].max(1))
        {
            alloc[bi] += 1;
            cur_sum += 1;
        } else {
            break;
        }
    }

    for (bi, bucket) in coarse.into_iter().enumerate() {
        let sub = fast_kmeans_subset(keys, &bucket, alloc[bi], 4, seed ^ ((bi as u64 + 1) * 0x9E37));
        result.extend(sub);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn probe_centroid(
    eng: &Engine,
    full_slots: &[FullSlotInfo],
    snap: &(Vec<u8>, Vec<u8>),
    bak: &KvFullBackup,
    n: usize,
    ref_toks: &[u32],
    ref_logits: &[Vec<f32>],
    has_decode_qcap: bool,
    lut: &[f32],
) {
    let d = ref_toks.len();
    let r0 = n.saturating_sub(EXT);
    let n_full = full_slots.len();
    println!("\n--------------------------------------------------------------------------------");
    println!("  PROBE 3A [centroid]: No-RoPE Weighted Centroid Merging ((k_c, v_c, ln w_c))");
    println!("--------------------------------------------------------------------------------");
    if r0 < 128 {
        println!("  [skip] need prompt > EXT+128 ({}) tokens for centroid merging (got n={n})", EXT + 128);
        return;
    }
    println!(
        "  Exploits zero RoPE & zero relbias for d>=1024: writing each cluster's (k_c, v_c) into its w_c"
    );
    println!(
        "  far-key slots in kv_full is mathematically identical to weighted-centroid attention with bias ln(w_c)."
    );
    println!(
        "  Head-level metrics (rel-L2, cos, far-LSE, cluster-TV) are exact on far keys + sink (no relbias in CPU eval);"
    );
    println!(
        "  E2E KL & argmax are exact GPU decode through Engine::encode_step (with exact relbias & all 48 slots)."
    );

    let (qcap_ptr, ncap) = {
        let qc = eng.qcap.borrow();
        let (buf, cap) = qc.as_ref().expect("qcap missing in probe_centroid");
        (buf.contents().as_ptr() as usize, *cap)
    };
    let q_eval: Vec<usize> = if has_decode_qcap {
        (n..n + d.min(16)).collect()
    } else {
        (n.saturating_sub(16)..n).collect()
    };

    let cap = eng.full_cap.get();
    let bases: Vec<(usize, usize)> = full_slots
        .iter()
        .map(|s| {
            let (kb, vb, _) = eng.kv_slot(s.slot);
            (kb as usize, vb as usize)
        })
        .collect();
    let kv_ptr = eng.kv_full.borrow().contents().as_ptr() as usize;

    let configs: Vec<(&str, usize, f64)> = vec![
        ("8x  (10% outlier)", 8, 0.10),
        ("16x (10% outlier)", 16, 0.10),
        ("16x ( 0% outlier)", 16, 0.00),
        ("32x (10% outlier)", 32, 0.10),
        ("64x (10% outlier)", 64, 0.10),
    ];

    println!(
        "\n  {:>18} | {:>5} {:>9} {:>9} | {:>7} {:>7} {:>7} {:>7} | {:>9} {:>9} {:>6}",
        "config", "M/hd", "256K FP16", "256K +Q4", "rel-L2", "cos", "LSE err", "cl-TV", "E2E KL", "E2E max", "agree"
    );

    for &(label, ratio, out_frac) in &configs {
        let target_m = (r0 / ratio).max(1);
        let n_heads = n_full * NKV;
        let next = AtomicUsize::new(0);
        // Per-head metrics: (rel_l2, cos, lse_err, cl_tv, actual_m)
        let mut head_m = vec![(0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize); n_heads];
        let hm_ptr = head_m.as_mut_ptr() as usize;
        let bases_ref = &bases;
        let q_eval_ref = &q_eval;

        std::thread::scope(|sc| {
            for _ in 0..N_THREADS {
                sc.spawn(|| loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= n_heads {
                        break;
                    }
                    let fi = idx / NKV;
                    let g = idx % NKV;
                    let mut keys = vec![0.0f32; n * HD];
                    let mut vals = vec![0.0f32; n * HD];
                    for j in 0..n {
                        let kr = bak.row(fi, 0, g, j);
                        let vr = bak.row(fi, 1, g, j);
                        let kd = &mut keys[j * HD..(j + 1) * HD];
                        let vd = &mut vals[j * HD..(j + 1) * HD];
                        for c in 0..HD {
                            kd[c] = lut[kr[c] as usize];
                            vd[c] = lut[vr[c] as usize];
                        }
                    }

                    let clusters = cluster_far_keys_2level(
                        &keys,
                        r0,
                        target_m,
                        out_frac,
                        0xC0DE_0000 ^ ((idx as u64) << 12) ^ ratio as u64,
                    );
                    let m_act = clusters.len();
                    let mut ck = vec![0.0f32; m_act * HD];
                    let mut cv = vec![0.0f32; m_act * HD];
                    let mut cw = vec![0usize; m_act];

                    // Write weighted centroids into kv_full (each cluster's member slots get (k_c, v_c))
                    let (kb, vb) = bases_ref[fi];
                    let k_dst_base = unsafe { (kv_ptr as *mut u16).add(kb + g * cap * HD) };
                    let v_dst_base = unsafe { (kv_ptr as *mut u16).add(vb + g * cap * HD) };
                    for (ci, cl) in clusters.iter().enumerate() {
                        let w = cl.len();
                        cw[ci] = w;
                        let mut km = [0.0f64; HD];
                        let mut vm = [0.0f64; HD];
                        for &j in cl {
                            let kr = &keys[j * HD..(j + 1) * HD];
                            let vr = &vals[j * HD..(j + 1) * HD];
                            for c in 0..HD {
                                km[c] += kr[c] as f64;
                                vm[c] += vr[c] as f64;
                            }
                        }
                        let inv = 1.0 / w as f64;
                        let mut kh = [0u16; HD];
                        let mut vh = [0u16; HD];
                        for c in 0..HD {
                            let kf = (km[c] * inv) as f32;
                            let vf = (vm[c] * inv) as f32;
                            kh[c] = f32_to_f16(kf);
                            vh[c] = f32_to_f16(vf);
                            ck[ci * HD + c] = lut[kh[c] as usize];
                            cv[ci * HD + c] = lut[vh[c] as usize];
                        }
                        for &j in cl {
                            unsafe {
                                std::ptr::copy_nonoverlapping(kh.as_ptr(), k_dst_base.add(j * HD), HD);
                                std::ptr::copy_nonoverlapping(vh.as_ptr(), v_dst_base.add(j * HD), HD);
                            }
                        }
                    }

                    // Restore local 1024 exact (in case anything touched it)
                    for j in r0..n {
                        unsafe {
                            std::ptr::copy_nonoverlapping(bak.row(fi, 0, g, j).as_ptr(), k_dst_base.add(j * HD), HD);
                            std::ptr::copy_nonoverlapping(bak.row(fi, 1, g, j).as_ptr(), v_dst_base.add(j * HD), HD);
                        }
                    }

                    // Exact far-key + local + sink head-level attention evaluation on q_eval
                    let mut sum_l2 = 0.0f64;
                    let mut sum_cos = 0.0f64;
                    let mut sum_lse = 0.0f64;
                    let mut sum_tv = 0.0f64;
                    let mut qcnt = 0usize;
                    let mut q = [0.0f32; HD];
                    let mut sc_exact = vec![0.0f32; n];

                    for &pos in q_eval_ref {
                        let k_end = (pos + 1).min(n);
                        for hh in 0..3 {
                            let sink = read_qcap_query(qcap_ptr, ncap, fi, pos, g * 3 + hh, lut, &mut q);
                            // Exact attention over [0..r0] (far) + [r0..k_end] (local) + sink
                            let mut mx_e = sink;
                            let mut mx_far_e = f32::NEG_INFINITY;
                            for j in 0..k_end {
                                let s = dot128(&q, &keys[j * HD..(j + 1) * HD]);
                                sc_exact[j] = s;
                                if s > mx_e {
                                    mx_e = s;
                                }
                                if j < r0 && s > mx_far_e {
                                    mx_far_e = s;
                                }
                            }
                            let mut z_e = (sink - mx_e).exp() as f64;
                            let mut z_far_e = 0.0f64;
                            let mut o_e = [0.0f64; HD];
                            for j in 0..k_end {
                                let w = (sc_exact[j] - mx_e).exp() as f64;
                                z_e += w;
                                let vr = &vals[j * HD..(j + 1) * HD];
                                for c in 0..HD {
                                    o_e[c] += w * vr[c] as f64;
                                }
                                if j < r0 {
                                    z_far_e += (sc_exact[j] - mx_far_e).exp() as f64;
                                }
                            }
                            let lse_far_e = mx_far_e as f64 + z_far_e.max(1e-30).ln();

                            // Approx attention: weighted centroids for far + exact local [r0..k_end] + sink
                            let mut mx_a = sink;
                            let mut mx_far_a = f32::NEG_INFINITY;
                            let mut sc_c = vec![0.0f32; m_act];
                            for ci in 0..m_act {
                                let s = dot128(&q, &ck[ci * HD..(ci + 1) * HD]) + (cw[ci] as f32).ln();
                                sc_c[ci] = s;
                                if s > mx_a {
                                    mx_a = s;
                                }
                                if s > mx_far_a {
                                    mx_far_a = s;
                                }
                            }
                            for j in r0..k_end {
                                if sc_exact[j] > mx_a {
                                    mx_a = sc_exact[j];
                                }
                            }
                            let mut z_a = (sink - mx_a).exp() as f64;
                            let mut z_far_a = 0.0f64;
                            let mut o_a = [0.0f64; HD];
                            for ci in 0..m_act {
                                let w = (sc_c[ci] - mx_a).exp() as f64;
                                z_a += w;
                                let vr = &cv[ci * HD..(ci + 1) * HD];
                                for c in 0..HD {
                                    o_a[c] += w * vr[c] as f64;
                                }
                                z_far_a += (sc_c[ci] - mx_far_a).exp() as f64;
                            }
                            for j in r0..k_end {
                                let w = (sc_exact[j] - mx_a).exp() as f64;
                                z_a += w;
                                let vr = &vals[j * HD..(j + 1) * HD];
                                for c in 0..HD {
                                    o_a[c] += w * vr[c] as f64;
                                }
                            }
                            let lse_far_a = mx_far_a as f64 + z_far_a.max(1e-30).ln();

                            // Output rel-L2 & cosine
                            let (mut diff2, mut ref2, mut ap2, mut dot_o) = (0.0, 0.0, 0.0, 0.0);
                            let (inv_ze, inv_za) = (1.0 / z_e.max(1e-30), 1.0 / z_a.max(1e-30));
                            for c in 0..HD {
                                let oe = o_e[c] * inv_ze;
                                let oa = o_a[c] * inv_za;
                                diff2 += (oa - oe) * (oa - oe);
                                ref2 += oe * oe;
                                ap2 += oa * oa;
                                dot_o += oe * oa;
                            }
                            sum_l2 += (diff2 / ref2.max(1e-18)).sqrt();
                            sum_cos += dot_o / (ref2 * ap2).sqrt().max(1e-18);
                            sum_lse += (lse_far_a - lse_far_e).abs();

                            // Cluster-level softmax TV distance on the full distribution
                            let mut tv = ((sink - mx_e).exp() as f64 * inv_ze
                                - (sink - mx_a).exp() as f64 * inv_za)
                                .abs();
                            for j in r0..k_end {
                                let pe = (sc_exact[j] - mx_e).exp() as f64 * inv_ze;
                                let pa = (sc_exact[j] - mx_a).exp() as f64 * inv_za;
                                tv += (pe - pa).abs();
                            }
                            for (ci, cl) in clusters.iter().enumerate() {
                                let pe: f64 = cl
                                    .iter()
                                    .map(|&j| (sc_exact[j] - mx_e).exp() as f64 * inv_ze)
                                    .sum();
                                let pa = (sc_c[ci] - mx_a).exp() as f64 * inv_za;
                                tv += (pe - pa).abs();
                            }
                            sum_tv += 0.5 * tv;
                            qcnt += 1;
                        }
                    }

                    let out = unsafe { &mut *(hm_ptr as *mut (f64, f64, f64, f64, usize)).add(idx) };
                    let inv_q = 1.0 / qcnt.max(1) as f64;
                    *out = (
                        sum_l2 * inv_q,
                        sum_cos * inv_q,
                        sum_lse * inv_q,
                        sum_tv * inv_q,
                        m_act,
                    );
                });
            }
        });

        // True end-to-end GPU decode on the weighted-centroid KV cache!
        let ev = eval_decode(eng, snap, n, ref_toks, ref_logits, d);

        let inv_h = 1.0 / n_heads as f64;
        let mean_l2: f64 = head_m.iter().map(|x| x.0).sum::<f64>() * inv_h;
        let mean_cos: f64 = head_m.iter().map(|x| x.1).sum::<f64>() * inv_h;
        let mean_lse: f64 = head_m.iter().map(|x| x.2).sum::<f64>() * inv_h;
        let mean_tv: f64 = head_m.iter().map(|x| x.3).sum::<f64>() * inv_h;
        let mean_m: usize = (head_m.iter().map(|x| x.4).sum::<usize>() as f64 * inv_h).round() as usize;

        // 256K memory extrapolation: local 1024 exact + (256K-1024)/ratio centroids (each centroid + 2-byte fp16 ln(w))
        let cent_256k = ((CTX_256K - EXT) as f64) / (ratio as f64);
        let loc_mib = (n_full * EXT * 2 * NKV * HD * 2) as f64 / 1048576.0;
        let cent_fp16_mib = (n_full as f64) * cent_256k * (2 * NKV * HD * 2 + NKV * 2) as f64 / 1048576.0;
        let cent_q4_mib = (n_full as f64) * cent_256k * ((2 * NKV * HD) as f64 * (4.125 / 8.0) + (NKV * 2) as f64) / 1048576.0;

        println!(
            "  {:>18} | {:5} {:7.1} Mi {:7.1} Mi | {:6.2}% {:7.4} {:7.4} {:7.4} | {:9.5} {:9.5} {:2}/{:<2}",
            label,
            mean_m,
            loc_mib + cent_fp16_mib,
            loc_mib + cent_q4_mib,
            mean_l2 * 100.0,
            mean_cos,
            mean_lse,
            mean_tv,
            ev.mean_kl,
            ev.worst_kl,
            ev.agree,
            d
        );
    }
}

// =============================================================================
// PROBE 3B: Two-Tier Sparse Top-K Page Retrieval + Merged Background Centroid
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum PageScorer {
    Quest,
    Ball,
    Wht2,
}

impl PageScorer {
    fn name(self) -> &'static str {
        match self {
            PageScorer::Quest => "Quest(min/max)",
            PageScorer::Ball => "Centroid+Rad",
            PageScorer::Wht2 => "WHT-2bit-sketch",
        }
    }
    /// Index bytes per far token (across 1 KV head, 128 dims) at 256K
    fn index_bytes_per_tok_head(self, blk: usize) -> f64 {
        match self {
            // k_min (fp16) + k_max (fp16) + bg (k_mean, v_mean fp16) per page
            PageScorer::Quest => (4 * HD * 2) as f64 / blk as f64,
            // k_mean (fp16) + v_mean (fp16) + radius (fp16) per page
            PageScorer::Ball => (2 * HD * 2 + 2) as f64 / blk as f64,
            // 2-bit packed keys (32B + 2B scale per tok) + bg (k_mean, v_mean fp16) per page
            PageScorer::Wht2 => 34.0 + (2 * HD * 2) as f64 / blk as f64,
        }
    }
}

struct PageMeta {
    k_min: [f32; HD],
    k_max: [f32; HD],
    k_mean: [f32; HD],
    v_mean: [f32; HD],
    radius: f32,
    count: usize,
}

fn build_pages(keys: &[f32], vals: &[f32], r0: usize, blk: usize) -> Vec<PageMeta> {
    let n_full_pages = r0 / blk;
    let mut pages = Vec::with_capacity(n_full_pages);
    for p in 0..n_full_pages {
        let j0 = p * blk;
        let j1 = j0 + blk;
        let mut k_min = [f32::INFINITY; HD];
        let mut k_max = [f32::NEG_INFINITY; HD];
        let mut km = [0.0f64; HD];
        let mut vm = [0.0f64; HD];
        for j in j0..j1 {
            let kr = &keys[j * HD..(j + 1) * HD];
            let vr = &vals[j * HD..(j + 1) * HD];
            for c in 0..HD {
                if kr[c] < k_min[c] {
                    k_min[c] = kr[c];
                }
                if kr[c] > k_max[c] {
                    k_max[c] = kr[c];
                }
                km[c] += kr[c] as f64;
                vm[c] += vr[c] as f64;
            }
        }
        let inv = 1.0 / blk as f64;
        let mut k_mean = [0.0f32; HD];
        let mut v_mean = [0.0f32; HD];
        for c in 0..HD {
            k_mean[c] = (km[c] * inv) as f32;
            v_mean[c] = (vm[c] * inv) as f32;
        }
        let mut max_r2 = 0.0f32;
        for j in j0..j1 {
            let kr = &keys[j * HD..(j + 1) * HD];
            let mut d2 = 0.0f32;
            for c in 0..HD {
                let df = kr[c] - k_mean[c];
                d2 += df * df;
            }
            if d2 > max_r2 {
                max_r2 = d2;
            }
        }
        pages.push(PageMeta {
            k_min,
            k_max,
            k_mean,
            v_mean,
            radius: max_r2.sqrt(),
            count: blk,
        });
    }
    pages
}

/// Score all pages for a GQA group (3 query heads `qs`) using margin-normalized max across the 3 heads.
fn score_pages_gqa(
    scorer: PageScorer,
    pages: &[PageMeta],
    qs: &[[f32; HD]; 3],
    rot_qs: &[[f32; HD]; 3],
    wht2_keys: &[u8],    // [n_far * 32] packed 2-bit codes (if Wht2)
    wht2_scales: &[f32], // [n_far]
    lm2: &[f32],
    blk: usize,
    out_scores: &mut [f32],
) {
    let np = pages.len();
    let mut head_scores = [vec![0.0f32; np], vec![0.0f32; np], vec![0.0f32; np]];
    for hh in 0..3 {
        let q = &qs[hh];
        let q_norm = q.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let rq = &rot_qs[hh];
        let hs = &mut head_scores[hh];
        match scorer {
            PageScorer::Quest => {
                for (p, pm) in pages.iter().enumerate() {
                    let mut s = 0.0f32;
                    for c in 0..HD {
                        let a = q[c] * pm.k_min[c];
                        let b = q[c] * pm.k_max[c];
                        s += if a > b { a } else { b };
                    }
                    hs[p] = s;
                }
            }
            PageScorer::Ball => {
                for (p, pm) in pages.iter().enumerate() {
                    hs[p] = dot128(q, &pm.k_mean) + q_norm * pm.radius;
                }
            }
            PageScorer::Wht2 => {
                for p in 0..np {
                    let j0 = p * blk;
                    let mut best = f32::NEG_INFINITY;
                    for j in j0..j0 + blk {
                        let packed = &wht2_keys[j * 32..(j + 1) * 32];
                        let mut acc = 0.0f32;
                        for b in 0..32 {
                            let byte = packed[b];
                            let c0 = b * 4;
                            acc += rq[c0] * lm2[(byte & 3) as usize]
                                + rq[c0 + 1] * lm2[((byte >> 2) & 3) as usize]
                                + rq[c0 + 2] * lm2[((byte >> 4) & 3) as usize]
                                + rq[c0 + 3] * lm2[((byte >> 6) & 3) as usize];
                        }
                        let s = acc * wht2_scales[j];
                        if s > best {
                            best = s;
                        }
                    }
                    hs[p] = best;
                }
            }
        }
        // Margin-normalize per query head so no single GQA head's logit offset dominates
        let mx = hs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for s in hs.iter_mut() {
            *s -= mx;
        }
    }
    for p in 0..np {
        out_scores[p] = head_scores[0][p]
            .max(head_scores[1][p])
            .max(head_scores[2][p]);
    }
}

#[allow(clippy::too_many_arguments)]
fn probe_sparse(
    eng: &Engine,
    full_slots: &[FullSlotInfo],
    snap: &(Vec<u8>, Vec<u8>),
    bak: &KvFullBackup,
    n: usize,
    ref_toks: &[u32],
    ref_logits: &[Vec<f32>],
    has_decode_qcap: bool,
    lut: &[f32],
) {
    let d = ref_toks.len();
    let r0 = n.saturating_sub(EXT);
    let n_full = full_slots.len();
    let n_heads = n_full * NKV;
    let lm = LmTables::new();

    println!("\n--------------------------------------------------------------------------------");
    println!("  PROBE 3B [sparse]: Two-Tier Sparse Top-K Page Retrieval + Background Centroid");
    println!("--------------------------------------------------------------------------------");
    if r0 < 256 {
        println!("  [skip] need prompt > EXT+256 ({}) tokens for sparse page retrieval (got n={n})", EXT + 256);
        return;
    }

    let (qcap_ptr, ncap) = {
        let qc = eng.qcap.borrow();
        let (buf, cap) = qc.as_ref().expect("qcap missing in probe_sparse");
        (buf.contents().as_ptr() as usize, *cap)
    };
    let q_eval: Vec<usize> = if has_decode_qcap {
        (n..n + d).collect()
    } else {
        (n.saturating_sub(d)..n).collect()
    };
    let n_q = q_eval.len();
    let fracs = [0.01f64, 0.02, 0.05, 0.10, 0.20];

    // 3B-1: Offline head-level evaluation across BLK in [32, 64] and scorers in [Quest, Ball, Wht2]
    println!(
        "\n[3B-1] Head-Level Page Recall (top-100 far mass), Attn Output Rel-L2 (FP16 / WHT-3b / WHT-2b),"
    );
    println!(
        "       and Inter-Step Page Overlap across {n_q} consecutive decode queries (all {n_heads} heads):"
    );

    // Also save per-step selected page orders for BLK=32 (Quest and Wht2) to drive true E2E GPU decode!
    let blk_e2e = 32usize;
    let np_e2e = r0 / blk_e2e;
    let max_k_e2e = ((np_e2e as f64 * 0.20).ceil() as usize).clamp(1, np_e2e);
    // Layout: [scorer_idx: 0=Quest, 1=Wht2][head_idx][step][rank] -> page_id
    let mut e2e_top_pages: [Vec<Vec<Vec<usize>>>; 2] = [
        vec![vec![Vec::new(); n_q]; n_heads],
        vec![vec![Vec::new(); n_q]; n_heads],
    ];

    let scorers = [PageScorer::Quest, PageScorer::Ball, PageScorer::Wht2];

    for &blk in &[32usize, 64usize] {
        let np = r0 / blk;
        let far_aligned = np * blk;
        if np < 4 {
            continue;
        }
        println!("\n  === Page size BLK={blk} ({np} far pages, {far_aligned} far tokens aligned) ===");
        println!(
            "  {:>15} {:>5} | {:>8} | {:>8} {:>8} {:>8} | {:>7} {:>11} | {:>9}",
            "scorer", "top%", "top100R", "L2 FP16", "L2 Q3b", "L2 Q2b", "overlap", "256K SSD/st", "256K idx"
        );

        // Per-head results: [scorer_idx: 0..3][frac_idx: 0..5] -> (top100_recall, l2_fp16, l2_q3, l2_q2, overlap)
        let mut per_head = vec![[[(0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64); 5]; 3]; n_heads];
        let ph_ptr = per_head.as_mut_ptr() as usize;
        let save_e2e = blk == blk_e2e;
        let e2e_ptr0 = e2e_top_pages[0].as_mut_ptr() as usize;
        let e2e_ptr1 = e2e_top_pages[1].as_mut_ptr() as usize;

        let next = AtomicUsize::new(0);
        let q_eval_ref = &q_eval;
        let lm_ref = &lm;
        std::thread::scope(|sc| {
            for _ in 0..N_THREADS {
                sc.spawn(|| loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= n_heads {
                        break;
                    }
                    let fi = idx / NKV;
                    let g = idx % NKV;
                    let mut keys = vec![0.0f32; n * HD];
                    let mut vals = vec![0.0f32; n * HD];
                    for j in 0..n {
                        let kr = bak.row(fi, 0, g, j);
                        let vr = bak.row(fi, 1, g, j);
                        let kd = &mut keys[j * HD..(j + 1) * HD];
                        let vd = &mut vals[j * HD..(j + 1) * HD];
                        for c in 0..HD {
                            kd[c] = lut[kr[c] as usize];
                            vd[c] = lut[vr[c] as usize];
                        }
                    }
                    let pages = build_pages(&keys, &vals, far_aligned, blk);
                    let mut tot_k = [0.0f64; HD];
                    let mut tot_v = [0.0f64; HD];
                    for pm in &pages {
                        for c in 0..HD {
                            tot_k[c] += pm.k_mean[c] as f64 * pm.count as f64;
                            tot_v[c] += pm.v_mean[c] as f64 * pm.count as f64;
                        }
                    }
                    let s1_k = rademacher_signs(((fi as u64) << 16) ^ g as u64 ^ 0x1111);
                    let s1_v = rademacher_signs(((fi as u64) << 16) ^ (1u64 << 8) ^ g as u64 ^ 0x1111);
                    let s2_dummy = [1.0f32; HD];

                    // Build compact WHT-2bit key sketch + WHT-3b/2b reconstructed far KV ONCE per head
                    let mut wht2_packed = vec![0u8; far_aligned * 32];
                    let mut wht2_scales = vec![0.0f32; far_aligned];
                    let mut k_q3 = vec![0.0f32; far_aligned * HD];
                    let mut v_q3 = vec![0.0f32; far_aligned * HD];
                    let mut k_q2 = vec![0.0f32; far_aligned * HD];
                    let mut v_q2 = vec![0.0f32; far_aligned * HD];
                    for j in 0..far_aligned {
                        let mut kv = [0.0f32; HD];
                        let mut vv = [0.0f32; HD];
                        kv.copy_from_slice(&keys[j * HD..(j + 1) * HD]);
                        vv.copy_from_slice(&vals[j * HD..(j + 1) * HD]);

                        let yk = rotate_fwd(&kv, &s1_k);
                        let rms_k = (yk.iter().map(|&v| v * v).sum::<f32>() / HD as f32).sqrt();
                        wht2_scales[j] = rms_k;
                        if rms_k > 1e-12 {
                            let inv = 1.0 / rms_k;
                            let mut yq2 = [0.0f32; HD];
                            let mut yq3 = [0.0f32; HD];
                            let dst = &mut wht2_packed[j * 32..(j + 1) * 32];
                            for b in 0..32 {
                                let c0 = b * 4;
                                let (i0, q0) = quant_nearest(yk[c0] * inv, &lm_ref.c[2]);
                                let (i1, q1) = quant_nearest(yk[c0 + 1] * inv, &lm_ref.c[2]);
                                let (i2, q2) = quant_nearest(yk[c0 + 2] * inv, &lm_ref.c[2]);
                                let (i3, q3) = quant_nearest(yk[c0 + 3] * inv, &lm_ref.c[2]);
                                dst[b] = (i0 as u8)
                                    | ((i1 as u8) << 2)
                                    | ((i2 as u8) << 4)
                                    | ((i3 as u8) << 6);
                                yq2[c0] = q0 * rms_k;
                                yq2[c0 + 1] = q1 * rms_k;
                                yq2[c0 + 2] = q2 * rms_k;
                                yq2[c0 + 3] = q3 * rms_k;
                                for u in 0..4 {
                                    yq3[c0 + u] = quant_nearest(yk[c0 + u] * inv, &lm_ref.c[3]).1 * rms_k;
                                }
                            }
                            k_q2[j * HD..(j + 1) * HD].copy_from_slice(&rotate_inv(&yq2, &s1_k));
                            k_q3[j * HD..(j + 1) * HD].copy_from_slice(&rotate_inv(&yq3, &s1_k));
                        } else {
                            k_q2[j * HD..(j + 1) * HD].copy_from_slice(&kv);
                            k_q3[j * HD..(j + 1) * HD].copy_from_slice(&kv);
                        }
                        v_q3[j * HD..(j + 1) * HD].copy_from_slice(&quantize_vec(
                            &vv,
                            QuantMode::WhtLloydMax(3),
                            &s1_v,
                            &s2_dummy,
                            lm_ref,
                        ));
                        v_q2[j * HD..(j + 1) * HD].copy_from_slice(&quantize_vec(
                            &vv,
                            QuantMode::WhtLloydMax(2),
                            &s1_v,
                            &s2_dummy,
                            lm_ref,
                        ));
                    }

                    let mut page_scores = vec![0.0f32; np];
                    let mut prev_sets: [[Vec<bool>; 5]; 3] =
                        std::array::from_fn(|_| std::array::from_fn(|_| vec![false; np]));
                    let mut orders: [Vec<usize>; 3] = std::array::from_fn(|_| Vec::new());
                    let mut acc = [[(0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64); 5]; 3];
                    let mut overlap_cnt = 0usize;
                    let mut q_cnt = 0usize;
                    let mut exact_sc = vec![0.0f32; n];
                    let mut saved_q = if save_e2e { vec![Vec::new(); n_q] } else { Vec::new() };
                    let mut saved_w = if save_e2e { vec![Vec::new(); n_q] } else { Vec::new() };

                    for (step_i, &pos) in q_eval_ref.iter().enumerate() {
                        let k_end = (pos + 1).min(n);
                        let mut qs = [[0.0f32; HD]; 3];
                        let mut rot_qs = [[0.0f32; HD]; 3];
                        let mut sinks = [0.0f32; 3];
                        for hh in 0..3 {
                            sinks[hh] = read_qcap_query(
                                qcap_ptr,
                                ncap,
                                fi,
                                pos,
                                g * 3 + hh,
                                lut,
                                &mut qs[hh],
                            );
                            rot_qs[hh] = rotate_fwd(&qs[hh], &s1_k);
                        }

                        // Score & rank pages for all 3 scorers on this step
                        for (si, &scorer) in scorers.iter().enumerate() {
                            score_pages_gqa(
                                scorer,
                                &pages,
                                &qs,
                                &rot_qs,
                                &wht2_packed,
                                &wht2_scales,
                                &lm_ref.c[2],
                                blk,
                                &mut page_scores,
                            );
                            let mut order: Vec<usize> = (0..np).collect();
                            order.sort_by(|&a, &b| page_scores[b].total_cmp(&page_scores[a]));
                            if save_e2e && si == 0 {
                                saved_q[step_i] = order[..max_k_e2e.min(np)].to_vec();
                            } else if save_e2e && si == 2 {
                                saved_w[step_i] = order[..max_k_e2e.min(np)].to_vec();
                            }
                            for (fi_frac, &frac) in fracs.iter().enumerate() {
                                let kp = ((np as f64 * frac).ceil() as usize).clamp(1, np);
                                let mut cur = vec![false; np];
                                let mut inter = 0usize;
                                for &p in order.iter().take(kp) {
                                    cur[p] = true;
                                    if step_i > 0 && prev_sets[si][fi_frac][p] {
                                        inter += 1;
                                    }
                                }
                                if step_i > 0 {
                                    acc[si][fi_frac].4 += inter as f64 / kp as f64;
                                }
                                prev_sets[si][fi_frac] = cur;
                            }
                            orders[si] = order;
                        }
                        if step_i > 0 {
                            overlap_cnt += 1;
                        }

                        // Evaluate attention output accuracy & top-100 recall on first 12 decode steps (reusing exact_sc across all 3 scorers!)
                        if step_i < 12 {
                            for hh in 0..3 {
                                let q = &qs[hh];
                                let sink = sinks[hh];
                                let mut mx_e = sink;
                                for j in 0..k_end {
                                    let s = dot128(q, &keys[j * HD..(j + 1) * HD]);
                                    exact_sc[j] = s;
                                    if s > mx_e {
                                        mx_e = s;
                                    }
                                }
                                let mut z_e = (sink - mx_e).exp() as f64;
                                let mut o_e = [0.0f64; HD];
                                for j in 0..k_end {
                                    let w = (exact_sc[j] - mx_e).exp() as f64;
                                    z_e += w;
                                    let vr = &vals[j * HD..(j + 1) * HD];
                                    for c in 0..HD {
                                        o_e[c] += w * vr[c] as f64;
                                    }
                                }
                                let inv_ze = 1.0 / z_e.max(1e-30);
                                let mut ref_norm2 = 0.0f64;
                                for c in 0..HD {
                                    o_e[c] *= inv_ze;
                                    ref_norm2 += o_e[c] * o_e[c];
                                }

                                let mut far_idx: Vec<usize> = (0..far_aligned).collect();
                                let top100 = 100.min(far_aligned);
                                far_idx.select_nth_unstable_by(top100 - 1, |&a, &b| {
                                    exact_sc[b].total_cmp(&exact_sc[a])
                                });
                                let mut top100_mass_tot = 0.0f64;
                                for &j in far_idx.iter().take(top100) {
                                    top100_mass_tot += (exact_sc[j] - mx_e).exp() as f64 * inv_ze;
                                }

                                for si in 0..3 {
                                    let order = &orders[si];
                                    for (fi_frac, &frac) in fracs.iter().enumerate() {
                                        let kp = ((np as f64 * frac).ceil() as usize).clamp(1, np);
                                        let is_kept = &prev_sets[si][fi_frac];

                                        let mut top100_hit = 0.0f64;
                                        for &j in far_idx.iter().take(top100) {
                                            if is_kept[j / blk] {
                                                top100_hit += (exact_sc[j] - mx_e).exp() as f64 * inv_ze;
                                            }
                                        }
                                        let rec = if top100_mass_tot > 1e-12 {
                                            top100_hit / top100_mass_tot
                                        } else {
                                            1.0
                                        };

                                        // O(kp) background centroid via subtracting kept pages from tot_k / tot_v
                                        let mut bg_k = tot_k;
                                        let mut bg_v = tot_v;
                                        let bg_cnt = far_aligned.saturating_sub(kp * blk);
                                        for &p in order.iter().take(kp) {
                                            let pm = &pages[p];
                                            let w = pm.count as f64;
                                            for c in 0..HD {
                                                bg_k[c] -= pm.k_mean[c] as f64 * w;
                                                bg_v[c] -= pm.v_mean[c] as f64 * w;
                                            }
                                        }
                                        let mut bg_kf = [0.0f32; HD];
                                        let mut bg_vf = [0.0f32; HD];
                                        if bg_cnt > 0 {
                                            let inv = 1.0 / bg_cnt as f64;
                                            for c in 0..HD {
                                                bg_kf[c] = (bg_k[c] * inv) as f32;
                                                bg_vf[c] = (bg_v[c] * inv) as f32;
                                            }
                                        }
                                        let bg_logit = if bg_cnt > 0 {
                                            dot128(q, &bg_kf) + (bg_cnt as f32).ln()
                                        } else {
                                            f32::NEG_INFINITY
                                        };

                                        let eval_variant = |fk: &[f32], fv: &[f32], use_exact_sc: bool| -> f64 {
                                            let mut mx = sink.max(bg_logit);
                                            for &p in order.iter().take(kp) {
                                                for j in p * blk..(p + 1) * blk {
                                                    let s = if use_exact_sc {
                                                        exact_sc[j]
                                                    } else {
                                                        dot128(q, &fk[j * HD..(j + 1) * HD])
                                                    };
                                                    if s > mx {
                                                        mx = s;
                                                    }
                                                }
                                            }
                                            for j in far_aligned..k_end {
                                                if exact_sc[j] > mx {
                                                    mx = exact_sc[j];
                                                }
                                            }
                                            let mut z = (sink - mx).exp() as f64;
                                            let mut o = [0.0f64; HD];
                                            if bg_cnt > 0 {
                                                let w = (bg_logit - mx).exp() as f64;
                                                z += w;
                                                for c in 0..HD {
                                                    o[c] += w * bg_vf[c] as f64;
                                                }
                                            }
                                            for &p in order.iter().take(kp) {
                                                for j in p * blk..(p + 1) * blk {
                                                    let s = if use_exact_sc {
                                                        exact_sc[j]
                                                    } else {
                                                        dot128(q, &fk[j * HD..(j + 1) * HD])
                                                    };
                                                    let w = (s - mx).exp() as f64;
                                                    z += w;
                                                    let vr = &fv[j * HD..(j + 1) * HD];
                                                    for c in 0..HD {
                                                        o[c] += w * vr[c] as f64;
                                                    }
                                                }
                                            }
                                            for j in far_aligned..k_end {
                                                let w = (exact_sc[j] - mx).exp() as f64;
                                                z += w;
                                                let vr = &vals[j * HD..(j + 1) * HD];
                                                for c in 0..HD {
                                                    o[c] += w * vr[c] as f64;
                                                }
                                            }
                                            let inv_z = 1.0 / z.max(1e-30);
                                            let mut d2 = 0.0f64;
                                            for c in 0..HD {
                                                let diff = o[c] * inv_z - o_e[c];
                                                d2 += diff * diff;
                                            }
                                            (d2 / ref_norm2.max(1e-18)).sqrt()
                                        };

                                        acc[si][fi_frac].0 += rec;
                                        acc[si][fi_frac].1 += eval_variant(&keys, &vals, true);
                                        acc[si][fi_frac].2 += eval_variant(&k_q3, &v_q3, false);
                                        acc[si][fi_frac].3 += eval_variant(&k_q2, &v_q2, false);
                                    }
                                }
                                q_cnt += 1;
                            }
                        }
                    }

                    let out = unsafe {
                        &mut *(ph_ptr as *mut [[(f64, f64, f64, f64, f64); 5]; 3]).add(idx)
                    };
                    let iq = 1.0 / q_cnt.max(1) as f64;
                    let io = 1.0 / overlap_cnt.max(1) as f64;
                    for si in 0..3 {
                        for f in 0..5 {
                            out[si][f] = (
                                acc[si][f].0 * iq,
                                acc[si][f].1 * iq,
                                acc[si][f].2 * iq,
                                acc[si][f].3 * iq,
                                acc[si][f].4 * io,
                            );
                        }
                    }
                    if save_e2e {
                        unsafe {
                            *(e2e_ptr0 as *mut Vec<Vec<usize>>).add(idx) = saved_q;
                            *(e2e_ptr1 as *mut Vec<Vec<usize>>).add(idx) = saved_w;
                        }
                    }
                });
            }
        });

        let inv_h = 1.0 / n_heads as f64;
        for (si, &scorer) in scorers.iter().enumerate() {
            let idx_256k_mib = (n_heads as f64)
                * ((CTX_256K - EXT) as f64)
                * scorer.index_bytes_per_tok_head(blk)
                / 1048576.0;
            for (fi_frac, &frac) in fracs.iter().enumerate() {
                let rec = per_head.iter().map(|x| x[si][fi_frac].0).sum::<f64>() * inv_h;
                let l2_16 = per_head.iter().map(|x| x[si][fi_frac].1).sum::<f64>() * inv_h;
                let l2_q3 = per_head.iter().map(|x| x[si][fi_frac].2).sum::<f64>() * inv_h;
                let l2_q2 = per_head.iter().map(|x| x[si][fi_frac].3).sum::<f64>() * inv_h;
                let ov = per_head.iter().map(|x| x[si][fi_frac].4).sum::<f64>() * inv_h;
                let pages_256k = (CTX_256K - EXT) as f64 / blk as f64;
                let delta_pages_per_head = pages_256k * frac * (1.0 - ov);
                let ssd_mib_step =
                    (n_heads as f64) * delta_pages_per_head * (blk * 2 * HD * 2) as f64 / 1048576.0;
                println!(
                    "  {:>15} {:4.0}% | {:7.2}% | {:7.2}% {:7.2}% {:7.2}% | {:6.2}% {:8.2} Mi | {:6.1} Mi",
                    scorer.name(),
                    frac * 100.0,
                    rec * 100.0,
                    l2_16 * 100.0,
                    l2_q3 * 100.0,
                    l2_q2 * 100.0,
                    ov * 100.0,
                    ssd_mib_step,
                    idx_256k_mib
                );
            }
        }
    }

    // 3B-2: True dynamic per-step E2E GPU decode KL via Engine's `Prune` bag (BLK=32)!
    if np_e2e >= 4 && has_decode_qcap {
        let far_aligned = np_e2e * blk_e2e;
        let d_e2e = d.min(n_q);
        // Precompute exact total far-key sum [head][kv][HD] once so background centroid is O(kp * blk)!
        let mut tot_far_sum = vec![[[0.0f64; HD]; 2]; n_heads];
        for idx in 0..n_heads {
            let fi = idx / NKV;
            let g = idx % NKV;
            for kv in 0..2 {
                let s = &mut tot_far_sum[idx][kv];
                for j in 0..far_aligned {
                    let row = bak.row(fi, kv, g, j);
                    for c in 0..HD {
                        s[c] += lut[row[c] as usize] as f64;
                    }
                }
            }
        }
        let tot_far_ref = &tot_far_sum;

        println!(
            "\n[3B-2] True Per-Step Dynamic E2E GPU Decode KL (BLK={blk_e2e}, {d_e2e} steps via Engine::Prune):"
        );
        println!(
            "  {:>16} {:>5} {:>6} {:>5} | {:>10} | {:>9} {:>9} {:>6}",
            "scorer", "top%", "kv_fmt", "bg_c", "256K active", "mean KL", "worst KL", "agree"
        );

        let e2e_cases: Vec<(usize, &str, f64, QuantMode, bool)> = vec![
            (0, "Quest(min/max)", 0.01, QuantMode::Exact, true),
            (0, "Quest(min/max)", 0.02, QuantMode::Exact, true),
            (0, "Quest(min/max)", 0.05, QuantMode::Exact, true),
            (0, "Quest(min/max)", 0.05, QuantMode::Exact, false),
            (0, "Quest(min/max)", 0.05, QuantMode::WhtLloydMax(3), true),
            (0, "Quest(min/max)", 0.05, QuantMode::WhtLloydMax(2), true),
            (0, "Quest(min/max)", 0.10, QuantMode::Exact, true),
            (0, "Quest(min/max)", 0.20, QuantMode::Exact, true),
            (1, "WHT-2bit-sketch", 0.01, QuantMode::Exact, true),
            (1, "WHT-2bit-sketch", 0.02, QuantMode::Exact, true),
            (1, "WHT-2bit-sketch", 0.05, QuantMode::Exact, true),
            (1, "WHT-2bit-sketch", 0.05, QuantMode::WhtLloydMax(3), true),
            (1, "WHT-2bit-sketch", 0.10, QuantMode::Exact, true),
            (1, "WHT-2bit-sketch", 0.20, QuantMode::Exact, true),
        ];

        for &(sc_idx, sc_name, frac, qmode, use_bg) in &e2e_cases {
            let kp = ((np_e2e as f64 * frac).ceil() as usize).clamp(1, np_e2e);
            let f_toks = kp * blk_e2e;
            let rows = f_toks + 1;
            let bg_cnt = far_aligned.saturating_sub(f_toks);
            let bag = eng.gpu.buffer(n_full * 2 * NKV * rows * HD * 2);
            let bag_ptr = bag.contents().as_ptr() as usize;
            *eng.prune.borrow_mut() = Some(Prune {
                bag,
                rows,
                r0: far_aligned,
                lastbias: (bg_cnt.max(1) as f32).ln(),
                use_sum: use_bg && bg_cnt > 0,
            });

            eng.restore(snap);
            let mut agree = 0usize;
            let mut worst_kl = 0.0f64;
            let mut sum_kl = 0.0f64;
            let mut lg = vec![0.0f32; eng.n_vocab];

            for step_i in 0..d_e2e {
                let next = AtomicUsize::new(0);
                let top_ref = &e2e_top_pages[sc_idx];
                let lm_ref = &lm;
                std::thread::scope(|sc| {
                    for _ in 0..N_THREADS {
                        sc.spawn(|| loop {
                            let idx = next.fetch_add(1, Ordering::Relaxed);
                            if idx >= n_heads {
                                break;
                            }
                            let fi = idx / NKV;
                            let g = idx % NKV;
                            let kept_pages = &top_ref[idx][step_i][..kp];
                            let s1_k = rademacher_signs(((fi as u64) << 16) ^ g as u64 ^ 0x1111);
                            let s1_v = rademacher_signs(((fi as u64) << 16) ^ (1u64 << 8) ^ g as u64 ^ 0x1111);
                            let s2_dummy = [1.0f32; HD];

                            for kv in 0..2 {
                                let s1 = if kv == 0 { &s1_k } else { &s1_v };
                                let dst_head = unsafe {
                                    (bag_ptr as *mut u16)
                                        .add(((fi * 2 + kv) * NKV + g) * rows * HD)
                                };
                                let mut bg_sum = tot_far_ref[idx][kv];
                                let mut row_out = 0usize;
                                for &p in kept_pages {
                                    for j in p * blk_e2e..(p + 1) * blk_e2e {
                                        let src = bak.row(fi, kv, g, j);
                                        if use_bg && bg_cnt > 0 {
                                            for c in 0..HD {
                                                bg_sum[c] -= lut[src[c] as usize] as f64;
                                            }
                                        }
                                        let dst = unsafe {
                                            std::slice::from_raw_parts_mut(
                                                dst_head.add(row_out * HD),
                                                HD,
                                            )
                                        };
                                        if matches!(qmode, QuantMode::Exact) {
                                            dst.copy_from_slice(src);
                                        } else {
                                            let mut x = [0.0f32; HD];
                                            for c in 0..HD {
                                                x[c] = lut[src[c] as usize];
                                            }
                                            let xq = quantize_vec(&x, qmode, s1, &s2_dummy, lm_ref);
                                            for c in 0..HD {
                                                dst[c] = f32_to_f16(xq[c]);
                                            }
                                        }
                                        row_out += 1;
                                    }
                                }
                                if use_bg && bg_cnt > 0 {
                                    let inv = 1.0 / bg_cnt as f64;
                                    let dst = unsafe {
                                        std::slice::from_raw_parts_mut(
                                            dst_head.add(f_toks * HD),
                                            HD,
                                        )
                                    };
                                    for c in 0..HD {
                                        dst[c] = f32_to_f16((bg_sum[c] * inv) as f32);
                                    }
                                }
                            }
                        });
                    }
                });

                eng.set_tokens(n + step_i, &[ref_toks[step_i]]);
                let cb = eng.gpu.cmd();
                eng.encode_step(&cb, n + step_i, true, false);
                cb.commit();
                cb.waitUntilCompleted();
                for (dst, &raw) in lg.iter_mut().zip(eng.logits().iter()) {
                    *dst = softcap_f32(raw, eng.softcap);
                }
                let kl = logit_kl_f64(&ref_logits[step_i], &lg);
                if argmax_f32(&lg) == argmax_f32(&ref_logits[step_i]) {
                    agree += 1;
                }
                worst_kl = worst_kl.max(kl);
                sum_kl += kl;
            }

            let fmt_str = match qmode {
                QuantMode::Exact => "FP16",
                QuantMode::WhtLloydMax(3) => "WHT-3b",
                QuantMode::WhtLloydMax(2) => "WHT-2b",
                _ => "other",
            };
            let active_256k_mib = ((n_full * EXT * 2 * NKV * HD * 2) as f64
                + (n_full as f64)
                    * ((CTX_256K - EXT) as f64)
                    * frac
                    * (2 * NKV * HD) as f64
                    * (qmode.eff_bits_per_val() / 8.0))
                / 1048576.0;
            println!(
                "  {:>16} {:4.0}% {:>6} {:>5} | {:7.1} Mi | {:9.5} {:9.5} {:2}/{:<2}",
                sc_name,
                frac * 100.0,
                fmt_str,
                if use_bg { "yes" } else { "no" },
                active_256k_mib,
                sum_kl / d_e2e as f64,
                worst_kl,
                agree,
                d_e2e
            );
            *eng.prune.borrow_mut() = None;
        }
    }
}

// =============================================================================
// PROBE 4: Cross-Pass Loop Predictability (`blk.11` & `blk.17` pass 0->1->2) & SVD
// =============================================================================

fn extract_strided_f32(bak: &KvFullBackup, fi: usize, kv: usize, g: usize, n_use: usize, max_rows: usize, lut: &[f32]) -> (Vec<f32>, usize) {
    let stride = (n_use / max_rows.max(1)).max(1);
    let mut rows = Vec::with_capacity(n_use.min(max_rows) * HD);
    let mut cnt = 0usize;
    for pos in (0..n_use).step_by(stride) {
        let r = bak.row(fi, kv, g, pos);
        for c in 0..HD {
            rows.push(lut[r[c] as usize]);
        }
        cnt += 1;
        if cnt >= max_rows {
            break;
        }
    }
    (rows, cnt)
}

/// Fit centered OLS `Y ~ X W` on `[0..n_train]` and evaluate out-of-sample on `[n_test_start..n_rows]`.
/// `x` is `[n_rows][dx]`, `y` is `[n_rows][HD]`. Returns `(oos_r2, res_var_ratio, test_residuals)`.
fn ols_crosspass_oos(
    x: &[f32],
    dx: usize,
    y: &[f32],
    n_rows: usize,
) -> (f64, f64, Vec<f32>) {
    let n_train = n_rows / 2;
    let gap = 4usize.min(n_rows.saturating_sub(n_train)); // avoid 4-tap conv leakage across split
    let test_start = (n_train + gap).min(n_rows);
    let n_test = n_rows.saturating_sub(test_start);
    if n_train < 16 || n_test < 16 {
        return (0.0, 1.0, Vec::new());
    }

    // Compute training means
    let mut mx = vec![0.0f64; dx];
    let mut my = [0.0f64; HD];
    for i in 0..n_train {
        let xr = &x[i * dx..(i + 1) * dx];
        let yr = &y[i * HD..(i + 1) * HD];
        for c in 0..dx {
            mx[c] += xr[c] as f64;
        }
        for c in 0..HD {
            my[c] += yr[c] as f64;
        }
    }
    for c in 0..dx {
        mx[c] /= n_train as f64;
    }
    for c in 0..HD {
        my[c] /= n_train as f64;
    }

    // Normal equations: (Xc^T Xc + ridge I) W = Xc^T Yc
    let mut xtx = vec![0.0f64; dx * dx];
    let mut xty = vec![0.0f64; dx * HD];
    let mut xc = vec![0.0f64; dx];
    let mut yc = [0.0f64; HD];
    for i in 0..n_train {
        let xr = &x[i * dx..(i + 1) * dx];
        let yr = &y[i * HD..(i + 1) * HD];
        for c in 0..dx {
            xc[c] = xr[c] as f64 - mx[c];
        }
        for c in 0..HD {
            yc[c] = yr[c] as f64 - my[c];
        }
        for r in 0..dx {
            let xr_v = xc[r];
            let row_off = r * dx;
            for c in r..dx {
                xtx[row_off + c] += xr_v * xc[c];
            }
            let y_off = r * HD;
            for c in 0..HD {
                xty[y_off + c] += xr_v * yc[c];
            }
        }
    }
    for r in 0..dx {
        for c in 0..r {
            xtx[r * dx + c] = xtx[c * dx + r];
        }
    }
    let tr: f64 = (0..dx).map(|i| xtx[i * dx + i]).sum::<f64>().max(1e-12);
    let ridge = 1e-4 * tr / dx as f64;
    for i in 0..dx {
        xtx[i * dx + i] += ridge;
    }

    if !cholesky_solve_multi(&xtx, &mut xty, dx, HD) {
        return (0.0, 1.0, Vec::new());
    }
    // xty now holds W [dx x HD]

    // Out-of-sample evaluation on [test_start..n_rows]
    let mut test_my = [0.0f64; HD];
    for i in test_start..n_rows {
        let yr = &y[i * HD..(i + 1) * HD];
        for c in 0..HD {
            test_my[c] += yr[c] as f64;
        }
    }
    for c in 0..HD {
        test_my[c] /= n_test as f64;
    }

    let mut tot_var = 0.0f64;
    let mut res_var = 0.0f64;
    let mut res_vecs = vec![0.0f32; n_test * HD];
    for (ti, i) in (test_start..n_rows).enumerate() {
        let xr = &x[i * dx..(i + 1) * dx];
        let yr = &y[i * HD..(i + 1) * HD];
        for c in 0..dx {
            xc[c] = xr[c] as f64 - mx[c];
        }
        let mut pred = my;
        for r in 0..dx {
            let xv = xc[r];
            let wrow = &xty[r * HD..(r + 1) * HD];
            for c in 0..HD {
                pred[c] += xv * wrow[c];
            }
        }
        for c in 0..HD {
            let yv = yr[c] as f64;
            let err = yv - pred[c];
            let dv = yv - test_my[c];
            res_var += err * err;
            tot_var += dv * dv;
            res_vecs[ti * HD + c] = err as f32;
        }
    }
    let ratio = (res_var / tot_var.max(1e-18)).clamp(0.0, 2.0);
    let r2 = (1.0 - ratio).clamp(-1.0, 1.0);
    (r2, ratio, res_vecs)
}

#[derive(Clone, Copy, Default)]
struct SvdSummary {
    r90: usize,
    r95: usize,
    r99: usize,
    wh_r95: usize,
    wh_r99: usize,
    top1_frac: f64,
    eff_rank: f64,
    mean_energy_frac: f64,
}

fn svd_ranks_from_eigs(eigs: &[f64]) -> (usize, usize, usize, f64, f64) {
    let tot: f64 = eigs.iter().sum::<f64>().max(1e-18);
    let mut cum = 0.0f64;
    let (mut r90, mut r95, mut r99) = (HD, HD, HD);
    let mut ent = 0.0f64;
    for (i, &lam) in eigs.iter().enumerate() {
        let p = lam / tot;
        cum += p;
        if r90 == HD && cum >= 0.90 {
            r90 = i + 1;
        }
        if r95 == HD && cum >= 0.95 {
            r95 = i + 1;
        }
        if r99 == HD && cum >= 0.99 {
            r99 = i + 1;
        }
        if p > 1e-15 {
            ent -= p * p.ln();
        }
    }
    (r90, r95, r99, eigs[0] / tot, ent.exp())
}

fn centered_svd_summary(data: &[f32], n_rows: usize) -> SvdSummary {
    if n_rows < 2 {
        return SvdSummary::default();
    }
    let mut mean = [0.0f64; HD];
    let mut raw_e = 0.0f64;
    for i in 0..n_rows {
        let r = &data[i * HD..(i + 1) * HD];
        for c in 0..HD {
            let v = r[c] as f64;
            mean[c] += v;
            raw_e += v * v;
        }
    }
    for c in 0..HD {
        mean[c] /= n_rows as f64;
    }
    let mean_e: f64 = mean.iter().map(|&m| m * m).sum::<f64>() * n_rows as f64;
    let mean_energy_frac = mean_e / raw_e.max(1e-18);

    let mut cov = vec![0.0f64; HD * HD];
    let mut xc = [0.0f64; HD];
    for i in 0..n_rows {
        let r = &data[i * HD..(i + 1) * HD];
        for c in 0..HD {
            xc[c] = r[c] as f64 - mean[c];
        }
        for a in 0..HD {
            let xa = xc[a];
            let off = a * HD;
            for b in a..HD {
                cov[off + b] += xa * xc[b];
            }
        }
    }
    let inv_n = 1.0 / n_rows as f64;
    for a in 0..HD {
        for b in a..HD {
            let v = cov[a * HD + b] * inv_n;
            cov[a * HD + b] = v;
            cov[b * HD + a] = v;
        }
    }

    let eigs = jacobi_eigenvalues_psd(&cov, HD);
    let (r90, r95, r99, top1_frac, eff_rank) = svd_ranks_from_eigs(&eigs);

    // Standardized (channel-whitened correlation matrix) SVD: R_ab = C_ab / sqrt(C_aa * C_bb)
    let mut std_inv = [0.0f64; HD];
    for a in 0..HD {
        let d = cov[a * HD + a];
        std_inv[a] = if d > 1e-18 { 1.0 / d.sqrt() } else { 0.0 };
    }
    let mut corr = vec![0.0f64; HD * HD];
    for a in 0..HD {
        for b in a..HD {
            let v = cov[a * HD + b] * std_inv[a] * std_inv[b];
            corr[a * HD + b] = v;
            corr[b * HD + a] = v;
        }
    }
    let wh_eigs = jacobi_eigenvalues_psd(&corr, HD);
    let (_, wh_r95, wh_r99, _, _) = svd_ranks_from_eigs(&wh_eigs);

    SvdSummary {
        r90,
        r95,
        r99,
        wh_r95,
        wh_r99,
        top1_frac,
        eff_rank,
        mean_energy_frac,
    }
}

fn probe_crosspass(full_slots: &[FullSlotInfo], bak: &KvFullBackup, n: usize, lut: &[f32]) {
    let r0 = n.saturating_sub(EXT);
    let n_use = if r0 >= 256 { r0 } else { n };
    let max_fit = 4096usize;

    println!("\n--------------------------------------------------------------------------------");
    println!("  PROBE 4 [crosspass]: Loop Cross-Pass Predictability & Centered/Whitened SVD");
    println!("--------------------------------------------------------------------------------");
    println!(
        "  Analyzing {n_use} {} positions (strided sample up to {max_fit}, 50/50 chronological split with 4-tok gap):",
        if r0 >= 256 { "far" } else { "all" }
    );

    // Find looped physical full-attention layers (phys with all N_PASS passes)
    let mut phys_layers: Vec<usize> = full_slots.iter().map(|s| s.phys).collect();
    phys_layers.sort_unstable();
    phys_layers.dedup();

    println!("\n[4A] Cross-Pass Linear Predictability (blk.11 & blk.17, pass 0 -> 1 -> 2, out-of-sample R^2):");
    println!(
        "  {:>6} {:>2} {:>2} | {:>7} {:>7} {:>7} | {:>7} {:>7} {:>6} | {:>7} {:>7} {:>6} | {:>7} | {:>5}",
        "layer", "KV", "g", "cos01", "cos12", "cos02", "R2 0>1", "resV01", "bit01", "R2 01>2", "resV012", "bit012", "R2 1>2", "res95"
    );

    for &phys in &phys_layers {
        let mut passes = [usize::MAX; N_PASS];
        for s in full_slots.iter().filter(|s| s.phys == phys) {
            if s.pass < N_PASS {
                passes[s.pass] = s.fi;
            }
        }
        if passes.iter().any(|&fi| fi == usize::MAX) {
            continue;
        }
        for kv in 0..2 {
            let kv_tag = if kv == 0 { "K" } else { "V" };
            let mut results = vec![[0.0f64; 11]; NKV];
            let res_ptr = results.as_mut_ptr() as usize;
            let next = AtomicUsize::new(0);
            std::thread::scope(|sc| {
                for _ in 0..N_THREADS.min(NKV) {
                    sc.spawn(|| loop {
                        let g = next.fetch_add(1, Ordering::Relaxed);
                        if g >= NKV {
                            break;
                        }
                        let (x0, m0) = extract_strided_f32(bak, passes[0], kv, g, n_use, max_fit, lut);
                        let (x1, _) = extract_strided_f32(bak, passes[1], kv, g, n_use, max_fit, lut);
                        let (x2, _) = extract_strided_f32(bak, passes[2], kv, g, n_use, max_fit, lut);

                        let mean_cos = |a: &[f32], b: &[f32]| -> f64 {
                            let mut s = 0.0f64;
                            for i in 0..m0 {
                                let ra = &a[i * HD..(i + 1) * HD];
                                let rb = &b[i * HD..(i + 1) * HD];
                                let d = dot128(ra, rb) as f64;
                                let na = dot128(ra, ra) as f64;
                                let nb = dot128(rb, rb) as f64;
                                s += d / (na * nb).sqrt().max(1e-18);
                            }
                            s / m0.max(1) as f64
                        };
                        let c01 = mean_cos(&x0, &x1);
                        let c12 = mean_cos(&x1, &x2);
                        let c02 = mean_cos(&x0, &x2);

                        let (r2_01, rv_01, res01) = ols_crosspass_oos(&x0, HD, &x1, m0);
                        let (r2_12, _rv_12, _) = ols_crosspass_oos(&x1, HD, &x2, m0);
                        let mut x01 = vec![0.0f32; m0 * 2 * HD];
                        for i in 0..m0 {
                            x01[i * 2 * HD..i * 2 * HD + HD].copy_from_slice(&x0[i * HD..(i + 1) * HD]);
                            x01[i * 2 * HD + HD..(i + 1) * 2 * HD].copy_from_slice(&x1[i * HD..(i + 1) * HD]);
                        }
                        let (r2_012, rv_012, _) = ols_crosspass_oos(&x01, 2 * HD, &x2, m0);
                        let res01_r95 = if !res01.is_empty() {
                            centered_svd_summary(&res01, res01.len() / HD).r95 as f64
                        } else {
                            HD as f64
                        };
                        let bit01 = 0.5 * (1.0 / rv_01.max(1e-6)).log2().max(0.0);
                        let bit012 = 0.5 * (1.0 / rv_012.max(1e-6)).log2().max(0.0);
                        unsafe {
                            *(res_ptr as *mut [f64; 11]).add(g) = [
                                c01,
                                c12,
                                c02,
                                r2_01,
                                rv_01,
                                bit01,
                                r2_012,
                                rv_012,
                                bit012,
                                r2_12.max(-1.0),
                                res01_r95,
                            ];
                        }
                    });
                }
            });

            let mut avg = [0.0f64; 11];
            for g in 0..NKV {
                let r = results[g];
                for k in 0..11 {
                    avg[k] += r[k] / NKV as f64;
                }
                println!(
                    "  blk.{phys:02} {:>2} {:2} | {:+7.4} {:+7.4} {:+7.4} | {:+7.4} {:7.4} {:5.2}b | {:+7.4} {:7.4} {:5.2}b | {:+7.4} | {:5.0}",
                    kv_tag, g, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10]
                );
            }
            println!(
                "  blk.{phys:02} {:>2} -- | {:+7.4} {:+7.4} {:+7.4} | {:+7.4} {:7.4} {:5.2}b | {:+7.4} {:7.4} {:5.2}b | {:+7.4} | {:5.1}  <-- MEAN",
                kv_tag, avg[0], avg[1], avg[2], avg[3], avg[4], avg[5], avg[6], avg[7], avg[8], avg[9], avg[10]
            );
        }
    }

    // 4B. Centered & Whitened SVD Spectrum of Far K and V across ALL 8 full slots
    println!(
        "\n[4B] Centered & Whitened SVD Spectrum of Far K & V (all {} full slots, mean across 8 KV heads):",
        full_slots.len()
    );
    println!(
        "  {:>12} {:>2} | {:>5} {:>12} {:>12} {:>6} {:>7} | {:>6} {:>6} | {:>7}",
        "layer(pass)", "KV", "r90", "r95 (mn-mx)", "r99 (mn-mx)", "top1%", "effRnk", "wh_r95", "wh_r99", "DC_eng%"
    );
    let n_full = full_slots.len();
    let n_jobs = n_full * 2 * NKV;
    let mut svd_all = vec![SvdSummary::default(); n_jobs];
    let svd_ptr = svd_all.as_mut_ptr() as usize;
    let next = AtomicUsize::new(0);
    std::thread::scope(|sc| {
        for _ in 0..N_THREADS {
            sc.spawn(|| loop {
                let j = next.fetch_add(1, Ordering::Relaxed);
                if j >= n_jobs {
                    break;
                }
                let fi = j / (2 * NKV);
                let rem = j % (2 * NKV);
                let kv = rem / NKV;
                let g = rem % NKV;
                let (x, m) = extract_strided_f32(bak, fi, kv, g, n_use, max_fit, lut);
                let s = centered_svd_summary(&x, m);
                unsafe {
                    *(svd_ptr as *mut SvdSummary).add(j) = s;
                }
            });
        }
    });

    for s in full_slots {
        for kv in 0..2 {
            let kv_tag = if kv == 0 { "K" } else { "V" };
            let slice = &svd_all[(s.fi * 2 + kv) * NKV..(s.fi * 2 + kv + 1) * NKV];
            let avg_r90 = slice.iter().map(|x| x.r90 as f64).sum::<f64>() / NKV as f64;
            let avg_r95 = slice.iter().map(|x| x.r95 as f64).sum::<f64>() / NKV as f64;
            let min_r95 = slice.iter().map(|x| x.r95).min().unwrap_or(0);
            let max_r95 = slice.iter().map(|x| x.r95).max().unwrap_or(0);
            let avg_r99 = slice.iter().map(|x| x.r99 as f64).sum::<f64>() / NKV as f64;
            let min_r99 = slice.iter().map(|x| x.r99).min().unwrap_or(0);
            let max_r99 = slice.iter().map(|x| x.r99).max().unwrap_or(0);
            let avg_wh95 = slice.iter().map(|x| x.wh_r95 as f64).sum::<f64>() / NKV as f64;
            let avg_wh99 = slice.iter().map(|x| x.wh_r99 as f64).sum::<f64>() / NKV as f64;
            let avg_t1 = slice.iter().map(|x| x.top1_frac).sum::<f64>() / NKV as f64;
            let avg_er = slice.iter().map(|x| x.eff_rank).sum::<f64>() / NKV as f64;
            let avg_dc = slice.iter().map(|x| x.mean_energy_frac).sum::<f64>() / NKV as f64;
            println!(
                "  {:>12} {:>2} | {:5.1} {:4.1}({:2}-{:3}) {:4.1}({:2}-{:3}) {:5.2}% {:7.1} | {:6.1} {:6.1} | {:6.2}%",
                s.label,
                kv_tag,
                avg_r90,
                avg_r95,
                min_r95,
                max_r95,
                avg_r99,
                min_r99,
                max_r99,
                avg_t1 * 100.0,
                avg_er,
                avg_wh95,
                avg_wh99,
                avg_dc * 100.0
            );
        }
    }
}

// =============================================================================
// PROBE 5: Real Expert Activation Zipf Skew Across All 24 Physical Layers
// =============================================================================

fn probe_zipf(
    eng: &Engine,
    pt: &[u32],
    ref_routes: &[Vec<[u8; TOPK]>],
    n: usize,
) {
    println!("\n--------------------------------------------------------------------------------");
    println!("  PROBE 5 [zipf]: Real Expert Activation Zipf Skew Across All 24 Physical Layers");
    println!("--------------------------------------------------------------------------------");

    // 1. Collect routing on sequential prompt tokens `0..n_pf` via fast `encode_step(..., logits=false)`
    let n_pf = n.min(384);
    eng.reset();
    eng.set_tokens(0, &pt[..n_pf]);
    let mut prompt_routes: Vec<Vec<[u8; TOPK]>> = Vec::with_capacity(n_pf);
    let t0 = Instant::now();
    for pos in 0..n_pf {
        let cb = eng.gpu.cmd();
        eng.encode_step(&cb, pos, false, false);
        cb.commit();
        cb.waitUntilCompleted();
        prompt_routes.push(eng.routes());
    }
    let pf_r_sec = t0.elapsed().as_secs_f64();

    // Combine prompt + decode routes (keep both separately too for comparison)
    let mut all_routes: Vec<&Vec<[u8; TOPK]>> = Vec::with_capacity(prompt_routes.len() + ref_routes.len());
    for r in &prompt_routes {
        all_routes.push(r);
    }
    for r in ref_routes {
        all_routes.push(r);
    }
    let n_toks = all_routes.len();
    println!(
        "  Collected routing traces: {n_pf} prompt tokens ({pf_r_sec:.2}s) + {} decode tokens = {n_toks} tokens x {N_SLOT} slots",
        ref_routes.len()
    );

    // Aggregate per physical layer (24 x 128) AND per pass for looped layers (12 x 3 x 128)
    let mut hist_phys = [[0u64; NEXP]; N_PHYS];
    let mut hist_pass = [[[0u64; NEXP]; N_PASS]; N_PHYS];
    let mut hist_dec = [[0u64; NEXP]; N_PHYS];
    // Per-token expert bitmask [phys][tok] (u128) for fast batch working-set simulation
    let mut tok_mask_phys = vec![vec![0u128; n_toks]; N_PHYS]; // union across passes of phys layer
    let mut tok_mask_p0 = vec![vec![0u128; n_toks]; N_PHYS];   // single pass (pass 0)

    for (ti, route_48) in all_routes.iter().enumerate() {
        let is_dec = ti >= prompt_routes.len();
        for (exec, exps) in route_48.iter().enumerate() {
            let (phys, pass, _slot) = exec_to_phys_pass(exec);
            let mut m = 0u128;
            for &e in exps {
                let ei = (e as usize).min(NEXP - 1);
                hist_phys[phys][ei] += 1;
                hist_pass[phys][pass][ei] += 1;
                if is_dec {
                    hist_dec[phys][ei] += 1;
                }
                m |= 1u128 << ei;
            }
            tok_mask_phys[phys][ti] |= m;
            if pass == 0 {
                tok_mask_p0[phys][ti] = m;
            }
        }
    }

    let batches = [1usize, 8, 32, 64, 128, 256];
    let n_mc = 200usize;

    let sim_batch_touched = |masks: &[u128], b: usize, seed: u64| -> f64 {
        let m = masks.len();
        if m == 0 {
            return 0.0;
        }
        let mut st = seed;
        let mut tot = 0u64;
        if b <= m {
            // Partial Fisher-Yates without replacement
            let mut idx: Vec<usize> = (0..m).collect();
            for _ in 0..n_mc {
                let mut u = 0u128;
                for i in 0..b {
                    let j = i + (splitmix64(&mut st) as usize % (m - i));
                    idx.swap(i, j);
                    u |= masks[idx[i]];
                }
                tot += u.count_ones() as u64;
            }
        } else {
            for _ in 0..n_mc {
                let mut u = 0u128;
                for _ in 0..b {
                    let j = splitmix64(&mut st) as usize % m;
                    u |= masks[j];
                }
                tot += u.count_ones() as u64;
            }
        }
        tot as f64 / n_mc as f64
    };

    println!("\n[5A] Per-Physical-Layer Expert Skew & Simulated Random-Batch Working Set (out of {NEXP} experts):");
    println!(
        "  {:>6} {:>4} | {:>5} {:>5} {:>5} | {:>5} {:>5} {:>5} {:>5} | {:>4} {:>5} {:>5} {:>5} {:>5} {:>5} | {:>6}",
        "layer", "loop", "H(b)", "2^H", "Zipf", "T16%", "T32%", "T64%", "T96%", "B=1", "B=8", "B=32", "B=64", "B=128", "B=256", "p0-2J"
    );

    let mut sum_ent = 0.0f64;
    let mut sum_cov = [0.0f64; 4];
    let mut sum_b_single = [0.0f64; 6];
    let mut sum_b_union = [0.0f64; 6];

    for phys in 0..N_PHYS {
        let is_loop = (LOOP_START..LOOP_END).contains(&phys);
        let h = &hist_phys[phys];
        let tot: u64 = h.iter().sum();
        let mut sorted: Vec<u64> = h.to_vec();
        sorted.sort_unstable_by(|a, b| b.cmp(a));

        let mut ent = 0.0f64;
        for &c in &sorted {
            if c > 0 {
                let p = c as f64 / tot.max(1) as f64;
                ent -= p * p.log2();
            }
        }
        let cov = |k: usize| -> f64 {
            sorted.iter().take(k).sum::<u64>() as f64 / tot.max(1) as f64 * 100.0
        };
        let (c16, c32, c64, c96) = (cov(16), cov(32), cov(64), cov(96));

        // Zipf exponent OLS fit on ranks 1..96 with count > 0
        let (mut sx, mut sy, mut sxx, mut sxy, mut cnt_z) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (r, &c) in sorted.iter().take(96).enumerate() {
            if c > 0 {
                let lx = ((r + 1) as f64).ln();
                let ly = (c as f64 / tot.max(1) as f64).ln();
                sx += lx;
                sy += ly;
                sxx += lx * lx;
                sxy += lx * ly;
                cnt_z += 1.0;
            }
        }
        let zipf_a = if cnt_z > 2.0 && (cnt_z * sxx - sx * sx).abs() > 1e-12 {
            -(cnt_z * sxy - sx * sy) / (cnt_z * sxx - sx * sx)
        } else {
            0.0
        };

        // Per-single-pass batch working set (relevant to one grouped GEMM dispatch per pass)
        let mut b_single = [0.0f64; 6];
        let mut b_union = [0.0f64; 6];
        for (bi, &b) in batches.iter().enumerate() {
            b_single[bi] = sim_batch_touched(&tok_mask_p0[phys], b, 0x51F0 ^ ((phys as u64) << 8) ^ bi as u64);
            b_union[bi] = if is_loop {
                sim_batch_touched(&tok_mask_phys[phys], b, 0x91F0 ^ ((phys as u64) << 8) ^ bi as u64)
            } else {
                b_single[bi]
            };
            sum_b_single[bi] += b_single[bi] / N_PHYS as f64;
            sum_b_union[bi] += b_union[bi] / N_PHYS as f64;
        }

        // Cross-pass Jaccard overlap of top-32 experts for looped layers (pass 0 vs pass 2)
        let jacc_str = if is_loop {
            let top32 = |counts: &[u64; NEXP]| -> u128 {
                let mut ord: Vec<usize> = (0..NEXP).collect();
                ord.sort_by(|&a, &b| counts[b].cmp(&counts[a]));
                let mut m = 0u128;
                for &e in ord.iter().take(32) {
                    m |= 1u128 << e;
                }
                m
            };
            let m0 = top32(&hist_pass[phys][0]);
            let m2 = top32(&hist_pass[phys][2]);
            let j = (m0 & m2).count_ones() as f64 / (m0 | m2).count_ones().max(1) as f64;
            format!("{:5.2}", j)
        } else {
            "  -  ".to_string()
        };

        sum_ent += ent / N_PHYS as f64;
        sum_cov[0] += c16 / N_PHYS as f64;
        sum_cov[1] += c32 / N_PHYS as f64;
        sum_cov[2] += c64 / N_PHYS as f64;
        sum_cov[3] += c96 / N_PHYS as f64;

        println!(
            "  blk.{phys:02} {:>4} | {:5.2} {:5.1} {:5.2} | {:4.1}% {:4.1}% {:4.1}% {:4.1}% | {:4.1} {:5.1} {:5.1} {:5.1} {:5.1} {:5.1} | {:>6}",
            if is_loop { "3x" } else { "1x" },
            ent,
            2f64.powf(ent),
            zipf_a,
            c16,
            c32,
            c64,
            c96,
            b_single[0],
            b_single[1],
            b_single[2],
            b_single[3],
            b_single[4],
            b_single[5],
            jacc_str
        );
    }

    println!("  -----------------------------------------------------------------------------------------------------------");
    println!(
        "  MEAN (1-pass) | {:5.2} {:5.1}   -   | {:4.1}% {:4.1}% {:4.1}% {:4.1}% | {:4.1} {:5.1} {:5.1} {:5.1} {:5.1} {:5.1} |",
        sum_ent,
        2f64.powf(sum_ent),
        sum_cov[0],
        sum_cov[1],
        sum_cov[2],
        sum_cov[3],
        sum_b_single[0],
        sum_b_single[1],
        sum_b_single[2],
        sum_b_single[3],
        sum_b_single[4],
        sum_b_single[5]
    );
    println!(
        "  MEAN (3p-uni) |       (union across passes 0..2 for blk.6..17)     | {:4.1} {:5.1} {:5.1} {:5.1} {:5.1} {:5.1} |",
        sum_b_union[0],
        sum_b_union[1],
        sum_b_union[2],
        sum_b_union[3],
        sum_b_union[4],
        sum_b_union[5]
    );

    // 5B. Decode-only vs Prompt+Decode skew summary + contiguous-window locality comparison
    let mut dec_ent_sum = 0.0f64;
    let mut dec_t32_sum = 0.0f64;
    for phys in 0..N_PHYS {
        let h = &hist_dec[phys];
        let tot: u64 = h.iter().sum();
        if tot == 0 {
            continue;
        }
        let mut s: Vec<u64> = h.to_vec();
        s.sort_unstable_by(|a, b| b.cmp(a));
        let mut ent = 0.0f64;
        for &c in &s {
            if c > 0 {
                let p = c as f64 / tot as f64;
                ent -= p * p.log2();
            }
        }
        dec_ent_sum += ent / N_PHYS as f64;
        dec_t32_sum += s.iter().take(32).sum::<u64>() as f64 / tot as f64 * 100.0 / N_PHYS as f64;
    }

    // Contiguous window B=8, 32, 64 (single-sequence temporal locality vs independent batch)
    let contig = |b: usize| -> f64 {
        if n_toks < b {
            return 0.0;
        }
        let mut tot = 0.0f64;
        let wins = n_toks - b + 1;
        for phys in 0..N_PHYS {
            let mut ltot = 0u64;
            for w in 0..wins {
                let mut u = 0u128;
                for i in w..w + b {
                    u |= tok_mask_p0[phys][i];
                }
                ltot += u.count_ones() as u64;
            }
            tot += ltot as f64 / wins as f64;
        }
        tot / N_PHYS as f64
    };
    println!(
        "\n[5B] Decode-only ({} tok) mean entropy: {:.2} bits (2^H={:.1}, Top-32={:.1}%)",
        ref_routes.len(),
        dec_ent_sum,
        2f64.powf(dec_ent_sum),
        dec_t32_sum
    );
    println!(
        "     Contiguous-window (temporal locality) mean experts/layer: B=8 -> {:.1} (vs rand {:.1}), B=32 -> {:.1} (vs rand {:.1}), B=64 -> {:.1} (vs rand {:.1})",
        contig(8),
        sum_b_single[1],
        contig(32),
        sum_b_single[2],
        contig(64),
        sum_b_single[3]
    );
}
