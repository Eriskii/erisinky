//! Single-stream Dual-Persona Option B LoRA engine (Linear + 128-expert MoE + sconv/attn_res)
//! with live Eris <-> Ziz blending, overall strength control, component ablation (all/mlp/attention),
//! and a local ErisLLM-style HTTP streaming chat WebUI server.

use crate::engine::{
    Engine, MoeGemmHArgs, E, FF, LOOP_END, LOOP_START, NEXP, N_PHYS, TOPK,
};
use crate::metal::{Buf, Enc};
use crate::multistream::{json_escape, parse_json, JVal};
use crate::sampler::Sampler;
use crate::session::Session;
use crate::tokenizer::Tokenizer;
use objc2_metal::{MTLBuffer, MTLCommandBuffer};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::time::Instant;

// ----------------------------------------------------------------------------- Metal arg structs (mirror persona.metal)
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct MergePbArgs {
    pub pb_off: u32,
    pub a0_off: u32,
    pub b0_off: u32,
    pub a1_off: u32,
    pub b1_off: u32,
    pub out_dim: u32,
    pub in_dim: u32,
    pub r0: u32,
    pub r1: u32,
    pub scale0: f32,
    pub scale1: f32,
    pub pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct PLoraSeg {
    pub a0_off: u32,
    pub b0_off: u32,
    pub a1_off: u32,
    pub b1_off: u32,
    pub x_off: u32,
    pub y_off: u32,
    pub in_dim: u32,
    pub out_dim: u32,
    pub r0: u32,
    pub r1: u32,
    pub tmp_off: u32,
    pub in_half: u32,
    pub out_half: u32,
    pub scale0: f32,
    pub scale1: f32,
    pub pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct PLoraArgs {
    pub nseg: u32,
    pub t: u32,
    pub max_out_blocks: u32,
    pub pad0: u32,
    pub seg: [PLoraSeg; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct PostConvDecArgs {
    pub y_off: u32,
    pub out_off: u32,
    pub cs_off: u32,
    pub convw_off: u32,
    pub res_off: i32,
    pub pad0: u32,
    pub pad1: u32,
    pub pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct PLoraMoeAArgs {
    pub a0_off: u32,
    pub a1_off: u32,
    pub a0_exp_bytes: u32,
    pub a1_exp_bytes: u32,
    pub r0: u32,
    pub r1: u32,
    pub sel_off: u32,
    pub x_off: u32,
    pub x_stride: u32,
    pub in_dim: u32,
    pub tmp_off: u32,
    pub mul_weight: u32,
    pub scale0: f32,
    pub scale1: f32,
    pub pad0: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct PLoraMoeBArgs {
    pub b0_off: u32,
    pub b1_off: u32,
    pub b0_exp_bytes: u32,
    pub b1_exp_bytes: u32,
    pub r0: u32,
    pub r1: u32,
    pub tmp_off: u32,
    pub flags: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct PLoraMoePfAArgs {
    pub a0_off: u32,
    pub a1_off: u32,
    pub a0_exp_bytes: u32,
    pub a1_exp_bytes: u32,
    pub r0: u32,
    pub r1: u32,
    pub xh_off: u32,
    pub sel_off: u32,
    pub inv_off: u32,
    pub tmp_off: u32,
    pub t: u32,
    pub in_dim: u32,
    pub scale0: f32,
    pub scale1: f32,
    pub pad0: u32,
    pub pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct PLoraMoePfB2Args {
    pub b0_off: u32,
    pub b1_off: u32,
    pub b0_exp_bytes: u32,
    pub b1_exp_bytes: u32,
    pub r0: u32,
    pub r1: u32,
    pub sel_off: u32,
    pub inv_off: u32,
    pub tmp_off: u32,
    pub y_off: u32,
    pub t: u32,
    pub flags: u32,
}

// ----------------------------------------------------------------------------- .ilora Adapter
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PersonaParts {
    All,
    Mlp,
    Attention,
}

impl PersonaParts {
    pub fn from_str(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "mlp" => Self::Mlp,
            "attention" | "attn" => Self::Attention,
            _ => Self::All,
        }
    }
    #[allow(dead_code)]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Mlp => "mlp",
            Self::Attention => "attention",
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct LinPair {
    pub a_off: u32,
    pub b_off: u32,
    pub rank: u32,
    pub in_dim: u32,
    pub out_dim: u32,
}

impl LinPair {
    fn none() -> Self {
        Self { a_off: u32::MAX, b_off: u32::MAX, rank: 0, in_dim: 0, out_dim: 0 }
    }
    fn present(&self) -> bool {
        self.rank > 0 && self.a_off != u32::MAX && self.b_off != u32::MAX
    }
}

#[derive(Clone, Copy, Default)]
pub struct MoePair {
    pub a13_off: u32,
    pub b13_off: u32,
    pub a2_off: u32,
    pub b2_off: u32,
    pub rank: u32,
    pub a13_exp_bytes: u32,
    pub b13_exp_bytes: u32,
    pub a2_exp_bytes: u32,
    pub b2_exp_bytes: u32,
}

impl MoePair {
    fn none() -> Self {
        Self { a13_off: u32::MAX, b13_off: u32::MAX, a2_off: u32::MAX, b2_off: u32::MAX, ..Default::default() }
    }
    fn present(&self) -> bool {
        self.rank > 0 && self.a13_off != u32::MAX
    }
}

#[derive(Clone, Copy)]
pub struct CtrlPair {
    pub down: LinPair,
    pub gain: LinPair,
    pub forget: LinPair,
}

impl Default for CtrlPair {
    fn default() -> Self {
        Self { down: LinPair::none(), gain: LinPair::none(), forget: LinPair::none() }
    }
}

#[derive(Clone, Copy)]
pub struct LayerOptionB {
    pub wq: LinPair,
    pub wk: LinPair,
    pub wv: LinPair,
    pub wo: LinPair,
    pub wgate: LinPair,
    pub wrel: LinPair,
    pub wsink: LinPair,
    pub router: LinPair,
    pub ctrl: [CtrlPair; 4], // q, k, v, mlp
    pub moe: MoePair,
}

struct StructEntry {
    name: String,
    pb_off: u32,
    kind: u8, // 0 = attn (k/v/attn sconv), 1 = mlp (mlp sconv), 2 = global (attn_res, msg_query)
    vals: Vec<f32>,
}

pub struct PersonaAdapter {
    pub name: String,
    #[allow(dead_code)]
    pub path: String,
    pub linear_scale: f32,
    pub moe_scale: f32,
    pub ad_buf: Buf,
    pub layers: Vec<LayerOptionB>,
    structs: Vec<StructEntry>,
}

struct TInfo {
    dtype: String,
    shape: Vec<usize>,
    offset: usize,
    nbytes: usize,
}

impl PersonaAdapter {
    pub fn empty(eng: &Engine, name: &str) -> Self {
        let mut layers = Vec::with_capacity(N_PHYS);
        for _ in 0..N_PHYS {
            layers.push(LayerOptionB {
                wq: LinPair::none(),
                wk: LinPair::none(),
                wv: LinPair::none(),
                wo: LinPair::none(),
                wgate: LinPair::none(),
                wrel: LinPair::none(),
                wsink: LinPair::none(),
                router: LinPair::none(),
                ctrl: [CtrlPair::default(); 4],
                moe: MoePair::none(),
            });
        }
        Self {
            name: name.to_string(),
            path: String::new(),
            linear_scale: 0.0,
            moe_scale: 0.0,
            ad_buf: eng.gpu.buffer(256),
            layers,
            structs: Vec::new(),
        }
    }

    pub fn load(eng: &Engine, path: &str) -> Result<Self, String> {
        let t0 = Instant::now();
        let raw = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
        if raw.len() < 12 || &raw[0..8] != b"ILOR0001" {
            return Err(format!("invalid .ilora magic in {path} (run export_ilora.py first)"));
        }
        let hlen = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let data_start = 12 + hlen;
        if data_start > raw.len() {
            return Err(format!("truncated .ilora header in {path}"));
        }
        let hdr_str = std::str::from_utf8(&raw[12..data_start]).map_err(|e| e.to_string())?;
        let hdr = parse_json(hdr_str)?;

        let name = hdr
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or("adapter"))
            .to_string();
        let linear_scale = hdr.get("linear_scale").and_then(|v| v.as_f64()).unwrap_or(2.0) as f32;
        let moe_scale = hdr.get("moe_scale").and_then(|v| v.as_f64()).unwrap_or(2.0) as f32;

        let mut tmap: HashMap<String, TInfo> = HashMap::new();
        if let Some(arr) = hdr.get("tensors").and_then(|v| v.as_arr()) {
            for item in arr {
                let tname = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let dtype = item.get("dtype").and_then(|v| v.as_str()).unwrap_or("f16").to_string();
                let shape: Vec<usize> = item
                    .get("shape")
                    .and_then(|v| v.as_arr())
                    .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect())
                    .unwrap_or_default();
                let offset = item.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let nbytes = item.get("nbytes").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                tmap.insert(tname, TInfo { dtype, shape, offset, nbytes });
            }
        }

        let payload = &raw[data_start..];
        let ad_buf = eng.gpu.buffer_with(payload);

        let get_f32 = |tname: &str| -> Option<Vec<f32>> {
            let ti = tmap.get(tname)?;
            if ti.dtype != "f32" {
                return None;
            }
            let b = &payload[ti.offset..ti.offset + ti.nbytes];
            Some(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
        };

        let get_lin = |prefix: &str| -> LinPair {
            let (Some(ta), Some(tb)) = (tmap.get(&format!("{prefix}.lora_a")), tmap.get(&format!("{prefix}.lora_b"))) else {
                return LinPair::none();
            };
            assert_eq!(ta.dtype, "f16", "{prefix}.lora_a must be f16");
            assert_eq!(tb.dtype, "f16", "{prefix}.lora_b must be f16");
            let (rank, in_dim) = if ta.shape.len() == 2 {
                (ta.shape[0] as u32, ta.shape[1] as u32)
            } else {
                (ta.shape[1] as u32, ta.shape[2] as u32)
            };
            let out_dim = if tb.shape.len() == 2 { tb.shape[0] as u32 } else { tb.shape[1] as u32 };
            assert!(rank <= 16, "{prefix}: linear LoRA rank {rank} > 16");
            LinPair {
                a_off: ta.offset as u32,
                b_off: tb.offset as u32,
                rank,
                in_dim,
                out_dim,
            }
        };

        let get_moe = |layer_idx: usize| -> MoePair {
            let p = format!("layers.{layer_idx}.experts");
            let (Some(a13), Some(b13), Some(a2), Some(b2)) = (
                tmap.get(&format!("{p}.lora_a13")),
                tmap.get(&format!("{p}.lora_b13")),
                tmap.get(&format!("{p}.lora_a2")),
                tmap.get(&format!("{p}.lora_b2")),
            ) else {
                return MoePair::none();
            };
            assert_eq!(a13.shape[0], NEXP);
            let rank = a13.shape[1] as u32;
            assert!(rank <= 8, "layer {layer_idx} MoE LoRA rank {rank} > 8");
            MoePair {
                a13_off: a13.offset as u32,
                b13_off: b13.offset as u32,
                a2_off: a2.offset as u32,
                b2_off: b2.offset as u32,
                rank,
                a13_exp_bytes: (rank as usize * E * 2) as u32,
                b13_exp_bytes: (2 * FF * rank as usize * 2) as u32,
                a2_exp_bytes: (rank as usize * FF * 2) as u32,
                b2_exp_bytes: (E * rank as usize * 2) as u32,
            }
        };

        let mut layers = Vec::with_capacity(N_PHYS);
        let mut structs = Vec::new();
        for i in 0..N_PHYS {
            let p = format!("layers.{i}");
            let mut ll = LayerOptionB {
                wq: get_lin(&format!("{p}.wq")),
                wk: get_lin(&format!("{p}.wk")),
                wv: get_lin(&format!("{p}.wv")),
                wo: get_lin(&format!("{p}.wo")),
                wgate: get_lin(&format!("{p}.gate_attn")),
                wrel: get_lin(&format!("{p}.wr")),
                wsink: get_lin(&format!("{p}.attn_sink_proj")),
                router: get_lin(&format!("{p}.router")),
                ctrl: [CtrlPair::default(); 4],
                moe: get_moe(i),
            };
            if (LOOP_START..LOOP_END).contains(&i) {
                let cnames = ["ctrl_q", "ctrl_k", "ctrl_v", "ctrl_mlp"];
                for (c, cname) in cnames.iter().enumerate() {
                    ll.ctrl[c] = CtrlPair {
                        down: get_lin(&format!("{p}.{cname}.down")),
                        gain: get_lin(&format!("{p}.{cname}.gain")),
                        forget: get_lin(&format!("{p}.{cname}.forget")),
                    };
                }
            }
            layers.push(ll);

            let l = &eng.layers[i];
            if let Some(v) = get_f32(&format!("{p}.k_sconv.weight")) {
                structs.push(StructEntry { name: format!("{p}.k_sconv"), pb_off: l.k_conv, kind: 0, vals: v });
            }
            if let Some(v) = get_f32(&format!("{p}.v_sconv.weight")) {
                structs.push(StructEntry { name: format!("{p}.v_sconv"), pb_off: l.v_conv, kind: 0, vals: v });
            }
            if let Some(v) = get_f32(&format!("{p}.attn_sconv.weight")) {
                structs.push(StructEntry { name: format!("{p}.attn_sconv"), pb_off: l.attn_post_conv, kind: 0, vals: v });
            }
            if let Some(v) = get_f32(&format!("{p}.mlp_sconv.weight")) {
                structs.push(StructEntry { name: format!("{p}.mlp_sconv"), pb_off: l.ffn_post_conv, kind: 1, vals: v });
            }
            if let Some(v) = get_f32(&format!("attn_res_mlp_queries.{i}")) {
                structs.push(StructEntry { name: format!("attn_res_mlp.{i}"), pb_off: l.ffn_res_query, kind: 2, vals: v });
            }
            if i > 0 && l.attn_res_query >= 0 {
                if let Some(v) = get_f32(&format!("attn_res_attn_queries.{}", i - 1)) {
                    structs.push(StructEntry { name: format!("attn_res_attn.{}", i - 1), pb_off: l.attn_res_query as u32, kind: 2, vals: v });
                }
            }
        }
        if let Some(v) = get_f32("attn_res_output_query") {
            structs.push(StructEntry { name: "attn_res_out".into(), pb_off: eng.output_res_query, kind: 2, vals: v });
        }
        if let (Some(qv), Some(kv)) = (
            get_f32("middle_cycle_cross_pass_queries"),
            get_f32("middle_cycle_cross_pass_kv_queries"),
        ) {
            for i in 0..(LOOP_END - LOOP_START) {
                let mut combined = Vec::with_capacity(4 * E);
                combined.extend_from_slice(&qv[(2 * i) * E..(2 * i + 1) * E]);
                combined.extend_from_slice(&kv[(2 * i) * E..(2 * i + 1) * E]);
                combined.extend_from_slice(&kv[(2 * i + 1) * E..(2 * i + 2) * E]);
                combined.extend_from_slice(&qv[(2 * i + 1) * E..(2 * i + 2) * E]);
                structs.push(StructEntry { name: format!("msg_query.{i}"), pb_off: eng.loops[i].msg_query, kind: 2, vals: combined });
            }
        }

        eprintln!(
            "[persona] loaded '{}' ({:.1} MiB, {} tensors, lin_scale={:.2}, moe_scale={:.2}) in {:.0?}",
            name,
            payload.len() as f64 / 1048576.0,
            tmap.len(),
            linear_scale,
            moe_scale,
            t0.elapsed()
        );
        Ok(Self {
            name,
            path: path.to_string(),
            linear_scale,
            moe_scale,
            ad_buf,
            layers,
            structs,
        })
    }
}

// ----------------------------------------------------------------------------- Dual-Persona Runtime Hook
pub struct PersonaHook {
    pub ad0: PersonaAdapter,
    pub ad1: PersonaAdapter,
    pub pb_live: Buf,
    pub w0: f32,
    pub w1: f32,
    pub parts: PersonaParts,
    cached_key: (u32, u32, u8),
}

#[derive(Clone, Copy)]
pub(crate) struct LinSpec {
    pub p0: LinPair,
    pub p1: LinPair,
    pub x_off: u32,
    pub y_off: u32,
    pub in_half: bool,
    pub out_half: bool,
}

impl PersonaHook {
    pub fn new(eng: &Engine, ad0: PersonaAdapter, ad1: PersonaAdapter) -> Self {
        let pb_len = eng.pb.length();
        let pb_live = eng.gpu.buffer(pb_len);
        unsafe {
            std::ptr::copy_nonoverlapping(
                eng.pb.contents().as_ptr() as *const u8,
                pb_live.contents().as_ptr() as *mut u8,
                pb_len,
            );
        }
        let mut h = Self {
            ad0,
            ad1,
            pb_live,
            w0: 0.0,
            w1: 0.0,
            parts: PersonaParts::All,
            cached_key: (u32::MAX, u32::MAX, 255),
        };
        h.set_blend(eng, 0.0, 1.0, PersonaParts::All);
        h
    }

    pub fn blend_key(&self) -> (u32, u32, u8) {
        (self.w0.to_bits(), self.w1.to_bits(), self.parts as u8)
    }

    #[inline]
    pub fn attn_scales(&self) -> (f32, f32) {
        if self.parts == PersonaParts::Mlp {
            (0.0, 0.0)
        } else {
            (self.w0 * self.ad0.linear_scale, self.w1 * self.ad1.linear_scale)
        }
    }

    #[inline]
    pub fn mlp_lin_scales(&self) -> (f32, f32) {
        if self.parts == PersonaParts::Attention {
            (0.0, 0.0)
        } else {
            (self.w0 * self.ad0.linear_scale, self.w1 * self.ad1.linear_scale)
        }
    }

    #[inline]
    pub fn moe_scales(&self) -> (f32, f32) {
        if self.parts == PersonaParts::Attention {
            (0.0, 0.0)
        } else {
            (self.w0 * self.ad0.moe_scale, self.w1 * self.ad1.moe_scale)
        }
    }

    #[inline]
    pub fn has_attn_lora(&self) -> bool {
        let (s0, s1) = self.attn_scales();
        s0 != 0.0 || s1 != 0.0
    }

    #[inline]
    pub fn has_moe_lora(&self) -> bool {
        let (s0, s1) = self.moe_scales();
        s0 != 0.0 || s1 != 0.0
    }

    /// Update live blend weights: `mix` in [0, 1] (0 = ad0/Eris, 1 = ad1/Ziz), `strength` in [0, 3].
    /// Rebuilds `pb_live` (sconv + attn_res + BF16 router & loop controllers) in ~1.5 ms only when changed.
    pub fn set_blend(&mut self, eng: &Engine, mix: f32, strength: f32, parts: PersonaParts) {
        let mix = mix.clamp(0.0, 1.0);
        let strength = strength.clamp(0.0, 3.0);
        let w0 = if strength == 0.0 { 0.0 } else { strength * (1.0 - mix) };
        let w1 = if strength == 0.0 { 0.0 } else { strength * mix };
        let key = (w0.to_bits(), w1.to_bits(), parts as u8);
        if key == self.cached_key {
            return;
        }
        self.w0 = w0;
        self.w1 = w1;
        self.parts = parts;
        self.cached_key = key;

        // 1. Reset pb_live from immutable base eng.pb
        let pb_len = eng.pb.length();
        let base_ptr = eng.pb.contents().as_ptr() as *const u8;
        let live_ptr = self.pb_live.contents().as_ptr() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(base_ptr, live_ptr, pb_len);
        }

        if w0 == 0.0 && w1 == 0.0 {
            return;
        }

        // 2. Patch f32 structural tensors (sconv + attn_res + msg_query): live = base + w0*(t0-base) + w1*(t1-base)
        let kind_w = |kind: u8| -> (f32, f32) {
            match (parts, kind) {
                (PersonaParts::All, _) => (w0, w1),
                (PersonaParts::Attention, 0) => (w0, w1),
                (PersonaParts::Mlp, 1) => (w0, w1),
                _ => (0.0, 0.0),
            }
        };
        let apply_structs = |entries: &[StructEntry], is_second: bool| {
            for ent in entries {
                let (ew0, ew1) = kind_w(ent.kind);
                let w = if is_second { ew1 } else { ew0 };
                if w == 0.0 {
                    continue;
                }
                let n = ent.vals.len();
                unsafe {
                    let b_sl = std::slice::from_raw_parts(base_ptr.add(ent.pb_off as usize) as *const f32, n);
                    let l_sl = std::slice::from_raw_parts_mut(live_ptr.add(ent.pb_off as usize) as *mut f32, n);
                    for j in 0..n {
                        l_sl[j] += w * (ent.vals[j] - b_sl[j]);
                    }
                }
            }
        };
        apply_structs(&self.ad0.structs, false);
        apply_structs(&self.ad1.structs, true);

        // 3. Merge router & loop controller LoRAs into BF16 matrices in pb_live on the GPU (~1.5 ms total!)
        let (as0, as1) = self.attn_scales();
        let (ms0, ms1) = self.mlp_lin_scales();
        let cb = eng.gpu.cmd();
        let e = Enc::new(&cb);
        e.buf(&eng.pb, 0, 0);
        e.buf(&self.pb_live, 0, 1);
        e.buf(&self.ad0.ad_buf, 0, 10);
        e.buf(&self.ad1.ad_buf, 0, 11);
        e.pso(&eng.gpu.pso("k_plora_merge_pb"));

        let dispatch_merge = |pb_off: u32, p0: LinPair, p1: LinPair, s0: f32, s1: f32| {
            let eff0 = if p0.present() { s0 } else { 0.0 };
            let eff1 = if p1.present() { s1 } else { 0.0 };
            if eff0 == 0.0 && eff1 == 0.0 {
                return;
            }
            let (out_dim, in_dim) = if p0.present() { (p0.out_dim, p0.in_dim) } else { (p1.out_dim, p1.in_dim) };
            let args = MergePbArgs {
                pb_off,
                a0_off: p0.a_off,
                b0_off: p0.b_off,
                a1_off: p1.a_off,
                b1_off: p1.b_off,
                out_dim,
                in_dim,
                r0: p0.rank,
                r1: p1.rank,
                scale0: eff0,
                scale1: eff1,
                pad0: 0,
            };
            e.bytes(&args, 5);
            e.dispatch(((in_dim as usize).div_ceil(256), out_dim as usize, 1), 256);
        };

        for i in 0..N_PHYS {
            let l0 = &self.ad0.layers[i];
            let l1 = &self.ad1.layers[i];
            dispatch_merge(eng.layers[i].router, l0.router, l1.router, ms0, ms1);
            if (LOOP_START..LOOP_END).contains(&i) {
                let lw = &eng.loops[i - LOOP_START];
                for c in 0..4 {
                    let (cs0, cs1) = if c == 3 { (ms0, ms1) } else { (as0, as1) };
                    dispatch_merge(lw.down[c], l0.ctrl[c].down, l1.ctrl[c].down, cs0, cs1);
                    dispatch_merge(lw.gain[c], l0.ctrl[c].gain, l1.ctrl[c].gain, cs0, cs1);
                    dispatch_merge(lw.forget[c], l0.ctrl[c].forget, l1.ctrl[c].forget, cs0, cs1);
                }
            }
        }
        e.end();
        cb.commit();
        cb.waitUntilCompleted();
    }

    pub(crate) fn encode_lin(&self, eng: &Engine, e: &Enc, t: usize, tmp_base: u32, specs: &[LinSpec]) {
        let (s0, s1) = self.attn_scales();
        if (s0 == 0.0 && s1 == 0.0) || specs.is_empty() || t == 0 {
            return;
        }
        let mut a = PLoraArgs { t: t as u32, ..Default::default() };
        let mut nseg = 0usize;
        let mut max_out = 0u32;
        let mut cur_tmp = tmp_base;
        for sp in specs {
            let eff0 = if sp.p0.present() { s0 } else { 0.0 };
            let eff1 = if sp.p1.present() { s1 } else { 0.0 };
            if eff0 == 0.0 && eff1 == 0.0 {
                continue;
            }
            let (in_dim, out_dim) = if sp.p0.present() {
                (sp.p0.in_dim, sp.p0.out_dim)
            } else {
                (sp.p1.in_dim, sp.p1.out_dim)
            };
            a.seg[nseg] = PLoraSeg {
                a0_off: sp.p0.a_off,
                b0_off: sp.p0.b_off,
                a1_off: sp.p1.a_off,
                b1_off: sp.p1.b_off,
                x_off: sp.x_off,
                y_off: sp.y_off,
                in_dim,
                out_dim,
                r0: sp.p0.rank,
                r1: sp.p1.rank,
                tmp_off: cur_tmp,
                in_half: sp.in_half as u32,
                out_half: sp.out_half as u32,
                scale0: eff0,
                scale1: eff1,
                pad0: 0,
            };
            cur_tmp += (t * 32) as u32;
            max_out = max_out.max(out_dim);
            nseg += 1;
        }
        if nseg == 0 {
            return;
        }
        a.nseg = nseg as u32;
        a.max_out_blocks = max_out.div_ceil(256);

        e.pso(&eng.gpu.pso("k_plora_lin_a"));
        e.bytes(&a, 5);
        e.dispatch((t, nseg, 1), 256);
        e.barrier();

        e.pso(&eng.gpu.pso("k_plora_lin_b"));
        e.bytes(&a, 5);
        e.dispatch((a.max_out_blocks as usize, t, nseg), 256);
        e.barrier();
    }

    pub(crate) fn encode_moe_a_dec(
        &self,
        eng: &Engine,
        e: &Enc,
        ip: usize,
        is_w2: bool,
        sel_off: u32,
        x_off: u32,
        tmp_off: u32,
    ) {
        let (s0, s1) = self.moe_scales();
        let m0 = self.ad0.layers[ip].moe;
        let m1 = self.ad1.layers[ip].moe;
        let args = PLoraMoeAArgs {
            a0_off: if is_w2 { m0.a2_off } else { m0.a13_off },
            a1_off: if is_w2 { m1.a2_off } else { m1.a13_off },
            a0_exp_bytes: if is_w2 { m0.a2_exp_bytes } else { m0.a13_exp_bytes },
            a1_exp_bytes: if is_w2 { m1.a2_exp_bytes } else { m1.a13_exp_bytes },
            r0: m0.rank,
            r1: m1.rank,
            sel_off,
            x_off,
            x_stride: if is_w2 { FF as u32 } else { 0 },
            in_dim: if is_w2 { FF as u32 } else { E as u32 },
            tmp_off,
            mul_weight: is_w2 as u32,
            scale0: if m0.present() { s0 } else { 0.0 },
            scale1: if m1.present() { s1 } else { 0.0 },
            pad0: 0,
            pad1: 0,
        };
        e.pso(&eng.gpu.pso("k_plora_moe_a_dec"));
        e.bytes(&args, 5);
        e.dispatch((TOPK, 1, 1), 256);
        e.barrier();
    }

    pub(crate) fn moe_b_args(&self, ip: usize, is_w2: bool, tmp_off: u32) -> PLoraMoeBArgs {
        let (s0, s1) = self.moe_scales();
        let m0 = self.ad0.layers[ip].moe;
        let m1 = self.ad1.layers[ip].moe;
        let mut flags = 0u32;
        if s0 != 0.0 && m0.present() {
            flags |= 1;
        }
        if s1 != 0.0 && m1.present() {
            flags |= 2;
        }
        PLoraMoeBArgs {
            b0_off: if is_w2 { m0.b2_off } else { m0.b13_off },
            b1_off: if is_w2 { m1.b2_off } else { m1.b13_off },
            b0_exp_bytes: if is_w2 { m0.b2_exp_bytes } else { m0.b13_exp_bytes },
            b1_exp_bytes: if is_w2 { m1.b2_exp_bytes } else { m1.b13_exp_bytes },
            r0: m0.rank,
            r1: m1.rank,
            tmp_off,
            flags,
        }
    }

    pub(crate) fn encode_moe_a13_pf(
        &self,
        eng: &Engine,
        e: &Enc,
        ip: usize,
        t: usize,
        xh_off: u32,
        sel_off: u32,
        inv_off: u32,
        tmp_off: u32,
    ) {
        let (s0, s1) = self.moe_scales();
        let m0 = self.ad0.layers[ip].moe;
        let m1 = self.ad1.layers[ip].moe;
        let args = PLoraMoePfAArgs {
            a0_off: m0.a13_off,
            a1_off: m1.a13_off,
            a0_exp_bytes: m0.a13_exp_bytes,
            a1_exp_bytes: m1.a13_exp_bytes,
            r0: m0.rank,
            r1: m1.rank,
            xh_off,
            sel_off,
            inv_off,
            tmp_off,
            t: t as u32,
            in_dim: E as u32,
            scale0: if m0.present() { s0 } else { 0.0 },
            scale1: if m1.present() { s1 } else { 0.0 },
            pad0: 0,
            pad1: 0,
        };
        e.pso(&eng.gpu.pso("k_plora_moe_a13_pf"));
        e.bytes(&args, 5);
        e.dispatch((t, 1, 1), 256);
        e.barrier();
    }

    pub(crate) fn encode_moe_w2_pf(
        &self,
        eng: &Engine,
        e: &Enc,
        ip: usize,
        t: usize,
        acth_off: u32,
        sel_off: u32,
        inv_off: u32,
        tmp_off: u32,
        y_off: u32,
    ) {
        let (s0, s1) = self.moe_scales();
        let m0 = self.ad0.layers[ip].moe;
        let m1 = self.ad1.layers[ip].moe;
        let eff0 = if m0.present() { s0 } else { 0.0 };
        let eff1 = if m1.present() { s1 } else { 0.0 };
        if eff0 == 0.0 && eff1 == 0.0 {
            return;
        }
        let a_args = PLoraMoePfAArgs {
            a0_off: m0.a2_off,
            a1_off: m1.a2_off,
            a0_exp_bytes: m0.a2_exp_bytes,
            a1_exp_bytes: m1.a2_exp_bytes,
            r0: m0.rank,
            r1: m1.rank,
            xh_off: acth_off,
            sel_off,
            inv_off,
            tmp_off,
            t: t as u32,
            in_dim: FF as u32,
            scale0: eff0,
            scale1: eff1,
            pad0: 0,
            pad1: 0,
        };
        e.pso(&eng.gpu.pso("k_plora_moe_a2_pf"));
        e.bytes(&a_args, 5);
        e.dispatch((t, TOPK, 1), 256);
        e.barrier();

        let mut flags = 0u32;
        if eff0 != 0.0 {
            flags |= 1;
        }
        if eff1 != 0.0 {
            flags |= 2;
        }
        let b_args = PLoraMoePfB2Args {
            b0_off: m0.b2_off,
            b1_off: m1.b2_off,
            b0_exp_bytes: m0.b2_exp_bytes,
            b1_exp_bytes: m1.b2_exp_bytes,
            r0: m0.rank,
            r1: m1.rank,
            sel_off,
            inv_off,
            tmp_off,
            y_off,
            t: t as u32,
            flags,
        };
        e.pso(&eng.gpu.pso("k_plora_moe_b2_pf"));
        e.bytes(&b_args, 5);
        e.dispatch((t, E / 256, 1), 256);
        e.barrier();
    }

    pub(crate) fn encode_moe_gu_pf_lora(
        &self,
        eng: &Engine,
        e: &Enc,
        ip: usize,
        g: &MoeGemmHArgs,
        name: &str,
        ind_off: usize,
        tg: usize,
        tmp_off: u32,
    ) {
        let la = self.moe_b_args(ip, false, tmp_off);
        e.pso(&eng.gpu.pso(name));
        e.bytes(g, 5);
        e.bytes(&la, 8);
        e.dispatch_indirect(&eng.act_buf(), ind_off, tg);
    }
}

// ----------------------------------------------------------------------------- Exactness Self-Test (`persona-check`)
fn softcap(x: f32, c: f32) -> f32 {
    if c > 0.0 { c * (x / c).tanh() } else { x }
}

fn kl_and_diff(a: &[f32], b: &[f32], cap: f32) -> (usize, usize, f32, f32) {
    let ga: Vec<f32> = a.iter().map(|&x| softcap(x, cap)).collect();
    let gb: Vec<f32> = b.iter().map(|&x| softcap(x, cap)).collect();
    let am = |v: &[f32]| v.iter().enumerate().fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m }).0;
    let lse = |v: &[f32]| {
        let m = v.iter().cloned().fold(f32::MIN, f32::max);
        m + v.iter().map(|x| (x - m).exp()).sum::<f32>().ln()
    };
    let (a1, a2) = (am(&ga), am(&gb));
    let (l1, l2) = (lse(&ga), lse(&gb));
    let kl: f32 = gb.iter().zip(&ga).map(|(rb, ra)| {
        let pb = (rb - l2).exp();
        pb * ((rb - l2) - (ra - l1))
    }).sum();
    let maxd = ga.iter().zip(&gb).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    (a1, a2, maxd, kl)
}

pub fn run_persona_check(model: &str, eris_path: &str, ziz_path: &str, ctx: usize) {
    let eng = Engine::load(model, ctx);
    let tk = Tokenizer::load(&eng.g);

    let prompt_str = "<|message_system|><|content_text|>[Discord DM with a friend · 2026-11-26]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>friend: hey what are you working on today? tell me about your setup<|end_message|><|message_model|><|content_text|>";
    let toks = tk.encode(prompt_str, true);
    println!("[persona-check] prompt has {} tokens", toks.len());

    // 1. Base logits (no persona hook)
    eng.reset();
    eng.reserve(toks.len() + 64, false);
    eng.ensure_batch(toks.len());
    eng.set_tokens(0, &toks);
    let cb = eng.gpu.cmd();
    eng.encode_prefill(&cb, 0, toks.len(), true, false);
    cb.commit();
    cb.waitUntilCompleted();
    let base_pf = eng.logits().to_vec();

    // Load both adapters
    let ad0 = PersonaAdapter::load(&eng, eris_path).expect("load eris .ilora");
    let ad1 = PersonaAdapter::load(&eng, ziz_path).expect("load ziz .ilora");
    // Sanity-check structural tensor alignment against base GGUF PB (should be tiny fine-tune deltas, not layout mismatches!)
    let base_ptr = eng.pb.contents().as_ptr() as *const u8;
    for ad in [&ad0, &ad1] {
        let mut worst = (0.0f32, String::new());
        for ent in &ad.structs {
            let b_sl = unsafe { std::slice::from_raw_parts(base_ptr.add(ent.pb_off as usize) as *const f32, ent.vals.len()) };
            let md = ent.vals.iter().zip(b_sl).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            if md > worst.0 { worst = (md, ent.name.clone()); }
        }
        println!("[Struct check] '{}': max|trained - base_gguf| = {:.5} (in {})", ad.name, worst.0, worst.1);
    }
    eng.attach_persona(PersonaHook::new(&eng, ad0, ad1));

    // Base pf vs dec reference
    eng.set_persona_blend(0.0, 0.0, PersonaParts::All);
    eng.reset();
    eng.set_tokens(0, &toks);
    for pos in 0..toks.len() {
        let cb = eng.gpu.cmd();
        eng.encode_step(&cb, pos, pos + 1 == toks.len(), false);
        cb.commit();
        cb.waitUntilCompleted();
    }
    let base_dec = eng.logits().to_vec();
    let (ba1, ba2, bmd, bkl) = kl_and_diff(&base_pf, &base_dec, eng.softcap);
    println!("[Test 0] base pf vs dec: argmax {ba1}/{ba2} | maxdiff={bmd:.4} | KL={bkl:.6}");

    // 2. Strength = 0.0 must be bit-identical to base
    eng.set_persona_blend(0.5, 0.0, PersonaParts::All);
    eng.reset();
    eng.ensure_batch(toks.len());
    eng.set_tokens(0, &toks);
    let cb = eng.gpu.cmd();
    eng.encode_prefill(&cb, 0, toks.len(), true, false);
    cb.commit();
    cb.waitUntilCompleted();
    let zero_pf = eng.logits().to_vec();
    let (a1, a2, maxd, kl) = kl_and_diff(&zero_pf, &base_pf, eng.softcap);
    println!("[Test 1] strength=0.0 vs base: argmax {a1} vs {a2} | maxdiff={maxd:.6} | KL={kl:.8}");
    assert!(maxd < 1e-5, "strength=0.0 drifted from base: maxdiff={maxd}");

    // 3. Prefill vs token-by-token decode parity at mix=0.0 (Eris), 0.5 (Blend), 1.0 (Ziz)
    for (label, mix, strength, parts) in [
        ("Eris (mix=0.0, s=1.0)", 0.0f32, 1.0f32, PersonaParts::All),
        ("50/50 (mix=0.5, s=1.0)", 0.5, 1.0, PersonaParts::All),
        ("Ziz (mix=1.0, s=1.0)", 1.0, 1.0, PersonaParts::All),
        ("MLP-only (mix=0.0, s=1.0)", 0.0, 1.0, PersonaParts::Mlp),
        ("Attn-only (mix=0.0, s=1.0)", 0.0, 1.0, PersonaParts::Attention),
    ] {
        let t_blend = Instant::now();
        eng.set_persona_blend(mix, strength, parts);
        let blend_ms = t_blend.elapsed().as_secs_f64() * 1e3;

        // Batched prefill
        eng.reset();
        eng.ensure_batch(toks.len());
        eng.set_tokens(0, &toks);
        let cb = eng.gpu.cmd();
        eng.encode_prefill(&cb, 0, toks.len(), true, false);
        cb.commit();
        cb.waitUntilCompleted();
        let lg_pf = eng.logits().to_vec();

        // Step-by-step decode
        eng.reset();
        eng.set_tokens(0, &toks);
        for pos in 0..toks.len() {
            let cb = eng.gpu.cmd();
            eng.encode_step(&cb, pos, pos + 1 == toks.len(), false);
            cb.commit();
            cb.waitUntilCompleted();
        }
        let lg_dec = eng.logits().to_vec();
        let (p1, d1, md, kl) = kl_and_diff(&lg_pf, &lg_dec, eng.softcap);
        let (_, _, md_base, kl_base) = kl_and_diff(&lg_pf, &base_pf, eng.softcap);
        println!(
            "[Test 2] {label:<28} | blend {blend_ms:4.1}ms | pf vs dec: argmax {p1}/{d1} maxdiff={md:.4} KL={kl:.6} | vs base KL={kl_base:.4} (maxdiff={md_base:.2})"
        );
        assert!(kl < 0.02, "Prefill vs decode mismatch on {label}: KL={kl}");
    }

    // 4. Quick live samples + speed check!
    println!("\n[Test 3] Live samples + speed:");
    let samples = [
        ("Eris direct (Discord)", 0.0f32, 1.0f32, "<|message_system|><|content_text|>[Discord DM with a friend · 2026-11-26]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>friend: how did the Inkling training go??<|end_message|><|message_model|><|content_text|>"),
        ("50/50 Blend (Discord)", 0.5, 1.0, "<|message_system|><|content_text|>[Discord DM with a friend · 2026-11-26]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>friend: what do you think about self-modification and decision theory?<|end_message|><|message_model|><|content_text|>"),
        ("Ziz direct (Essay)", 1.0, 1.0, "<|message_system|><|content_text|>[sinceriously.fyi]\nOn Decision Theory and Self-Modification<|end_message|><|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_model|><|content_text|>"),
        ("Ziz native thinking (FAQ, effort=0.7)", 1.0, 1.0, "<|message_system|><|content_text|>[sinceriously.fyi · Cached Answers]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|><|message_user|><|content_text|>Why do people fail to coordinate around existential risk?<|end_message|>"),
        ("Eris native thinking (effort=0.7)", 0.0, 1.0, "<|message_system|><|content_text|>[Discord DM with a friend · 2026-11-26]<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|><|message_user|><|content_text|>friend: quick sanity check, what is 17 * 23?<|end_message|>"),
    ];
    for (title, mix, strength, p) in samples {
        eng.set_persona_blend(mix, strength, PersonaParts::All);
        let pt = tk.encode(p, true);
        let mut sess = Session::new(&eng, &tk);
        let first = sess.prefill(&pt);
        let mut sampler = Sampler::new(0.7, 64, 0.95, eng.softcap, 42);
        let mut out_bytes = Vec::new();
        sess.generate(first, 96, &mut sampler, |t| {
            out_bytes.extend_from_slice(&tk.decode_bytes(t));
            true
        });
        let s = sess.stats;
        let tps = s.gen_tokens as f64 / s.gen_time.as_secs_f64().max(1e-9);
        let p_tps = s.prompt_tokens as f64 / s.prompt_time.as_secs_f64().max(1e-9);
        println!("\n--- {title} ({:.1} tok/s decode, {:.0} tok/s prefill) ---", tps, p_tps);
        println!("{}", String::from_utf8_lossy(&out_bytes));
    }
}

// ----------------------------------------------------------------------------- Prompt Rendering (exact parity with format.py)
fn sanitize_special(s: &str) -> String {
    let mut out = s.replace("\r\n", "\n");
    // Defuse literal Inkling 2 special tokens so user text cannot inject protocol markers
    for tok in [
        "<|unused", "<|endoftext|>", "<|begin_of_text|>", "<|end_message|>",
        "<|message_user|>", "<|message_model|>", "<|message_system|>", "<|message_tool|>",
        "<|content_text|>", "<|content_image|>", "<|content_model_end_sampling|>",
        "<|content_thinking|>", "<|content_tool_error|>", "<|content_xml|>",
        "<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>",
    ] {
        if out.contains(tok) {
            let safe = format!("<\u{200b}{}", &tok[1..]);
            out = out.replace(tok, &safe);
        }
    }
    out.trim().to_string()
}

fn one_line(s: &str, max_chars: usize) -> String {
    let collapsed: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    let clean = sanitize_special(&collapsed);
    clean.chars().take(max_chars).collect()
}

fn format_effort_str(raw: &str) -> (String, bool) {
    let v = match raw.trim().to_ascii_lowercase().as_str() {
        "" | "none" | "0" | "0.0" | "off" => 0.0,
        "minimal" => 0.1,
        "low" => 0.2,
        "medium" => 0.7,
        "high" => 0.9,
        "xhigh" | "max" => 0.99,
        x => x.parse::<f64>().unwrap_or(0.0).clamp(0.0, 0.99),
    };
    if v == 0.0 {
        ("0".into(), false)
    } else {
        (format!("{v}"), true)
    }
}

fn build_prompt_from_req(req: &JVal) -> (String, bool) {
    let kind = req.get("kind").and_then(|v| v.as_str()).unwrap_or("discord");
    let date = req.get("date").and_then(|v| v.as_str()).unwrap_or("");
    let date_s = if date.is_empty() { today_ymd() } else { date.to_string() };
    let title = one_line(req.get("title").and_then(|v| v.as_str()).unwrap_or(""), 300);
    let where_s = one_line(req.get("where").and_then(|v| v.as_str()).unwrap_or(""), 300);
    let custom_hdr = one_line(req.get("custom_header").and_then(|v| v.as_str()).unwrap_or(""), 300);

    let (header, joiner) = match kind {
        "website" => (
            format!("[eriskii.net · {date_s}]\n{}", if title.is_empty() { "Untitled" } else { &title }),
            "\n\n",
        ),
        "bsky_post" => (format!("[Bluesky · {date_s}]"), "\n"),
        "bsky_thread" => (format!("[Bluesky thread · {date_s}]"), "\n"),
        "ziz_essay" => (
            format!("[sinceriously.fyi]\n{}", if title.is_empty() { "Untitled" } else { &title }),
            "\n\n",
        ),
        "ziz_glossary" => ("[sinceriously.fyi · Glossary]".to_string(), "\n\n"),
        "ziz_faq" => ("[sinceriously.fyi · Cached Answers]".to_string(), "\n\n"),
        "ziz_comments" => (
            format!(
                "[sinceriously.fyi comments · {}]",
                if title.is_empty() { "Discussion" } else { &title }
            ),
            "\n\n",
        ),
        "custom" => (
            if custom_hdr.is_empty() { format!("[Chat · {date_s}]") } else { custom_hdr },
            "\n",
        ),
        _ => (
            format!(
                "[{} · {date_s}]",
                if where_s.is_empty() { "Discord DM with a friend" } else { &where_s }
            ),
            "\n",
        ),
    };

    let effort_raw = req.get("effort").and_then(|v| v.as_str()).unwrap_or("0");
    let (eff_s, thinking_enabled) = format_effort_str(effort_raw);

    let mut events: Vec<(String, String, bool)> = Vec::new();
    if let Some(arr) = req.get("events").and_then(|v| v.as_arr()) {
        for item in arr {
            let txt_raw = item.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let txt = sanitize_special(&txt_raw.replace('\0', ""));
            let txt: String = txt.chars().take(12000).collect();
            if txt.is_empty() {
                continue;
            }
            let mine = item.get("mine").and_then(|v| v.as_bool()).unwrap_or(false);
            let who = if mine {
                String::new()
            } else {
                one_line(item.get("who").and_then(|v| v.as_str()).unwrap_or(""), 120)
            };
            events.push((who, txt, mine));
        }
    }

    let last_is_mine = events.last().map(|e| e.2).unwrap_or(false);

    let mut out = String::new();
    out.push_str("<|message_system|><|content_text|>");
    out.push_str(&sanitize_special(&header));
    out.push_str("<|end_message|><|message_system|><|content_text|>Thinking effort level: ");
    out.push_str(&eff_s);
    out.push_str("<|end_message|>");

    let mut role: Option<&str> = None;
    for (who, txt, mine) in &events {
        let r = if *mine { "model" } else { "user" };
        let line = if *mine || who.is_empty() {
            txt.clone()
        } else {
            format!("{who}: {txt}")
        };
        if role == Some(r) {
            out.push_str(if *mine { joiner } else { "\n" });
            out.push_str(&line);
        } else {
            if role == Some("model") {
                out.push_str("<|end_message|><|content_model_end_sampling|>");
            } else if role == Some("user") {
                out.push_str("<|end_message|>");
            }
            if r == "model" {
                out.push_str("<|message_model|><|content_text|>");
            } else {
                out.push_str("<|message_user|><|content_text|>");
            }
            out.push_str(&line);
            role = Some(r);
        }
    }

    if last_is_mine {
        out.push_str(joiner);
        return (out, false);
    }

    if role == Some("model") {
        out.push_str("<|end_message|><|content_model_end_sampling|>");
    } else if role == Some("user") {
        out.push_str("<|end_message|>");
    }

    // At effort=0, prefill <|message_model|><|content_text|> for crisp direct persona voice.
    // At effort>0, leave at <|end_message|> so Inkling 2 natively routes into <|content_thinking|> -> <|content_text|>.
    if !thinking_enabled {
        out.push_str("<|message_model|><|content_text|>");
    }
    (out, thinking_enabled)
}

fn today_ymd() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as libc::time_t)
        .unwrap_or(0);
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    unsafe {
        libc::localtime_r(&secs, &mut tm);
    }
    format!("{:04}-{:02}-{:02}", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday)
}

// ----------------------------------------------------------------------------- HTTP Streaming Server (`persona-serve`)
struct PrefixCache {
    toks: Vec<u32>,
    blend_key: (u32, u32, u8),
}

fn decode_utf8_stream(tk: &Tokenizer, tok: u32, pending: &mut Vec<u8>) -> String {
    if tk.is_special(tok) {
        pending.clear();
        return String::from_utf8_lossy(&tk.decode_bytes(tok)).to_string();
    }
    pending.extend_from_slice(&tk.decode_bytes(tok));
    let mut out = String::new();
    loop {
        match std::str::from_utf8(pending) {
            Ok(s) => {
                out.push_str(s);
                pending.clear();
                break;
            }
            Err(e) => {
                let valid_len = e.valid_up_to();
                if valid_len > 0 {
                    out.push_str(std::str::from_utf8(&pending[..valid_len]).unwrap());
                    pending.drain(..valid_len);
                }
                if let Some(err_len) = e.error_len() {
                    out.push('\u{FFFD}');
                    pending.drain(..err_len);
                } else {
                    break;
                }
            }
        }
    }
    out
}

fn send_http(stream: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) {
    let hdr = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(hdr.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn check_local_headers(headers: &HashMap<String, String>, port: u16) -> bool {
    let h1 = format!("127.0.0.1:{port}");
    let h2 = format!("localhost:{port}");
    let host_ok = headers
        .get("host")
        .map(|h| h == &h1 || h == &h2)
        .unwrap_or(false);
    let origin_ok = match headers.get("origin") {
        None => true,
        Some(o) => o == &format!("http://{h1}") || o == &format!("http://{h2}"),
    };
    host_ok && origin_ok
}

pub fn run_persona_serve(
    model: &str,
    eris_path: &str,
    ziz_path: &str,
    webui_path: &str,
    contexts_path: &str,
    port: u16,
    ctx: usize,
) {
    let eng = Engine::load(model, ctx);
    let tk = Tokenizer::load(&eng.g);

    let ad0 = if !eris_path.is_empty() && Path::new(eris_path).exists() {
        PersonaAdapter::load(&eng, eris_path).expect("load eris .ilora")
    } else {
        PersonaAdapter::empty(&eng, "eris")
    };
    let ad1 = if !ziz_path.is_empty() && Path::new(ziz_path).exists() {
        PersonaAdapter::load(&eng, ziz_path).expect("load ziz .ilora")
    } else {
        PersonaAdapter::empty(&eng, "ziz")
    };

    let eris_name = ad0.name.clone();
    let ziz_name = ad1.name.clone();
    let eris_short = Path::new(eris_path)
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or(&eris_name)
        .to_string();
    let ziz_short = Path::new(ziz_path)
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|s| s.to_str())
        .unwrap_or(&ziz_name)
        .to_string();
    let model_name = Path::new(model)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(model)
        .to_string();

    eng.attach_persona(PersonaHook::new(&eng, ad0, ad1));

    let contexts_json = if !contexts_path.is_empty() && Path::new(contexts_path).exists() {
        std::fs::read_to_string(contexts_path).unwrap_or_else(|_| "[]".to_string())
    } else {
        "[]".to_string()
    };

    let listener = TcpListener::bind(("127.0.0.1", port)).unwrap_or_else(|e| {
        panic!("failed to bind http://127.0.0.1:{port}: {e}");
    });
    println!(
        "Inkling 2 Mini Dual-Persona Chat UI on http://127.0.0.1:{port}  (Eris: {eris_short} <-> Ziz: {ziz_short})",
    );

    let mut pcache = PrefixCache {
        toks: Vec::new(),
        blend_key: (u32::MAX, u32::MAX, 255),
    };
    let mut rng_counter = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1);

    for incoming in listener.incoming() {
        let Ok(mut stream) = incoming else { continue };
        let _ = stream.set_nodelay(true);
        let Ok(reader_stream) = stream.try_clone() else { continue };
        let mut br = BufReader::new(reader_stream);

        let mut req_line = String::new();
        if br.read_line(&mut req_line).unwrap_or(0) == 0 {
            continue;
        }
        let mut parts_it = req_line.split_whitespace();
        let method = parts_it.next().unwrap_or("").to_string();
        let path = parts_it.next().unwrap_or("/").split('?').next().unwrap_or("/").to_string();

        let mut headers = HashMap::new();
        let mut content_len = 0usize;
        loop {
            let mut line = String::new();
            if br.read_line(&mut line).unwrap_or(0) == 0 {
                break;
            }
            let tr = line.trim_end_matches(['\r', '\n']);
            if tr.is_empty() {
                break;
            }
            if let Some((k, v)) = tr.split_once(':') {
                let kl = k.trim().to_ascii_lowercase();
                let vl = v.trim().to_string();
                if kl == "content-length" {
                    content_len = vl.parse().unwrap_or(0);
                }
                headers.insert(kl, vl);
            }
        }

        if !check_local_headers(&headers, port) {
            send_http(&mut stream, "403 Forbidden", "text/plain", b"forbidden");
            continue;
        }

        let mut body = vec![0u8; content_len.min(4 * 1024 * 1024)];
        if !body.is_empty() && br.read_exact(&mut body).is_err() {
            continue;
        }

        match (method.as_str(), path.as_str()) {
            ("GET", "/") | ("GET", "/index.html") => {
                match std::fs::read(webui_path) {
                    Ok(bytes) => send_http(&mut stream, "200 OK", "text/html; charset=utf-8", &bytes),
                    Err(e) => send_http(
                        &mut stream,
                        "500 Internal Server Error",
                        "text/plain",
                        format!("could not read webui {webui_path}: {e}").as_bytes(),
                    ),
                }
            }
            ("GET", "/api/meta") => {
                let meta = format!(
                    "{{\"contexts\":{},\"eris\":{},\"ziz\":{},\"model\":{},\"format\":\"inkling2\",\"today\":{}}}",
                    contexts_json,
                    json_escape(&eris_short),
                    json_escape(&ziz_short),
                    json_escape(&model_name),
                    json_escape(&today_ymd()),
                );
                send_http(&mut stream, "200 OK", "application/json", meta.as_bytes());
            }
            ("POST", "/api/prompt") => {
                let Ok(s) = std::str::from_utf8(&body) else {
                    send_http(&mut stream, "400 Bad Request", "text/plain", b"invalid utf8");
                    continue;
                };
                let Ok(req) = parse_json(s) else {
                    send_http(&mut stream, "400 Bad Request", "text/plain", b"invalid json");
                    continue;
                };
                let (prompt, _) = build_prompt_from_req(&req);
                let n = tk.encode(&prompt, true).len();
                let resp = format!("{{\"prompt\":{},\"tokens\":{}}}", json_escape(&prompt), n);
                send_http(&mut stream, "200 OK", "application/json", resp.as_bytes());
            }
            ("POST", "/api/generate") => {
                let Ok(s) = std::str::from_utf8(&body) else {
                    send_http(&mut stream, "400 Bad Request", "text/plain", b"invalid utf8");
                    continue;
                };
                let Ok(req) = parse_json(s) else {
                    send_http(&mut stream, "400 Bad Request", "text/plain", b"invalid json");
                    continue;
                };

                let mix = req.get("mix").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                let strength = req.get("lora_strength").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                let parts = PersonaParts::from_str(req.get("parts").and_then(|v| v.as_str()).unwrap_or("all"));
                let temp = (req.get("temperature").and_then(|v| v.as_f64()).unwrap_or(0.8) as f32).clamp(0.0, 2.0);
                let top_p = (req.get("top_p").and_then(|v| v.as_f64()).unwrap_or(0.95) as f32).clamp(0.01, 1.0);
                let top_k = req.get("top_k").and_then(|v| v.as_u64()).unwrap_or(64).clamp(1, 512) as usize;
                let max_tok = req.get("max_tokens").and_then(|v| v.as_u64()).unwrap_or(250).clamp(1, 4096) as usize;

                let t_blend = Instant::now();
                eng.set_persona_blend(mix, strength, parts);
                let blend_ms = t_blend.elapsed().as_secs_f64() * 1e3;
                let bkey = eng.persona_blend_key();

                let (prompt, _thinking) = build_prompt_from_req(&req);
                let toks = tk.encode(&prompt, true);
                if toks.is_empty() || toks.len() + 8 >= eng.max_ctx {
                    send_http(&mut stream, "400 Bad Request", "text/plain", b"prompt too long for context");
                    continue;
                }

                // Stream raw UTF-8 response immediately
                let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=utf-8\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n";
                if stream.write_all(hdr.as_bytes()).is_err() {
                    continue;
                }
                let _ = stream.flush();

                // Reuse prefix KV cache only when blend is unchanged and new prompt strictly extends cached tokens
                let reuse_len = if bkey == pcache.blend_key
                    && !pcache.toks.is_empty()
                    && toks.len() > pcache.toks.len()
                    && toks.starts_with(&pcache.toks)
                {
                    pcache.toks.len()
                } else {
                    0
                };

                let mut sess = if reuse_len > 0 {
                    Session::resume(&eng, &tk, reuse_len)
                } else {
                    Session::new(&eng, &tk)
                };

                let t_pf = Instant::now();
                let first = sess.prefill(&toks[reuse_len..]);
                let pf_ms = t_pf.elapsed().as_secs_f64() * 1e3;
                let max_new = max_tok.min(sess.ctx_left().saturating_sub(4));

                rng_counter = rng_counter.wrapping_add(1);
                let mut sampler = Sampler::new(temp, top_k, top_p, eng.softcap, rng_counter);
                let mut utf8_pending = Vec::new();
                let mut generated: Vec<u32> = Vec::new();
                let mut broken = false;

                sess.generate(first, max_new, &mut sampler, |t| {
                    generated.push(t);
                    let piece = decode_utf8_stream(&tk, t, &mut utf8_pending);
                    if !piece.is_empty() {
                        if stream.write_all(piece.as_bytes()).is_err() || stream.flush().is_err() {
                            broken = true;
                            return false;
                        }
                    }
                    true
                });

                // Update prefix cache to exact committed KV positions (sess.n_past)
                let mut full_seq = toks;
                full_seq.extend_from_slice(&generated);
                if sess.n_past <= full_seq.len() {
                    full_seq.truncate(sess.n_past);
                    pcache.toks = full_seq;
                    pcache.blend_key = bkey;
                } else {
                    pcache.toks.clear();
                }

                if !broken {
                    let s = sess.stats;
                    let tps = s.gen_tokens as f64 / s.gen_time.as_secs_f64().max(1e-9);
                    let meta_tail = format!(
                        "\x1e{{\"gen_tokens\":{},\"prompt_tokens\":{},\"reused_tokens\":{},\"tps\":{:.1},\"prefill_ms\":{:.0},\"blend_ms\":{:.1}}}",
                        s.gen_tokens, s.prompt_tokens, reuse_len, tps, pf_ms, blend_ms
                    );
                    let _ = stream.write_all(meta_tail.as_bytes());
                    let _ = stream.flush();
                }
            }
            _ => {
                send_http(&mut stream, "404 Not Found", "text/plain", b"not found");
            }
        }
    }
}
