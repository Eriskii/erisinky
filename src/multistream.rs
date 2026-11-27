//! 7-channel Multi-Stream inference engine (arXiv:2605.12460 + Option B sparse optimization)
//! with pass-specific LoRA, channel embeddings, silence/tool token adapters, column-mode short conv,
//! and shared non-silent KV cache over Inkling 2 Mini's zero-copy mmapped GGUF weights.

use crate::engine::{
    slot_of, CtrlBArgs, CtrlBJob, Engine, GemmHArgs, HSeg, Layer, LoopW, MixJob, MoeGemmHArgs,
    ToHalfArgs, TopkBArgs, VecAlloc, BLOCK, B_NVEC, E, EXT, FF, HD, LOOP_END, LOOP_START, NCH,
    NEXP, NH, NKV, NREL, N_PASS, N_PHYS, N_SLOT, TOPK,
};
use crate::metal::{Buf, Enc};
use objc2_metal::{MTLBuffer, MTLCommandBuffer};
use std::collections::HashMap;
use std::time::Instant;

pub const MS_CHANNELS: usize = 7;
pub const MS_MAX_M: usize = 8; // at most 3 inputs + 4 outputs = 7 active tokens per tick
pub const CH_USER_IN: usize = 0;
pub const CH_AGENT_IN: usize = 1;
pub const CH_TOOL_IN: usize = 2;
#[allow(dead_code)]
pub const CH_THINKING: usize = 3;
#[allow(dead_code)]
pub const CH_USER_OUT: usize = 4;
#[allow(dead_code)]
pub const CH_AGENT_OUT: usize = 5;
#[allow(dead_code)]
pub const CH_TOOL_CALLS: usize = 6;

#[allow(dead_code)]
pub const CHANNEL_NAMES: [&str; MS_CHANNELS] = [
    "user_input",
    "agent_msg_input",
    "tool_input",
    "thinking",
    "user_output",
    "agent_msg_output",
    "tool_calls",
];

const NTC_B: usize = 4;

// ----------------------------------------------------------------------------- Metal arg structs
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Route4Args {
    sel_off: u32,
    t: u32,
    perm_off: u32,
    inv_off: u32,
    aw_off: u32,
    ind_off: u32,
    gu_y: [u32; NTC_B],
    dn_y: [u32; NTC_B],
    tiles_off: [u32; NTC_B],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MoeCombPostArgs {
    yd_off: u32,
    inv_off: u32,
    res_off: u32,
    out_off: u32,
    layer_slot: u32,
    max_batch: u32,
    convw_t_off: u32,
    seq_cs_off: u32,
    seq_pos_off: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct LmHeadArgs {
    w_lo: u32,
    w_hi: u32,
    row_bytes: u32,
    wtype: u32,
    n_vocab: u32,
    k: u32,
    xh_off: u32,
    part_off: u32,
    b: u32,
    nvch: u32,
    vstart: u32,
    vcount: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MsEmbedArgs {
    emb_lo: u32,
    emb_hi: u32,
    row_bytes: u32,
    norm_off: u32,
    out_off: u32,
    silence_id: u32,
    tool_id: u32,
    peer_id: u32,
    ch_emb_off: u32,
    sil_emb_off: u32,
    tool_emb_off: u32,
    peer_emb_off: u32,
    eps: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MsLoraSeg {
    a_off: u32,
    b_off: u32,
    x_off: u32,
    y_off: u32,
    in_dim: u32,
    out_dim: u32,
    rank: u32,
    tmp_off: u32,
    in_half: u32,
    out_half: u32,
    pad0: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MsLoraArgs {
    nseg: u32,
    m: u32,
    max_rank: u32,
    max_out_blocks: u32,
    seg: [MsLoraSeg; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MsCtrlGfJob {
    gain_w: u32,
    forget_w: u32,
    gain_b_off: u32,
    forget_b_off: u32,
    gh_off: u32,
    tmp_gain_off: u32,
    tmp_forget_off: u32,
    u_off: u32,
    m_off: u32,
    raw_off: u32,
    rank: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MsCtrlGfArgs {
    m: u32,
    njobs: u32,
    pad0: u32,
    pad1: u32,
    job: [MsCtrlGfJob; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MsAttnArgs {
    m: u32,
    kv_cap: u32,
    slot_idx: u32,
    is_swa: u32,
    cur_tick: u32,
    n_kv: u32,
    j_start: u32,
    n_vis_past: u32,
    layer_slot: u32,
    max_channels: u32,
    seq_cs_off: u32,
    seq_pos_off: u32,
    kv_write_off: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    r_off: u32,
    sink_off: u32,
    kdyn_off: i32,
    vdyn_off: i32,
    kconv_t_w: u32,
    vconv_t_w: u32,
    qclip_w: u32,
    relclip_w: u32,
    relbias_w: u32,
    log_alpha: f32,
    log_floor: f32,
    gate_off: u32,
    out_off: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MsWoPostArgs {
    wo_y_off: u32,
    res_off: u32,
    out_off: u32,
    has_res: u32,
    layer_slot: u32,
    max_channels: u32,
    convw_t_off: u32,
    seq_cs_off: u32,
    seq_pos_off: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct MsOutPrepArgs {
    fin_off: u32,
    out_f32_off: u32,
    out_half_off: u32,
    out_ch_emb_off: u32,
    m_start: u32,
    n_out: u32,
    head_rank: u32,
    pad0: u32,
    head_w1_off: u32,
    head_b1_off: u32,
    head_w2_off: u32,
    pad1: u32,
}

// ----------------------------------------------------------------------------- Minimal Zero-Dependency JSON Parser
#[derive(Debug, Clone)]
pub enum JVal {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<JVal>),
    Obj(Vec<(String, JVal)>),
}

impl JVal {
    pub fn get(&self, key: &str) -> Option<&JVal> {
        if let JVal::Obj(pairs) = self {
            for (k, v) in pairs {
                if k == key {
                    return Some(v);
                }
            }
        }
        None
    }
    pub fn as_str(&self) -> Option<&str> {
        if let JVal::Str(s) = self { Some(s) } else { None }
    }
    pub fn as_f64(&self) -> Option<f64> {
        if let JVal::Num(n) = self { Some(*n) } else { None }
    }
    pub fn as_u64(&self) -> Option<u64> {
        self.as_f64().map(|n| n as u64)
    }
    pub fn as_bool(&self) -> Option<bool> {
        if let JVal::Bool(b) = self { Some(*b) } else { None }
    }
    pub fn as_arr(&self) -> Option<&[JVal]> {
        if let JVal::Arr(v) = self { Some(v) } else { None }
    }
}

pub fn parse_json(s: &str) -> Result<JVal, String> {
    let b = s.as_bytes();
    let mut i = 0usize;
    let v = parse_jval(b, &mut i)?;
    Ok(v)
}

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && matches!(b[*i], b' ' | b'\t' | b'\r' | b'\n') {
        *i += 1;
    }
}

fn parse_jstr(b: &[u8], i: &mut usize) -> Result<String, String> {
    if *i >= b.len() || b[*i] != b'"' {
        return Err(format!("expected '\"' at {}", *i));
    }
    *i += 1;
    let mut out = String::new();
    while *i < b.len() {
        let c = b[*i];
        *i += 1;
        if c == b'"' {
            return Ok(out);
        }
        if c == b'\\' {
            if *i >= b.len() {
                return Err("unterminated escape".into());
            }
            let e = b[*i];
            *i += 1;
            match e {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'b' => out.push('\x08'),
                b'f' => out.push('\x0c'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'u' => {
                    if *i + 4 > b.len() {
                        return Err("short \\u escape".into());
                    }
                    let hex = std::str::from_utf8(&b[*i..*i + 4]).map_err(|e| e.to_string())?;
                    let code = u16::from_str_radix(hex, 16).map_err(|e| e.to_string())?;
                    *i += 4;
                    if (0xD800..=0xDBFF).contains(&code)
                        && *i + 6 <= b.len()
                        && b[*i] == b'\\'
                        && b[*i + 1] == b'u'
                    {
                        let hex2 = std::str::from_utf8(&b[*i + 2..*i + 6]).map_err(|e| e.to_string())?;
                        if let Ok(lo) = u16::from_str_radix(hex2, 16) {
                            if (0xDC00..=0xDFFF).contains(&lo) {
                                *i += 6;
                                let cp = 0x10000 + (((code as u32 - 0xD800) << 10) | (lo as u32 - 0xDC00));
                                if let Some(ch) = char::from_u32(cp) {
                                    out.push(ch);
                                    continue;
                                }
                            }
                        }
                    }
                    if let Some(ch) = char::from_u32(code as u32) {
                        out.push(ch);
                    }
                }
                _ => out.push(e as char),
            }
        } else {
            // Decode UTF-8 multi-byte characters cleanly
            let start = *i - 1;
            if c < 0x80 {
                out.push(c as char);
            } else {
                let len = if c & 0xE0 == 0xC0 {
                    2
                } else if c & 0xF0 == 0xE0 {
                    3
                } else {
                    4
                };
                let end = (start + len).min(b.len());
                *i = end;
                if let Ok(seg) = std::str::from_utf8(&b[start..end]) {
                    out.push_str(seg);
                }
            }
        }
    }
    Err("unterminated string".into())
}

fn parse_jval(b: &[u8], i: &mut usize) -> Result<JVal, String> {
    skip_ws(b, i);
    if *i >= b.len() {
        return Err("unexpected EOF".into());
    }
    match b[*i] {
        b'"' => Ok(JVal::Str(parse_jstr(b, i)?)),
        b'{' => {
            *i += 1;
            let mut pairs = Vec::new();
            skip_ws(b, i);
            if *i < b.len() && b[*i] == b'}' {
                *i += 1;
                return Ok(JVal::Obj(pairs));
            }
            loop {
                skip_ws(b, i);
                let k = parse_jstr(b, i)?;
                skip_ws(b, i);
                if *i >= b.len() || b[*i] != b':' {
                    return Err(format!("expected ':' at {}", *i));
                }
                *i += 1;
                let v = parse_jval(b, i)?;
                pairs.push((k, v));
                skip_ws(b, i);
                if *i < b.len() && b[*i] == b',' {
                    *i += 1;
                    continue;
                }
                if *i < b.len() && b[*i] == b'}' {
                    *i += 1;
                    return Ok(JVal::Obj(pairs));
                }
                return Err(format!("expected ',' or '}}' at {}", *i));
            }
        }
        b'[' => {
            *i += 1;
            let mut arr = Vec::new();
            skip_ws(b, i);
            if *i < b.len() && b[*i] == b']' {
                *i += 1;
                return Ok(JVal::Arr(arr));
            }
            loop {
                arr.push(parse_jval(b, i)?);
                skip_ws(b, i);
                if *i < b.len() && b[*i] == b',' {
                    *i += 1;
                    continue;
                }
                if *i < b.len() && b[*i] == b']' {
                    *i += 1;
                    return Ok(JVal::Arr(arr));
                }
                return Err(format!("expected ',' or ']' at {}", *i));
            }
        }
        b't' => {
            if b[*i..].starts_with(b"true") {
                *i += 4;
                Ok(JVal::Bool(true))
            } else {
                Err("invalid literal".into())
            }
        }
        b'f' => {
            if b[*i..].starts_with(b"false") {
                *i += 5;
                Ok(JVal::Bool(false))
            } else {
                Err("invalid literal".into())
            }
        }
        b'n' => {
            if b[*i..].starts_with(b"null") {
                *i += 4;
                Ok(JVal::Null)
            } else {
                Err("invalid literal".into())
            }
        }
        _ => {
            let start = *i;
            while *i < b.len()
                && matches!(b[*i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
            {
                *i += 1;
            }
            let s = std::str::from_utf8(&b[start..*i]).map_err(|e| e.to_string())?;
            let n: f64 = s.parse().map_err(|_| format!("bad number '{s}'"))?;
            Ok(JVal::Num(n))
        }
    }
}

pub fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ----------------------------------------------------------------------------- .imsa Adapter Representation
#[derive(Clone, Copy, Default)]
struct LoraPair {
    a_off: u32, // byte offset in ad_buf
    b_off: u32, // byte offset in ad_buf
    num_passes: u32,
    rank: u32,
    in_dim: u32,
    out_dim: u32,
}

impl LoraPair {
    #[inline]
    fn present(&self) -> bool {
        self.rank > 0
    }
    #[inline]
    fn pass_offsets(&self, pass: usize) -> (u32, u32) {
        let p = if self.num_passes > 1 { pass as u32 } else { 0 };
        let a = self.a_off + p * self.rank * self.in_dim * 2;
        let b = self.b_off + p * self.out_dim * self.rank * 2;
        (a, b)
    }
}

#[derive(Clone, Copy, Default)]
struct CtrlLora {
    down: LoraPair,
    gain: LoraPair,
    forget: LoraPair,
}

#[derive(Clone, Copy, Default)]
struct LayerLora {
    wq: LoraPair,
    wk: LoraPair,
    wv: LoraPair,
    wo: LoraPair,
    gate: LoraPair,
    wr: LoraPair,
    sink: LoraPair,
    ctrl: [CtrlLora; 4], // q, k, v, mlp
}

#[allow(dead_code)]
pub struct MsAdapter {
    pub name: String,
    pub path: String,
    pub silence_id: u32,
    pub tool_id: u32,
    pub peer_id: u32,
    pub eom_id: u32,
    pub lora_alpha: f32,
    pub lora_rank: u32,
    pub has_tool_adapter: bool,
    pub has_peer_adapter: bool,
    pub out_head_rank: u32,
    pub native_framing: bool,
    pub preamble: String,
    pub ad_buf: Buf,
    pub pb_buf: Buf,
    ch_emb_off: u32,
    out_ch_emb_off: u32,
    out_head_w1_off: u32,
    out_head_b1_off: u32,
    out_head_w2_off: u32,
    sil_emb_off: u32,
    tool_emb_off: u32,
    peer_emb_off: u32,
    sil_unembed_delta: Vec<f32>,
    sil_unembed_bias: f32,
    tool_unembed_delta: Vec<f32>,
    tool_unembed_bias: f32,
    peer_unembed_delta: Vec<f32>,
    peer_unembed_bias: f32,
    layers: Vec<LayerLora>,
}

struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    offset: usize,
    nbytes: usize,
}

impl MsAdapter {
    pub fn load(eng: &Engine, path: &str) -> Result<MsAdapter, String> {
        let t0 = Instant::now();
        let raw = std::fs::read(path).map_err(|e| format!("read adapter {path}: {e}"))?;
        if raw.len() < 12 || &raw[0..8] != b"IMSA0001" {
            return Err(format!("invalid .imsa magic in {path}"));
        }
        let hlen = u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize;
        let data_start = 12 + hlen;
        if data_start > raw.len() {
            return Err(format!("truncated .imsa header in {path}"));
        }
        let hdr_str = std::str::from_utf8(&raw[12..data_start]).map_err(|e| e.to_string())?;
        let hdr = parse_json(hdr_str)?;

        let silence_id = hdr.get("silence_token_id").and_then(|v| v.as_u64()).unwrap_or(200007) as u32;
        let tool_id = hdr.get("tool_token_id").and_then(|v| v.as_u64()).unwrap_or(200049) as u32;
        let peer_id = hdr.get("peer_token_id").and_then(|v| v.as_u64()).unwrap_or(200009) as u32;
        let eom_id = hdr.get("end_message_token_id").and_then(|v| v.as_u64()).unwrap_or(200010) as u32;
        let lora_rank = hdr.get("lora_rank").and_then(|v| v.as_u64()).unwrap_or(32) as u32;
        let lora_alpha = hdr.get("lora_alpha").and_then(|v| v.as_f64()).unwrap_or(64.0) as f32;
        let has_tool_adapter = hdr.get("has_tool_adapter").and_then(|v| v.as_bool()).unwrap_or(false);
        let has_peer_adapter = hdr.get("has_peer_adapter").and_then(|v| v.as_bool()).unwrap_or(false);
        let native_framing = hdr.get("native_framing").and_then(|v| v.as_bool()).unwrap_or(false);
        let preamble = hdr.get("preamble").and_then(|v| v.as_str()).unwrap_or("").to_string();

        let mut tmap: HashMap<String, TensorInfo> = HashMap::new();
        if let Some(arr) = hdr.get("tensors").and_then(|v| v.as_arr()) {
            for item in arr {
                let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let dtype = item.get("dtype").and_then(|v| v.as_str()).unwrap_or("f32").to_string();
                let shape: Vec<usize> = item
                    .get("shape")
                    .and_then(|v| v.as_arr())
                    .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect())
                    .unwrap_or_default();
                let offset = item.get("offset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let nbytes = item.get("nbytes").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                tmap.insert(name, TensorInfo { dtype, shape, offset, nbytes });
            }
        }

        let payload = &raw[data_start..];
        let ad_buf = eng.gpu.buffer_with(payload);

        // Clone base Engine.pb into adapter's own mutable pb_buf and patch updated structural fp32 tensors
        let pb_len = eng.pb.length();
        let pb_buf = eng.gpu.buffer(pb_len);
        unsafe {
            std::ptr::copy_nonoverlapping(
                eng.pb.contents().as_ptr() as *const u8,
                pb_buf.contents().as_ptr() as *mut u8,
                pb_len,
            );
        }
        let pb_ptr = pb_buf.contents().as_ptr() as *mut u8;

        let slice_of = |name: &str| -> Option<&[u8]> {
            let ti = tmap.get(name)?;
            Some(&payload[ti.offset..ti.offset + ti.nbytes])
        };

        let patch_f32 = |dst_off: u32, bytes: &[u8]| unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), pb_ptr.add(dst_off as usize), bytes.len());
        };
        let patch_conv_soa = |dst_raw_off: u32, dst_t_off: u32, bytes: &[u8]| unsafe {
            // Raw [dim, 4] f32
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), pb_ptr.add(dst_raw_off as usize), bytes.len());
            // Transposed [4, dim] SoA f32
            let dim = (bytes.len() / 4) / 4;
            let dst = pb_ptr.add(dst_t_off as usize);
            for c in 0..dim {
                for k in 0..4 {
                    let s = (c * 4 + k) * 4;
                    let d = (k * dim + c) * 4;
                    std::ptr::copy_nonoverlapping(bytes.as_ptr().add(s), dst.add(d), 4);
                }
            }
        };

        // 1. Patch per-layer short conv weights + attn_res queries
        for i in 0..N_PHYS {
            let l = &eng.layers[i];
            if let Some(b) = slice_of(&format!("layers.{i}.k_sconv.weight")) {
                patch_conv_soa(l.k_conv, l.k_conv_t, b);
            }
            if let Some(b) = slice_of(&format!("layers.{i}.v_sconv.weight")) {
                patch_conv_soa(l.v_conv, l.v_conv_t, b);
            }
            if let Some(b) = slice_of(&format!("layers.{i}.attn_sconv.weight")) {
                patch_conv_soa(l.attn_post_conv, l.attn_post_conv_t, b);
            }
            if let Some(b) = slice_of(&format!("layers.{i}.mlp_sconv.weight")) {
                patch_conv_soa(l.ffn_post_conv, l.ffn_post_conv_t, b);
            }
            if let Some(b) = slice_of(&format!("attn_res_mlp_queries.{i}")) {
                patch_f32(l.ffn_res_query, b);
            }
            if i > 0 && l.attn_res_query >= 0 {
                if let Some(b) = slice_of(&format!("attn_res_attn_queries.{}", i - 1)) {
                    patch_f32(l.attn_res_query as u32, b);
                }
            }
        }
        if let Some(b) = slice_of("attn_res_output_query") {
            patch_f32(eng.output_res_query, b);
        }
        // 2. Patch looped core cross-pass message queries: [q[2*i], kv[2*i], kv[2*i+1], q[2*i+1]]
        if let (Some(q_bytes), Some(kv_bytes)) = (
            slice_of("middle_cycle_cross_pass_queries"),
            slice_of("middle_cycle_cross_pass_kv_queries"),
        ) {
            let row_b = E * 4;
            for i in 0..(LOOP_END - LOOP_START) {
                let dst = eng.loops[i].msg_query;
                unsafe {
                    std::ptr::copy_nonoverlapping(q_bytes.as_ptr().add((2 * i) * row_b), pb_ptr.add(dst as usize), row_b);
                    std::ptr::copy_nonoverlapping(kv_bytes.as_ptr().add((2 * i) * row_b), pb_ptr.add(dst as usize + row_b), row_b);
                    std::ptr::copy_nonoverlapping(kv_bytes.as_ptr().add((2 * i + 1) * row_b), pb_ptr.add(dst as usize + 2 * row_b), row_b);
                    std::ptr::copy_nonoverlapping(q_bytes.as_ptr().add((2 * i + 1) * row_b), pb_ptr.add(dst as usize + 3 * row_b), row_b);
                }
            }
        }

        // 3. Collect LoRA pairs
        let get_lora = |prefix: &str| -> LoraPair {
            let (Some(ta), Some(tb)) = (tmap.get(&format!("{prefix}.lora_a")), tmap.get(&format!("{prefix}.lora_b"))) else {
                return LoraPair::default();
            };
            assert_eq!(ta.dtype, "f16", "{prefix}.lora_a must be f16");
            assert_eq!(tb.dtype, "f16", "{prefix}.lora_b must be f16");
            assert_eq!(ta.shape.len(), 3);
            assert_eq!(tb.shape.len(), 3);
            LoraPair {
                a_off: ta.offset as u32,
                b_off: tb.offset as u32,
                num_passes: ta.shape[0] as u32,
                rank: ta.shape[1] as u32,
                in_dim: ta.shape[2] as u32,
                out_dim: tb.shape[1] as u32,
            }
        };

        let mut layers = Vec::with_capacity(N_PHYS);
        for i in 0..N_PHYS {
            let p = format!("layers.{i}");
            let mut ll = LayerLora {
                wq: get_lora(&format!("{p}.wq")),
                wk: get_lora(&format!("{p}.wk")),
                wv: get_lora(&format!("{p}.wv")),
                wo: get_lora(&format!("{p}.wo")),
                gate: get_lora(&format!("{p}.gate_attn")),
                wr: get_lora(&format!("{p}.wr")),
                sink: get_lora(&format!("{p}.attn_sink_proj")),
                ctrl: [CtrlLora::default(); 4],
            };
            if (LOOP_START..LOOP_END).contains(&i) {
                let cnames = ["ctrl_q", "ctrl_k", "ctrl_v", "ctrl_mlp"];
                for (c, cname) in cnames.iter().enumerate() {
                    ll.ctrl[c] = CtrlLora {
                        down: get_lora(&format!("{p}.{cname}.down")),
                        gain: get_lora(&format!("{p}.{cname}.gain")),
                        forget: get_lora(&format!("{p}.{cname}.forget")),
                    };
                }
            }
            layers.push(ll);
        }

        let f32_vec = |name: &str| -> Vec<f32> {
            match slice_of(name) {
                Some(b) => b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
                None => Vec::new(),
            }
        };

        let ch_emb_off = tmap.get("channel_embedding.weight").map(|t| t.offset as u32).unwrap_or(u32::MAX);
        let out_ch_emb_off = tmap.get("output_channel_embedding.weight").map(|t| t.offset as u32).unwrap_or(u32::MAX);
        let out_head_w1_off = tmap.get("output_channel_head.w1").map(|t| t.offset as u32).unwrap_or(u32::MAX);
        let out_head_b1_off = tmap.get("output_channel_head.b1").map(|t| t.offset as u32).unwrap_or(u32::MAX);
        let out_head_w2_off = tmap.get("output_channel_head.w2").map(|t| t.offset as u32).unwrap_or(u32::MAX);
        let out_head_rank = tmap
            .get("output_channel_head.w1")
            .and_then(|t| t.shape.get(1).copied())
            .unwrap_or(0) as u32;
        let sil_emb_off = tmap.get("silence_adapter.embed_delta").map(|t| t.offset as u32).unwrap_or(u32::MAX);
        let tool_emb_off = tmap.get("tool_adapter.embed_delta").map(|t| t.offset as u32).unwrap_or(u32::MAX);
        let peer_emb_off = tmap.get("peer_adapter.embed_delta").map(|t| t.offset as u32).unwrap_or(u32::MAX);

        let sil_unembed_delta = f32_vec("silence_adapter.unembed_delta");
        let sil_unembed_bias = f32_vec("silence_adapter.unembed_bias").first().copied().unwrap_or(0.0);
        let tool_unembed_delta = f32_vec("tool_adapter.unembed_delta");
        let tool_unembed_bias = f32_vec("tool_adapter.unembed_bias").first().copied().unwrap_or(0.0);
        let peer_unembed_delta = f32_vec("peer_adapter.unembed_delta");
        let peer_unembed_bias = f32_vec("peer_adapter.unembed_bias").first().copied().unwrap_or(0.0);

        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("adapter")
            .to_string();

        eprintln!(
            "[multistream] loaded adapter {} ({:.1} MiB, r={}, alpha={}, tool={}, peer={}, out_head_r={}, native_framing={}) in {:.2?}",
            name,
            payload.len() as f64 / (1 << 20) as f64,
            lora_rank,
            lora_alpha,
            !tool_unembed_delta.is_empty(),
            !peer_unembed_delta.is_empty(),
            out_head_rank,
            native_framing,
            t0.elapsed()
        );

        Ok(MsAdapter {
            name,
            path: path.to_string(),
            silence_id,
            tool_id,
            peer_id,
            eom_id,
            lora_alpha,
            lora_rank,
            has_tool_adapter: has_tool_adapter || !tool_unembed_delta.is_empty(),
            has_peer_adapter: has_peer_adapter || !peer_unembed_delta.is_empty(),
            out_head_rank,
            native_framing,
            preamble,
            ad_buf,
            pb_buf,
            ch_emb_off,
            out_ch_emb_off,
            out_head_w1_off,
            out_head_b1_off,
            out_head_w2_off,
            sil_emb_off,
            tool_emb_off,
            peer_emb_off,
            sil_unembed_delta,
            sil_unembed_bias,
            tool_unembed_delta,
            tool_unembed_bias,
            peer_unembed_delta,
            peer_unembed_bias,
            layers,
        })
    }
}

// ----------------------------------------------------------------------------- Compact Multi-Stream Arena (~8 MiB!)
#[derive(Clone, Copy, Default)]
struct MsArena {
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
    kdyn: u32,
    vdyn: u32,
    agh: u32,
    wo_y: u32,
    lora_tmp: u32, // 8 segments * MS_MAX_M * 64 floats
    fin: u32,
    fh: u32,
    rlog: u32,
    sel: u32,
    perm: u32,
    inv: u32,
    aw: u32,
    tl: [u32; NTC_B],
    ind: u32,
    acth: u32,
    xgh: u32,
    yd: u32,
    out_f32: u32,  // 4 * E f32
    out_half: u32, // 4 * E / 2 f32 words
    logits: u32,   // 4 * n_vocab f32 (~3.2 MiB!)
    total: u32,
}

impl MsArena {
    fn new(n_vocab: usize) -> MsArena {
        let mut o = 0u32;
        let take = |o: &mut u32, n: usize| -> u32 {
            let r = *o;
            *o += (n.div_ceil(64) * 64) as u32;
            r
        };
        let cap = MS_MAX_M;
        let mut a = MsArena::default();
        a.vec0 = take(&mut o, B_NVEC * cap * E);
        for i in 0..4 {
            a.msg[i] = take(&mut o, cap * E);
        }
        for i in 0..4 {
            a.cath[i] = take(&mut o, cap * E); // 2*E halfs = E f32 words
        }
        a.lora_tmp = take(&mut o, 8 * cap * 64);

        // Attention scratch
        for i in 0..3 {
            a.inp[i] = take(&mut o, cap * E);
        }
        for i in 0..4 {
            a.gain_o[i] = take(&mut o, cap * E);
        }
        for i in 0..4 {
            a.gh[i] = take(&mut o, cap * 32); // 64 halfs = 32 f32 words
        }
        for i in 0..3 {
            a.xh[i] = take(&mut o, cap * E / 2);
        }
        a.q = take(&mut o, cap * NH * HD / 2);
        a.gate = take(&mut o, cap * NH * HD / 2);
        a.k = take(&mut o, cap * NKV * HD);
        a.v = take(&mut o, cap * NKV * HD);
        a.rel = take(&mut o, cap * NH * NREL);
        a.sink = take(&mut o, cap * NH);
        a.kdyn = take(&mut o, cap * 32);
        a.vdyn = take(&mut o, cap * 32);
        a.agh = take(&mut o, cap * NH * HD / 2);
        a.wo_y = take(&mut o, cap * E);

        // FFN scratch
        a.fin = take(&mut o, cap * E);
        a.fh = take(&mut o, cap * E / 2);
        a.rlog = take(&mut o, cap * NEXP);
        a.sel = take(&mut o, cap * 16);
        a.perm = take(&mut o, cap * TOPK);
        a.inv = take(&mut o, cap * TOPK);
        a.aw = take(&mut o, cap * TOPK);
        a.tl = [
            take(&mut o, (cap * TOPK / 128 + NEXP + 1) * 4),
            take(&mut o, (NEXP + 1) * 4),
            take(&mut o, (NEXP + 1) * 4),
            take(&mut o, (NEXP + 1) * 4),
        ];
        a.ind = take(&mut o, 64);
        a.acth = take(&mut o, cap * TOPK * FF / 2);
        a.xgh = take(&mut o, cap * TOPK * E / 2);
        a.yd = a.xgh;

        // Output head (4 output channels)
        a.out_f32 = take(&mut o, 4 * E);
        a.out_half = take(&mut o, 4 * E / 2);
        a.logits = take(&mut o, 4 * n_vocab);
        a.total = o;
        a
    }

    #[inline]
    fn vec(&self, i: usize) -> u32 {
        self.vec0 + (i * MS_MAX_M * E) as u32
    }
}

// ----------------------------------------------------------------------------- Multi-Stream Sampler
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct MsSamplerCfg {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub silence_penalty: f32,
    pub seed: u64,
    rng_state: u64,
}

impl MsSamplerCfg {
    pub fn new(temp: f32, top_k: usize, top_p: f32, silence_penalty: f32, seed: u64) -> Self {
        Self {
            temp,
            top_k: top_k.max(1),
            top_p: top_p.clamp(0.01, 1.0),
            silence_penalty,
            seed,
            rng_state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_f32(&mut self) -> f32 {
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Sample one token from pre-softcap raw logits (after silence/tool deltas are added).
    /// Returns (token_id, silence_prob_estimate).
    fn sample_one(
        &mut self,
        logits: &mut [f32],
        softcap: f32,
        silence_id: u32,
    ) -> (u32, f32) {
        let sil_idx = silence_id as usize;
        let apply_cap = |x: f32| if softcap > 0.0 { softcap * (x / softcap).tanh() } else { x };

        if self.temp <= 0.0 {
            let mut best_i = 0u32;
            let mut best_v = f32::NEG_INFINITY;
            let mut best_non_sil_v = f32::NEG_INFINITY;
            let mut sil_v = f32::NEG_INFINITY;
            for (i, &raw) in logits.iter().enumerate() {
                let mut v = apply_cap(raw);
                if i == sil_idx {
                    v -= self.silence_penalty;
                    sil_v = v;
                } else if v > best_non_sil_v {
                    best_non_sil_v = v;
                }
                if v > best_v {
                    best_v = v;
                    best_i = i as u32;
                }
            }
            let diff = (best_non_sil_v - sil_v).clamp(-30.0, 30.0);
            let sil_p = 1.0 / (1.0 + diff.exp());
            return (best_i, sil_p);
        }

        // Top-k over capped logits (with silence_penalty applied to silence_id)
        let k = self.top_k.min(logits.len()).min(128);
        let mut cand: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
        let mut floor = f32::NEG_INFINITY;
        let mut sil_val = f32::NEG_INFINITY;
        for (i, &raw) in logits.iter().enumerate() {
            let mut v = apply_cap(raw);
            if i == sil_idx {
                v -= self.silence_penalty;
                sil_val = v;
            }
            if v <= floor && cand.len() == k {
                continue;
            }
            if cand.len() < k {
                cand.push((v, i as u32));
                if cand.len() == k {
                    floor = cand.iter().map(|c| c.0).fold(f32::INFINITY, f32::min);
                }
            } else {
                let j = cand.iter().position(|c| c.0 == floor).unwrap();
                cand[j] = (v, i as u32);
                floor = cand.iter().map(|c| c.0).fold(f32::INFINITY, f32::min);
            }
        }
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let t = self.temp.max(1e-5);
        let m = cand[0].0 / t;
        let mut probs: Vec<f32> = cand.iter().map(|&(v, _)| (v / t - m).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= sum;
        }
        let sil_p = cand
            .iter()
            .zip(probs.iter())
            .find(|(c, _)| c.1 == silence_id)
            .map(|(_, &p)| p)
            .unwrap_or_else(|| ((sil_val / t - m).exp() / sum).clamp(0.0, 1.0));

        // Top-p truncation
        let mut cum = 0.0f32;
        let mut n = probs.len();
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum >= self.top_p {
                n = i + 1;
                break;
            }
        }
        let tot: f32 = probs[..n].iter().sum();
        let mut r = self.next_f32() * tot;
        for i in 0..n {
            r -= probs[i];
            if r <= 0.0 {
                return (cand[i].1, sil_p);
            }
        }
        (cand[n - 1].1, sil_p)
    }
}

// ----------------------------------------------------------------------------- Tick Result
#[derive(Clone, Debug)]
pub struct TickOut {
    pub tick: u32,
    pub inputs: [Option<u32>; 3],
    pub emitted: [u32; 4], // [thinking, user_output, agent_msg_output, tool_calls]
    pub sil_probs: [f32; 4],
    pub n_kv: usize,
    pub active_m: usize,
    pub elapsed_ms: f64,
}

struct PreambleSnapshot {
    n_kv: usize,
    kv_swa_compact: Vec<u8>,  // n_swa * 2 * NKV * n_kv * HD * 2
    kv_full_compact: Vec<u8>, // n_full * 2 * NKV * n_kv * HD * 2
    cs_bytes: Vec<u8>,
    kv_ticks: Vec<u32>,
    ch_pos: [u32; MS_CHANNELS],
    cur_tick: u32,
}

// ----------------------------------------------------------------------------- Multi-Stream Engine
pub struct MsEngine<'a> {
    pub eng: &'a Engine,
    pub adapter: MsAdapter,
    arena: MsArena,
    act: Buf,
    cs_pool: Buf,
    kv_swa: Buf,
    kv_full: Buf,
    kv_ticks_buf: Buf,
    kv_cap: usize,
    max_kv: usize,
    n_swa: usize,
    n_full: usize,
    toks_buf: Buf,
    meta_buf: Buf, // [0..8] seq_cs (=chs), [16..24] seq_pos, [32..40] kv_write_idx
    pub kv_ticks: Vec<u32>,
    pub ch_pos: [u32; MS_CHANNELS],
    pub cur_tick: u32,
    pub prev_outputs: [u32; 4],
    logits_scratch: Vec<f32>,
    preamble_snap: Option<PreambleSnapshot>,
}

const META_CS_OFF: u32 = 0;
const META_POS_OFF: u32 = 16;
const META_KVW_OFF: u32 = 32;

impl<'a> MsEngine<'a> {
    pub fn new(eng: &'a Engine, adapter: MsAdapter, max_kv: usize) -> MsEngine<'a> {
        let max_kv = max_kv.clamp(512, 8192);
        let init_cap = 512usize.min(max_kv);
        let n_swa = eng.slot_kind.iter().filter(|k| k.0).count();
        let n_full = N_SLOT - n_swa;

        let arena = MsArena::new(eng.n_vocab);
        let act = eng.gpu.buffer(arena.total as usize * 4);
        let cs_pool = eng.gpu.buffer(N_SLOT * MS_CHANNELS * 3 * NCH * 4);
        let kv_swa = eng.gpu.buffer(n_swa * 2 * NKV * init_cap * HD * 2);
        let kv_full = eng.gpu.buffer(n_full * 2 * NKV * init_cap * HD * 2);
        let kv_ticks_buf = eng.gpu.buffer(init_cap * 4);
        let toks_buf = eng.gpu.buffer(MS_MAX_M * 4);
        let meta_buf = eng.gpu.buffer(64 * 4);

        let sil = adapter.silence_id;
        // Pre-warm our custom multistream pipelines so first tick is fast
        for k in &[
            "k_ms_embed",
            "k_ms_lora_a",
            "k_ms_lora_b",
            "k_ms_ctrl_gf",
            "k_ms_kv_attn",
            "k_ms_wo_post",
            "k_ms_out_prep",
        ] {
            eng.gpu.pso(k);
        }

        let mut ms = MsEngine {
            eng,
            adapter,
            arena,
            act,
            cs_pool,
            kv_swa,
            kv_full,
            kv_ticks_buf,
            kv_cap: init_cap,
            max_kv,
            n_swa,
            n_full,
            toks_buf,
            meta_buf,
            kv_ticks: Vec::with_capacity(init_cap),
            ch_pos: [0; MS_CHANNELS],
            cur_tick: 0,
            prev_outputs: [sil; 4],
            logits_scratch: vec![0.0f32; eng.n_vocab],
            preamble_snap: None,
        };
        ms.reset();
        ms
    }

    /// Pre-run a native-framed adapter's system preamble once and snapshot its KV + conv state
    /// so every subsequent `reset()` restores the warm preamble in ~2 ms!
    pub fn warm_preamble(&mut self, preamble_toks: &[u32]) {
        self.preamble_snap = None;
        self.reset();
        if preamble_toks.is_empty() {
            return;
        }
        let t0 = Instant::now();
        let sil = self.adapter.silence_id;
        let mut dummy = MsSamplerCfg::new(0.0, 1, 1.0, 0.0, 1);
        for &tok in preamble_toks {
            self.prev_outputs = [sil; 4];
            let _ = self.step_tick([Some(tok), None, None], &mut dummy);
        }
        self.prev_outputs = [sil; 4];

        let n_kv = self.kv_ticks.len();
        let row_bytes = n_kv * HD * 2;
        let cap_bytes = self.kv_cap * HD * 2;
        let snap_kv = |buf: &Buf, nslots: usize| -> Vec<u8> {
            let nblks = nslots * 2 * NKV;
            let mut out = vec![0u8; nblks * row_bytes];
            unsafe {
                let src = buf.contents().as_ptr() as *const u8;
                for blk in 0..nblks {
                    std::ptr::copy_nonoverlapping(
                        src.add(blk * cap_bytes),
                        out.as_mut_ptr().add(blk * row_bytes),
                        row_bytes,
                    );
                }
            }
            out
        };
        let cs_bytes = unsafe {
            std::slice::from_raw_parts(self.cs_pool.contents().as_ptr() as *const u8, self.cs_pool.length()).to_vec()
        };
        self.preamble_snap = Some(PreambleSnapshot {
            n_kv,
            kv_swa_compact: snap_kv(&self.kv_swa, self.n_swa),
            kv_full_compact: snap_kv(&self.kv_full, self.n_full),
            cs_bytes,
            kv_ticks: self.kv_ticks.clone(),
            ch_pos: self.ch_pos,
            cur_tick: self.cur_tick,
        });
        eprintln!(
            "[multistream] cached {}-token native preamble snapshot ({:.1} MiB) in {:.2?}",
            preamble_toks.len(),
            (n_kv * N_SLOT * 2 * NKV * HD * 2 + self.cs_pool.length()) as f64 / (1 << 20) as f64,
            t0.elapsed()
        );
    }

    pub fn kv_near_full(&self) -> bool {
        self.kv_ticks.len() + MS_CHANNELS >= self.max_kv
    }

    pub fn mem_bytes(&self) -> usize {
        self.adapter.ad_buf.length()
            + self.adapter.pb_buf.length()
            + self.act.length()
            + self.cs_pool.length()
            + self.kv_swa.length()
            + self.kv_full.length()
    }

    pub fn reset(&mut self) {
        if let Some(snap) = &self.preamble_snap {
            let n_kv = snap.n_kv;
            let row_bytes = n_kv * HD * 2;
            let cap_bytes = self.kv_cap * HD * 2;
            let restore_kv = |buf: &Buf, compact: &[u8], nslots: usize| unsafe {
                let nblks = nslots * 2 * NKV;
                let dst = buf.contents().as_ptr() as *mut u8;
                for blk in 0..nblks {
                    std::ptr::copy_nonoverlapping(
                        compact.as_ptr().add(blk * row_bytes),
                        dst.add(blk * cap_bytes),
                        row_bytes,
                    );
                }
            };
            restore_kv(&self.kv_swa, &snap.kv_swa_compact, self.n_swa);
            restore_kv(&self.kv_full, &snap.kv_full_compact, self.n_full);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    snap.cs_bytes.as_ptr(),
                    self.cs_pool.contents().as_ptr() as *mut u8,
                    snap.cs_bytes.len(),
                );
                let kp = self.kv_ticks_buf.contents().as_ptr() as *mut u32;
                for (i, &tk) in snap.kv_ticks.iter().enumerate() {
                    *kp.add(i) = tk;
                }
            }
            self.kv_ticks = snap.kv_ticks.clone();
            self.ch_pos = snap.ch_pos;
            self.cur_tick = snap.cur_tick;
            self.prev_outputs = [self.adapter.silence_id; 4];
            return;
        }
        unsafe {
            std::ptr::write_bytes(self.cs_pool.contents().as_ptr() as *mut u8, 0, self.cs_pool.length());
        }
        self.kv_ticks.clear();
        self.ch_pos = [0; MS_CHANNELS];
        self.cur_tick = 0;
        self.prev_outputs = [self.adapter.silence_id; 4];
    }

    #[allow(dead_code)]
    pub fn swap_adapter(&mut self, adapter: MsAdapter) {
        self.adapter = adapter;
        self.preamble_snap = None;
        self.reset();
    }

    fn ensure_kv_cap(&mut self, need: usize) {
        if need <= self.kv_cap {
            return;
        }
        let new_cap = ((need + 511) / 512 * 512).min(self.max_kv);
        assert!(need <= new_cap, "Multi-stream KV cache full ({} non-silent keys)", self.max_kv);
        let old = self.kv_cap;
        let grow = |old_buf: &Buf, nslots: usize| -> Buf {
            let nb = self.eng.gpu.buffer(nslots * 2 * NKV * new_cap * HD * 2);
            unsafe {
                let src = old_buf.contents().as_ptr() as *const u8;
                let dst = nb.contents().as_ptr() as *mut u8;
                for blk in 0..nslots * 2 * NKV {
                    std::ptr::copy_nonoverlapping(
                        src.add(blk * old * HD * 2),
                        dst.add(blk * new_cap * HD * 2),
                        old * HD * 2,
                    );
                }
            }
            nb
        };
        self.kv_swa = grow(&self.kv_swa, self.n_swa);
        self.kv_full = grow(&self.kv_full, self.n_full);
        let nt = self.eng.gpu.buffer(new_cap * 4);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.kv_ticks_buf.contents().as_ptr() as *const u8,
                nt.contents().as_ptr() as *mut u8,
                old * 4,
            );
        }
        self.kv_ticks_buf = nt;
        self.kv_cap = new_cap;
    }

    // ------------------------------------------------------------------------- Low-level GPU helpers
    #[inline]
    fn go<T>(&self, e: &Enc, pso: &str, args: &T, grid: (usize, usize, usize), tg: usize) {
        e.pso(&self.eng.gpu.pso(pso));
        e.bytes(args, 5);
        e.dispatch(grid, tg);
        e.barrier();
    }

    fn mix_b(&self, e: &Enc, jobs: &[MixJob], m: usize) {
        for j in jobs {
            let j = MixJob { ntok: m as u32, ..*j };
            let name = if j.q_off < 0 {
                "k_mix_b_id".to_string()
            } else {
                format!("k_mix_b_{}_{}", j.n_s, j.n_q)
            };
            e.pso(&self.eng.gpu.pso(&name));
            e.bytes(&j, 5);
            e.dispatch((m, 1, 1), 32);
        }
        e.barrier();
    }

    fn gemm_h(&self, e: &Enc, segs: &[HSeg], m: usize) {
        let mut a = GemmHArgs { nseg: segs.len() as u32, t: m as u32, ..Default::default() };
        let mut tiles = 0u32;
        for (i, sg) in segs.iter().enumerate() {
            a.seg[i] = *sg;
            a.seg[i].tile_start = tiles;
            tiles += sg.nrows.div_ceil(64);
        }
        self.go(e, "k_gemm_sg4", &a, (1, tiles as usize, 1), 128);
    }

    fn lora_add_segs(&self, e: &Enc, segs: &[MsLoraSeg], m: usize, run_b: bool) {
        if segs.is_empty() {
            return;
        }
        let mut a = MsLoraArgs { nseg: segs.len() as u32, m: m as u32, ..Default::default() };
        for (i, &sg) in segs.iter().enumerate() {
            a.seg[i] = sg;
            a.max_rank = a.max_rank.max(sg.rank);
            a.max_out_blocks = a.max_out_blocks.max(sg.out_dim.div_ceil(256));
        }
        e.buf(&self.adapter.ad_buf, 0, 3);
        self.go(e, "k_ms_lora_a", &a, (a.max_rank as usize, m, segs.len()), 256);
        if run_b {
            self.go(e, "k_ms_lora_b", &a, (a.max_out_blocks as usize, m, segs.len()), 256);
        }
    }

    fn make_lora_seg(
        &self,
        lp: &LoraPair,
        pass: usize,
        x_off: u32,
        y_off: u32,
        in_half: bool,
        out_half: bool,
        slot_idx: usize,
    ) -> Option<MsLoraSeg> {
        if !lp.present() {
            return None;
        }
        let (a_off, b_off) = lp.pass_offsets(pass);
        Some(MsLoraSeg {
            a_off,
            b_off,
            x_off,
            y_off,
            in_dim: lp.in_dim,
            out_dim: lp.out_dim,
            rank: lp.rank,
            tmp_off: self.arena.lora_tmp + (slot_idx * MS_MAX_M * 64) as u32,
            in_half: in_half as u32,
            out_half: out_half as u32,
            pad0: 0,
            scale: self.adapter.lora_alpha / (lp.rank.max(1) as f32),
        })
    }

    fn ctrl_ms(
        &self,
        e: &Enc,
        lw: &LoopW,
        ll: &LayerLora,
        pass: usize,
        jobs: &[(usize, u32, u32, u32)], // (c_idx, u_off, m_off, norm_w)
        m: usize,
        hout: &[u32],
    ) {
        let d = self.arena;
        // 1. Base down GEMM: [M, 2*E] half -> [M, 64] half
        let down: Vec<HSeg> = jobs
            .iter()
            .map(|&(c, _, _, _)| {
                let mut s = Engine::hseg_pb(lw.down[c], 64, (2 * E) as u32, d.cath[c], d.gh[c]);
                s.hout = 1;
                s
            })
            .collect();
        self.gemm_h(e, &down, m);

        // 2. Down LoRA add onto d.gh[c] (half)
        let mut lsegs = Vec::new();
        for (i, &(c, _, _, _)) in jobs.iter().enumerate() {
            if let Some(sg) = self.make_lora_seg(&ll.ctrl[c].down, pass, d.cath[c], d.gh[c], true, true, i) {
                lsegs.push(sg);
            }
        }
        self.lora_add_segs(e, &lsegs, m, true);

        // 3. Gain & Forget LoRA-A projections from d.gh[c] (half) -> lora_tmp[0..2*njobs]
        let mut gf_a = Vec::new();
        for (i, &(c, _, _, _)) in jobs.iter().enumerate() {
            if let Some(sg) = self.make_lora_seg(&ll.ctrl[c].gain, pass, d.gh[c], 0, true, false, 2 * i) {
                gf_a.push(sg);
            }
            if let Some(sg) = self.make_lora_seg(&ll.ctrl[c].forget, pass, d.gh[c], 0, true, false, 2 * i + 1) {
                gf_a.push(sg);
            }
        }
        self.lora_add_segs(e, &gf_a, m, false);

        // 4. Exact fused gain/forget BF16 + LoRA-B + u*(1 - tanh(F)) + tanh(G)*m -> d.gain_o[c]
        let mut ga = MsCtrlGfArgs { m: m as u32, njobs: jobs.len() as u32, ..Default::default() };
        for (i, &(c, u, mv, _)) in jobs.iter().enumerate() {
            let gl = &ll.ctrl[c].gain;
            let fl = &ll.ctrl[c].forget;
            let rank = if gl.present() && fl.present() { gl.rank } else { 0 };
            let (_, gb_off) = if rank > 0 { gl.pass_offsets(pass) } else { (0, 0) };
            let (_, fb_off) = if rank > 0 { fl.pass_offsets(pass) } else { (0, 0) };
            ga.job[i] = MsCtrlGfJob {
                gain_w: lw.gain[c],
                forget_w: lw.forget[c],
                gain_b_off: gb_off,
                forget_b_off: fb_off,
                gh_off: d.gh[c],
                tmp_gain_off: d.lora_tmp + (2 * i * MS_MAX_M * 64) as u32,
                tmp_forget_off: d.lora_tmp + ((2 * i + 1) * MS_MAX_M * 64) as u32,
                u_off: u,
                m_off: mv,
                raw_off: d.gain_o[c],
                rank,
                pad0: 0,
            };
        }
        e.buf(&self.adapter.ad_buf, 0, 3);
        self.go(e, "k_ms_ctrl_gf", &ga, (E / 256, m, jobs.len()), 256);

        // 5. RMSNorm raw_off -> half hout[i]
        let mut ca = CtrlBArgs { n: jobs.len() as u32, ..Default::default() };
        for (i, &(c, _, _, nw)) in jobs.iter().enumerate() {
            ca.job[i] = CtrlBJob {
                u_off: d.gain_o[c],
                m_off: u32::MAX,
                g_off: 0,
                f_off: 0,
                out_off: 0,
                norm_w: nw,
                h_off: hout[i] as i32,
                nof32: 1,
            };
        }
        self.go(e, "k_ctrl_apply_b", &ca, (m, jobs.len(), 1), 256);
    }

    fn moe_4class(&self, e: &Enc, l: &Layer, m: usize) {
        let d = self.arena;
        self.gemm_h(e, &[Engine::hseg_pb(l.router, NEXP as u32, E as u32, d.fh, d.rlog)], m);
        let ta = TopkBArgs {
            logits_off: d.rlog,
            bias_off: l.exp_bias,
            sel_off: d.sel,
            t: m as u32,
            wscale: self.eng.wscale,
            pad: [0; 3],
        };
        self.go(e, "k_topk_b", &ta, (m, 1, 1), 32);
        let ra = Route4Args {
            sel_off: d.sel,
            t: m as u32,
            perm_off: d.perm,
            inv_off: d.inv,
            aw_off: d.aw,
            ind_off: d.ind,
            gu_y: [(FF / 32) as u32, (FF / 32) as u32, (FF / 4) as u32, (FF / 4) as u32],
            dn_y: [(E / 64) as u32, (E / 64) as u32, (E / 8) as u32, (E / 8) as u32],
            tiles_off: d.tl,
        };
        self.go(e, "k_moe_route4", &ra, (1, 1, 1), 1024);

        let tha = ToHalfArgs {
            src_off: d.fh,
            dst_off: d.xgh,
            k: E as u32,
            rows: (m * TOPK) as u32,
            perm_off: d.perm,
            gather: 3,
            src_stride: E as u32,
            dst_stride: E as u32,
        };
        self.go(e, "k_tohalf", &tha, (E.div_ceil(1024), m * TOPK, 1), 256);

        let gu_cls: [(&str, usize); NTC_B] = [
            ("k_moe_sg8", 256),
            ("k_moe_sg4", 128),
            ("k_moe_dir_gu2", 128),
            ("k_moe_dir_gu1", 128),
        ];
        let dn_cls: [(&str, usize); NTC_B] = [
            ("k_moe_sg8", 256),
            ("k_moe_sg4", 128),
            ("k_moe_dir_dn2", 128),
            ("k_moe_dir_dn1", 128),
        ];

        for (c, &(name, tg)) in gu_cls.iter().enumerate() {
            let g = MoeGemmHArgs {
                w_lo: l.gu.off as u32,
                w_hi: (l.gu.off >> 32) as u32,
                expert_bytes: l.gu.expert_bytes as u32,
                row_bytes: l.gu.row_bytes,
                wtype: l.gu.ty,
                nrows: (2 * FF) as u32,
                k: E as u32,
                xh_off: d.xgh,
                tiles_off: d.tl[c],
                ntiles_off: d.ind + (c * 4) as u32,
                out_off: d.acth,
                ldo: FF as u32,
                situ: 1,
                aw_off: d.aw,
                beta: self.eng.situ_beta,
                lbeta: self.eng.situ_lbeta,
            };
            e.pso(&self.eng.gpu.pso(name));
            e.bytes(&g, 5);
            e.dispatch_indirect(&self.act, (d.ind as usize + c * 4) * 4, tg);
        }
        e.barrier();

        for (c, &(name, tg)) in dn_cls.iter().enumerate() {
            let dn = MoeGemmHArgs {
                w_lo: l.dn.off as u32,
                w_hi: (l.dn.off >> 32) as u32,
                expert_bytes: l.dn.expert_bytes as u32,
                row_bytes: l.dn.row_bytes,
                wtype: l.dn.ty,
                nrows: E as u32,
                k: FF as u32,
                xh_off: d.acth,
                tiles_off: d.tl[c],
                ntiles_off: d.ind + (c * 4) as u32,
                out_off: d.yd,
                ldo: E as u32,
                situ: 0,
                aw_off: 0,
                beta: 0.0,
                lbeta: 0.0,
            };
            e.pso(&self.eng.gpu.pso(name));
            e.bytes(&dn, 5);
            e.dispatch_indirect(&self.act, (d.ind as usize + (NTC_B + c) * 4) * 4, tg);
        }
        e.barrier();
    }

    #[allow(clippy::too_many_arguments)]
    fn run_layer(
        &self,
        e: &Enc,
        m: usize,
        swa_j_start: u32,
        slot: usize,
        ip: usize,
        pass: usize,
        cur: usize,
        blocks: &mut Vec<usize>,
        msg: Option<(usize, usize)>,
        va: &mut VecAlloc,
        pinned: &[usize],
    ) -> usize {
        let eng = self.eng;
        let l = &eng.layers[ip];
        let ll = &self.adapter.layers[ip];
        let d = self.arena;
        let ctl = msg.map(|_| &eng.loops[ip - LOOP_START]);
        let es = E as u32;

        // ---- 1. Attention input depth mix + optional loop controllers -> half d.xh[0..3]
        let norms: [i32; 3] = if l.swa {
            [l.attn_norm; 3]
        } else {
            [l.q_inp_norm, l.k_inp_norm, l.v_inp_norm]
        };
        let mut st: Vec<u32> = blocks.iter().map(|&v| d.vec(v)).collect();
        st.push(d.vec(cur));
        let nq = if l.attn_res_query >= 0 { l.n_attn_q } else { 1 };
        if let Some(lw) = ctl {
            let (b0, b1) = msg.unwrap();
            let mut ja = eng.mix_job_off(&st, l.attn_res_query, nq, &d.inp[..nq as usize], &[], es);
            let mut jm = eng.mix_job_off(&[d.vec(b0), d.vec(b1)], lw.msg_query as i32, 4, &d.msg, &[], es);
            if l.swa {
                ja.hcopy[0] = [d.cath[0] as i32, d.cath[1] as i32, d.cath[2] as i32];
            } else {
                for j in 0..3 {
                    ja.hcopy[j][0] = d.cath[j] as i32;
                }
            }
            for j in 0..4 {
                jm.hcopy[j][0] = (d.cath[j] + (E / 2) as u32) as i32;
            }
            self.mix_b(e, &[ja, jm], m);
            let jobs: Vec<(usize, u32, u32, u32)> = (0..3)
                .map(|j| (j, if l.swa { d.inp[0] } else { d.inp[j] }, d.msg[j], norms[j] as u32))
                .collect();
            self.ctrl_ms(e, lw, ll, pass, &jobs, m, &[d.xh[0], d.xh[1], d.xh[2]]);
        } else {
            let nn: Vec<i32> = if l.swa { vec![l.attn_norm] } else { norms.to_vec() };
            let mut ja = eng.mix_job_off(&st, l.attn_res_query, nq, &d.inp[..nq as usize], &nn, es);
            for q in 0..nq as usize {
                ja.hout[q] = d.xh[q] as i32;
            }
            ja.nof32 = 0xF;
            self.mix_b(e, &[ja], m);
        }

        // ---- 2. QKV + Gate + Rel + Sink (+ dyn conv taps) base GEMM + pass-specific LoRA
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
        self.gemm_h(e, &hs, m);

        let mut qkv_lora = Vec::with_capacity(6);
        let qkv_specs: [(&LoraPair, u32, u32, bool); 6] = [
            (&ll.wq, d.xh[xi[0]], d.q, true),
            (&ll.gate, d.xh[xi[0]], d.gate, true),
            (&ll.wr, d.xh[xi[0]], d.rel, false),
            (&ll.sink, d.xh[xi[0]], d.sink, false),
            (&ll.wk, d.xh[xi[1]], d.k, false),
            (&ll.wv, d.xh[xi[2]], d.v, false),
        ];
        for (idx, (lp, x_off, y_off, out_half)) in qkv_specs.iter().enumerate() {
            if let Some(sg) = self.make_lora_seg(lp, pass, *x_off, *y_off, true, *out_half, idx) {
                qkv_lora.push(sg);
            }
        }
        self.lora_add_segs(e, &qkv_lora, m, true);

        // ---- 3. Multi-Stream column-mode K/V conv + tick-causal shared-KV attention + conditional KV append
        let (is_swa, slot_idx) = eng.slot_kind[slot];
        if is_swa {
            e.buf(&self.kv_swa, 0, 3);
        } else {
            e.buf(&self.kv_full, 0, 3);
        }
        e.buf(&self.kv_ticks_buf, 0, 6);
        let aa = MsAttnArgs {
            m: m as u32,
            kv_cap: self.kv_cap as u32,
            slot_idx: slot_idx as u32,
            is_swa: is_swa as u32,
            cur_tick: self.cur_tick,
            n_kv: self.kv_ticks.len() as u32,
            j_start: if is_swa { swa_j_start } else { 0 },
            n_vis_past: self.kv_ticks.len() as u32,
            layer_slot: slot as u32,
            max_channels: MS_CHANNELS as u32,
            seq_cs_off: META_CS_OFF,
            seq_pos_off: META_POS_OFF,
            kv_write_off: META_KVW_OFF,
            q_off: d.q,
            k_off: d.k,
            v_off: d.v,
            r_off: d.rel,
            sink_off: d.sink,
            kdyn_off: if l.swa { -1 } else { d.kdyn as i32 },
            vdyn_off: if l.swa { -1 } else { d.vdyn as i32 },
            kconv_t_w: l.k_conv_t,
            vconv_t_w: l.v_conv_t,
            qclip_w: l.q_clip,
            relclip_w: l.rel_clip,
            relbias_w: l.rel_bias,
            log_alpha: eng.log_alpha,
            log_floor: eng.log_floor,
            gate_off: d.gate,
            out_off: d.agh,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        self.go(e, "k_ms_kv_attn", &aa, (NKV, m, 1), 128);

        // ---- 4. wo GEMM -> d.wo_y + wo LoRA + column-mode attn_sconv + optional residual -> vec(mid)
        let block_start = ip % BLOCK == 0;
        let mut live: Vec<usize> = blocks.clone();
        live.push(cur);
        live.extend_from_slice(pinned);
        let mid = va.alloc(&live);

        self.gemm_h(e, &[Engine::hseg(&l.wo, d.agh, d.wo_y)], m);
        if let Some(sg) = self.make_lora_seg(&ll.wo, pass, d.agh, d.wo_y, true, false, 0) {
            self.lora_add_segs(e, &[sg], m, true);
        }
        let wp = MsWoPostArgs {
            wo_y_off: d.wo_y,
            res_off: d.vec(cur),
            out_off: d.vec(mid),
            has_res: (!block_start) as u32,
            layer_slot: slot as u32,
            max_channels: MS_CHANNELS as u32,
            convw_t_off: l.attn_post_conv_t,
            seq_cs_off: META_CS_OFF,
            seq_pos_off: META_POS_OFF,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        self.go(e, "k_ms_wo_post", &wp, (m, E.div_ceil(1024), 1), 256);
        if block_start {
            blocks.push(cur);
        }

        // ---- 5. FFN input depth mix + optional loop controller -> half d.fh
        let mut st: Vec<u32> = blocks.iter().map(|&v| d.vec(v)).collect();
        st.push(d.vec(mid));
        if let Some(lw) = ctl {
            let mut jf = eng.mix_job_off(&st, l.ffn_res_query as i32, 1, &[d.fin], &[], es);
            jf.hcopy[0][0] = d.cath[3] as i32;
            self.mix_b(e, &[jf], m);
            self.ctrl_ms(e, lw, ll, pass, &[(3, d.fin, d.msg[3], l.ffn_norm)], m, &[d.fh]);
        } else {
            let mut jf = eng.mix_job_off(&st, l.ffn_res_query as i32, 1, &[d.fin], &[l.ffn_norm as i32], es);
            jf.hout[0] = d.fh as i32;
            jf.nof32 = 1;
            self.mix_b(e, &[jf], m);
        }

        // ---- 6. 4-class MoE (router + topk + route4 + gather + gate_up + down)
        self.moe_4class(e, l, m);

        // ---- 7. MoE combine + column-mode mlp_sconv + residual -> vec(out)
        let mut live: Vec<usize> = blocks.clone();
        live.push(mid);
        live.extend_from_slice(pinned);
        let out = va.alloc(&live);
        let cp = MoeCombPostArgs {
            yd_off: d.yd,
            inv_off: d.inv,
            res_off: d.vec(mid),
            out_off: d.vec(out),
            layer_slot: slot as u32,
            max_batch: MS_CHANNELS as u32,
            convw_t_off: l.ffn_post_conv_t,
            seq_cs_off: META_CS_OFF,
            seq_pos_off: META_POS_OFF,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        self.go(e, "k_moe_comb_post_b", &cp, (m, E.div_ceil(1024), 1), 256);
        out
    }

    // ------------------------------------------------------------------------- Execute ONE 7-channel tick
    pub fn step_tick(&mut self, inputs: [Option<u32>; 3], sampler: &mut MsSamplerCfg) -> TickOut {
        let t0 = Instant::now();
        let sil = self.adapter.silence_id;

        // Build Option-B active token list: non-silent inputs [0..3) + dense outputs [3..7)
        let mut active_toks: Vec<u32> = Vec::with_capacity(MS_CHANNELS);
        let mut active_chs: Vec<u32> = Vec::with_capacity(MS_CHANNELS);
        for ch in 0..3 {
            if let Some(tok) = inputs[ch] {
                if tok != sil {
                    active_toks.push(tok);
                    active_chs.push(ch as u32);
                }
            }
        }
        let m_out_start = active_toks.len();
        for out_i in 0..4 {
            active_toks.push(self.prev_outputs[out_i]);
            active_chs.push((3 + out_i) as u32);
        }
        let m = active_toks.len();

        // Assign non-silent KV write slots (Option B skips writing <|silence|> into KV cache!)
        let n_kv = self.kv_ticks.len();
        let mut n_new = 0usize;
        let mut kv_write_idx = [u32::MAX; MS_MAX_M];
        for i in 0..m {
            if active_toks[i] != sil {
                kv_write_idx[i] = (n_kv + n_new) as u32;
                n_new += 1;
            }
        }
        self.ensure_kv_cap(n_kv + n_new);

        // Compute smallest key index still inside SWA window [cur_tick - EXT + 1 .. cur_tick)
        let swa_min_tick = self.cur_tick.saturating_sub(EXT as u32 - 1);
        let swa_j_start = self.kv_ticks.partition_point(|&tk| tk < swa_min_tick) as u32;

        // Upload toks, meta, and new kv_ticks
        unsafe {
            let tp = self.toks_buf.contents().as_ptr() as *mut i32;
            let mp = self.meta_buf.contents().as_ptr() as *mut u32;
            for i in 0..m {
                *tp.add(i) = active_toks[i] as i32;
                *mp.add(META_CS_OFF as usize + i) = active_chs[i];
                *mp.add(META_POS_OFF as usize + i) = self.ch_pos[active_chs[i] as usize];
                *mp.add(META_KVW_OFF as usize + i) = kv_write_idx[i];
            }
            let kp = self.kv_ticks_buf.contents().as_ptr() as *mut u32;
            for j in 0..n_new {
                *kp.add(n_kv + j) = self.cur_tick;
            }
        }

        // Encode entire 48-slot forward + 4-channel lm_head in ONE command buffer
        let cb = self.eng.gpu.cmd();
        {
            let e = Enc::new(&cb);
            let d = self.arena;
            let eng = self.eng;

            e.buf(&eng.w, 0, 0);
            e.buf(&self.adapter.pb_buf, 0, 1);
            e.buf(&self.act, 0, 2);
            e.buf(&self.adapter.ad_buf, 0, 3);
            e.buf(&self.cs_pool, 0, 4);
            e.buf(&self.toks_buf, 0, 6);
            e.buf(&self.meta_buf, 0, 7);

            // 1. Multi-Stream Embedding -> vec(0)
            let ea = MsEmbedArgs {
                emb_lo: eng.tok_embd.off as u32,
                emb_hi: (eng.tok_embd.off >> 32) as u32,
                row_bytes: eng.tok_embd.row_bytes,
                norm_off: eng.tok_norm,
                out_off: d.vec(0),
                silence_id: self.adapter.silence_id,
                tool_id: self.adapter.tool_id,
                peer_id: self.adapter.peer_id,
                ch_emb_off: self.adapter.ch_emb_off,
                sil_emb_off: self.adapter.sil_emb_off,
                tool_emb_off: self.adapter.tool_emb_off,
                peer_emb_off: self.adapter.peer_emb_off,
                eps: eng.eps,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            };
            self.go(&e, "k_ms_embed", &ea, (m, 1, 1), 256);

            // 2. 48 layer executions (prelude 0..6, 3x looped core 6..18, coda 18..24)
            let mut va = VecAlloc { next: 1, n: B_NVEC };
            let mut blocks: Vec<usize> = Vec::new();
            let mut cur = 0usize;
            for ip in 0..LOOP_START {
                cur = self.run_layer(&e, m, swa_j_start, ip, ip, 0, cur, &mut blocks, None, &mut va, &[]);
            }
            let prelude = cur;
            let prelude_blocks = blocks.clone();
            let mut bank = (0usize, 0usize);
            for pass in 0..N_PASS {
                if pass > 0 {
                    cur = prelude;
                    blocks = prelude_blocks.clone();
                }
                let msg = if pass > 0 { Some(bank) } else { None };
                let pinned: Vec<usize> = {
                    let mut p = vec![prelude];
                    if pass > 0 {
                        p.push(bank.0);
                        p.push(bank.1);
                    }
                    p
                };
                for ip in LOOP_START..LOOP_END {
                    cur = self.run_layer(
                        &e,
                        m,
                        swa_j_start,
                        slot_of(pass, ip),
                        ip,
                        pass,
                        cur,
                        &mut blocks,
                        msg,
                        &mut va,
                        &pinned,
                    );
                }
                bank = (blocks[prelude_blocks.len() + 1], cur);
            }
            for ip in LOOP_END..N_PHYS {
                cur = self.run_layer(&e, m, swa_j_start, ip, ip, 0, cur, &mut blocks, None, &mut va, &[]);
            }

            // 3. Final output depth mix + output_norm -> f32 d.fin[0..m]
            let mut st: Vec<u32> = blocks.iter().map(|&v| d.vec(v)).collect();
            st.push(d.vec(cur));
            let jo = eng.mix_job_off(&st, eng.output_res_query as i32, 1, &[d.fin], &[eng.output_norm as i32], E as u32);
            self.mix_b(&e, &[jo], m);

            // 4. Add output_channel_embedding + optional per-channel gated head for the 4 output tokens -> out_f32 + out_half
            e.buf(&self.adapter.ad_buf, 0, 3);
            let op = MsOutPrepArgs {
                fin_off: d.fin,
                out_f32_off: d.out_f32,
                out_half_off: d.out_half,
                out_ch_emb_off: self.adapter.out_ch_emb_off,
                m_start: m_out_start as u32,
                n_out: 4,
                head_rank: self.adapter.out_head_rank,
                pad0: 0,
                head_w1_off: self.adapter.out_head_w1_off,
                head_b1_off: self.adapter.out_head_b1_off,
                head_w2_off: self.adapter.out_head_w2_off,
                pad1: 0,
            };
            self.go(&e, "k_ms_out_prep", &op, (4, 1, 1), 256);

            // 5. Full lm_head GEMM for the 4 output tokens -> f32 d.logits[4 * n_vocab]
            let w = &eng.output;
            let la = LmHeadArgs {
                w_lo: w.off as u32,
                w_hi: (w.off >> 32) as u32,
                row_bytes: w.row_bytes,
                wtype: w.ty,
                n_vocab: eng.n_vocab as u32,
                k: w.k,
                xh_off: d.out_half,
                part_off: d.logits,
                b: 4,
                nvch: 0,
                vstart: 0,
                vcount: eng.n_vocab as u32,
            };
            self.go(&e, "k_lmhead_slice_sg4", &la, (1, eng.n_vocab.div_ceil(64), 1), 128);
            e.end();
        }
        cb.commit();
        cb.waitUntilCompleted();

        // CPU exact silence_adapter + tool_adapter logit patching + softcap + sampling (<0.3 ms!)
        let nv = self.eng.n_vocab;
        let act_ptr = self.act.contents().as_ptr() as *const f32;
        let mut emitted = [sil; 4];
        let mut sil_probs = [0.0f32; 4];
        for out_i in 0..4 {
            let h = unsafe {
                std::slice::from_raw_parts(act_ptr.add(self.arena.out_f32 as usize + out_i * E), E)
            };
            let lg_gpu = unsafe {
                std::slice::from_raw_parts(act_ptr.add(self.arena.logits as usize + out_i * nv), nv)
            };
            self.logits_scratch.copy_from_slice(lg_gpu);

            if !self.adapter.sil_unembed_delta.is_empty() && (self.adapter.silence_id as usize) < nv {
                let mut dot = self.adapter.sil_unembed_bias;
                for (a, b) in h.iter().zip(self.adapter.sil_unembed_delta.iter()) {
                    dot += a * b;
                }
                self.logits_scratch[self.adapter.silence_id as usize] += dot;
            }
            if !self.adapter.tool_unembed_delta.is_empty() && (self.adapter.tool_id as usize) < nv {
                let mut dot = self.adapter.tool_unembed_bias;
                for (a, b) in h.iter().zip(self.adapter.tool_unembed_delta.iter()) {
                    dot += a * b;
                }
                self.logits_scratch[self.adapter.tool_id as usize] += dot;
            }
            if !self.adapter.peer_unembed_delta.is_empty() && (self.adapter.peer_id as usize) < nv {
                let mut dot = self.adapter.peer_unembed_bias;
                for (a, b) in h.iter().zip(self.adapter.peer_unembed_delta.iter()) {
                    dot += a * b;
                }
                self.logits_scratch[self.adapter.peer_id as usize] += dot;
            }

            let (tok, sp) = sampler.sample_one(&mut self.logits_scratch, self.eng.softcap, sil);
            emitted[out_i] = tok;
            sil_probs[out_i] = sp;
        }

        // Advance tick state
        for _ in 0..n_new {
            self.kv_ticks.push(self.cur_tick);
        }
        for &ch in &active_chs {
            self.ch_pos[ch as usize] += 1;
        }
        let tick_done = self.cur_tick;
        self.cur_tick += 1;
        self.prev_outputs = emitted;

        TickOut {
            tick: tick_done,
            inputs,
            emitted,
            sil_probs,
            n_kv: self.kv_ticks.len(),
            active_m: m,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
        }
    }
}
