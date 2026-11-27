//! Zero-dependency HTTP + SSE server for TRUE parallel 7-channel Multi-Stream inference (`MsEngine`).
//!
//! Automatically supports both:
//!   - Native-framed V4+ adapters (`native_framing = true`): caches the native `tool_declare` +
//!     `Thinking effort` preamble KV snapshot (~2 ms reset!), wraps ch0/ch1/ch2 inputs with
//!     Inkling-2-Mini's native role tokens (`<|message_user|>`, `<|message_tool|>`), and strips
//!     boilerplate protocol headers (`<|message_model|>`, `<|content_thinking|>`, `<|content_text|>`)
//!     from the 4 output card text views while keeping every token visible on the 7-track timeline.
//!   - Legacy raw-framed V3 adapters (`native_framing = false`).

use crate::engine::Engine;
use crate::multistream::{
    json_escape, parse_json, MsAdapter, MsEngine, MsSamplerCfg, TickOut, CH_AGENT_IN, CH_TOOL_IN,
    CH_USER_IN,
};
use crate::tokenizer::Tokenizer;
use std::collections::{BTreeMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::Duration;

const UI_HTML: &str = include_str!("multistream_ui.html");
const MAX_HISTORY: usize = 500;
const IDLE_AUTO_PAUSE_TICKS: usize = 12;

#[derive(Clone)]
pub struct AdapterEntry {
    pub name: String,
    pub path: String,
}

struct ScheduledInj {
    at_tick: u32,
    ch: usize,
    toks: Vec<u32>,
}

struct Shared {
    adapters: Vec<AdapterEntry>,
    active_adapter: String,
    native_framing: bool,
    pending_adapter: Option<String>,
    running: bool,
    step_budget: usize,
    pending_reset: bool,
    pending_note: Option<String>,
    mem_mib: f64,
    // Sampler & runtime tuning
    temp: f32,
    top_k: usize,
    top_p: f32,
    silence_penalty: f32,
    tick_delay_ms: u64,
    auto_pause_idle: bool,
    auto_peer: bool,
    peer_latency: u32,
    peer_kv: String,
    auto_tool: bool,
    tool_latency: u32,
    mem_files: BTreeMap<String, String>,
    // Live async queues & scheduled injections
    input_queues: [VecDeque<u32>; 3],
    scheduled: Vec<ScheduledInj>,
    // History & SSE subscribers
    history: VecDeque<String>,
    subscribers: Vec<mpsc::Sender<String>>,
    cur_tick: u32,
}

impl Shared {
    fn broadcast(&mut self, json_msg: String) {
        self.subscribers.retain(|tx| tx.send(json_msg.clone()).is_ok());
    }

    fn has_pending_work(&self) -> bool {
        self.input_queues.iter().any(|q| !q.is_empty()) || !self.scheduled.is_empty()
    }
}

/// Discover all `.imsa` adapters alongside `primary` (putting `primary` first).
pub fn discover_adapters(primary: &str) -> Vec<AdapterEntry> {
    let mut out: Vec<AdapterEntry> = Vec::new();
    let p = Path::new(primary);
    let canon = std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let primary_str = canon.to_string_lossy().to_string();
    if canon.exists() {
        let name = canon.file_stem().and_then(|s| s.to_str()).unwrap_or("adapter").to_string();
        out.push(AdapterEntry { name, path: primary_str.clone() });
    }
    if let Some(dir) = canon.parent() {
        if let Ok(rd) = std::fs::read_dir(dir) {
            let mut others = Vec::new();
            for entry in rd.flatten() {
                let ep = entry.path();
                if ep.extension().and_then(|e| e.to_str()) == Some("imsa") {
                    let s = ep.to_string_lossy().to_string();
                    if s != primary_str {
                        let n = ep.file_stem().and_then(|x| x.to_str()).unwrap_or("adapter").to_string();
                        others.push(AdapterEntry { name: n, path: s });
                    }
                }
            }
            others.sort_by(|a, b| a.name.cmp(&b.name));
            out.extend(others);
        }
    }
    out
}

fn encode_input_message(
    tk: &Tokenizer,
    ch: usize,
    text: &str,
    eom: bool,
    eom_id: u32,
    native_framing: bool,
    tool_reply_name: Option<&str>,
) -> Vec<u32> {
    if !native_framing {
        let mut toks = tk.encode(text, true);
        if eom {
            toks.push(eom_id);
        }
        return toks;
    }
    let framed = match ch {
        CH_USER_IN => format!("<|message_user|><|content_text|>{text}"),
        CH_AGENT_IN => {
            if let Some(tname) = tool_reply_name {
                format!("<|message_tool|>{tname}<|content_text|>{text}")
            } else {
                format!("<|message_user|><|content_text|>[Peer Agent]: {text}")
            }
        }
        CH_TOOL_IN => {
            let tname = tool_reply_name.unwrap_or("calc");
            format!("<|message_tool|>{tname}<|content_text|>{text}")
        }
        _ => format!("<|message_user|><|content_text|>{text}"),
    };
    let mut toks = tk.encode(&framed, true);
    if eom {
        toks.push(eom_id);
    }
    toks
}

fn decode_token_utf8(tk: &Tokenizer, tok: u32, sil: u32, pending: &mut Vec<u8>) -> String {
    if tok == sil {
        return String::new();
    }
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

fn format_tok_json(id: Option<u32>, sil: u32, txt: &str) -> String {
    match id {
        None => "{\"sil\":true}".to_string(),
        Some(t) if t == sil => format!("{{\"id\":{t},\"sil\":true,\"txt\":\"\"}}"),
        Some(t) => format!("{{\"id\":{t},\"sil\":false,\"txt\":{}}}", json_escape(txt)),
    }
}

fn format_tick_json(
    out: &TickOut,
    display_tick: u32,
    in_txts: &[String; 3],
    out_txts: &[String; 4],
    sil: u32,
    running: bool,
    mem_mib: f64,
) -> String {
    format!(
        "{{\"type\":\"tick\",\"tick\":{},\"n_kv\":{},\"m\":{},\"ms\":{:.2},\"running\":{},\"mem_mib\":{:.1},\"ins\":[{},{},{}],\"outs\":[{},{},{},{}],\"sil_p\":[{:.3},{:.3},{:.3},{:.3}]}}",
        display_tick,
        out.n_kv,
        out.active_m,
        out.elapsed_ms,
        running,
        mem_mib,
        format_tok_json(out.inputs[0], sil, &in_txts[0]),
        format_tok_json(out.inputs[1], sil, &in_txts[1]),
        format_tok_json(out.inputs[2], sil, &in_txts[2]),
        format_tok_json(Some(out.emitted[0]), sil, &out_txts[0]),
        format_tok_json(Some(out.emitted[1]), sil, &out_txts[1]),
        format_tok_json(Some(out.emitted[2]), sil, &out_txts[2]),
        format_tok_json(Some(out.emitted[3]), sil, &out_txts[3]),
        out.sil_probs[0],
        out.sil_probs[1],
        out.sil_probs[2],
        out.sil_probs[3],
    )
}

// ----------------------------------------------------------------------------- Safe Arithmetic Evaluator + Auto-Responders
fn eval_arith(expr: &str) -> Option<f64> {
    let bytes: Vec<u8> = expr.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let mut pos = 0usize;
    let val = parse_expr(&bytes, &mut pos)?;
    if pos == bytes.len() && val.is_finite() { Some(val) } else { None }
}

fn parse_expr(b: &[u8], p: &mut usize) -> Option<f64> {
    let mut lhs = parse_term(b, p)?;
    while *p < b.len() && (b[*p] == b'+' || b[*p] == b'-') {
        let op = b[*p];
        *p += 1;
        let rhs = parse_term(b, p)?;
        lhs = if op == b'+' { lhs + rhs } else { lhs - rhs };
    }
    Some(lhs)
}

fn parse_term(b: &[u8], p: &mut usize) -> Option<f64> {
    let mut lhs = parse_factor(b, p)?;
    while *p < b.len() && (b[*p] == b'*' || b[*p] == b'/') {
        let op = b[*p];
        *p += 1;
        let rhs = parse_factor(b, p)?;
        if op == b'*' {
            lhs *= rhs;
        } else {
            if rhs == 0.0 { return None; }
            lhs /= rhs;
        }
    }
    Some(lhs)
}

fn parse_factor(b: &[u8], p: &mut usize) -> Option<f64> {
    if *p < b.len() && b[*p] == b'-' {
        *p += 1;
        return Some(-parse_factor(b, p)?);
    }
    if *p < b.len() && b[*p] == b'+' {
        *p += 1;
        return parse_factor(b, p);
    }
    if *p < b.len() && b[*p] == b'(' {
        *p += 1;
        let v = parse_expr(b, p)?;
        if *p < b.len() && b[*p] == b')' {
            *p += 1;
            return Some(v);
        }
        return None;
    }
    let start = *p;
    while *p < b.len() && (b[*p].is_ascii_digit() || b[*p] == b'.') {
        *p += 1;
    }
    if *p == start { return None; }
    std::str::from_utf8(&b[start..*p]).ok()?.parse::<f64>().ok()
}

fn strip_to_json(raw: &str) -> &str {
    if let Some(idx) = raw.find('{') {
        &raw[idx..]
    } else {
        raw.trim()
    }
}

fn resolve_peer_reply(request: &str, peer_kv: &str) -> Option<String> {
    let raw = request.trim();
    if raw.contains("<|content_text|>") || raw.contains("<|content_thinking|>") {
        return None;
    }
    let mut query = raw.to_string();
    if let Ok(jv) = parse_json(strip_to_json(&query)) {
        if let Some(name) = jv.get("name").and_then(|v| v.as_str()) {
            if !name.is_empty() && name != "message_peer" {
                return None;
            }
        }
        if let Some(m) = jv
            .get("args")
            .and_then(|a| a.get("message"))
            .and_then(|v| v.as_str())
            .or_else(|| jv.get("arguments").and_then(|a| a.get("message")).and_then(|v| v.as_str()))
            .or_else(|| jv.get("message").and_then(|v| v.as_str()))
        {
            query = m.to_string();
        }
    }
    let req_lower = query.to_ascii_lowercase();
    for pair in peer_kv.split(',') {
        let pair = pair.trim();
        if let Some((k, v)) = pair.split_once('=') {
            let k_t = k.trim();
            let v_t = v.trim();
            if !k_t.is_empty() && req_lower.contains(&k_t.to_ascii_lowercase()) {
                return Some(format!("{k_t}={v_t}"));
            }
        }
    }
    if let Some(idx) = req_lower.find("code_") {
        let rest = &query[idx + 5..];
        let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() {
            let n: u32 = digits.parse().unwrap_or(1);
            let val = (n * 43 + 57) % 900 + 100;
            return Some(format!("code_{digits}={val}"));
        }
    }
    Some(format!("Peer ACK: received '{}'", query.trim().chars().take(60).collect::<String>()))
}

fn default_mem_files() -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert("/memory/profile.md".into(), "# User Profile\nHome city: Portland\nOffice: Downtown Portland\nPreferred tone: concise and practical.".into());
    m.insert("/memory/diet.md".into(), "# Dietary Restrictions\nSTRICT ALLERGY: peanuts (must always avoid!).\nLikes: Italian and Mediterranean food.".into());
    m.insert("/memory/contacts.md".into(), "# Friends & Family\n- Sarah: birthday next Tuesday. Loves ceramics pottery class. Favorite cuisine: Italian.\n- Mark: coworker, free after 6pm Tuesday.".into());
    m.insert("/memory/budget.md".into(), "# Personal Budget Rules\nMax spend for friend birthday dinner + gift combined: $120.".into());
    m.insert("/memory/calendar.md".into(), "# Calendar\n- Mon: Dentist 3pm\n- Tue: Team standup 10am, free evening after 5:30pm\n- Wed: Flight 8am".into());
    m.insert("/memory/projects.md".into(), "# Projects\nWorking on Inkling 2 Mini Multi-Stream inference in Rust + Metal.".into());
    m
}

fn resolve_tool_reply(call_text: &str, mem_files: &mut BTreeMap<String, String>) -> Option<(String, String)> {
    let trimmed = strip_to_json(call_text);
    if let Ok(jv) = parse_json(trimmed) {
        let name = jv.get("name").and_then(|v| v.as_str()).unwrap_or("");
        let args = jv.get("args").or_else(|| jv.get("arguments")).unwrap_or(&jv);
        match name {
            "message_peer" => return None,
            "fs_search" => {
                let q = args.get("query").and_then(|v| v.as_str()).unwrap_or("").trim().to_ascii_lowercase();
                if q.is_empty() || q == "*" || q == "all" || q == "ls" || q == "/" {
                    let ks: Vec<String> = mem_files.keys().map(|k| json_escape(k)).collect();
                    return Some(("fs_search".into(), format!("{{\"files\":[{}]}}", ks.join(","))));
                }
                let terms: Vec<&str> = q.split(|c: char| c.is_whitespace() || c == ',' || c == '/').filter(|w| w.len() >= 2).collect();
                let mut hits = Vec::new();
                for (path, content) in mem_files.iter() {
                    let low = format!("{path}\n{content}").to_ascii_lowercase();
                    if low.contains(&q) || (!terms.is_empty() && terms.iter().any(|t| low.contains(t))) {
                        let snip: String = content.lines().filter(|ln| {
                            let l = ln.to_ascii_lowercase();
                            l.contains(&q) || terms.iter().any(|t| l.contains(t))
                        }).take(3).collect::<Vec<_>>().join(" | ");
                        let snip: String = if snip.is_empty() {
                            content.chars().take(120).collect()
                        } else {
                            snip.chars().take(160).collect()
                        };
                        hits.push(format!("{{\"path\":{},\"snippet\":{}}}", json_escape(path), json_escape(&snip)));
                    }
                }
                if hits.is_empty() {
                    let ks: Vec<String> = mem_files.keys().map(|k| json_escape(k)).collect();
                    return Some(("fs_search".into(), format!("{{\"matches\":[],\"available_files\":[{}]}}", ks.join(","))));
                }
                return Some(("fs_search".into(), format!("{{\"matches\":[{}]}}", hits.join(","))));
            }
            "fs_read" => {
                let p = args.get("path").and_then(|v| v.as_str()).unwrap_or("").trim();
                if let Some(c) = mem_files.get(p) {
                    return Some(("fs_read".into(), c.clone()));
                }
                let pnorm = p.trim_start_matches('/');
                for (k, v) in mem_files.iter() {
                    if k.trim_start_matches('/') == pnorm || k.ends_with(&format!("/{pnorm}")) {
                        return Some(("fs_read".into(), v.clone()));
                    }
                }
                let ks: Vec<String> = mem_files.keys().map(|k| json_escape(k)).collect();
                return Some(("fs_read".into(), format!("{{\"error\":\"file not found\",\"available_files\":[{}]}}", ks.join(","))));
            }
            "env_action" => {
                let act = args.get("action").and_then(|v| v.as_str()).unwrap_or("").trim().to_ascii_lowercase();
                let arg = args.get("arg").and_then(|v| v.as_str()).unwrap_or("").trim();
                if act == "write_memory" {
                    if let Some((path, body)) = arg.split_once('|') {
                        mem_files.insert(path.trim().to_string(), body.trim().to_string());
                        return Some(("env_action".into(), format!("{{\"saved\":{}}}", json_escape(path.trim()))));
                    }
                    mem_files.entry("/memory/profile.md".into()).or_default().push_str(&format!("\n{arg}"));
                    return Some(("env_action".into(), "{\"appended\":\"/memory/profile.md\"}".into()));
                }
                return Some(("env_action".into(), format!("{{\"ack\":{},\"arg\":{}}}", json_escape(&act), json_escape(arg))));
            }
            "calc" | "" => {
                if let Some(expr) = args.get("expr").and_then(|v| v.as_str()) {
                    if let Some(val) = eval_arith(expr) {
                        if (val - val.round()).abs() < 1e-9 {
                            return Some(("calc".into(), format!("{{\"result\":{}}}", val.round() as i64)));
                        }
                        return Some(("calc".into(), format!("{{\"result\":{val:.6}}}")));
                    }
                }
            }
            _ => return None,
        }
    }
    if let Some(val) = eval_arith(trimmed) {
        if (val - val.round()).abs() < 1e-9 {
            return Some(("calc".into(), format!("{{\"result\":{}}}", val.round() as i64)));
        }
        return Some(("calc".into(), format!("{{\"result\":{val:.6}}}")));
    }
    None
}

// ----------------------------------------------------------------------------- HTTP Server
fn write_http(stream: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) {
    let hdr = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let _ = stream.write_all(hdr.as_bytes());
    let _ = stream.write_all(body);
    let _ = stream.flush();
}

fn handle_client(
    mut stream: TcpStream,
    pair: Arc<(Mutex<Shared>, Condvar)>,
    tk: Arc<Tokenizer>,
    eom_id: u32,
) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut req_line = String::new();
    if reader.read_line(&mut req_line).unwrap_or(0) == 0 {
        return;
    }
    let parts: Vec<&str> = req_line.split_whitespace().collect();
    if parts.len() < 2 {
        return;
    }
    let method = parts[0];
    let path = parts[1].split('?').next().unwrap_or(parts[1]);

    let mut content_len = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let l = line.trim();
        if l.is_empty() {
            break;
        }
        if let Some((k, v)) = l.split_once(':') {
            if k.eq_ignore_ascii_case("content-length") {
                content_len = v.trim().parse().unwrap_or(0).min(1 << 20);
            }
        }
    }
    let mut body = vec![0u8; content_len];
    if content_len > 0 {
        let _ = reader.read_exact(&mut body);
    }
    let body_str = String::from_utf8_lossy(&body).to_string();

    match (method, path) {
        ("GET", "/") | ("GET", "/index.html") => {
            write_http(&mut stream, "200 OK", "text/html; charset=utf-8", UI_HTML.as_bytes());
        }
        ("GET", "/api/state") => {
            let st = pair.0.lock().unwrap();
            let adapters_json: Vec<String> = st
                .adapters
                .iter()
                .map(|a| format!("{{\"name\":{},\"path\":{}}}", json_escape(&a.name), json_escape(&a.path)))
                .collect();
            let hist_json: Vec<&str> = st.history.iter().map(|s| s.as_str()).collect();
            let json = format!(
                "{{\"active_adapter\":{},\"native_framing\":{},\"adapters\":[{}],\"running\":{},\"tick\":{},\"mem_mib\":{:.1},\"temp\":{},\"top_k\":{},\"top_p\":{},\"silence_penalty\":{},\"tick_delay_ms\":{},\"auto_pause_idle\":{},\"auto_peer\":{},\"peer_latency\":{},\"peer_kv\":{},\"auto_tool\":{},\"tool_latency\":{},\"history\":[{}]}}",
                json_escape(&st.active_adapter),
                st.native_framing,
                adapters_json.join(","),
                st.running,
                st.cur_tick,
                st.mem_mib,
                st.temp,
                st.top_k,
                st.top_p,
                st.silence_penalty,
                st.tick_delay_ms,
                st.auto_pause_idle,
                st.auto_peer,
                st.peer_latency,
                json_escape(&st.peer_kv),
                st.auto_tool,
                st.tool_latency,
                hist_json.join(",")
            );
            write_http(&mut stream, "200 OK", "application/json", json.as_bytes());
        }
        ("GET", "/api/events") => {
            let _ = stream.set_read_timeout(None);
            let _ = stream.set_write_timeout(Some(Duration::from_secs(5)));
            let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";
            if stream.write_all(hdr.as_bytes()).is_err() {
                return;
            }
            let _ = stream.flush();
            let (tx, rx) = mpsc::channel::<String>();
            {
                let mut st = pair.0.lock().unwrap();
                st.subscribers.push(tx);
            }
            loop {
                match rx.recv_timeout(Duration::from_secs(15)) {
                    Ok(msg) => {
                        let chunk = format!("data: {msg}\n\n");
                        if stream.write_all(chunk.as_bytes()).is_err() || stream.flush().is_err() {
                            break;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        if stream.write_all(b": ping\n\n").is_err() || stream.flush().is_err() {
                            break;
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        }
        ("POST", "/api/send") => {
            let Ok(jv) = parse_json(&body_str) else {
                write_http(&mut stream, "400 Bad Request", "application/json", b"{\"error\":\"bad json\"}");
                return;
            };
            let ch = jv.get("channel").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let text = jv.get("text").and_then(|v| v.as_str()).unwrap_or("");
            let eom = jv.get("eom").and_then(|v| v.as_bool()).unwrap_or(true);
            let auto_start = jv.get("auto_start").and_then(|v| v.as_bool()).unwrap_or(true);
            let delay = jv.get("delay_ticks").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            if ch >= 3 {
                write_http(&mut stream, "400 Bad Request", "application/json", b"{\"error\":\"channel must be 0..2\"}");
                return;
            }
            let ntok = {
                let mut st = pair.0.lock().unwrap();
                let toks = encode_input_message(&tk, ch, text, eom, eom_id, st.native_framing, None);
                let n = toks.len();
                if delay > 0 {
                    let at = st.cur_tick + delay;
                    st.scheduled.push(ScheduledInj { at_tick: at, ch, toks });
                } else {
                    for t in toks {
                        st.input_queues[ch].push_back(t);
                    }
                }
                if auto_start {
                    st.running = true;
                }
                pair.1.notify_all();
                n
            };
            let resp = format!("{{\"ok\":true,\"queued_tokens\":{ntok}}}");
            write_http(&mut stream, "200 OK", "application/json", resp.as_bytes());
        }
        ("POST", "/api/control") => {
            let Ok(jv) = parse_json(&body_str) else {
                write_http(&mut stream, "400 Bad Request", "application/json", b"{\"error\":\"bad json\"}");
                return;
            };
            let action = jv.get("action").and_then(|v| v.as_str()).unwrap_or("");
            let n = jv.get("n").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
            {
                let mut st = pair.0.lock().unwrap();
                match action {
                    "start" => st.running = true,
                    "pause" => {
                        st.running = false;
                        st.step_budget = 0;
                        let m = st.mem_mib;
                        st.broadcast(format!("{{\"type\":\"status\",\"running\":false,\"mem_mib\":{m:.1}}}"));
                    }
                    "step" => {
                        st.running = false;
                        st.step_budget += n.clamp(1, 256);
                    }
                    "reset" => {
                        st.running = false;
                        st.step_budget = 0;
                        st.input_queues = [VecDeque::new(), VecDeque::new(), VecDeque::new()];
                        st.scheduled.clear();
                        st.pending_reset = true;
                        st.pending_note = Some("Session reset".into());
                    }
                    _ => {}
                }
                pair.1.notify_all();
            }
            write_http(&mut stream, "200 OK", "application/json", b"{\"ok\":true}");
        }
        ("POST", "/api/config") => {
            if let Ok(jv) = parse_json(&body_str) {
                let mut st = pair.0.lock().unwrap();
                if let Some(v) = jv.get("temp").and_then(|v| v.as_f64()) {
                    st.temp = (v as f32).clamp(0.0, 2.0);
                }
                if let Some(v) = jv.get("top_k").and_then(|v| v.as_u64()) {
                    st.top_k = (v as usize).clamp(1, 128);
                }
                if let Some(v) = jv.get("top_p").and_then(|v| v.as_f64()) {
                    st.top_p = (v as f32).clamp(0.05, 1.0);
                }
                if let Some(v) = jv.get("silence_penalty").and_then(|v| v.as_f64()) {
                    st.silence_penalty = (v as f32).clamp(-5.0, 5.0);
                }
                if let Some(v) = jv.get("tick_delay_ms").and_then(|v| v.as_u64()) {
                    st.tick_delay_ms = v.min(1000);
                }
                if let Some(v) = jv.get("auto_pause_idle").and_then(|v| v.as_bool()) {
                    st.auto_pause_idle = v;
                }
                if let Some(v) = jv.get("auto_peer").and_then(|v| v.as_bool()) {
                    st.auto_peer = v;
                }
                if let Some(v) = jv.get("peer_latency").and_then(|v| v.as_u64()) {
                    st.peer_latency = (v as u32).clamp(1, 64);
                }
                if let Some(v) = jv.get("peer_kv").and_then(|v| v.as_str()) {
                    st.peer_kv = v.to_string();
                }
                if let Some(v) = jv.get("auto_tool").and_then(|v| v.as_bool()) {
                    st.auto_tool = v;
                }
                if let Some(v) = jv.get("tool_latency").and_then(|v| v.as_u64()) {
                    st.tool_latency = (v as u32).clamp(1, 64);
                }
            }
            write_http(&mut stream, "200 OK", "application/json", b"{\"ok\":true}");
        }
        ("POST", "/api/adapter") => {
            if let Ok(jv) = parse_json(&body_str) {
                if let Some(p) = jv.get("path").and_then(|v| v.as_str()) {
                    let mut st = pair.0.lock().unwrap();
                    st.running = false;
                    st.step_budget = 0;
                    st.pending_adapter = Some(p.to_string());
                    pair.1.notify_all();
                }
            }
            write_http(&mut stream, "200 OK", "application/json", b"{\"ok\":true}");
        }
        ("POST", "/api/preset") => {
            let Ok(jv) = parse_json(&body_str) else {
                write_http(&mut stream, "400 Bad Request", "application/json", b"{\"error\":\"bad json\"}");
                return;
            };
            let name = jv.get("name").and_then(|v| v.as_str()).unwrap_or("math");
            let mut st = pair.0.lock().unwrap();
            st.pending_reset = true;
            st.running = true;
            st.step_budget = 0;
            st.input_queues = [VecDeque::new(), VecDeque::new(), VecDeque::new()];
            st.scheduled.clear();
            let nf = st.native_framing;
            let enc_u = |s: &str| encode_input_message(&tk, CH_USER_IN, s, true, eom_id, nf, None);

            match name {
                "math" => {
                    let prompt = "Compute 18 * 7 + 27.";
                    st.silence_penalty = 0.0;
                    for t in enc_u(prompt) { st.input_queues[CH_USER_IN].push_back(t); }
                    st.pending_note = Some(format!("Preset [Streaming Math]: {prompt}"));
                }
                "interrupt" => {
                    let p1 = if nf { "Add 28 and 11 and explain the steps." } else { "Add 28 and 11." };
                    let p2 = "Wait, subtract the second from the first instead.";
                    st.silence_penalty = 0.0;
                    let int_t = if nf { 22 } else { 18 };
                    for t in enc_u(p1) { st.input_queues[CH_USER_IN].push_back(t); }
                    st.scheduled.push(ScheduledInj { at_tick: int_t, ch: CH_USER_IN, toks: enc_u(p2) });
                    st.pending_note = Some(format!("Preset [Mid-Stream Interrupt]: '{p1}' -> tick {int_t}: '{p2}'"));
                }
                "peer" => {
                    let prompt = if nf {
                        "Check with the peer agent for code_42 and tell me what it is."
                    } else {
                        "Ask the peer agent for code_18 and report it to me."
                    };
                    st.auto_peer = true;
                    st.silence_penalty = if nf { 0.0 } else { 0.5 };
                    for t in enc_u(prompt) { st.input_queues[CH_USER_IN].push_back(t); }
                    st.pending_note = Some(format!("Preset [Async Peer Round-Trip]: {prompt}"));
                }
                "tool" => {
                    let prompt = "Calculate 412 * 37 + 89 using calc.";
                    st.auto_tool = true;
                    st.silence_penalty = 0.0;
                    for t in enc_u(prompt) { st.input_queues[CH_USER_IN].push_back(t); }
                    st.pending_note = Some(format!("Preset [Async Tool Calc]: {prompt}"));
                }
                "chat" => {
                    let prompt = "Explain why the sky is blue in two clear sentences.";
                    st.silence_penalty = 0.0;
                    for t in enc_u(prompt) { st.input_queues[CH_USER_IN].push_back(t); }
                    st.pending_note = Some(format!("Preset [Concurrent Chat]: {prompt}"));
                }
                "code" => {
                    let prompt = "Write a short Python function to check if a number is prime.";
                    st.silence_penalty = 0.0;
                    for t in enc_u(prompt) { st.input_queues[CH_USER_IN].push_back(t); }
                    st.pending_note = Some(format!("Preset [Concurrent Python]: {prompt}"));
                }
                "duplex" => {
                    let p1 = "What do you think of badgers? Do you think that's a good animal?";
                    let p2 = "oh and birds generally";
                    st.silence_penalty = 0.0;
                    for t in enc_u(p1) { st.input_queues[CH_USER_IN].push_back(t); }
                    st.scheduled.push(ScheduledInj { at_tick: 24, ch: CH_USER_IN, toks: enc_u(p2) });
                    st.pending_note = Some(format!("Preset [Duplex Interject]: '{p1}' -> tick 24: '{p2}'"));
                }
                "fs_memory" => {
                    let p1 = "Hey! Help me plan a birthday dinner and gift idea for Sarah next Tuesday. Check my /memory files first so you respect my diet, budget, and what Sarah likes.";
                    let p2 = "Oh and also check in my notes whether Mark is free to join us on Tuesday!";
                    st.auto_tool = true;
                    st.silence_penalty = 0.0;
                    st.mem_files = default_mem_files();
                    for t in enc_u(p1) { st.input_queues[CH_USER_IN].push_back(t); }
                    st.scheduled.push(ScheduledInj { at_tick: 38, ch: CH_USER_IN, toks: enc_u(p2) });
                    st.pending_note = Some("Preset [FS Memory Chat]: Searching /memory/*.md + live interjection".into());
                }
                _ => {
                    let prompt = "Hi";
                    st.silence_penalty = 0.0;
                    for t in enc_u(prompt) { st.input_queues[CH_USER_IN].push_back(t); }
                }
            }
            pair.1.notify_all();
            write_http(&mut stream, "200 OK", "application/json", b"{\"ok\":true}");
        }
        _ => {
            write_http(&mut stream, "404 Not Found", "text/plain", b"not found");
        }
    }
}

// ----------------------------------------------------------------------------- Main Server Loop (100% True Parallel MsEngine!)
pub fn run_server(
    eng: &Engine,
    tk: Tokenizer,
    adapter_path: &str,
    port: u16,
    max_kv: usize,
    init_temp: f32,
    seed: u64,
) {
    let adapters = discover_adapters(adapter_path);
    assert!(!adapters.is_empty(), "no .imsa adapter found near {adapter_path}");
    let active_path = adapters[0].path.clone();
    let adapter = MsAdapter::load(eng, &active_path).expect("load .imsa adapter");
    let eom_id = adapter.eom_id;
    let native_framing = adapter.native_framing;
    let preamble_str = adapter.preamble.clone();

    let mut ms = MsEngine::new(eng, adapter, max_kv);
    if native_framing && !preamble_str.is_empty() {
        let ptoks = tk.encode(&preamble_str, true);
        ms.warm_preamble(&ptoks);
    }
    let mut base_tick_offset = ms.cur_tick;
    let mem_mib = ms.mem_bytes() as f64 / (1 << 20) as f64;

    let shared = Arc::new((
        Mutex::new(Shared {
            adapters,
            active_adapter: active_path,
            native_framing,
            pending_adapter: None,
            running: false,
            step_budget: 0,
            pending_reset: false,
            pending_note: None,
            mem_mib,
            temp: init_temp,
            top_k: 64,
            top_p: 0.95,
            silence_penalty: 0.0,
            tick_delay_ms: 0,
            auto_pause_idle: true,
            auto_peer: true,
            peer_latency: 5,
            peer_kv: "code_18=831, code_42=519, status=ready".to_string(),
            auto_tool: true,
            tool_latency: 5,
            mem_files: default_mem_files(),
            input_queues: [VecDeque::new(), VecDeque::new(), VecDeque::new()],
            scheduled: Vec::new(),
            history: VecDeque::with_capacity(MAX_HISTORY),
            subscribers: Vec::new(),
            cur_tick: 0,
        }),
        Condvar::new(),
    ));

    let tk = Arc::new(tk);
    let addr = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| panic!("bind {addr}: {e}"));

    eprintln!("[webui] Inkling 2 Mini 7-Channel Multi-Stream UI ready at http://{addr}/");
    eprintln!("[webui] Adapter: {} (native_framing={}) | GPU working: {:.1} MiB (+ mmapped GGUF)",
        ms.adapter.name, native_framing, mem_mib);
    eprintln!("[webui] Press Ctrl-C to stop.");

    let http_pair = Arc::clone(&shared);
    let http_tk = Arc::clone(&tk);
    std::thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            let p = Arc::clone(&http_pair);
            let t = Arc::clone(&http_tk);
            std::thread::spawn(move || handle_client(conn, p, t, eom_id));
        }
    });

    let mut sampler = MsSamplerCfg::new(init_temp, 64, 0.95, 0.0, seed);
    let mut utf8_pending: [Vec<u8>; 7] = Default::default();
    let mut peer_out_buf: Vec<u32> = Vec::new();
    let mut tool_out_buf: Vec<u32> = Vec::new();
    let mut consecutive_idle = 0usize;

    loop {
        let (do_step, delay_ms) = {
            let (lock, cvar) = &*shared;
            let mut st = lock.lock().unwrap();
            loop {
                if st.pending_adapter.is_some() || st.pending_reset {
                    break (false, 0);
                }
                if st.running || st.step_budget > 0 {
                    if st.step_budget > 0 {
                        st.step_budget -= 1;
                    }
                    let d = st.tick_delay_ms;
                    sampler.temp = st.temp;
                    sampler.top_k = st.top_k;
                    sampler.top_p = st.top_p;
                    sampler.silence_penalty = st.silence_penalty;
                    break (true, d);
                }
                st = cvar.wait(st).unwrap();
            }
        };

        let swap_path = {
            let mut st = shared.0.lock().unwrap();
            st.pending_adapter.take()
        };
        if let Some(new_path) = swap_path {
            match MsAdapter::load(eng, &new_path) {
                Ok(new_ad) => {
                    let nf = new_ad.native_framing;
                    let pre = new_ad.preamble.clone();
                    let aname = new_ad.name.clone();
                    ms.swap_adapter(new_ad);
                    if nf && !pre.is_empty() {
                        let ptoks = tk.encode(&pre, true);
                        ms.warm_preamble(&ptoks);
                    }
                    base_tick_offset = ms.cur_tick;
                    for p in utf8_pending.iter_mut() { p.clear(); }
                    peer_out_buf.clear();
                    tool_out_buf.clear();
                    consecutive_idle = 0;
                    let mem = ms.mem_bytes() as f64 / (1 << 20) as f64;
                    let mut st = shared.0.lock().unwrap();
                    st.active_adapter = new_path.clone();
                    st.native_framing = nf;
                    st.cur_tick = 0;
                    st.mem_mib = mem;
                    st.history.clear();
                    st.input_queues = [VecDeque::new(), VecDeque::new(), VecDeque::new()];
                    st.scheduled.clear();
                    let note = format!("Switched adapter to {aname} (native_framing={nf})");
                    let sil_pen = st.silence_penalty;
                    st.broadcast(format!(
                        "{{\"type\":\"reset\",\"running\":false,\"mem_mib\":{:.1},\"silence_penalty\":{:.2},\"note\":{}}}",
                        mem,
                        sil_pen,
                        json_escape(&note)
                    ));
                }
                Err(e) => eprintln!("[webui] adapter load error: {e}"),
            }
            continue;
        }

        let do_reset = {
            let mut st = shared.0.lock().unwrap();
            if st.pending_reset {
                st.pending_reset = false;
                true
            } else {
                false
            }
        };
        if do_reset {
            ms.reset();
            base_tick_offset = ms.cur_tick;
            for p in utf8_pending.iter_mut() { p.clear(); }
            peer_out_buf.clear();
            tool_out_buf.clear();
            consecutive_idle = 0;
            let mem = ms.mem_bytes() as f64 / (1 << 20) as f64;
            let mut st = shared.0.lock().unwrap();
            st.cur_tick = 0;
            st.mem_mib = mem;
            st.history.clear();
            let note = st.pending_note.take().unwrap_or_else(|| "Session reset".into());
            let run_flag = st.running;
            let sil_pen = st.silence_penalty;
            st.broadcast(format!(
                "{{\"type\":\"reset\",\"running\":{},\"mem_mib\":{:.1},\"silence_penalty\":{:.2},\"note\":{}}}",
                run_flag,
                mem,
                sil_pen,
                json_escape(&note)
            ));
            continue;
        }

        if !do_step {
            continue;
        }
        if ms.kv_near_full() {
            let mut st = shared.0.lock().unwrap();
            st.running = false;
            st.step_budget = 0;
            let m = st.mem_mib;
            st.broadcast(format!("{{\"type\":\"status\",\"running\":false,\"mem_mib\":{m:.1}}}"));
            eprintln!("[webui] KV cache reached max_kv ({}) — auto-paused; click Reset to continue.", ms.kv_ticks.len());
            continue;
        }

        let disp_tick = ms.cur_tick.saturating_sub(base_tick_offset);
        let inputs: [Option<u32>; 3] = {
            let mut st = shared.0.lock().unwrap();
            let mut remaining = Vec::new();
            let mut due = Vec::new();
            for inj in st.scheduled.drain(..) {
                if inj.at_tick <= disp_tick { due.push(inj); } else { remaining.push(inj); }
            }
            st.scheduled = remaining;
            for inj in due {
                if inj.ch < 3 {
                    for t in inj.toks { st.input_queues[inj.ch].push_back(t); }
                }
            }
            [
                st.input_queues[0].pop_front(),
                st.input_queues[1].pop_front(),
                st.input_queues[2].pop_front(),
            ]
        };

        let tick_out = ms.step_tick(inputs, &mut sampler);
        let sil = ms.adapter.silence_id;
        let eom = ms.adapter.eom_id;
        let tool_tok = ms.adapter.tool_id;
        let peer_tok = ms.adapter.peer_id;
        let nf = ms.adapter.native_framing;

        let in_txts: [String; 3] = [
            inputs[0].map_or(String::new(), |t| decode_token_utf8(&tk, t, sil, &mut utf8_pending[0])),
            inputs[1].map_or(String::new(), |t| decode_token_utf8(&tk, t, sil, &mut utf8_pending[1])),
            inputs[2].map_or(String::new(), |t| decode_token_utf8(&tk, t, sil, &mut utf8_pending[2])),
        ];
        let out_txts: [String; 4] = [
            decode_token_utf8(&tk, tick_out.emitted[0], sil, &mut utf8_pending[3]),
            decode_token_utf8(&tk, tick_out.emitted[1], sil, &mut utf8_pending[4]),
            decode_token_utf8(&tk, tick_out.emitted[2], sil, &mut utf8_pending[5]),
            decode_token_utf8(&tk, tick_out.emitted[3], sil, &mut utf8_pending[6]),
        ];

        let mut peer_completed: Option<String> = None;
        let p_tok = tick_out.emitted[2];
        if p_tok != sil {
            if p_tok == eom {
                if !peer_out_buf.is_empty() {
                    let bytes: Vec<u8> = peer_out_buf.iter().flat_map(|&t| tk.decode_bytes(t)).collect();
                    peer_completed = Some(String::from_utf8_lossy(&bytes).to_string());
                    peer_out_buf.clear();
                }
            } else if p_tok != peer_tok {
                peer_out_buf.push(p_tok);
            }
        }

        let mut tool_completed: Option<String> = None;
        let t_tok = tick_out.emitted[3];
        if t_tok != sil {
            if t_tok == eom {
                if !tool_out_buf.is_empty() {
                    let bytes: Vec<u8> = tool_out_buf.iter().flat_map(|&t| tk.decode_bytes(t)).collect();
                    tool_completed = Some(String::from_utf8_lossy(&bytes).to_string());
                    tool_out_buf.clear();
                }
            } else if t_tok != tool_tok {
                tool_out_buf.push(t_tok);
            }
        }

        let any_active = inputs.iter().any(|x| x.is_some())
            || tick_out.emitted.iter().any(|&t| t != sil);
        let mem = ms.mem_bytes() as f64 / (1 << 20) as f64;
        let next_disp_tick = ms.cur_tick.saturating_sub(base_tick_offset);
        {
            let mut st = shared.0.lock().unwrap();
            st.cur_tick = next_disp_tick;
            st.mem_mib = mem;

            if let Some(req_txt) = peer_completed {
                if st.auto_peer {
                    if let Some(reply) = resolve_peer_reply(&req_txt, &st.peer_kv) {
                        let rtoks = encode_input_message(&tk, CH_AGENT_IN, &reply, true, eom, nf, Some("message_peer"));
                        let at = next_disp_tick + st.peer_latency;
                        st.scheduled.push(ScheduledInj { at_tick: at, ch: CH_AGENT_IN, toks: rtoks });
                        eprintln!("[auto-peer] '{}' -> '{}' @ tick {at}", req_txt.trim().chars().take(80).collect::<String>(), reply);
                    }
                }
            }
            if let Some(call_txt) = tool_completed {
                if st.auto_tool {
                    if let Some((tname, reply)) = resolve_tool_reply(&call_txt, &mut st.mem_files) {
                        let rtoks = encode_input_message(&tk, CH_TOOL_IN, &reply, true, eom, nf, Some(&tname));
                        let at = next_disp_tick + st.tool_latency;
                        st.scheduled.push(ScheduledInj { at_tick: at, ch: CH_TOOL_IN, toks: rtoks });
                        eprintln!("[auto-tool:{tname}] '{}' -> '{}' @ tick {at}", call_txt.trim(), reply.chars().take(100).collect::<String>());
                    }
                }
            }

            if any_active || st.has_pending_work() {
                consecutive_idle = 0;
            } else {
                consecutive_idle += 1;
                if st.auto_pause_idle && st.running && consecutive_idle >= IDLE_AUTO_PAUSE_TICKS {
                    st.running = false;
                }
            }

            let json = format_tick_json(&tick_out, disp_tick, &in_txts, &out_txts, sil, st.running, mem);
            if st.history.len() >= MAX_HISTORY {
                st.history.pop_front();
            }
            st.history.push_back(json.clone());
            st.broadcast(json);
        }

        if delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(delay_ms));
        }
    }
}
