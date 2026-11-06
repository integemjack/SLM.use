//! An OpenAI-compatible HTTP server. The model is not shared between threads: it stays
//! on the thread that created it, which runs one request at a time in arrival order,
//! and each connection gets its own thread to read the request and send the answer.
//!
//! Each response closes its connection, which is what lets a stream end without chunked
//! encoding.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokenizers::Tokenizer;

use crate::api::{self, Defaults, Meta, Params, StopFilter, Usage};
use crate::model::{Model, Session, CHUNK};
use crate::ops::Ops;
use crate::sample::{pick, Rng};
use crate::text::{encode, encode_chars, Detokenizer};

const MAX_HEAD: usize = 64 * 1024;
const MAX_BODY: usize = 16 * 1024 * 1024;
const READ_TIMEOUT: Duration = Duration::from_secs(30);

pub struct Options {
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub model_id: String,
    pub defaults: Defaults,
    pub loops: Option<usize>,
    pub seed: u64,
}

enum Event {
    Text(String),
    Done { finish: &'static str, usage: Usage },
    Failed(String),
}

struct Job {
    params: Params,
    events: Sender<Event>,
}

/// What every connection thread needs.
struct Shared {
    jobs: Sender<Job>,
    api_key: Option<String>,
    model_id: String,
    defaults: Defaults,
    started: u64,
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

struct Request {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: Vec<u8>,
}

enum ReadError {
    Closed,
    Refused(u16, &'static str),
}

fn read_request(conn: &mut TcpStream) -> Result<Request, ReadError> {
    let mut reader = BufReader::new(conn.try_clone().map_err(|_| ReadError::Closed)?);
    let mut line = String::new();
    let mut head = 0;
    let mut read_line = |line: &mut String| -> Result<(), ReadError> {
        line.clear();
        match reader.read_line(line) {
            Ok(0) | Err(_) => return Err(ReadError::Closed),
            Ok(n) => head += n,
        }
        if head > MAX_HEAD {
            return Err(ReadError::Refused(431, "the request head is too large"));
        }
        Ok(())
    };
    read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (method, target) = match (parts.next(), parts.next()) {
        (Some(m), Some(t)) => (m.to_owned(), t.to_owned()),
        _ => return Err(ReadError::Refused(400, "malformed request line")),
    };
    let path = target.split('?').next().unwrap_or_default().to_owned();
    let mut headers = HashMap::new();
    loop {
        read_line(&mut line)?;
        let text = line.trim_end();
        if text.is_empty() {
            break;
        }
        if let Some((k, v)) = text.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_owned());
        }
    }
    if headers
        .get("transfer-encoding")
        .is_some_and(|t| !t.eq_ignore_ascii_case("identity"))
    {
        return Err(ReadError::Refused(411, "send a Content-Length"));
    }
    let length = headers
        .get("content-length")
        .map_or(Ok(0), |l| l.parse::<usize>());
    let length = length.map_err(|_| ReadError::Refused(400, "invalid Content-Length"))?;
    if length > MAX_BODY {
        return Err(ReadError::Refused(413, "the request body is too large"));
    }
    if headers
        .get("expect")
        .is_some_and(|e| e.eq_ignore_ascii_case("100-continue"))
    {
        conn.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
            .map_err(|_| ReadError::Closed)?;
    }
    let mut body = vec![0; length];
    reader
        .read_exact(&mut body)
        .map_err(|_| ReadError::Closed)?;
    Ok(Request {
        method,
        path,
        headers,
        body,
    })
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Content Too Large",
        431 => "Request Header Fields Too Large",
        _ => "Internal Server Error",
    }
}

fn send_json(conn: &mut TcpStream, status: u16, body: &Value) {
    let text = body.to_string();
    let head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        reason(status),
        text.len()
    );
    let _ = conn
        .write_all(head.as_bytes())
        .and_then(|_| conn.write_all(text.as_bytes()));
}

fn send_error(conn: &mut TcpStream, status: u16, message: &str) {
    let kind = if status == 401 {
        "authentication_error"
    } else {
        "invalid_request_error"
    };
    send_json(conn, status, &api::error(message, kind));
}

fn authorized(req: &Request, key: &Option<String>) -> bool {
    match key {
        None => true,
        Some(k) => req
            .headers
            .get("authorization")
            .is_some_and(|h| h.strip_prefix("Bearer ") == Some(k)),
    }
}

/// Paths with and without the `/v1` prefix both work, since clients differ on it.
fn route(path: &str) -> &str {
    path.strip_prefix("/v1")
        .unwrap_or(path)
        .trim_end_matches('/')
}

fn collect(conn: &mut TcpStream, events: Receiver<Event>, meta: &Meta) {
    let mut text = String::new();
    for event in events {
        match event {
            Event::Text(t) => text.push_str(&t),
            Event::Done { finish, usage } => {
                return send_json(conn, 200, &meta.full(&text, finish, usage))
            }
            Event::Failed(m) => return send_json(conn, 500, &api::error(&m, "server_error")),
        }
    }
    send_json(conn, 500, &api::error("generation stopped", "server_error"));
}

fn chunk(conn: &mut TcpStream, value: &Value) -> std::io::Result<()> {
    conn.write_all(format!("data: {value}\n\n").as_bytes())?;
    conn.flush()
}

/// Sends server-sent events. A failed write means the client left: returning drops the
/// receiver, and that stops generation on the model thread.
fn stream(
    conn: &mut TcpStream,
    events: Receiver<Event>,
    meta: &Meta,
    include_usage: bool,
) -> std::io::Result<()> {
    conn.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
    )?;
    if meta.chat {
        chunk(conn, &meta.delta(Some(""), true, None))?;
    }
    for event in events {
        match event {
            Event::Text(t) => chunk(conn, &meta.delta(Some(&t), false, None))?,
            Event::Done { finish, usage } => {
                chunk(conn, &meta.delta(None, false, Some(finish)))?;
                if include_usage {
                    chunk(conn, &meta.usage_chunk(usage))?;
                }
                break;
            }
            Event::Failed(m) => {
                chunk(conn, &api::error(&m, "server_error"))?;
                break;
            }
        }
    }
    conn.write_all(b"data: [DONE]\n\n")?;
    conn.flush()
}

fn generate(conn: &mut TcpStream, s: &Shared, req: &Request, chat: bool) {
    let body: Value = match serde_json::from_slice(&req.body) {
        Ok(v) => v,
        Err(e) => return send_error(conn, 400, &format!("the body is not valid JSON: {e}")),
    };
    let parsed = if chat {
        api::parse_chat(&body, &s.defaults)
    } else {
        api::parse_completion(&body, &s.defaults)
    };
    let params = match parsed {
        Ok(p) => p,
        Err(m) => return send_error(conn, 400, &m),
    };
    let (is_stream, include_usage) = (params.stream, params.include_usage);
    let meta = Meta {
        id: format!(
            "{}-{:016x}",
            if chat { "chatcmpl" } else { "cmpl" },
            unique()
        ),
        created: now(),
        model: body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(&s.model_id)
            .to_owned(),
        chat,
    };
    let (events, rx) = channel();
    if s.jobs.send(Job { params, events }).is_err() {
        return send_json(
            conn,
            500,
            &api::error("the model thread has stopped", "server_error"),
        );
    }
    if is_stream {
        let _ = stream(conn, rx, &meta, include_usage);
    } else {
        collect(conn, rx, &meta);
    }
}

/// Distinct within a run, which is all a response id needs.
fn unique() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(n, |d| d.as_nanos() as u64 ^ (n << 48))
}

fn handle(mut conn: TcpStream, s: Arc<Shared>) {
    let _ = conn.set_nodelay(true);
    let _ = conn.set_read_timeout(Some(READ_TIMEOUT));
    let req = match read_request(&mut conn) {
        Ok(r) => r,
        Err(ReadError::Closed) => return,
        Err(ReadError::Refused(status, m)) => return send_error(&mut conn, status, m),
    };
    let path = route(&req.path);
    if path == "/health" {
        return send_json(&mut conn, 200, &json!({ "status": "ok" }));
    }
    if !authorized(&req, &s.api_key) {
        return send_error(&mut conn, 401, "invalid API key");
    }
    match (req.method.as_str(), path) {
        ("GET", "/models") => send_json(&mut conn, 200, &api::models(&s.model_id, s.started)),
        ("POST", "/chat/completions") => generate(&mut conn, &s, &req, true),
        ("POST", "/completions") => generate(&mut conn, &s, &req, false),
        (_, "/models" | "/chat/completions" | "/completions") => {
            send_error(&mut conn, 405, "method not allowed")
        }
        _ => send_error(&mut conn, 404, "no such endpoint"),
    }
}

/// A session and the tokens it has read, kept so that a chat can continue where the
/// last answer ended instead of reading the whole history again.
struct Kept<O: Ops> {
    session: Session<O>,
    tokens: Vec<u32>,
}

fn answer<O: Ops>(
    model: &Model<O>,
    tk: &Tokenizer,
    cache: &mut Option<Kept<O>>,
    loops: Option<usize>,
    job: &Job,
    rng: &mut Rng,
) -> Result<()> {
    let p = &job.params;
    let ids = if p.chars {
        encode_chars(tk, &p.prompt)?
    } else {
        encode(tk, &p.prompt)?
    };
    if ids.is_empty() {
        anyhow::bail!("the prompt is empty");
    }
    let chosen = p.think.map(|t| model.cfg.think_loops(t)).or(loops);
    let loops = model.cfg.loops(chosen)?;
    // Every loop has its own state, so another level cannot continue from this one.
    let new_tokens = match cache {
        Some(c)
            if c.session.loops == loops
                && ids.len() > c.tokens.len()
                && ids.starts_with(&c.tokens) =>
        {
            c.tokens.len()
        }
        _ => 0,
    };
    if new_tokens == 0 {
        // Dropped first, so the old state is gone before the new one is allocated.
        *cache = None;
        *cache = Some(Kept {
            session: model.session(Some(loops))?,
            tokens: Vec::new(),
        });
    }
    let Some(c) = cache else {
        unreachable!("the context was just set")
    };
    model.feed(&mut c.session, &ids[new_tokens..], CHUNK)?;
    c.tokens = ids;
    let prompt_tokens = c.tokens.len();

    let eot = tk.token_to_id("<|endoftext|>");
    let mut own_rng = p.seed.map(Rng::new);
    let mut text = Detokenizer::new(tk);
    let mut stops = StopFilter::new(p.stop.clone());
    let mut made = 0;
    let mut finish = "length";
    while made < p.max_tokens {
        let logits = model.logits(&c.session)?;
        let id = pick(
            logits,
            &c.tokens,
            &p.sampling,
            own_rng.as_mut().unwrap_or(rng),
        );
        if Some(id) == eot {
            finish = "stop";
            break;
        }
        c.tokens.push(id);
        model.step(&mut c.session, id)?;
        made += 1;
        let safe = stops.push(&text.push(id)?);
        if !safe.is_empty() && job.events.send(Event::Text(safe)).is_err() {
            return Ok(());
        }
        if stops.hit {
            finish = "stop";
            break;
        }
    }
    let rest = stops.finish();
    if !rest.is_empty() {
        let _ = job.events.send(Event::Text(rest));
    }
    let usage = Usage {
        prompt_tokens,
        completion_tokens: made,
    };
    let _ = job.events.send(Event::Done { finish, usage });
    Ok(())
}

/// Serves until the process ends. Prints the address once it accepts connections.
pub fn run<O: Ops>(model: Model<O>, tk: Tokenizer, o: Options) -> Result<()> {
    let listener = TcpListener::bind((o.host.as_str(), o.port))
        .with_context(|| format!("listening on {}:{}", o.host, o.port))?;
    let addr = listener.local_addr()?;
    let (jobs, queue) = channel();
    let shared = Arc::new(Shared {
        jobs,
        api_key: o.api_key,
        model_id: o.model_id,
        defaults: o.defaults,
        started: now(),
    });
    thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            let s = Arc::clone(&shared);
            thread::spawn(move || handle(conn, s));
        }
    });
    println!("{}, listening on http://{addr}/v1", model.device());
    std::io::stdout().flush()?;

    let mut cache = None;
    let mut rng = Rng::new(o.seed);
    for job in queue {
        if let Err(e) = answer(&model, &tk, &mut cache, o.loops, &job, &mut rng) {
            // The state may be half-written, so the next request starts clean.
            cache = None;
            let _ = job.events.send(Event::Failed(format!("{e:#}")));
        }
    }
    Ok(())
}
