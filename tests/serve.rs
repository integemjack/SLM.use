//! Starts the real binary on the tiny fixture and talks to it over a socket.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::process::{Child, Command, Stdio};

use serde_json::{json, Value};

struct Server {
    child: Child,
    addr: String,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start(extra: &[&str]) -> Server {
    let mut child = Command::new(env!("CARGO_BIN_EXE_slm"))
        .args([
            "--serve",
            "-m",
            "tests/fixtures/tiny",
            "-d",
            "cpu",
            "--port",
            "0",
            "-n",
            "8",
        ])
        .args(extra)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    out.read_line(&mut line).unwrap();
    let addr = line
        .split("http://")
        .nth(1)
        .and_then(|rest| rest.split('/').next())
        .unwrap_or_else(|| panic!("no address in {line:?}"))
        .to_owned();
    Server { child, addr }
}

/// Returns the status and body.
fn call(s: &Server, method: &str, path: &str, body: &str, header: &str) -> (u16, String) {
    let mut conn = TcpStream::connect(&s.addr).unwrap();
    write!(
        conn,
        "{method} {path} HTTP/1.1\r\nHost: test\r\nContent-Length: {}\r\n{header}\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut text = String::new();
    conn.read_to_string(&mut text).unwrap();
    let (head, rest) = text.split_once("\r\n\r\n").unwrap();
    let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    (status, rest.to_owned())
}

fn post(s: &Server, path: &str, body: Value) -> (u16, Value) {
    let (status, text) = call(s, "POST", path, &body.to_string(), "");
    (status, serde_json::from_str(&text).unwrap())
}

fn events(text: &str) -> Vec<&str> {
    text.split("\n\n")
        .filter_map(|e| e.strip_prefix("data: "))
        .collect()
}

#[test]
fn speaks_the_openai_protocol() {
    let s = start(&[]);

    let (status, text) = call(&s, "GET", "/v1/models", "", "");
    assert_eq!(status, 200);
    let models: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(models["data"][0]["id"], "tiny");

    let (status, reply) = post(
        &s,
        "/v1/completions",
        json!({ "prompt": "Hi", "max_tokens": 4 }),
    );
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["object"], "text_completion");
    assert_eq!(reply["usage"]["prompt_tokens"], 2);
    assert_eq!(reply["usage"]["completion_tokens"], 4);
    assert_eq!(reply["choices"][0]["finish_reason"], "length");
    assert!(reply["choices"][0]["text"].is_string());

    let chat = json!({ "model": "any", "messages": [{ "role": "user", "content": "Hi" }], "max_tokens": 3 });
    let (status, reply) = post(&s, "/v1/chat/completions", chat);
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["object"], "chat.completion");
    assert_eq!(reply["model"], "any");
    assert_eq!(reply["choices"][0]["message"]["role"], "assistant");
    assert_eq!(reply["usage"]["completion_tokens"], 3);

    // The max_tokens default comes from -n on the command line.
    let (_, reply) = post(&s, "/v1/completions", json!({ "prompt": "Hi" }));
    assert_eq!(reply["usage"]["completion_tokens"], 8);
}

#[test]
fn a_fixed_seed_repeats_and_streams_the_same_text() {
    let s = start(&[]);
    let body = json!({ "prompt": "Hello", "max_tokens": 6, "seed": 7, "temperature": 1.0 });
    let (_, one) = post(&s, "/v1/completions", body.clone());
    let (_, two) = post(&s, "/v1/completions", body.clone());
    let text = one["choices"][0]["text"].as_str().unwrap();
    assert_eq!(text, two["choices"][0]["text"]);

    let mut body = body;
    body["stream"] = json!(true);
    body["stream_options"] = json!({ "include_usage": true });
    let (status, raw) = call(&s, "POST", "/v1/completions", &body.to_string(), "");
    assert_eq!(status, 200);
    let list = events(&raw);
    assert_eq!(list.last(), Some(&"[DONE]"));
    let chunks: Vec<Value> = list[..list.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap())
        .collect();
    let streamed: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["text"].as_str())
        .collect();
    assert_eq!(streamed, text);
    assert_eq!(chunks.last().unwrap()["usage"]["completion_tokens"], 6);
}

#[test]
fn chat_streams_open_with_the_role() {
    let s = start(&[]);
    let body = json!({ "messages": [{ "role": "user", "content": "Hi" }], "max_tokens": 3, "stream": true });
    let (status, raw) = call(&s, "POST", "/v1/chat/completions", &body.to_string(), "");
    assert_eq!(status, 200);
    let list = events(&raw);
    let first: Value = serde_json::from_str(list[0]).unwrap();
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");
    let last: Value = serde_json::from_str(list[list.len() - 2]).unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "length");
}

#[test]
fn stop_strings_end_the_text() {
    let s = start(&[]);
    let body = json!({ "prompt": "Hello", "max_tokens": 20, "seed": 3, "repeat_penalty": 1.0 });
    let (_, full) = post(&s, "/v1/completions", body.clone());
    let text = full["choices"][0]["text"].as_str().unwrap().to_owned();
    let mut chars = text.chars();
    let (a, b) = (chars.next().unwrap(), chars.next().unwrap());
    let mut body = body;
    body["stop"] = json!(format!("{a}{b}"));
    let (_, cut) = post(&s, "/v1/completions", body);
    assert_eq!(cut["choices"][0]["text"], "");
    assert_eq!(cut["choices"][0]["finish_reason"], "stop");
}

#[test]
fn a_chat_can_continue_from_its_last_answer() {
    let s = start(&[]);
    let first = json!({ "prompt": "Hello", "max_tokens": 4, "seed": 1, "temperature": 0.0 });
    let (_, one) = post(&s, "/v1/completions", first);
    let text = one["choices"][0]["text"].as_str().unwrap().to_owned();
    // The second prompt starts with the first prompt and answer, so it reuses their state.
    let longer = format!("Hello{text} more");
    let body = json!({ "prompt": longer, "max_tokens": 4, "seed": 1, "temperature": 0.0 });
    let (_, two) = post(&s, "/v1/completions", body.clone());
    assert_eq!(two["usage"]["prompt_tokens"], longer.len());
    // A fresh server reads the whole prompt, and must agree.
    let fresh = start(&[]);
    let (_, three) = post(&fresh, "/v1/completions", body);
    assert_eq!(two["choices"][0]["text"], three["choices"][0]["text"]);
}

#[test]
fn reasoning_effort_sets_the_loops_of_one_request() {
    let s = start(&[]);
    let plain = json!({ "prompt": "Hello", "max_tokens": 6, "seed": 1, "temperature": 0.0 });
    let mut high = plain.clone();
    high["reasoning_effort"] = json!("high");

    let (_, deep) = post(&s, "/v1/completions", high);
    let started_high = start(&["--think", "high"]);
    let (_, same) = post(&started_high, "/v1/completions", plain.clone());
    assert_eq!(deep["choices"][0]["text"], same["choices"][0]["text"]);

    // The request above must not leave the server at that level.
    let (_, after) = post(&s, "/v1/completions", plain.clone());
    let (_, fresh) = post(&start(&[]), "/v1/completions", plain);
    assert_eq!(after["choices"][0]["text"], fresh["choices"][0]["text"]);
    assert_ne!(deep["choices"][0]["text"], after["choices"][0]["text"]);

    let bad = json!({ "prompt": "Hello", "reasoning_effort": "huge" });
    assert_eq!(post(&s, "/v1/completions", bad).0, 400);
}

#[test]
fn another_level_does_not_continue_the_last_state() {
    let s = start(&[]);
    let first = json!({ "prompt": "Hello", "max_tokens": 4, "temperature": 0.0, "reasoning_effort": "low" });
    let (_, one) = post(&s, "/v1/completions", first);
    let text = one["choices"][0]["text"].as_str().unwrap().to_owned();
    let longer = json!({
        "prompt": format!("Hello{text} more"), "max_tokens": 4, "temperature": 0.0,
        "reasoning_effort": "max",
    });
    let (status, two) = post(&s, "/v1/completions", longer.clone());
    assert_eq!(status, 200, "{two}");
    let (_, three) = post(&start(&[]), "/v1/completions", longer);
    assert_eq!(two["choices"][0]["text"], three["choices"][0]["text"]);
}

#[test]
fn chars_gives_a_chat_only_the_last_question() {
    let s = start(&[]);
    let body = |messages: Value| json!({ "messages": messages, "max_tokens": 2, "chars": true });
    let (status, reply) = post(
        &s,
        "/v1/chat/completions",
        body(json!([
            { "role": "user", "content": "Hi" },
            { "role": "assistant", "content": "There" },
            { "role": "user", "content": "Bye" },
        ])),
    );
    assert_eq!(status, 200, "{reply}");
    assert_eq!(reply["usage"]["prompt_tokens"], 3);
}

#[test]
fn errors_use_the_openai_shape() {
    let s = start(&[]);
    let (status, reply) = post(&s, "/v1/completions", json!({ "max_tokens": 1 }));
    assert_eq!(status, 400);
    assert!(reply["error"]["message"].is_string());
    let (status, text) = call(&s, "POST", "/v1/completions", "not json", "");
    assert_eq!(status, 400, "{text}");
    assert_eq!(call(&s, "GET", "/v1/nothing", "", "").0, 404);
    assert_eq!(call(&s, "GET", "/v1/completions", "", "").0, 405);
    assert_eq!(call(&s, "GET", "/health", "", "").0, 200);
}

#[test]
fn an_api_key_guards_the_endpoints() {
    let s = start(&["--api-key", "secret"]);
    assert_eq!(call(&s, "GET", "/v1/models", "", "").0, 401);
    assert_eq!(
        call(
            &s,
            "GET",
            "/v1/models",
            "",
            "Authorization: Bearer wrong\r\n"
        )
        .0,
        401
    );
    assert_eq!(
        call(
            &s,
            "GET",
            "/v1/models",
            "",
            "Authorization: Bearer secret\r\n"
        )
        .0,
        200
    );
    assert_eq!(call(&s, "GET", "/health", "", "").0, 200);
}

#[test]
fn a_client_that_leaves_does_not_block_the_next_one() {
    let s = start(&[]);
    let body = json!({ "prompt": "Hi", "max_tokens": 100000, "stream": true }).to_string();
    let mut conn = TcpStream::connect(&s.addr).unwrap();
    write!(
        conn,
        "POST /v1/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut some = [0u8; 256];
    conn.read_exact(&mut some).unwrap();
    drop(conn);
    let (status, _) = post(
        &s,
        "/v1/completions",
        json!({ "prompt": "Hi", "max_tokens": 1 }),
    );
    assert_eq!(status, 200);
}
