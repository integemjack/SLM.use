//! The OpenAI wire format: request parsing and response bodies, free of any I/O.

use clap::ValueEnum;
use serde_json::{json, Value};

use crate::config::Think;
use crate::sample::Sampling;

/// What a request may leave out.
#[derive(Clone, Debug)]
pub struct Defaults {
    pub max_tokens: usize,
    pub sampling: Sampling,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub prompt: String,
    pub max_tokens: usize,
    pub sampling: Sampling,
    pub stop: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
    pub seed: Option<u64>,
    pub think: Option<Think>,
    pub chat: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

impl Usage {
    fn json(&self) -> Value {
        json!({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        })
    }
}

fn number(v: &Value, key: &str) -> Result<Option<f64>, String> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(x) => x
            .as_f64()
            .map(Some)
            .ok_or_else(|| format!("`{key}` must be a number")),
    }
}

fn stops(v: &Value) -> Result<Vec<String>, String> {
    match v.get("stop") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "`stop` must hold strings".to_owned())
            })
            .collect(),
        Some(_) => Err("`stop` must be a string or an array of strings".to_owned()),
    }
}

fn think(v: &Value) -> Result<Option<Think>, String> {
    match v.get("reasoning_effort") {
        None | Some(Value::Null) => Ok(None),
        Some(x) => x
            .as_str()
            .and_then(|s| <Think as ValueEnum>::from_str(s, true).ok())
            .map(Some)
            .ok_or_else(|| "`reasoning_effort` must be low, medium, high or max".to_owned()),
    }
}

fn common(v: &Value, d: &Defaults, prompt: String, chat: bool) -> Result<Params, String> {
    if !v.is_object() {
        return Err("the body must be a JSON object".to_owned());
    }
    if let Some(n) = number(v, "n")? {
        if n != 1.0 {
            return Err("only `n` = 1 is supported".to_owned());
        }
    }
    let mut s = d.sampling.clone();
    if let Some(t) = number(v, "temperature")? {
        s.temperature = t as f32;
    }
    if let Some(p) = number(v, "top_p")? {
        s.top_p = p as f32;
    }
    if let Some(k) = number(v, "top_k")? {
        s.top_k = k as usize;
    }
    if let Some(r) = number(v, "repeat_penalty")? {
        s.repeat_penalty = r as f32;
    }
    let limit = number(v, "max_completion_tokens")?.or(number(v, "max_tokens")?);
    Ok(Params {
        prompt,
        max_tokens: limit.map_or(d.max_tokens, |n| n as usize),
        sampling: s,
        stop: stops(v)?,
        stream: v.get("stream").and_then(Value::as_bool).unwrap_or(false),
        include_usage: v
            .pointer("/stream_options/include_usage")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        seed: number(v, "seed")?.map(|n| n as u64),
        think: think(v)?,
        chat,
    })
}

/// `/v1/completions`. A list of prompts is not supported, because one request makes one text.
pub fn parse_completion(v: &Value, d: &Defaults) -> Result<Params, String> {
    let prompt = match v.get("prompt") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(a)) if a.len() == 1 && a[0].is_string() => {
            a[0].as_str().unwrap_or_default().to_owned()
        }
        _ => return Err("`prompt` must be a string".to_owned()),
    };
    common(v, d, prompt, false)
}

fn content(m: &Value) -> Result<String, String> {
    match m.get("content") {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(s.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for p in parts {
                match p.get("text").and_then(Value::as_str) {
                    Some(t) => text.push_str(t),
                    None => return Err("only text content is supported".to_owned()),
                }
            }
            Ok(text)
        }
        Some(_) => Err("`content` must be a string or an array of text parts".to_owned()),
    }
}

/// The models are trained on plain text, so a chat becomes a labelled transcript that
/// the model continues after the final `Assistant:`.
pub fn render_chat(messages: &[Value]) -> Result<String, String> {
    let mut out = String::new();
    for m in messages {
        let text = content(m)?;
        match m.get("role").and_then(Value::as_str) {
            Some("system" | "developer") => {
                out.push_str(&text);
                out.push_str("\n\n");
            }
            Some("user") => out.push_str(&format!("User: {text}\n")),
            Some("assistant") => out.push_str(&format!("Assistant: {text}\n")),
            Some(other) => return Err(format!("role `{other}` is not supported")),
            None => return Err("every message needs a `role`".to_owned()),
        }
    }
    out.push_str("Assistant:");
    Ok(out)
}

/// `/v1/chat/completions`.
pub fn parse_chat(v: &Value, d: &Defaults) -> Result<Params, String> {
    let messages = v
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("`messages` must be an array")?;
    if messages.is_empty() {
        return Err("`messages` must not be empty".to_owned());
    }
    let mut p = common(v, d, render_chat(messages)?, true)?;
    // Without this the model goes on to write the user's next turn itself.
    p.stop.push("\nUser:".to_owned());
    Ok(p)
}

/// Holds back text that may turn out to be the start of a stop string.
pub struct StopFilter {
    stops: Vec<String>,
    held: String,
    pub hit: bool,
}

impl StopFilter {
    pub fn new(stops: Vec<String>) -> Self {
        let stops = stops.into_iter().filter(|s| !s.is_empty()).collect();
        Self {
            stops,
            held: String::new(),
            hit: false,
        }
    }

    /// Takes new text and returns what is now safe to send.
    pub fn push(&mut self, text: &str) -> String {
        self.held.push_str(text);
        if let Some(at) = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()))
            .min()
        {
            self.hit = true;
            self.held.truncate(at);
            return std::mem::take(&mut self.held);
        }
        let keep = self
            .stops
            .iter()
            .map(|s| partial_suffix(&self.held, s))
            .max()
            .unwrap_or(0);
        let rest = self.held.split_off(self.held.len() - keep);
        std::mem::replace(&mut self.held, rest)
    }

    /// The text still held back when generation ends.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

/// Bytes at the end of `text` that are the beginning of `stop`.
fn partial_suffix(text: &str, stop: &str) -> usize {
    (1..stop.len().min(text.len() + 1))
        .rev()
        .find(|n| stop.is_char_boundary(*n) && text.ends_with(&stop[..*n]))
        .unwrap_or(0)
}

/// One response body or stream chunk's fixed fields.
#[derive(Clone, Debug)]
pub struct Meta {
    pub id: String,
    pub created: u64,
    pub model: String,
    pub chat: bool,
}

impl Meta {
    fn object(&self, stream: bool) -> &'static str {
        match (self.chat, stream) {
            (true, false) => "chat.completion",
            (true, true) => "chat.completion.chunk",
            (false, _) => "text_completion",
        }
    }

    fn envelope(&self, stream: bool, choices: Value) -> Value {
        json!({
            "id": self.id,
            "object": self.object(stream),
            "created": self.created,
            "model": self.model,
            "choices": choices,
        })
    }

    /// The whole answer of a non-streaming request.
    pub fn full(&self, text: &str, finish: &str, usage: Usage) -> Value {
        let choice = if self.chat {
            json!({
                "index": 0,
                "message": { "role": "assistant", "content": text },
                "finish_reason": finish,
            })
        } else {
            json!({ "index": 0, "text": text, "logprobs": null, "finish_reason": finish })
        };
        let mut body = self.envelope(false, json!([choice]));
        body["usage"] = usage.json();
        body
    }

    /// A stream chunk. Chat streams open with the role.
    pub fn delta(&self, text: Option<&str>, role: bool, finish: Option<&str>) -> Value {
        let choice = if self.chat {
            let mut delta = json!({});
            if role {
                delta["role"] = json!("assistant");
            }
            if let Some(t) = text {
                delta["content"] = json!(t);
            }
            json!({ "index": 0, "delta": delta, "finish_reason": finish })
        } else {
            json!({ "index": 0, "text": text.unwrap_or(""), "logprobs": null, "finish_reason": finish })
        };
        self.envelope(true, json!([choice]))
    }

    /// The closing stream chunk that `stream_options.include_usage` asks for.
    pub fn usage_chunk(&self, usage: Usage) -> Value {
        let mut body = self.envelope(true, json!([]));
        body["usage"] = usage.json();
        body
    }
}

pub fn error(message: &str, kind: &str) -> Value {
    json!({ "error": { "message": message, "type": kind, "param": null, "code": null } })
}

pub fn models(id: &str, created: u64) -> Value {
    json!({
        "object": "list",
        "data": [{ "id": id, "object": "model", "created": created, "owned_by": "local" }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> Defaults {
        Defaults {
            max_tokens: 16,
            sampling: Sampling {
                temperature: 0.8,
                top_p: 1.0,
                top_k: 40,
                repeat_penalty: 1.1,
                repeat_window: 64,
            },
        }
    }

    #[test]
    fn chat_becomes_a_transcript() {
        let m = json!([
            { "role": "system", "content": "Be brief." },
            { "role": "user", "content": "Hi" },
            { "role": "assistant", "content": "Hello" },
            { "role": "user", "content": [{ "type": "text", "text": "Bye" }] },
        ]);
        let text = render_chat(m.as_array().unwrap()).unwrap();
        assert_eq!(
            text,
            "Be brief.\n\nUser: Hi\nAssistant: Hello\nUser: Bye\nAssistant:"
        );
    }

    #[test]
    fn request_fields_override_defaults() {
        let v = json!({
            "prompt": "x", "temperature": 0, "top_p": 0.5, "max_tokens": 3,
            "stop": "END", "stream": true, "seed": 9,
            "stream_options": { "include_usage": true },
        });
        let p = parse_completion(&v, &defaults()).unwrap();
        assert_eq!(
            (p.max_tokens, p.sampling.temperature, p.sampling.top_p),
            (3, 0.0, 0.5)
        );
        assert_eq!(p.sampling.top_k, 40);
        assert_eq!(p.stop, ["END"]);
        assert!(p.stream && p.include_usage);
        assert_eq!(p.seed, Some(9));
    }

    #[test]
    fn reasoning_effort_picks_a_level() {
        let d = defaults();
        let level =
            |e: Value| parse_completion(&json!({ "prompt": "x", "reasoning_effort": e }), &d);
        assert_eq!(level(json!(null)).unwrap().think, None);
        assert_eq!(level(json!("high")).unwrap().think, Some(Think::High));
        assert_eq!(level(json!("minimal")).unwrap().think, Some(Think::Low));
        assert_eq!(level(json!("xhigh")).unwrap().think, Some(Think::Max));
        assert!(level(json!("huge")).is_err());
        assert!(level(json!(3)).is_err());
    }

    #[test]
    fn max_completion_tokens_wins() {
        let v = json!({ "prompt": "x", "max_tokens": 3, "max_completion_tokens": 5 });
        assert_eq!(parse_completion(&v, &defaults()).unwrap().max_tokens, 5);
    }

    #[test]
    fn bad_requests_are_refused() {
        let d = defaults();
        assert!(parse_completion(&json!({}), &d).is_err());
        assert!(parse_completion(&json!({ "prompt": "x", "n": 2 }), &d).is_err());
        assert!(parse_completion(&json!({ "prompt": "x", "temperature": "hot" }), &d).is_err());
        assert!(parse_chat(&json!({ "messages": [] }), &d).is_err());
        assert!(parse_chat(&json!({ "messages": [{ "content": "x" }] }), &d).is_err());
        let image =
            json!({ "messages": [{ "role": "user", "content": [{ "type": "image_url" }] }] });
        assert!(parse_chat(&image, &d).is_err());
    }

    #[test]
    fn chat_stops_before_the_next_turn() {
        let v = json!({ "messages": [{ "role": "user", "content": "Hi" }], "stop": "x" });
        assert_eq!(parse_chat(&v, &defaults()).unwrap().stop, ["x", "\nUser:"]);
    }

    fn run_filter(stops: &[&str], parts: &[&str]) -> (String, bool) {
        let mut f = StopFilter::new(stops.iter().map(|s| s.to_string()).collect());
        let mut out = String::new();
        for p in parts {
            out.push_str(&f.push(p));
            if f.hit {
                return (out, true);
            }
        }
        out.push_str(&f.finish());
        (out, false)
    }

    #[test]
    fn stop_across_pieces_is_cut_out() {
        assert_eq!(
            run_filter(&["END"], &["ab", "cE", "N", "Dzz"]),
            ("abc".to_owned(), true)
        );
    }

    #[test]
    fn false_start_of_a_stop_is_released() {
        assert_eq!(
            run_filter(&["END"], &["aEN", "x"]),
            ("aENx".to_owned(), false)
        );
        assert_eq!(run_filter(&["END"], &["aEN"]), ("aEN".to_owned(), false));
    }

    #[test]
    fn earliest_stop_wins_and_chinese_is_safe() {
        assert_eq!(
            run_filter(&["界", "世"], &["你好世界"]),
            ("你好".to_owned(), true)
        );
        assert_eq!(
            run_filter(&["世界"], &["你好世", "人"]),
            ("你好世人".to_owned(), false)
        );
    }

    #[test]
    fn nothing_is_held_without_stops() {
        let mut f = StopFilter::new(Vec::new());
        assert_eq!(f.push("abc"), "abc");
    }

    #[test]
    fn response_shapes_match_openai() {
        let usage = Usage {
            prompt_tokens: 2,
            completion_tokens: 3,
        };
        let chat = Meta {
            id: "c".into(),
            created: 1,
            model: "m".into(),
            chat: true,
        };
        let full = chat.full("hi", "stop", usage);
        assert_eq!(full["object"], "chat.completion");
        assert_eq!(full["choices"][0]["message"]["content"], "hi");
        assert_eq!(full["usage"]["total_tokens"], 5);
        let chunk = chat.delta(Some("h"), true, None);
        assert_eq!(chunk["object"], "chat.completion.chunk");
        assert_eq!(chunk["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunk["choices"][0]["delta"]["content"], "h");
        assert!(chunk["choices"][0]["finish_reason"].is_null());
        let text = Meta {
            chat: false,
            ..chat
        };
        assert_eq!(text.full("hi", "length", usage)["choices"][0]["text"], "hi");
        assert_eq!(
            text.delta(Some("h"), false, None)["object"],
            "text_completion"
        );
    }
}
