use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Result};
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use slm::api::Defaults;
use slm::cpu::Cpu;
use slm::model::{Model, Session, CHUNK};
use slm::ops::Ops;
use slm::sample::{pick, Rng, Sampling};
use slm::serve;
use slm::text::{self, encode, Detokenizer};
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(version, about = "Run Looped Hybrid Reasoner language models locally")]
#[command(args_conflicts_with_subcommands = true)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    /// Start the OpenAI-compatible API server
    #[arg(long)]
    serve: bool,
    #[command(flatten)]
    common: Common,
    #[command(flatten)]
    gen: Gen,
    #[command(flatten)]
    net: Net,
}

#[derive(Clone, Copy, PartialEq, ValueEnum)]
enum Device {
    Auto,
    Cuda,
    Mlx,
    Cpu,
}

#[derive(Args, Clone)]
struct Common {
    /// Model folder with model.safetensors, config.json and tokenizer.json
    #[arg(short, long, default_value = "model")]
    model: PathBuf,
    #[arg(short, long, value_enum, default_value_t = Device::Auto)]
    device: Device,
    /// Thinking loops per token; more are slower. Defaults to the model's own value
    #[arg(short = 'R', long)]
    loops: Option<usize>,
    /// Use 16-bit floats on MLX, which halves the decode state
    #[arg(long)]
    fp16: bool,
}

#[derive(Args, Clone)]
struct Net {
    /// Address to listen on; use 0.0.0.0 to accept other machines
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Require this key as a Bearer token
    #[arg(long)]
    api_key: Option<String>,
}

#[derive(Args, Clone)]
struct Gen {
    #[arg(short = 'n', long, default_value_t = 256)]
    max_tokens: usize,
    /// 0 always picks the most likely token
    #[arg(short, long, default_value_t = 0.8)]
    temperature: f32,
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    /// Smallest set of tokens holding this much probability; 1 disables it
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,
    /// 1 disables it
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,
    #[arg(long, default_value_t = 256)]
    repeat_window: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

#[derive(Subcommand)]
enum Cmd {
    /// Continue a prompt and stream the text
    Run {
        prompt: String,
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        gen: Gen,
    },
    /// Interactive session; every line continues the same context
    Repl {
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        gen: Gen,
    },
    /// Measure decode and prompt speed
    Bench {
        #[command(flatten)]
        common: Common,
        /// Loop counts to measure
        #[arg(long, num_args = 1.., default_values_t = [4usize, 8, 16])]
        loop_counts: Vec<usize>,
        #[arg(long, default_value_t = 4096)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 256)]
        decode_tokens: usize,
    },
    /// Start the OpenAI-compatible API server, the same as --serve
    Serve {
        #[command(flatten)]
        common: Common,
        #[command(flatten)]
        gen: Gen,
        #[command(flatten)]
        net: Net,
    },
    /// Show the model and which device would run it
    Info {
        #[command(flatten)]
        common: Common,
    },
    /// Compare logits with the PyTorch reference saved by tools/make_fixture.py
    Verify {
        #[command(flatten)]
        common: Common,
    },
}

fn sampling(g: &Gen) -> Sampling {
    Sampling {
        temperature: g.temperature,
        top_k: g.top_k,
        top_p: g.top_p,
        repeat_penalty: g.repeat_penalty,
        repeat_window: g.repeat_window,
    }
}

/// Generates until the token limit or the end-of-text token.
fn generate<O: Ops>(
    model: &Model<O>,
    se: &mut Session<O>,
    tk: &Tokenizer,
    history: &mut Vec<u32>,
    g: &Gen,
    rng: &mut Rng,
) -> Result<usize> {
    let s = sampling(g);
    let eot = tk.token_to_id("<|endoftext|>");
    let mut out = Detokenizer::new(tk);
    let mut made = 0;
    while made < g.max_tokens {
        let id = pick(model.logits(se)?, history, &s, rng);
        if Some(id) == eot {
            break;
        }
        print!("{}", out.push(id)?);
        std::io::stdout().flush()?;
        history.push(id);
        model.step(se, id)?;
        made += 1;
    }
    Ok(made)
}

fn feed<O: Ops>(model: &Model<O>, se: &mut Session<O>, ids: &[u32]) -> Result<f64> {
    let t = Instant::now();
    model.feed(se, ids, CHUNK)?;
    model.logits(se)?;
    Ok(ids.len() as f64 / t.elapsed().as_secs_f64())
}

/// Greedy generation without tokenizer work, reading every token back as a stream would.
fn decode_speed<O: Ops>(model: &Model<O>, se: &mut Session<O>, tokens: usize) -> Result<f64> {
    let s = Sampling {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repeat_penalty: 1.0,
        repeat_window: 0,
    };
    let mut rng = Rng::new(0);
    let t = Instant::now();
    for _ in 0..tokens {
        let id = pick(model.logits(se)?, &[], &s, &mut rng);
        model.step(se, id)?;
    }
    Ok(tokens as f64 / t.elapsed().as_secs_f64())
}

fn run<O: Ops>(ops: O, cmd: Cmd) -> Result<()> {
    match cmd {
        Cmd::Run {
            prompt,
            common,
            gen,
        } => {
            let tk = text::load(&common.model)?;
            let model = Model::load(ops, &common.model)?;
            let mut se = model.session(common.loops)?;
            let mut history = encode(&tk, &prompt)?;
            if history.is_empty() {
                bail!("the prompt is empty");
            }
            let fill = feed(&model, &mut se, &history)?;
            print!("{prompt}");
            let t = Instant::now();
            let made = generate(
                &model,
                &mut se,
                &tk,
                &mut history,
                &gen,
                &mut Rng::new(gen.seed),
            )?;
            println!();
            eprintln!(
                "[{}, R={}: prompt {:.0} tok/s, decode {:.0} tok/s]",
                model.device(),
                se.loops,
                fill,
                made as f64 / t.elapsed().as_secs_f64()
            );
        }
        Cmd::Repl { common, gen } => {
            let tk = text::load(&common.model)?;
            let model = Model::load(ops, &common.model)?;
            let mut se = model.session(common.loops)?;
            let mut history = Vec::new();
            let mut rng = Rng::new(gen.seed);
            eprintln!(
                "{}, R={}. An empty line or Ctrl-D quits.",
                model.device(),
                se.loops
            );
            let stdin = std::io::stdin();
            loop {
                print!("> ");
                std::io::stdout().flush()?;
                let mut line = String::new();
                if stdin.lock().read_line(&mut line)? == 0 || line.trim().is_empty() {
                    return Ok(());
                }
                let ids = encode(&tk, line.trim_end_matches(['\r', '\n']))?;
                feed(&model, &mut se, &ids)?;
                history.extend(&ids);
                generate(&model, &mut se, &tk, &mut history, &gen, &mut rng)?;
                println!();
            }
        }
        Cmd::Bench {
            common,
            loop_counts,
            prompt_tokens,
            decode_tokens,
        } => {
            let tk = text::load(&common.model)?;
            let model = Model::load(ops, &common.model)?;
            let unit = encode(
                &tk,
                "The quick brown fox jumps over the lazy dog. 敏捷的棕色狐狸跳过了懒狗。",
            )?;
            let prompt: Vec<u32> = unit.iter().cycle().take(prompt_tokens).copied().collect();
            println!(
                "{}, {:.1}M parameters",
                model.device(),
                model.params() as f64 / 1e6
            );
            println!(
                "{:>3} {:>14} {:>14} {:>12}",
                "R", "prompt tok/s", "decode tok/s", "state MB"
            );
            for r in loop_counts {
                let mut se = model.session(Some(r))?;
                feed(&model, &mut se, &prompt[..prompt.len().min(16)])?;
                let fill = feed(&model, &mut se, &prompt)?;
                let dec = decode_speed(&model, &mut se, decode_tokens)?;
                let mb = model.state_bytes(&se) as f64 / (1 << 20) as f64;
                println!("{r:>3} {fill:>14.1} {dec:>14.1} {mb:>12.1}");
            }
        }
        Cmd::Serve { common, gen, net } => {
            let tk = text::load(&common.model)?;
            let model = Model::load(ops, &common.model)?;
            let name = std::fs::canonicalize(&common.model).ok();
            let model_id = name
                .as_ref()
                .and_then(|p| p.file_name())
                .map_or("slm".to_owned(), |n| n.to_string_lossy().into_owned());
            let options = serve::Options {
                host: net.host,
                port: net.port,
                api_key: net.api_key,
                model_id,
                defaults: Defaults {
                    max_tokens: gen.max_tokens,
                    sampling: sampling(&gen),
                },
                loops: common.loops,
                seed: gen.seed,
            };
            serve::run(model, tk, options)?;
        }
        Cmd::Info { common } => {
            let model = Model::load(ops, &common.model)?;
            let c = &model.cfg;
            println!("device:     {}", model.device());
            println!("parameters: {:.1}M", model.params() as f64 / 1e6);
            println!(
                "layers:     prelude {} / core {} / coda {}",
                c.prelude, c.core, c.coda
            );
            println!(
                "d_model:    {}, heads {}, window {}, vocab {}",
                c.d_model, c.n_heads, c.window, c.vocab_size
            );
            println!(
                "loops:      default {}, max {}",
                c.default_loops, c.max_loops
            );
        }
        Cmd::Verify { common } => {
            let model = Model::load(ops, &common.model)?;
            let (plain, prompt) = slm::verify::run(&model, &common.model)?;
            for (name, r) in [("decode", &plain), ("prompt", &prompt)] {
                println!(
                    "{}: {name}: {} steps, {} checked, max |diff| {:.6}, same argmax {}",
                    model.device(),
                    r.steps,
                    r.checked,
                    r.max_diff,
                    r.same_argmax
                );
                if r.max_diff > 1e-2 || !r.same_argmax {
                    bail!("logits differ from the reference");
                }
            }
        }
    }
    Ok(())
}

fn common(cmd: &Cmd) -> &Common {
    match cmd {
        Cmd::Run { common, .. }
        | Cmd::Repl { common, .. }
        | Cmd::Bench { common, .. }
        | Cmd::Serve { common, .. }
        | Cmd::Info { common }
        | Cmd::Verify { common } => common,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cmd = match (cli.cmd, cli.serve) {
        (Some(cmd), _) => cmd,
        (None, true) => Cmd::Serve {
            common: cli.common,
            gen: cli.gen,
            net: cli.net,
        },
        (None, false) => {
            Cli::command().print_help()?;
            return Ok(());
        }
    };
    let want = common(&cmd).device;

    #[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
    if want == Device::Auto || want == Device::Cuda {
        match slm::cuda::Cuda::probe()? {
            Some(ops) => return run(ops, cmd),
            None if want == Device::Cuda => bail!("no NVIDIA GPU or driver found"),
            None => {}
        }
    }
    #[cfg(all(feature = "mlx", target_os = "macos", target_arch = "aarch64"))]
    if want == Device::Auto || want == Device::Mlx {
        let fp16 = common(&cmd).fp16;
        return run(slm::mlx::Mlx::new(fp16)?, cmd);
    }
    if want == Device::Cuda || want == Device::Mlx {
        bail!("this build does not include that device");
    }
    run(Cpu, cmd)
}
