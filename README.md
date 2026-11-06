# SLM.use

用 Rust 在本地运行循环混合推理器（Looped Hybrid Reasoner）语言模型的命令行工具。一个可执行文件，不需要 Python，也不需要 CUDA Toolkit。

| 平台 | 加速 | 下载 |
|---|---|---|
| Windows x64 | CUDA，没有 N 卡时用 CPU | `slm-windows-x64.zip` |
| Windows ARM64 | CPU | `slm-windows-arm64.zip` |
| Linux x64 | CUDA，没有 N 卡时用 CPU | `slm-linux-x64.tar.gz` |
| Linux ARM64 | CUDA，没有 N 卡时用 CPU | `slm-linux-arm64.tar.gz` |
| macOS Apple 芯片 | MLX | `slm-macos-arm64.tar.gz` |
| macOS Intel | CPU | `slm-macos-x64.tar.gz` |

到 [Releases](https://github.com/integemjack/SLM.use/releases) 下载后解压即可。CUDA 只要求装好 NVIDIA 显卡驱动，要求 Pascal 或更新的显卡。macOS 版的 `mlx.metallib` 要和 `slm` 放在同一个目录。

## 准备模型

仓库和发布包都不含模型权重。用 PyTorch 训练出来的 `model.pt` 或 `ckpt.pt` 转换一次：

```bash
pip install torch safetensors
python tools/convert.py runs/g86-f/model.pt model --tokenizer data/pretrain/tokenizer.json
```

得到的 `model` 目录里有 `model.safetensors`、`config.json` 和 `tokenizer.json`。

## 使用

```bash
slm run "集中采购和分散采购的区别是" -m model -n 200
slm run "集中采购和分散采购的区别是" -m model --think high
slm repl -m model
slm bench -m model
slm info -m model
```

常用参数：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `-m, --model` | 模型目录 | `model` |
| `-d, --device` | `auto`、`cuda`、`mlx`、`cpu` | `auto` |
| `--think` | 思考档位：`low`、`medium`、`high`、`max`，不能和 `-R` 同时用 | `medium` |
| `-R, --loops` | 直接指定每个 token 循环几轮，越多越慢 | 模型配置里的值 |
| `-n, --max-tokens` | 最多生成多少 token | 256 |
| `-t, --temperature` | 采样温度，0 表示总取概率最大的 | 0.8 |
| `--top-k` | 只从概率最高的 k 个里采样 | 40 |
| `--top-p` | 只从累计概率达到 p 的 token 里采样，1 表示关闭 | 1 |
| `--repeat-penalty` | 重复惩罚，1 表示关闭 | 1.1 |
| `--seed` | 随机种子 | 0 |
| `--fp16` | MLX 上用 16 位浮点，解码状态减半 | 关 |

这是预训练模型，会续写文本，不会对话。

## 思考档位

模型每生成一个 token 都要循环几轮，一轮大约相当于一步推理。档位就是选循环几轮：`low` 是模型默认值的一半，`medium` 是默认值，`high` 是两倍，`max` 是模型允许的最大值。`slm info` 会列出当前模型每档对应的轮数。

```bash
slm run "集中采购和分散采购的区别是" -m model --think high
slm repl -m model --think low
slm --serve -m model --think high
```

`pt-loop-s0` 在 RTX 5080 上的实测：

| 档位 | 轮数 | 解码 | 解码状态 | 文本 loss | 全部答对的 chase | 全部答对的 arith |
|---|---|---|---|---|---|---|
| `low` | 2 | 约 300 tok/s | 137 MB | 3.23 | 2 步 | 1 步 |
| `medium` | 4 | 约 200 tok/s | 205 MB | 3.12 | 3 步 | 3 步 |
| `high` | 8 | 约 120 tok/s | 342 MB | 3.10 | 4 步 | 6 步 |
| `max` | 16 | 约 65 tok/s | 616 MB | 3.19 | 4 步 | 7 步 |

- 文本 loss 越低越好，在验证集上测，用的是普通分词。`high` 只比 `medium` 低 0.01，速度却慢将近一半，普通续写用默认档就够。`max` 的 loss 反而变差，语言部分训练时只转 2～10 轮。
- chase 是指针追踪，arith 是嵌套算术。超过表里步数的题也能答对一部分，比如 `max` 在 chase 5 步上约 0.7，在 arith 10 步上约 0.87。
- **推理题的结果只在按字符切 token 时成立。** 训练时题目是一个字符一个 token，而普通分词会把 `((`、`...` 这类连续标点合成一个 token。用 `slm run` 或 API 直接输入题目时，四个档位都和瞎猜差不多。所以目前档位对日常使用的影响主要是上面的文本 loss 和速度。


## OpenAI 兼容 API

```bash
slm --serve -m model --port 8080
```

默认只监听本机。要让别的机器访问，加 `--host 0.0.0.0`，这时请同时用 `--api-key` 设置密钥。

用官方客户端时，把地址指向这里即可：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="不校验时随便填")
reply = client.chat.completions.create(
    model="model",
    messages=[{"role": "user", "content": "集中采购和分散采购的区别是什么？"}],
    max_tokens=200,
)
print(reply.choices[0].message.content)
```

| 接口 | 说明 |
|---|---|
| `POST /v1/chat/completions` | 支持 `stream` |
| `POST /v1/completions` | 支持 `stream` |
| `GET /v1/models` | 列出当前模型 |
| `GET /health` | 不需要密钥 |

支持的请求字段：`max_tokens`、`max_completion_tokens`、`temperature`、`top_p`、`stop`、`seed`、`stream`、`stream_options.include_usage`，`reasoning_effort`，以及非标准的 `top_k` 和 `repeat_penalty`。请求没有给的值用命令行参数的值，比如 `-n`、`-t`。

`reasoning_effort` 只影响这一次请求，取值是 `low`、`medium`、`high`、`max`；OpenAI 的 `none` 和 `minimal` 按 `low` 算，`xhigh` 按 `max` 算。没有给时用启动时的 `--think` 或 `-R`。

- 这些模型没有用对话数据训练过。聊天消息会拼成 `User: ...` / `Assistant:` 文本让模型续写，回答的质量只是续写的质量。遇到 `\nUser:` 时停止，免得模型替用户写下一轮。
- 请求按到达顺序一个一个处理，一次只跑一个。
- 新请求的开头和上一次的提示词加回答完全相同时，只读新增的部分，长对话不用每次重读。换了档位时要重读整个提示词，因为每一轮都有自己的状态。
- `seed` 只保证同一台机器、同一后端上可复现。
- 不支持 `n` 大于 1、`logprobs`、工具调用和图片。`n` 大于 1 和图片会返回 400，其余字段会被忽略。

## 速度

RTX 5080，86M 模型（`pt-loop-s0`，窗口 4096），fp32，单位 tok/s。解码时每生成一个 token 就读回一次，上下文已占满窗口：

| R | 读提示词 | 解码 | PyTorch 解码 | 解码状态 |
|---:|---:|---:|---:|---:|
| 4 | 2819 | 209 | 154 | 205 MB |
| 8 | 1630 | 123 | 89 | 342 MB |
| 16 | 884 | 67 | 48 | 616 MB |

提示词每次最多处理 256 个 token，解码状态和逐 token 处理完全一致。4096 token 的提示词在 R=4 时约 1.5 秒；上下文短时更快，512 token 的提示词是 5860 tok/s，解码是 248 tok/s。

Ryzen 9 9950X 的 CPU 上，读提示词 R=4 时 1789 tok/s、R=16 时 534 tok/s（2048 token），此时解码是 67 和 22 tok/s。

带 G 层的模型（`g86-f`）上下文越长越慢：读 16384 token 的提示词时，R=4 约 1505 tok/s，之后解码约 151 tok/s。

## 实现

模型逻辑只写一份（`src/model.rs`），基于 `src/ops.rs` 里的一组运算；`src/cpu.rs`、`src/cuda.rs`、`src/mlx.rs` 分别实现这组运算。

解码和读提示词走同一段代码：模型每次前进 m 个 token，m=1 就是解码。滑动窗口的缓存比窗口多 256 个位置，这样一块里靠前的 token 仍能读到被同一块覆盖的旧内容。

- CUDA：内核在 `kernels/slm.cu`，预编译成 `kernels/slm.ptx` 嵌进程序，运行时只加载显卡驱动。改了内核要运行 `tools/build_kernels.sh`，这一步需要 CUDA 12.x 的 `nvcc`。
- MLX：通过 [mlx-rs](https://github.com/oxiglade/mlx-rs)。
- 不支持 `loop_lora`、`state_loops` 和 `global_rope_base`，加载这类模型时会报错。

## 从源码构建

```bash
cargo build --release --features cuda   # Windows、Linux
cargo build --release --features mlx    # Apple 芯片，需要 Xcode 和 CMake
cargo build --release                   # 只用 CPU
```

## 测试

`tests/fixtures/tiny` 是一个随机权重的小模型，包含所有层类型，并带有 PyTorch 算出的逐位置 logits。`cargo test` 用它核对每个后端，CI 在所有平台上都会跑。

`cargo test` 还会把小模型的前缀按不同块大小读入，再逐 token 解码，核对块处理后的状态。要核对真实模型，先用 `tools/make_fixture.py` 生成参考值，再运行 `slm verify -m model`，它同时检查解码和读提示词。
