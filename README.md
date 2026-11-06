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
slm repl -m model
slm bench -m model
slm info -m model
```

常用参数：

| 参数 | 含义 | 默认值 |
|---|---|---|
| `-m, --model` | 模型目录 | `model` |
| `-d, --device` | `auto`、`cuda`、`mlx`、`cpu` | `auto` |
| `-R, --loops` | 每个 token 循环几轮，越多越慢 | 模型配置里的值 |
| `-n, --max-tokens` | 最多生成多少 token | 256 |
| `-t, --temperature` | 采样温度，0 表示总取概率最大的 | 0.8 |
| `--top-k` | 只从概率最高的 k 个里采样 | 40 |
| `--repeat-penalty` | 重复惩罚，1 表示关闭 | 1.1 |
| `--seed` | 随机种子 | 0 |
| `--fp16` | MLX 上用 16 位浮点，解码状态减半 | 关 |

这是预训练模型，会续写文本，不会对话。

## 速度

RTX 5080，86M 模型（窗口 4096），fp32，每生成一个 token 就读回一次，单位 tok/s：

| R | SLM.use CUDA | PyTorch CUDA | 解码状态 |
|---:|---:|---:|---:|
| 4 | 253 | 154 | 193 MB |
| 8 | 150 | 89 | 322 MB |
| 16 | 82 | 48 | 580 MB |

Ryzen 9 9950X 的 CPU 上，R=4 时 80 tok/s，同一台机器上 PyTorch 是 25 tok/s。

提示词也是逐 token 处理的，速度和解码相同。4096 token 的提示词在 R=4 时要约 16 秒。

## 实现

模型逻辑只写一份（`src/model.rs`），基于 `src/ops.rs` 里的一组运算；`src/cpu.rs`、`src/cuda.rs`、`src/mlx.rs` 分别实现这组运算。

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

要核对真实模型，先用 `tools/make_fixture.py` 生成参考值，再运行 `slm verify -m model`。
