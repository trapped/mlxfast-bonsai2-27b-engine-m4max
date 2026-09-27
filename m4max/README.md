# m4max — Bonsai 2 27B on an M4 Max

A local OpenAI-compatible server for **Ternary Bonsai 2 27B** on
[`Layr-Labs/mlxfast-bonsai2-27b-engine`](https://github.com/Layr-Labs/mlxfast-bonsai2-27b-engine),
set up for coding agents (Pi) on Apple Silicon.

```
Pi / any OpenAI client ──HTTP :8000──▶ m4max/server/bonsai_openai.py ──NDJSON──▶ bonsai-serve (Swift, persistent CBv2 engine)
   chat template, <think> → reasoning_content,                               packed 2-bit Bonsai tower, hybrid prefix cache,
   <tool_call> XML → tool_calls, SSE streaming                                optional DFlash 2 speculation
```

## Pins

| What | Pin |
|---|---|
| Engine | this fork: `Layr-Labs/mlxfast-bonsai2-27b-engine` main + the `m4max:` commits (rebase with `m4max/sync-upstream.sh`) |
| Target | `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` @ `3f926b4` (sha256 manifest in upstream `fixtures/`) |
| DFlash 2 drafter | `z-lab/Qwen3.8-27B-DFlash2` @ `50307d4` |
| MTP head | `EigenLabs/Qwen3.8-27B-MTP-4bit` @ `329261c` (downloaded; not used, see below) |
| Python | `m4max/uv.lock` (transformers 5.17.0 for the tokenizer and chat template) |

## Quick start

```bash
m4max/setup.sh             # toolchain check, upstream build, ~12 GB of pinned downloads, transform, bonsai-serve
m4max/run-server.sh            # foreground on 127.0.0.1:8000, or:
m4max/scripts/launchd.sh install # persistent: starts at login, restarts on crash (m4max/logs/server.log)
(cd m4max && uv run python bench/bench.py   # benchmark the running server; appends to bench/results.md)
m4max/pi/install.sh              # add provider "bonsai" to Pi (backs up ~/.pi/agent first)
```

Endpoints: `GET /v1/models`, `POST /v1/chat/completions` (stream and non-stream,
tools, `stop`, `max_tokens`), `GET /health`, `GET /v1/stats` (stats for the last request).
Decoding is always greedy, because the engine only does argmax. Temperature and top_p are ignored.
Thinking is on by default. Turn it off with `"enable_thinking": false` or
`"chat_template_kwargs": {"enable_thinking": false}`.

Defaults, overridable in `m4max/config.env` or the environment: context `BONSAI_CONTEXT=49152`,
max output `BONSAI_MAX_OUTPUT=16384`, KV pool `BONSAI_KV_BYTES=14GiB`,
prefix cache `BONSAI_PREFIX_CACHE_BYTES=7GiB`, prefill chunk `BONSAI_PREFILL_CHUNK=512`,
decoder `BONSAI_SPEC=serial|dflash` with `BONSAI_DEPTH`.

## What had to be fixed for this Mac (M4 Max, 36 GB)

`scripts/detect-hw.sh` reports the chip. `setup.sh` applies the `the `m4max:` commits` patches
and writes `m4max/config.env` only when the chip is not NAX-capable (pre-M5). On an M5 the upstream
behaviour is left alone.

1. **Garbage output on M4.** Upstream builds NAX (Metal 4 tensor-op) code into the *plain*
   quantized kernels whenever the **compiler** supports Metal 4, not when the **GPU** has
   NAX. Bisected to `b0552f2`. the `m4max:` NAX-gate commit makes those bodies opt-in.
2. **Garbage output on M4, second cause.** `Qwen35TensorPackedMatmul` (commit `f5960db`) uses
   tensor ops unconditionally. `m4max/config.env` sets `DARKBLOOM_BONSAI_TENSOR_ROUTE=0`.
3. **DFlash acceptance of 0%.** The drafter's `DFlash2TensorMatmul` has the same problem.
   `DARKBLOOM_DFLASH2_TENSOR_MATMUL=0` restores 25–80% acceptance, depending on depth.
4. **Speculation looked slower than serial on M4.** The cause was verify-row padding, fixed below:
   DFlash is now the default and is about 40% faster than serial on code.
5. **The MTP head crashes** (a precondition in `Qwen35InlineMTPAssistant.observeCommittedTarget`),
   so it isn't offered.
6. **Every agent turn re-prefilled the whole prompt** at about 150 tok/s. bench-worker builds a
   fresh engine per request with the prefix cache off. `Vendor/mlx-swift-lm/Executables/bonsai-serve/main.swift` (`bonsai-serve`)
   keeps one engine alive and turns on the engine's own hybrid (KV + recurrent-state) prefix
   cache (the `m4max:` engine commit). A follow-up turn on a 27k-token history went from 229 s to 14 s TTFT.
7. The 4096-token decode cap is now configurable (the `m4max:` engine commit).

### Kernel optimizations for this model (the `m4max:` engine commit, knobs in `m4max/config.env`)

The approach followed the upstream AGENTS.md guidance: kernel work on the prefill and decode paths,
no re-quantization, measurements on a cooled machine, and upstream's correctness gate re-run
(64/64 tokens). The analysis tools are in `Vendor/mlx-swift-lm/Executables/bonsai-kbench/` (a microbenchmark of the packed matmul and
the transform at this model's shapes) and `MLXFAST_TRACE_KERNELS=1`, which logs every Metal
dispatch by kernel name.

| Change | Why | Result (cooled A/B) |
|---|---|---|
| `MLXFAST_QMM_BM/BN=64`: 64×64 tiles for `qmm_t` at prompt width on non-NAX GPUs | Prefill is bound by the dequantizing matmul (about 88% of FP16 peak). Bigger tiles reuse each dequantized weight tile across twice the rows. K order is unchanged, so tokens are bit-identical. | 4k-token prefill 195 → 214 tok/s (+10%) |
| `MLXFAST_BONSAI_QMV2=1`: new `affine_bonsai_qmv2` single-row 2-bit GEMV that reads the pack's FP16 scales and offsets as stored (widened exactly in-register) instead of FP32-widened copies. x is pre-scaled by 4⁻ⁱ, so each weight costs one and, one convert and one FMA. | Decode is about 1,350 dispatches per token, 401 of them this GEMV, and they are bandwidth-bound. FP32 constants add about 25% to the bytes read. Tokens are bit-identical. | decode step 28.4 → 26.5 ms (+7%) |

| `MLXFAST_MATRIX_MIN_ROWS=1`: don't pad speculative-verify rows to 13 | The 13-row threshold targets M5's NAX kernels. On M4 it moves an 8-row verify from `qmv_wide` (3.2× a GEMV) to the 32-row tile (6×). | DFlash depth 3 on code 16.6 → 52.7 tok/s |
| DFlash 2 prefix checkpoints (`Qwen35DFlash2Assistant+PrefixCheckpoint.swift`) | Without them the engine turns the prefix cache off whenever the drafter is armed. A checkpoint holds the drafter's pending context rows, exactly the state of a cold run at that position. Restored output matched cold output token for token. | DFlash turn-2 TTFT 27.8 s → 2.4 s (5.5k prompt) |
| Default decoder now `BONSAI_SPEC=dflash`, `BONSAI_DEPTH=3` | Serial 37.6 tok/s. DFlash depth 1/2/3/4/5 on code: 36.7/44.6/**52.7**/39.2/40.6; on prose: 31.9/36.0/34.7/27.4/28.6. Raw data in `m4max/bench/spec-ab.py` runs. | Pi multi-step task 49 s → 34 s, agent turns 42–53 tok/s at 72–93% acceptance |

On `84a5f3e`, DFlash is faster than serial at every context tested up to 16k tokens
(16k: 28.0 vs 15.3 tok/s). See `m4max/bench/results.md` for the upstream-bump comparison. One engine serves one decoder, and switching per request would drop the prefix
cache, so the default is a fixed choice. For very long sessions set `BONSAI_SPEC=serial`, for example
`BONSAI_SPEC=serial m4max/run-server.sh` (environment variables override `m4max/config.env`).

DFlash verify reads FP16 activations (upstream's verify route), so a rare near-tie can differ from
serial FP32 decoding. In 5 of 7 prompt×depth runs the output was identical to serial. Set
`BONSAI_SPEC=serial` for strict serial greedy.

Tried and dropped, with measurements:
- **Scalar few-row "qmvm" verify kernels (two designs):** 2.5× slower than the stock `qmv_wide`,
  which uses the matrix units.
- **FP16 activations at decode:** not faster, and tokens diverged.
- **Two packs per thread in stock `qmv`:** slower.
- **16-byte-load GEMV:** register spill, 3× slower.
- **128-wide qmm tiles:** slower.
- **BK=64:** crashes.
- **Faster prefill:** there is little left to gain. Prefill runs at the matmul's rate, and this machine reads memory at about 360 GB/s at most (1 GiB reduction), which puts the decode floor near 20 ms per token.

Upstream's own correctness gate passes with fixes 1–2: `./benchmark.sh --local-iterate`
matched 64/64 teacher-forced tokens against the shipped golden.

## Pi

`m4max/pi/install.sh` merges `m4max/pi/bonsai-provider.json` into `~/.pi/agent/models.json` as provider
`bonsai`, model `bonsai2-27b` (reasoning on, `thinkingFormat: qwen-chat-template`, context 49152,
max output 16384). Other providers and the default model (`lmstudio/qwen3-coder-30b-a3b-instruct-mlx`)
are left untouched. The first run's backup is `m4max/pi/backup/original`, and `m4max/pi/revert.sh` restores it.

```bash
pi --provider bonsai --model bonsai2-27b      # this session only
# or inside Pi: /model  ->  bonsai/bonsai2-27b
```

Note: with `-p` (print mode), give Pi `< /dev/null` when stdin isn't a terminal, or it waits on stdin.

## Benchmarks

See `m4max/bench/results.md` (HTTP end-to-end) and `m4max/bench/sweep-*.jsonl` (decoder and depth sweeps
straight against the engine). Numbers depend on memory pressure: on a 36 GB machine with a
browser open, macOS swaps, and decode drops.

Note: `weights/` and the drafter directories under `reference_weights/` must be real directories.
The engine's loaders reject symlinked checkpoint and drafter paths.

## Tracking upstream

This fork keeps upstream's history and stacks three commits on top of it:
1. `m4max: keep NAX bodies out of the plain quantized kernels`, the correctness fix without which an M4 outputs garbage.
2. `m4max: engine optimizations and a persistent serving worker`, with the kernels, verify padding, prefix cache and `bonsai-serve`.
3. `m4max: serving stack`, which adds this directory: the OpenAI server, Pi config, benchmarks and scripts.

To follow upstream, run `m4max/sync-upstream.sh`. It fetches upstream, rebases the commits, rebuilds,
re-transforms the weights and runs upstream's correctness gate. Compare speed with `m4max/bench/`, then
`git push --force-with-lease`. Upstream's CI/benchmark workflows target its ranked runners and stay
disabled on this fork.
