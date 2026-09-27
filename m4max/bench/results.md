# Benchmark results

- date: 2026-09-26 20:49
- hardware: chip="Apple M4 Max" generation=M4 memory_gb=36 macos=26.7 nax_capable=no
- worker: device= spec_modes=None
- server: spec=serial depth=7 context=49152; greedy decoding

| case | prompt_tok | cached_tok | gen_tok | ttft_s | prefill_tok_s | decode_tok_s | accept_rate | tok_per_round | mlx_peak_gb |
|---|---|---|---|---|---|---|---|---|---|
| code-short | 38 | 0 | 405 | 0.34 | 111.6 | 34.8 | None | None | 14.56 |
| explain | 32 | 0 | 512 | 0.2 | 158.1 | 33.7 | None | None | 15.04 |
| cold-prefill-1000 | 882 | 0 | 128 | 4.59 | 192.4 | 32.4 | None | None | 16.03 |
| cold-prefill-4000 | 3466 | 0 | 128 | 18.8 | 184.4 | 28.6 | None | None | 17.37 |
| cold-prefill-16000 | 13721 | 0 | 127 | 103.25 | 132.9 | 10.9 | None | None | 20.17 |
| cold-prefill-32000 | 27405 | 0 | 128 | 224.51 | 122.1 | 11.7 | None | None | 25.55 |
| agent-turn-after-largest | 27435 | 27136 | 7 | 8.42 | 35.7 | 12.7 | None | None | 26.54 |

machine RAM used during run (system): 22.3 GiB

## Speculative decoding sweep (DFlash 2, engine-direct, 256 greedy tokens)

`uv run python bench/sweep.py dflash 0,1,3,7,15` with the M4 `config.env` knobs. Raw data: `bench/sweep-dflash.jsonl`.

| depth | code tok/s | code accept | code tok/round | prose tok/s | prose accept | prose tok/round |
|---|---|---|---|---|---|---|
| 0 (serial) | 27.1 | – | 1.00 | 25.5 | – | 1.00 |
| 1 | 8.5 | 0.855 | 1.86 | 7.2 | 0.620 | 1.62 |
| 3 | 13.7 | 0.733 | 3.20 | 10.0 | 0.472 | 2.42 |
| 7 | 17.6 | 0.477 | 4.34 | 10.4 | 0.238 | 2.67 |
| 15 | 15.8 | 0.218 | 4.27 | 10.0 | 0.113 | 2.69 |

On M4 the drafter is accepted well (up to 4.3 tokens per round on code), but each verify
round costs about 200 ms without NAX tensor kernels, so serial decoding stays faster. The MTP head
crashes the engine (upstream precondition failure), so it wasn't measured. Upstream's own gate
(`./benchmark.sh --local-iterate`, DFlash depth 15 from `mtp-head.manifest.json`) passed
correctness 64/64, with prefill at 4.71 ms/token, decode at 48.2 ms/token, and peak RAM of 15.0 GB.
# Benchmark results

- date: 2026-09-26 22:15
- hardware: chip="Apple M4 Max" generation=M4 memory_gb=36 macos=26.7 nax_capable=no
- worker: device= spec_modes=None
- server: spec=serial depth=7 context=49152; greedy decoding

| case | prompt_tok | cached_tok | gen_tok | ttft_s | prefill_tok_s | decode_tok_s | accept_rate | tok_per_round | mlx_peak_gb |
|---|---|---|---|---|---|---|---|---|---|
| code-short | 38 | 0 | 405 | 0.34 | 111.8 | 37.5 | None | None | 12.96 |
| explain | 32 | 0 | 512 | 0.2 | 159.2 | 36.1 | None | None | 13.44 |
| cold-prefill-1000 | 882 | 0 | 128 | 4.22 | 209.4 | 34.9 | None | None | 14.43 |
| cold-prefill-4000 | 3466 | 0 | 128 | 19.25 | 180.1 | 32.5 | None | None | 15.77 |
| cold-prefill-16000 | 13721 | 0 | 127 | 90.53 | 151.6 | 15.6 | None | None | 18.57 |
| cold-prefill-32000 | 27405 | 0 | 128 | 260.95 | 105.0 | 14.4 | None | None | 23.95 |
| agent-turn-after-largest | 27435 | 27136 | 7 | 3.32 | 91.1 | 3.3 | None | None | 24.94 |

machine RAM used during run (system): 19.3 GiB

## Before vs after the kernel optimizations (same HTTP benchmark, 2026-09-26)

| case | decode tok/s before → after | prefill tok/s before → after | TTFT s before → after |
|---|---|---|---|
| code-short | 34.8 → 37.5 | – | 0.34 → 0.34 |
| explain | 33.7 → 36.1 | – | 0.20 → 0.20 |
| cold-prefill-1000 | 32.4 → 34.9 | 192 → 209 | 4.59 → 4.22 |
| cold-prefill-4000 | 28.6 → 32.5 | 184 → 180 | 18.8 → 19.3 |
| cold-prefill-16000 | 10.9 → 15.6 | 133 → 152 | 103 → 91 |
| cold-prefill-32000 | 11.7 → 14.4 | 122 → 105 * | 225 → 261 * |
| agent-turn-after-largest | – | – | 8.4 → 3.3 |

\* This case runs last, after roughly 10 minutes of back-to-back load, and the GPU throttles by then:
upstream's thermal gate refused a timing run at 43 °C around this time. The isolated, cooled A/Bs
(`bench/ab-decode.sh`, `bench/ab-prefill.sh`) are more reliable. They show +10% prefill (4k-token prompt,
195 → 214 tok/s) and +7% decode (step 28.4 → 26.5 ms), with bit-identical tokens.
Upstream `--local-iterate` correctness with all knobs on: 64/64 passed.
# Benchmark results

- date: 2026-09-27 05:25
- hardware: chip="Apple M4 Max" generation=M4 memory_gb=36 macos=26.7 nax_capable=no
- worker: device= spec_modes=None
- server: spec=dflash depth=3 context=49152; greedy decoding

| case | prompt_tok | cached_tok | gen_tok | ttft_s | prefill_tok_s | decode_tok_s | accept_rate | tok_per_round | mlx_peak_gb |
|---|---|---|---|---|---|---|---|---|---|
| code-short | 38 | 0 | 405 | 0.33 | 117.1 | 54.4 | 0.782 | 3.35 | 22.0 |
| explain | 32 | 0 | 512 | 0.19 | 167.2 | 42.2 | 0.585 | 2.75 | 22.0 |
| cold-prefill-1000 | 882 | 0 | 128 | 4.37 | 202.4 | 31.7 | 0.414 | 2.25 | 22.2 |
| cold-prefill-4000 | 3466 | 0 | 128 | 18.27 | 189.9 | 7.7 | 0.417 | 2.29 | 23.89 |
| cold-prefill-16000 | 13721 | 0 | 127 | 82.59 | 166.2 | 11.4 | 0.346 | 2.05 | 26.96 |
| cold-prefill-32000 | 27405 | 0 | 128 | 263.28 | 104.1 | 7.2 | 0.393 | 2.21 | 32.61 |
| agent-turn-after-largest | 27435 | 27136 | 7 | 4.59 | 65.6 | 5.6 | 0.833 | 3.5 | 32.61 |

machine RAM used during run (system): 24.2 GiB

## Speculative decoding after the verify-padding fix (2026-09-27)

Engine-direct measurements, cooled and interleaved (`bench/spec-ab.py`), greedy, 256 tokens:

| prompt | serial tok/s | DFlash d=1 | d=2 | **d=3** | d=4 | d=5 |
|---|---|---|---|---|---|---|
| code (chat) | 37.7 | 36.7 | 44.6 | **52.7** | 39.2 | 40.6 |
| prose (chat) | 36.7 | 31.9 | 36.0 | **34.9** | 27.4 | 28.6 |

By context length (raw code file prompt, depth 3, cooled): 4k tokens gives DFlash 40.1 vs serial 30.8 tok/s.
At 8k, rounds are ~75–128 ms (about 3.4 tokens each) against ~31 ms per serial token, roughly even.
At 27k (HTTP agent turn), decode is 17.1 vs 20.4 tok/s.

HTTP, same session, serial then DFlash (`BONSAI_SPEC` env): code-short 37.8 → 51.5, explain 37.0 → 43.0,
agent turn at 27k 20.4 → 17.1 tok/s (TTFT 3.3 → 3.5 s, prefix cache reused 27,136 tokens in both).
Peak MLX memory with the drafter resident: ~20 GB short, ~26 GB at 27k context.

Thermal note: after sustained runs this machine throttles by up to about 50% (DFlash rounds at 256 context
measured 95 ms hot and 61–63 ms cool). Only cooled, interleaved numbers are compared above.
## Upstream bump 30152dc → 84a5f3e (27 new accepted/validated submissions, 2026-09-27)

The new commits change Swift model/engine code only; there are no Metal kernel source changes.
They mostly target the ranked M5 box: DFlash drafter and verify work, query-head folding in the verify
attention GEMMs, fused GDN prework, prompt-lookup drafting, and more tensor-route kernels that
`config.env` keeps disabled. The weight format changed, so re-transform is required
(`setup.sh` now redoes it on a pin change). Our patches apply cleanly and `--local-iterate` passes 64/64.

Cooled, engine-direct, same machine:

| measurement | 30152dc (+patches) | 84a5f3e (+patches) |
|---|---|---|
| serial decode, chat prompts | 37.7 / 36.7 tok/s | 36.9 / 36.1 (same tokens) |
| DFlash d=3, chat code / prose | 52.7 / 34.9 | 53.1 / 36.8 (same tokens) |
| DFlash d=3 after 1k / 4k / 8k tokens | 32.5 / 18.5 / 25.6 | 34.9 / 27.0 / 22.2 |
| DFlash d=3 after 16k tokens (serial 15.3) | – | 28.0 |
| prefill 4k / 8k | 213.6 / 196–197 | 215.3 / 197 |
| MLX active memory at load | ~14 GB | 12.4 GB |
| 27k prompt (DFlash) peak memory | 26.0 GB | 26.0 GB |

In short: DFlash now beats serial at every context tested (up to 16k); prefill and serial decode are
unchanged; weights resident at load are ~1.6 GB smaller. Prompt-lookup drafting (on by default)
proposes a copied prompt span when the output quotes the prompt, which helps agent edits.
Two 27k-prompt runs looked slower on the new tip, but they were thermally throttled. The alternating
8k A/B is identical.
Context stays at 49,152: at 27k the peak is already 26 GB with the drafter resident.
## Upstream bump 84a5f3e → 831fae7 (12 more accepted submissions, 2026-09-27)

What changed: prompt-lookup drafting now skips the drafter's block forward while the output quotes
the prompt, and splices the drafter's block onto the prompt span it has started to quote. KV appends
are written in place by the producing kernel. There is more MTP/verify work, and an M5-only NAX attention
kernel change that doesn't affect this machine. The MTP head still crashes here (exit 133), so it stays unused.

New agent-edit benchmark `bench/quote-ab.py` (the model re-emits a 1.4k-token file with a rename, 1,386 tokens):

| | tok/s (two cooled runs) | tokens/round |
|---|---|---|
| serial | 32.8 | 1 |
| DFlash d=3 @ 84a5f3e | 54.8 / 50.4 | 3.97 |
| **DFlash d=3 @ 831fae7** | **67.3 / 65.7** | 3.98 |

Output was identical in all five runs. Short chat prompts are unchanged (DFlash code 53.1 → 53.3, serial ~36.8).
Long-context DFlash (4k/16k) differed by ±25% run to run with no consistent direction.
`--local-iterate` correctness passed 64/64.
