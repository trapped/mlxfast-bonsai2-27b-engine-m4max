"""Benchmark the running local server: prefill, TTFT, decode tok/s, RAM, speculative acceptance.

Usage: uv run python bench/bench.py [--url http://127.0.0.1:8000] [--out bench/results.md]
The server must be running (./run-server.sh). Decoding is greedy.
"""

import argparse
import json
import platform
import subprocess
import time
import urllib.request

import psutil

FILLER = open(__file__).read()  # realistic code-ish filler for long prompts

TASKS = {
    "code-short": "Write a Python function `merge_intervals(intervals)` that merges overlapping intervals. Include a docstring and 3 doctests.",
    "explain": "Explain, step by step, how a B-tree insertion with node splitting works. Be thorough.",
}


def post(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ch = json.loads(line[6:])
            d = ch["choices"][0]["delta"] if ch.get("choices") else {}
            if first is None and (d.get("content") or d.get("reasoning_content") or d.get("tool_calls")):
                first = time.perf_counter() - t0
            if ch.get("usage"):
                usage = ch["usage"]
    return first, time.perf_counter() - t0, usage


def worker_rss_gb():
    tot = 0
    for p in psutil.process_iter(["name", "cmdline"]):
        if "bonsai-serve" in " ".join(p.info["cmdline"] or []):
            tot += p.memory_info().rss
    return tot / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="bench/results.md")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prefill-sizes", default="1000,4000,16000,32000")
    a = ap.parse_args()
    health = json.load(urllib.request.urlopen(a.url + "/health"))
    rows = []

    def run(name, messages, max_tokens, thinking=False, warm=True):
        if warm:
            post(a.url, {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
                         "stream": True, "enable_thinking": False})
        ttft, total, u = post(a.url, {"messages": messages, "max_tokens": max_tokens, "stream": True,
                                      "enable_thinking": thinking})
        s = u["bonsai_stats"]
        rows.append({"case": name, "prompt_tok": s["prompt_tokens"], "cached_tok": s["cached_prompt_tokens"],
                     "gen_tok": s["completion_tokens"],
                     "ttft_s": round(ttft or 0, 2), "prefill_tok_s": s["prefill_tok_s"],
                     "decode_tok_s": s["decode_tok_s"], "accept_rate": s["accept_rate"],
                     "tok_per_round": s["mean_tokens_per_round"], "mlx_peak_gb": s["mlx_peak_gb"]})
        print(rows[-1], flush=True)

    for name, prompt in TASKS.items():
        run(name, [{"role": "user", "content": prompt}], a.max_tokens)
    last = None
    for n in [int(x) for x in a.prefill_sizes.split(",")]:
        chars = n * 3
        filler = (FILLER * (chars // len(FILLER) + 1))[:chars]
        # unique first line so the prefix cache cannot serve a cold-prefill case
        last = [{"role": "user", "content": f"[run {time.time_ns()}] Here is a file:\n```\n" + filler +
                 "\n```\nSummarize what this file does in 3 bullet points."}]
        run(f"cold-prefill-{n}", last, 128)
    # agent-style follow-up turn: same history + one more exchange -> prefix cache hit
    last = last + [{"role": "assistant", "content": "It is a benchmark script."},
                   {"role": "user", "content": "Which function measures RAM? Answer in one line."}]
    run("agent-turn-after-largest", last, 64, warm=False)

    hw = subprocess.run(["scripts/detect-hw.sh"], capture_output=True, text=True).stdout.strip()
    cols = list(rows[0].keys())
    md = [f"# Benchmark results\n", f"- date: {time.strftime('%Y-%m-%d %H:%M')}", f"- hardware: {hw}",
          f"- worker: device={health['hello'].get('device')} spec_modes={health['hello'].get('spec_modes')}",
          f"- server: spec={health['spec']} depth={health['depth']} context={health['context']}; greedy decoding\n",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    md.append(f"\nmachine RAM used during run (system): {round(psutil.virtual_memory().used / 2**30, 1)} GiB")
    open(a.out, "a").write("\n".join(md) + "\n\n")
    print(f"appended to {a.out}")


if __name__ == "__main__":
    main()
