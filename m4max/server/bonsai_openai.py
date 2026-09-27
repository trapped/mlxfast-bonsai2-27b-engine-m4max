"""OpenAI-compatible HTTP front end for the mlxfast Bonsai 2 27B engine.

The engine runs in `bonsai-serve` (serve-worker/main.swift): one persistent
CBv2 engine with the hybrid prefix cache, token ids in and out over NDJSON,
greedy decoding, optional DFlash 2 speculation. This process owns that worker,
renders the model's own chat template, parses reasoning / tool calls, and
exposes /v1/chat/completions.

Requests are served one at a time (the engine is single-stream).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from transformers import AutoTokenizer

EOS_IDS = {248044, 248046}  # <|endoftext|>, <|im_end|>


# --------------------------------------------------------------------------- worker


class Worker:
    """Owns one `bonsai-serve` process (persistent engine + hybrid prefix cache)."""

    def __init__(self, cmd: list[str], env: dict[str, str]):
        self.cmd, self.env = cmd, env
        self.proc: subprocess.Popen | None = None
        self.hello: dict = {}
        self.start()

    def start(self):
        log(f"starting worker: {' '.join(self.cmd)}")
        self.proc = subprocess.Popen(
            self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, text=True, bufsize=1, env=self.env,
        )
        self.hello = self.read()
        log(f"worker ready: {json.dumps(self.hello)}")

    def read(self) -> dict:
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited (rc={self.proc.poll()})")
        return json.loads(line)

    def send(self, obj: dict):
        if self.proc.poll() is not None:
            self.start()
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()


# --------------------------------------------------------------------------- parsing

TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FUNC_RE = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.S)
PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def coerce(value: str, schema: dict | None):
    t = (schema or {}).get("type")
    if t == "string":
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def parse_tool_calls(text: str, tools: list[dict] | None) -> tuple[str, list[dict]]:
    schemas = {}
    for t in tools or []:
        fn = t.get("function", {})
        schemas[fn.get("name")] = (fn.get("parameters") or {}).get("properties", {})
    calls = []
    for block in TOOL_RE.findall(text):
        block = block.strip()
        m = FUNC_RE.search(block)
        if m:
            name = m.group(1).strip()
            props = schemas.get(name, {})
            args = {k.strip(): coerce(v, props.get(k.strip())) for k, v in PARAM_RE.findall(m.group(2))}
        else:  # JSON style fallback
            try:
                obj = json.loads(block)
                name, args = obj["name"], obj.get("arguments", {})
            except Exception:
                continue
        calls.append({
            "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
        })
    content = TOOL_RE.sub("", text)
    # An unterminated trailing <tool_call> (hit max_tokens) is dropped from content.
    if "<tool_call>" in content:
        content = content.split("<tool_call>")[0]
    return content.strip(), calls


def split_reasoning(text: str, thinking: bool) -> tuple[str, str]:
    """The generation prompt already ends with '<think>\\n' when thinking is on."""
    if thinking or "</think>" in text:
        if "</think>" in text:
            r, c = text.split("</think>", 1)
            return r.replace("<think>", "").strip(), c.lstrip()
        return text.replace("<think>", "").strip(), ""
    return "", text


# --------------------------------------------------------------------------- engine


class Engine:
    def __init__(self, args):
        self.args = args
        self.tok = AutoTokenizer.from_pretrained(args.tokenizer)
        env = dict(os.environ)
        cmd = [args.worker, "--weights", args.weights, "--kv-bytes", str(args.kv_bytes),
               "--prefix-cache-bytes", str(args.prefix_cache_bytes),
               "--prefill-chunk", str(args.prefill_chunk)]
        if args.spec != "serial":
            cmd += ["--drafter", args.drafter]
        self.worker = Worker(cmd, env)
        self.lock = threading.Lock()
        self.last_stats: dict = {}

    def render(self, body: dict) -> tuple[list[int], bool]:
        msgs = []
        for m in body["messages"]:
            m = dict(m)
            c = m.get("content")
            if isinstance(c, list):  # flatten OpenAI content parts
                m["content"] = "".join(p.get("text", "") for p in c if p.get("type") in ("text", "input_text"))
            if m.get("role") == "developer":
                m["role"] = "system"
            if m.get("content") is None:
                m["content"] = ""
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except Exception:
                        fn["arguments"] = {}
            msgs.append(m)
        kw = dict(body.get("chat_template_kwargs") or {})
        thinking = kw.get("enable_thinking", body.get("enable_thinking"))
        if thinking is None:
            effort = body.get("reasoning_effort")
            thinking = self.args.default_thinking if effort is None else effort not in ("none", "off")
        kw["enable_thinking"] = bool(thinking)
        text = self.tok.apply_chat_template(
            msgs, tools=body.get("tools") or None, add_generation_prompt=True,
            tokenize=False, **kw)
        return self.tok.encode(text, add_special_tokens=False), bool(thinking)

    def generate(self, ids: list[int], max_tokens: int, stop: list[str]):
        """Yields decoded text deltas; sets self.last_stats at the end."""
        if len(ids) + max_tokens > self.args.context:
            max_tokens = self.args.context - len(ids)
            if max_tokens <= 0:
                raise ValueError(f"prompt is {len(ids)} tokens; context limit is {self.args.context}")
        max_tokens = min(max_tokens, self.args.max_output)
        w = self.worker
        t0 = time.perf_counter()
        w.send({"prompt_tokens": ids, "max_tokens": max_tokens, "stop_tokens": sorted(EOS_IDS),
                "decoder": self.args.spec, "depth": self.args.depth if self.args.spec != "serial" else 0})
        out: list[int] = []
        emitted = ""
        stopped = False
        fin = None
        try:
            while True:
                ev = w.read()
                if ev["event"] == "finished":
                    fin = ev
                    if not stopped:
                        text = self.tok.decode(out, skip_special_tokens=False)
                        if len(text) > len(emitted) and text.startswith(emitted):
                            yield text[len(emitted):]
                    break
                if stopped:
                    continue  # draining after a stop string
                out.extend(t for t in ev["tokens"] if t not in EOS_IDS)
                text = self.tok.decode(out, skip_special_tokens=False)
                hit = next((x for x in stop if x in text), None)
                if hit:
                    text = text[: text.index(hit)]
                    stopped = True
                    w.send({"cancel": True})
                else:
                    if text.endswith("\ufffd"):
                        text = text[:-1]  # incomplete UTF-8 sequence; wait for more tokens
                    hold = max((partial_suffix(text, x) for x in stop), default=0)
                    text = text[: len(text) - hold]  # could still become a stop string
                if len(text) > len(emitted) and text.startswith(emitted):
                    yield text[len(emitted):]
                    emitted = text
        finally:
            if fin is None:  # client went away mid-stream: stop the engine and drain
                w.send({"cancel": True})
                while w.read()["event"] != "finished":
                    pass
        total = time.perf_counter() - t0
        reason = fin.get("reason", "error")
        if reason.startswith("error"):
            raise RuntimeError(reason)
        ttft = fin.get("ttft_ms", 0) / 1000
        n = fin.get("completion_tokens", len(out))
        decode_s = fin.get("total_ms", total * 1000) / 1000 - ttft
        computed = len(ids) - fin.get("prefix_hit_tokens", 0)
        self.last_stats = {
            "prompt_tokens": len(ids), "completion_tokens": n,
            "cached_prompt_tokens": fin.get("prefix_hit_tokens", 0),
            "ttft_s": round(ttft, 3),
            "prefill_tok_s": round(computed / ttft, 1) if ttft else None,
            "decode_tok_s": round((n - 1) / decode_s, 1) if decode_s > 0 and n > 1 else None,
            "drafted": fin.get("drafted", 0), "accepted": fin.get("accepted", 0),
            "accept_rate": round(fin["accepted"] / fin["drafted"], 3) if fin.get("drafted") else None,
            "mean_tokens_per_round": round(n / fin["rounds"], 2) if fin.get("rounds") else None,
            "mlx_peak_gb": round(fin.get("mlx_peak_gb", 0), 2),
            "finish_reason": "stop" if stopped or reason in ("stop", "cancelled") else reason,
        }
        log(f"stats {json.dumps(self.last_stats)}")


# --------------------------------------------------------------------------- http


def log(msg):
    print(f"[bonsai-openai {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class Handler(BaseHTTPRequestHandler):
    engine: Engine = None
    model_id = "bonsai2-27b"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        log(fmt % a)

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            ctx = self.engine.args.context
            return self._json(200, {"object": "list", "data": [{
                "id": self.model_id, "object": "model", "owned_by": "local",
                "context_length": ctx, "max_model_len": ctx}]})
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            a = self.engine.args
            return self._json(200, {"status": "ok", "spec": a.spec, "depth": a.depth, "context": a.context,
                                    "max_output": a.max_output, "hello": self.engine.worker.hello})
        if self.path.rstrip("/") == "/v1/stats":
            return self._json(200, self.engine.last_stats)
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            return self._json(404, {"error": {"message": "not found"}})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except Exception as e:
            return self._json(400, {"error": {"message": f"bad json: {e}"}})
        eng = self.engine
        with eng.lock:
            try:
                ids, thinking = eng.render(body)
            except Exception as e:
                return self._json(400, {"error": {"message": f"template: {e}"}})
            max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or eng.args.max_output
            stop = body.get("stop") or []
            stop = [stop] if isinstance(stop, str) else stop
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            if body.get("stream"):
                self._stream(body, ids, thinking, max_tokens, stop, cid, created)
            else:
                try:
                    text = "".join(eng.generate(ids, max_tokens, stop))
                except ValueError as e:
                    return self._json(400, {"error": {"message": str(e), "code": "context_length_exceeded"}})
                except Exception as e:
                    return self._json(500, {"error": {"message": str(e)}})
                reasoning, content = split_reasoning(text, thinking)
                content, calls = parse_tool_calls(content, body.get("tools"))
                st = eng.last_stats
                msg = {"role": "assistant", "content": content or (None if calls else "")}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                if calls:
                    msg["tool_calls"] = calls
                self._json(200, {
                    "id": cid, "object": "chat.completion", "created": created, "model": self.model_id,
                    "choices": [{"index": 0, "message": msg,
                                 "finish_reason": "tool_calls" if calls else st["finish_reason"]}],
                    "usage": self._usage(st)})

    def _usage(self, st):
        return {"prompt_tokens": st["prompt_tokens"], "completion_tokens": st["completion_tokens"],
                "total_tokens": st["prompt_tokens"] + st["completion_tokens"], "bonsai_stats": st}

    def _stream(self, body, ids, thinking, max_tokens, stop, cid, created):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send(delta, finish=None, usage=None):
            chunk = {"id": cid, "object": "chat.completion.chunk", "created": created,
                     "model": self.model_id,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            if usage:
                chunk["usage"] = usage
            self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        send({"role": "assistant", "content": ""})
        in_reason = thinking
        buf = ""        # text not yet classified
        content_all = ""
        try:
            for delta in self.engine.generate(ids, max_tokens, stop):
                buf += delta
                while buf:
                    if in_reason:
                        if "</think>" in buf:
                            r, buf = buf.split("</think>", 1)
                            if r.replace("<think>", ""):
                                send({"reasoning_content": r.replace("<think>", "")})
                            in_reason = False
                            buf = buf.lstrip("\n")
                            continue
                        keep = partial_suffix(buf, "</think>")
                        out, buf = buf[: len(buf) - keep], buf[len(buf) - keep:]
                        if out.replace("<think>", ""):
                            send({"reasoning_content": out.replace("<think>", "")})
                        break
                    # content: stream until a tool call starts; from then on buffer
                    if "<tool_call>" in content_all + buf:
                        content_all += buf
                        buf = ""
                        break
                    keep = partial_suffix(buf, "<tool_call>")
                    out, buf = buf[: len(buf) - keep], buf[len(buf) - keep:]
                    if not content_all and not out.strip():
                        buf = ""  # drop leading whitespace
                        break
                    if out:
                        send({"content": out})
                        content_all += out
                    break
            content_all += buf
            tail = ""
            calls = []
            if "<tool_call>" in content_all:
                pre, rest = content_all.split("<tool_call>", 1)
                _, calls = parse_tool_calls("<tool_call>" + rest, body.get("tools"))
                tail = ""
            for i, c in enumerate(calls):
                send({"tool_calls": [{"index": i, **c}]})
            st = self.engine.last_stats
            send({}, "tool_calls" if calls else st["finish_reason"], self._usage(st))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            log("client disconnected")
        except Exception as e:
            log(f"stream error: {e}")
            try:
                self.wfile.write(f"data: {json.dumps({'error': {'message': str(e)}})}\n\n".encode())
            except Exception:
                pass


def partial_suffix(s: str, tag: str) -> int:
    """Length of the longest suffix of s that is a proper prefix of tag."""
    for k in range(min(len(s), len(tag) - 1), 0, -1):
        if s.endswith(tag[:k]):
            return k
    return 0


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    up = os.path.dirname(root)  # the engine repo root (m4max/..)
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("BONSAI_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("BONSAI_PORT", 8000)))
    p.add_argument("--worker", default=os.path.join(up, ".build-worker/release/bonsai-serve"))
    p.add_argument("--weights", default=os.path.join(up, "weights"))
    p.add_argument("--tokenizer", default=os.path.join(up, "reference_weights/Ternary-Bonsai-2-27B-mlx-2bit"))
    p.add_argument("--drafter", default=os.path.join(up, "reference_weights/Qwen3.8-27B-DFlash2"))
    p.add_argument("--spec", choices=["dflash", "serial"], default=os.environ.get("BONSAI_SPEC", "serial"))
    p.add_argument("--depth", type=int, default=int(os.environ.get("BONSAI_DEPTH", 7)))
    p.add_argument("--context", type=int, default=int(os.environ.get("BONSAI_CONTEXT", 49152)))
    p.add_argument("--max-output", type=int, default=int(os.environ.get("BONSAI_MAX_OUTPUT", 16384)))
    p.add_argument("--kv-bytes", type=int, default=int(os.environ.get("BONSAI_KV_BYTES", 14 << 30)))
    p.add_argument("--prefix-cache-bytes", type=int, default=int(os.environ.get("BONSAI_PREFIX_CACHE_BYTES", 7 << 30)))
    p.add_argument("--prefill-chunk", type=int, default=int(os.environ.get("BONSAI_PREFILL_CHUNK", 512)),
                   help="prefill chunk; prefix-cache checkpoints land on chunk boundaries")
    p.add_argument("--model-id", default="bonsai2-27b")
    p.add_argument("--default-thinking", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    Handler.engine = Engine(args)
    Handler.model_id = args.model_id
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    log(f"listening on http://{args.host}:{args.port}/v1 model={args.model_id} spec={args.spec}:{args.depth} ctx={args.context}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
