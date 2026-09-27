// bonsai-serve — persistent single-stream serving worker for bonsai-fast.
//
// Unlike bench-worker (one fresh engine per timed phase, no prefix cache, no
// stop tokens), this keeps ONE CBv2 engine per decoder alive for the process
// lifetime with the hybrid (KV + recurrent-state) prefix cache enabled, so an
// agent's next turn only prefills the tokens it appended.
//
// NDJSON on stdin/stdout.
//   -> {"prompt_tokens":[..], "max_tokens":N, "stop_tokens":[..], "decoder":"serial"|"dflash", "depth":D}
//   <- {"event":"delta","tokens":[..]}                       (0..n times)
//   <- {"event":"finished","reason":"stop|length|cancelled|error", ...stats}
//   -> {"cancel":true}   cancels the in-flight request
// First line out is {"event":"ready", ...}.

import Foundation
import MLX
import MLXLMCommon
import MLXRunners

func emit(_ obj: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: obj)
    FileHandle.standardOutput.write(data + Data("\n".utf8))
}

func log(_ s: String) { FileHandle.standardError.write(Data("bonsai-serve: \(s)\n".utf8)) }

func arg(_ name: String) -> String? {
    let a = CommandLine.arguments
    guard let i = a.firstIndex(of: name), i + 1 < a.count else { return nil }
    return a[i + 1]
}

guard let weightsPath = arg("--weights") else {
    log("usage: bonsai-serve --weights DIR [--drafter DIR] [--kv-bytes N] [--prefix-cache-bytes N]")
    exit(2)
}
let weights = URL(fileURLWithPath: weightsPath)
let drafter = arg("--drafter").map { URL(fileURLWithPath: $0) }
let kvBytes = Int(arg("--kv-bytes") ?? "") ?? (8 << 30)
let prefixBytes = Int(arg("--prefix-cache-bytes") ?? "") ?? (2 << 30)
let chunk = Int(arg("--prefill-chunk") ?? "") ?? 2048

let launchArgs = ["runtime-worker", "--weights", weightsPath] + (drafter.map { ["--drafter", $0.path] } ?? [])
let launch: BenchWorkerLaunchOptions
let runner: any Runner
do {
    launch = try BenchWorkerLaunchOptions.parse(launchArgs)
    let t = Date()
    runner = try await launch.resolveRunner().load(
        weights,
        options: RunnerLoadOptions(drafterDirectory: drafter, kvBytesCapacity: kvBytes, resources: launch.resources))
    log(String(format: "loaded in %.1fs", Date().timeIntervalSince(t)))
} catch {
    log("load failed: \(error)")
    exit(2)
}

final class State: @unchecked Sendable {
    var engines: [String: any CBv2Engine] = [:]
    var nextID: UInt64 = 1
    var inflight: CBv2RequestID?
    var inflightEngine: (any CBv2Engine)?
    let lock = NSLock()
    func setInflight(_ id: CBv2RequestID?, _ e: (any CBv2Engine)?) {
        lock.withLock { inflight = id; inflightEngine = e }
    }
    func cancelInflight() {
        lock.withLock { if let id = inflight { inflightEngine?.cancel(id) } }
    }
}
let state = State()

func engine(decoder: String, depth: Int) throws -> any CBv2Engine {
    let key = "\(decoder):\(depth)"
    if let e = state.engines[key] { return e }
    // Keep one engine resident: drop others so their KV pools are released.
    state.engines.removeAll()
    MLX.Stream().synchronize()
    Memory.clearCache()
    let id = DecoderID(decoder)
    var mtp = CBv2MTPConfig(enabled: false)
    if id != .serial {
        let d = max(1, depth)
        mtp = CBv2MTPConfig(
            enabled: true, maxDraftTokens: d, maxSpeculativeBatch: 1, fixedDraftTokens: d,
            verificationMode: .automatic, maxAutomaticRectangularTokens: 1 + d,
            draftTokenCeiling: id == .dflash
                ? CBv2MTPConfig.testedMaxBlockDraftTokens : CBv2MTPConfig.testedMaxDraftTokens)
    }
    var build = EngineBuild(
        kvBackend: .contiguous,
        kvBytesCapacity: kvBytes,
        schedulerConfig: CBv2SchedulerConfig(
            maxConcurrentRequests: 1, prefillChunkSize: chunk, maxWaiting: 1,
            enablePrefixCache: true),
        loopConfig: CBv2EngineLoopConfig(),
        prefixCache: nil,
        decoder: id,
        mtpConfig: mtp,
        environment: [:])
    build.hybridPrefixCache = CBv2HybridPrefixCacheConfig(
        maximumBytes: prefixBytes, maximumEntries: 16, maximumCheckpointsPerRequest: 4,
        modelID: weightsPath, promptContractID: "bonsai-fast-chatml", buildID: "bonsai-serve-1")
    let e = try runner.makeEngine(build)
    state.engines[key] = e
    return e
}

func reasonString(_ r: CBv2FinishReason) -> String {
    switch r {
    case .stop: return "stop"
    case .length: return "length"
    case .cancelled: return "cancelled"
    case .error(let m): return "error: \(m)"
    default: return "error: \(r)"
    }
}

func serve(_ req: [String: Any]) async {
    let prompt = (req["prompt_tokens"] as? [Int]) ?? []
    let maxTokens = (req["max_tokens"] as? Int) ?? 4096
    let stops = Set((req["stop_tokens"] as? [Int]) ?? [])
    let decoder = (req["decoder"] as? String) ?? "serial"
    let depth = (req["depth"] as? Int) ?? 0
    do {
        let e = try engine(decoder: decoder, depth: depth)
        let before = (e as? any CBv2MTPCountersReporting)?.mtpMetricsSnapshot()
        let id = CBv2RequestID(state.nextID); state.nextID += 1
        state.setInflight(id, e)
        var r = CBv2Request(id: id, promptTokens: prompt, maxTokens: maxTokens)
        r.sampling = CBv2SamplingParams(temperature: 0, topP: 1, topK: 0)
        r.stopTokens = stops
        let t0 = DispatchTime.now().uptimeNanoseconds
        var first: UInt64 = 0
        for await ev in try e.submit(r) {
            switch ev {
            case .delta(_, let tokens, _):
                if first == 0 { first = DispatchTime.now().uptimeNanoseconds }
                if !tokens.isEmpty { emit(["event": "delta", "tokens": tokens]) }
            case .finished(let reason, let usage):
                let after = (e as? any CBv2MTPCountersReporting)?.mtpMetricsSnapshot()
                let now = DispatchTime.now().uptimeNanoseconds
                emit([
                    "event": "finished", "reason": reasonString(reason),
                    "prompt_tokens": usage.promptTokens, "completion_tokens": usage.completionTokens,
                    "prefix_outcome": "\(usage.prefixCacheOutcome)",
                    "prefix_hit_tokens": usage.prefixCachePrefillTokensSaved,
                    "ttft_ms": first == 0 ? 0 : Double(first - t0) / 1e6,
                    "total_ms": Double(now - t0) / 1e6,
                    "drafted": (after?.draftedTokens ?? 0) - (before?.draftedTokens ?? 0),
                    "accepted": (after?.acceptedTokens ?? 0) - (before?.acceptedTokens ?? 0),
                    "rounds": (after?.rounds ?? 0) - (before?.rounds ?? 0),
                    "mlx_active_gb": Double(Memory.activeMemory) / 1e9,
                    "mlx_peak_gb": Double(Memory.peakMemory) / 1e9,
                ])
            }
        }
    } catch {
        emit(["event": "finished", "reason": "error: \(error)"])
    }
    state.setInflight(nil, nil)
    if CBv2StepProfiler.enabled {
        log("step profile\n" + CBv2StepProfiler.summaryTable())
        if let walls = CBv2StepProfiler.snapshot()["v2.step.wall"] {
            log("step walls ms: " + walls.map { String(format: "%.0f", $0 * 1000) }.joined(separator: " "))
        }
        CBv2StepProfiler.reset()
    }
}

emit(["event": "ready", "device": ProcessInfo.processInfo.environment["BONSAI_DEVICE"] ?? "",
      "decoders": runner.loadedDecoders.map(\.rawValue)])

// Requests are serialized; a cancel line may arrive while one is in flight.
let (requests, requestSink) = AsyncStream<String>.makeStream()
Thread.detachNewThread {
    while let line = readLine(strippingNewline: true) {
        guard let data = line.data(using: .utf8),
            let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { continue }
        if obj["cancel"] != nil {
            state.cancelInflight()
        } else {
            requestSink.yield(line)
        }
    }
    requestSink.finish()
}
for await line in requests {
    if let obj = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] {
        await serve(obj)
    }
}
