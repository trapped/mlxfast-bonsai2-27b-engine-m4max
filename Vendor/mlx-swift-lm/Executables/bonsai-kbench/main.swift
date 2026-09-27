// bonsai-kbench: time the packed 2-bit matmul and the Hadamard transform at
// Bonsai 2 27B shapes (hidden 5120, intermediate 17408, gs128, bits 2).
import Foundation
import MLX
import MLXNN

func time(_ label: String, flops: Double, bytes: Double, iters: Int = 20, _ body: () -> MLXArray) {
    for _ in 0..<3 { eval(body()) }
    MLX.Stream().synchronize()
    let t = Date()
    for _ in 0..<iters { eval(body()) }
    MLX.Stream().synchronize()
    let dt = Date().timeIntervalSince(t) / Double(iters)
    print(String(format: "%-44@ %8.3f ms  %7.2f TFLOP/s  %7.1f GB/s", label as NSString, dt * 1e3,
                 flops / dt / 1e12, bytes / dt / 1e9))
}

func packed(n: Int, k: Int) -> (MLXArray, MLXArray, MLXArray) {
    let w = MLXRandom.randInt(low: 0, high: Int32.max, [n, k * 2 / 32]).asType(.uint32)
    let s = (MLXRandom.uniform(low: 0.001, high: 0.01, [n, k / 128])).asType(.float16)
    let b = (MLXRandom.uniform(low: -0.01, high: 0.0, [n, k / 128])).asType(.float16)
    eval(w, s, b)
    return (w, s, b)
}

if CommandLine.arguments.contains("verify") {
    for (n, k, name) in [(17408, 5120, "gate/up"), (5120, 17408, "down"), (10240, 5120, "qkv-ish")] {
        let ws = (0..<16).map { _ in packed(n: n, k: k) }
        for m in [1, 8, 13] {
            let x = MLXRandom.normal([m, k]).asType(.float16); eval(x)
            time("x16 verify \(name) M=\(m) f16", flops: 32.0 * Double(m * n * k),
                 bytes: 16 * Double(n * k) / 4, iters: 10) {
                stacked(ws.map { quantizedMM(x, $0.0, scales: $0.1, biases: $0.2, transpose: true, groupSize: 128, bits: 2).sum(axis: 0) }).sum()
            }
        }
    }
    exit(0)
}
if CommandLine.arguments.contains("bw") {
    let a = MLXRandom.randInt(low: 0, high: 100, [256 << 20]).asType(.uint32); eval(a)  // 1 GiB
    let f = MLXRandom.normal([256 << 20]); eval(f)
    time("sum uint32 1GiB", flops: 0, bytes: Double(1 << 30)) { a.sum() }
    time("sum float32 1GiB", flops: 0, bytes: Double(1 << 30)) { f.sum() }
    time("max float32 1GiB", flops: 0, bytes: Double(1 << 30)) { f.max() }
    exit(0)
}
// Decode shapes: 16 independent matmuls per eval so launch/sync latency is amortized.
if CommandLine.arguments.contains("decode") {
    for (n, k, name) in [(17408, 5120, "gate/up"), (5120, 17408, "down"), (10240, 5120, "qkv-ish"), (248320, 5120, "lm_head")] {
        let ws = (0..<16).map { _ in packed(n: n, k: k) }
        let ws32 = ws.map { ($0.0, $0.1.asType(.float32), $0.2.asType(.float32)) }
        eval(ws32.flatMap { [$0.1, $0.2] })
        do {
            let x = MLXRandom.normal([1, k]); eval(x)
            time("x16 qmv f32-consts \(name) N=\(n) K=\(k)", flops: 32.0 * Double(n * k),
                 bytes: 16 * Double(n * k) / 4, iters: 10) {
                stacked(ws32.map { quantizedMM(x, $0.0, scales: $0.1, biases: $0.2, transpose: true, groupSize: 128, bits: 2) }).sum()
            }
        }
        for dt in [DType.float16, .float32] {
            let x = MLXRandom.normal([1, k]).asType(dt); eval(x)
            time("x16 qmv \(name) N=\(n) K=\(k) \(dt)", flops: 32.0 * Double(n * k),
                 bytes: 16 * Double(n * k) / 4, iters: 10) {
                stacked(ws.map { quantizedMM(x, $0.0, scales: $0.1, biases: $0.2, transpose: true, groupSize: 128, bits: 2) }).sum()
            }
        }
    }
    for (n, k, name) in [(17408, 5120, "gate/up"), (5120, 17408, "down"), (10240, 5120, "qkv-ish")] {
        let ws = (0..<16).map { _ in packed(n: n, k: k) }
        let x = MLXRandom.normal([1, k]); eval(x)
        let ref = quantizedMM(x, ws[0].0, scales: ws[0].1.asType(.float32), biases: ws[0].2.asType(.float32), transpose: true, groupSize: 128, bits: 2)
        for (r, sgs, wide) in [(2, 4, false), (4, 8, false), (8, 2, false), (1, 8, true), (2, 4, true), (2, 8, true), (4, 4, true)] {
            let got = Bonsai2BitGemv.apply(x, ws[0].0, ws[0].1, ws[0].2, R: r, SGS: sgs, wide: wide)
            let err = (abs(got - ref).max() / abs(ref).max()).item(Float.self)
            time("x16 gemv2 \(name) R=\(r) SGS=\(sgs) W=\(wide) relerr=\(err)", flops: 32.0 * Double(n * k),
                 bytes: 16 * Double(n * k) / 4, iters: 10) {
                stacked(ws.map { Bonsai2BitGemv.apply(x, $0.0, $0.1, $0.2, R: r, SGS: sgs, wide: wide) }).sum()
            }
        }
    }
    exit(0)
}
let rows = CommandLine.arguments.dropFirst().compactMap { Int($0) }
for (n, k, name) in [(17408, 5120, "gate/up"), (5120, 17408, "down"), (10240, 5120, "qkv-ish")] {
    let (w, s, b) = packed(n: n, k: k)
    for m in rows.isEmpty ? [1, 512, 2048] : rows {
        for dt in [DType.float16, .float32] {
            let x = MLXRandom.normal([m, k]).asType(dt); eval(x)
            time("qmm \(name) N=\(n) K=\(k) M=\(m) \(dt)", flops: 2.0 * Double(m * n * k),
                 bytes: Double(n * k) / 4 + Double(m * k * dt.size)) {
                quantizedMM(x, w, scales: s, biases: b, transpose: true, groupSize: 128, bits: 2)
            }
        }
    }
}
let signs = (0..<17408).map { _ in Float(Bool.random() ? 1 : -1) }
for (width, block) in [(5120, 1024), (17408, 1024)] {
    let h = try! SignedBlockHadamard(blockSize: block, signs: Array(signs[0..<width]))
    for m in [1, 512, 2048] {
        let x = MLXRandom.normal([m, width]); eval(x)
        time("hadamard w=\(width) M=\(m) (op chain / fused if installed)", flops: Double(m * width) * 10,
             bytes: Double(m * width * 8)) { h(x) }
    }
}
