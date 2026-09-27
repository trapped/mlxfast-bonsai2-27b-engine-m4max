import Foundation
import MLX

/// Custom 2-bit affine GEMV (group 128) for one activation row: FP32 math,
/// FP16 constants read as stored, R output rows per simdgroup.
enum Bonsai2BitGemv {
    static func source(R: Int, SGS: Int, wide: Bool = false) -> String {
        if wide { return sourceWide(R: R, SGS: SGS) }
        return """
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        constexpr int K = KDIM;
        constexpr int KW = K / 16;
        constexpr int KG = K / 128;
        constexpr int R = \(R);
        const int row0 = (int(threadgroup_position_in_grid.x) * \(SGS) + int(sg)) * R;
        const device uint2* wr[R];
        #pragma unroll
        for (int r = 0; r < R; r++) wr[r] = (const device uint2*)(w + (row0 + r) * KW);
        float acc[R];
        #pragma unroll
        for (int r = 0; r < R; r++) acc[r] = 0.0f;
        for (int c = int(lane); c < K / 32; c += 32) {
            // x prescaled by 4^-i so a masked (unshifted) code multiplies it directly.
            float xv[32];
            float xs = 0.0f;
            const device float4* xp = (const device float4*)(x + c * 32);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                float4 v = xp[i];
                xs += (v.x + v.y) + (v.z + v.w);
                const int j = (4 * i) & 15;
                xv[4*i]   = v.x * exp2(-2.0f * float(j));
                xv[4*i+1] = v.y * exp2(-2.0f * float(j + 1));
                xv[4*i+2] = v.z * exp2(-2.0f * float(j + 2));
                xv[4*i+3] = v.w * exp2(-2.0f * float(j + 3));
            }
            const int g = c >> 2;
            #pragma unroll
            for (int r = 0; r < R; r++) {
                const uint2 wv = wr[r][c];
                float d0 = 0.0f, d1 = 0.0f;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    d0 = fma(float(wv.x & (3u << (2 * i))), xv[i], d0);
                    d1 = fma(float(wv.y & (3u << (2 * i))), xv[16 + i], d1);
                }
                const int row = row0 + r;
                acc[r] += float(scales[row * KG + g]) * (d0 + d1) + float(biases[row * KG + g]) * xs;
            }
        }
        #pragma unroll
        for (int r = 0; r < R; r++) {
            float v = simd_sum(acc[r]);
            if (lane == 0) y[row0 + r] = v;
        }
        """
    }
    static func sourceWide(R: Int, SGS: Int) -> String {
        """
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        constexpr int K = KDIM;
        constexpr int KW = K / 16;
        constexpr int KG = K / 128;
        constexpr int R = \(R);
        const int row0 = (int(threadgroup_position_in_grid.x) * \(SGS) + int(sg)) * R;
        float acc[R];
        #pragma unroll
        for (int r = 0; r < R; r++) acc[r] = 0.0f;
        for (int c = int(lane); c < K / 64; c += 32) {
            float xv[64];
            float xs = 0.0f;
            const device float4* xp = (const device float4*)(x + c * 64);
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                float4 v = xp[i];
                xs += (v.x + v.y) + (v.z + v.w);
                const int j = (4 * i) & 15;
                xv[4*i]   = v.x * exp2(-2.0f * float(j));
                xv[4*i+1] = v.y * exp2(-2.0f * float(j + 1));
                xv[4*i+2] = v.z * exp2(-2.0f * float(j + 2));
                xv[4*i+3] = v.w * exp2(-2.0f * float(j + 3));
            }
            const int g = c >> 1;
            #pragma unroll
            for (int r = 0; r < R; r++) {
                const int row = row0 + r;
                const uint4 wv = ((const device uint4*)(w + row * KW))[c];
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    const uint m = 3u << (2 * i);
                    d0 = fma(float(wv.x & m), xv[i], d0);
                    d1 = fma(float(wv.y & m), xv[16 + i], d1);
                    d2 = fma(float(wv.z & m), xv[32 + i], d2);
                    d3 = fma(float(wv.w & m), xv[48 + i], d3);
                }
                acc[r] += float(scales[row * KG + g]) * ((d0 + d1) + (d2 + d3)) + float(biases[row * KG + g]) * xs;
            }
        }
        #pragma unroll
        for (int r = 0; r < R; r++) {
            float v = simd_sum(acc[r]);
            if (lane == 0) y[row0 + r] = v;
        }
        """
    }
    nonisolated(unsafe) static var kernels: [String: MLXFast.MLXFastKernel] = [:]
    static func apply(_ x: MLXArray, _ w: MLXArray, _ s: MLXArray, _ b: MLXArray, R: Int = 4, SGS: Int = 4, wide: Bool = false) -> MLXArray {
        let key = "\(R)_\(SGS)_\(wide)"
        let k = kernels[key] ?? MLXFast.metalKernel(
            name: "bonsai_gemv2_r\(R)_s\(SGS)_w\(wide ? 1 : 0)", inputNames: ["w", "scales", "biases", "x"],
            outputNames: ["y"], source: source(R: R, SGS: SGS, wide: wide))
        kernels[key] = k
        let n = w.dim(0), kdim = x.dim(-1)
        return k([w, s, b, x.reshaped(kdim)], template: [("KDIM", kdim)],
                 grid: (n / R / SGS * 32 * SGS, 1, 1), threadGroup: (32 * SGS, 1, 1),
                 outputShapes: [[1, n]], outputDTypes: [.float32])[0]
    }
}
