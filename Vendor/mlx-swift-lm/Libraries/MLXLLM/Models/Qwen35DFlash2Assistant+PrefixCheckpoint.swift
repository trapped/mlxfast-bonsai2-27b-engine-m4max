import Foundation
import MLX
import MLXLMCommon

/// bonsai-fast: prefix-cache checkpoints for the DFlash 2 drafter, so a
/// speculative engine keeps the hybrid prefix cache (agent turns reuse the
/// previous prompt instead of re-prefilling it).
///
/// During prefill the drafter only queues committed target context rows
/// (`pending`, capped at its sliding window) and absorbs them into its caches
/// at the first block. A checkpoint therefore holds exactly what a cold run
/// holds at that position: a compact copy of the pending rows and the absolute
/// row count. Restore hands them to a fresh request state with fresh caches.
/// A state that has already absorbed rows (after the first block) declines.
extension Qwen35DFlash2Assistant: CBv2MTPPrefixCheckpointDrafter {
    private final class PrefixCheckpoint: CBv2MTPPrefixCheckpoint {
        let owner: ObjectIdentifier
        let targetInputCount: Int
        let rows: MLXArray?

        init(owner: ObjectIdentifier, count: Int, rows: MLXArray?) {
            self.owner = owner
            self.targetInputCount = count
            self.rows = rows
        }

        var materializedBytes: Int { rows?.nbytes ?? 0 }
        var evaluationTargets: [MLXArray] { rows.map { [$0] } ?? [] }
    }

    public func capturePrefixCheckpoint(
        requestState: any CBv2MTPRequestState, targetInputCount: Int
    ) -> (any CBv2MTPPrefixCheckpoint)? {
        guard let state = requestState as? RequestState,
            !state.isReleased, !state.cacheSeeded, !state.contextPrefetched,
            state.roots.isEmpty, state.observedRows == targetInputCount
        else { return nil }
        var rows: MLXArray?
        if !state.pending.isEmpty {
            let joined = concatenated(state.pending, axis: 1)
            guard joined.dim(1) == state.pendingRows else { return nil }
            // Compact byte-preserving copy: drops views of later prompt rows.
            rows = MLX.where(MLXArray(true), joined, joined)
        }
        return PrefixCheckpoint(owner: ObjectIdentifier(self), count: targetInputCount, rows: rows)
    }

    public func restorePrefixCheckpoint(
        _ checkpoint: any CBv2MTPPrefixCheckpoint
    ) -> (any CBv2MTPRequestState)? {
        guard let checkpoint = checkpoint as? PrefixCheckpoint,
            checkpoint.owner == ObjectIdentifier(self),
            let state = makeRequestState() as? RequestState
        else { return nil }
        if let rows = checkpoint.rows {
            state.pending = [rows]
            state.pendingRows = rows.dim(1)
        }
        state.observedRows = checkpoint.targetInputCount
        return state
    }
}
