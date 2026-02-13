# CPU Usage Reduction — VRAD-RTX Analysis

Current GPU path timing (validation map, 30301 faces, 386 lights, RTX 3090):

| Phase | Wall Clock | CPU Utilization |
|---|---|---|
| Direct Lighting (total) | 57.32s | 73.7% peak |
| ↳ BuildFacelights (CPU) | 43.61s | — |
| ↳ SceneData Upload | 1.09s | — |
| ↳ GPU Direct Light kernel | 0.78s | — |
| ↳ SS + PatchLights (CPU) | 11.84s | — |
| Visibility Matrix | 28.82s | 67.7% peak |
| ↳ CPU Prep (est) | 26.04s (90.4%) | — |
| ↳ GPU Wait (est) | 2.78s (9.6%) | — |
| Bounces | 3.08s | 3.1% |
| **Total** | **104.72s** | — |

---

## Strategy 1: Eliminate Redundant `BuildFacelights` Work on GPU Path

**The problem:** On the GPU path, [BuildFacelights](file:///e:/GitHub/vrad-rtx/src/utils/vrad/lightmap.cpp#L3579-L3724) still runs for all 30,301 faces. It does:
1. Sample position generation (`InitLightinfo` → `CalcPoints` → `InitSampleInfo`)
2. Per-group illumination point/normal computation (`ComputeIlluminationPointAndNormalsSSE`)
3. Sky/ambient light processing via `GatherSampleLight_CollectGPURays` — but this **skips standard lights** (line 2634: `if (isStandardLight) continue;`)
4. Shadow ray buffer management + flush threshold checks

The GPU kernel at [LaunchGPUDirectLighting](file:///e:/GitHub/vrad-rtx/src/utils/vrad/direct_lighting_gpu_vrad.cpp#L367-L387) then independently computes all standard light contributions (point, surface, spot) with inline shadow tracing in 0.31s — the same work item (1) was preparing for.

**What's actually needed:** Steps 1-2 are mandatory — they create the `facelight[]` sample arrays that everything downstream uses. Step 3 is genuinely needed for sky lights only. Step 4 is **dead code** on the GPU path — `GatherSampleLight_CollectGPURays` skips all standard lights, so `g_threadRays` is never populated and `FlushThreadShadowRays` flushes nothing (`FlushShadowRays: 0.00s`).

### Sub-strategy 1a: Remove dead shadow ray batching

The entire `DeferredShadowRay_t` / `g_threadRays` / `FlushThreadShadowRays` infrastructure is vestigial in the current GPU path. Standard lights hit `continue` at line 2634, the else branch at line 2668-2719 that would defer rays is unreachable (it's guarded by `if (isStandardLight)` which was already `continue`'d). The flush at line 3681 and `FlushAllThreadShadowRays` in `vrad.cpp:2198` are no-ops.

- **Estimated savings:** ~0-1s (allocation/bookkeeping overhead only)
- **Effort:** Low — remove dead code paths
- **Risk:** None — functionally dead code
- **Verdict:** ✅ **Do it** (cleanup, marginal speed)

### Sub-strategy 1b: Skip `GatherSampleLight_CollectGPURays` cluster merging for sky-only iteration

Currently, `GatherSampleLight_CollectGPURays` iterates the per-cluster light list (avg 119 lights/cluster), checks each light's type, skips ~98% of them (standard lights), and only processes sky/ambient. This is O(faces × samples × lights_per_cluster) with a very high discard rate. 

Instead: build a separate **sky-only light list** at init time (typically 1-5 sky lights total), and iterate only that list in the GPU path. This eliminates the per-cluster lookup, the merged-cluster `CUtlVector::Find()` O(n²) path, and 14.6M wasted light iterations.

- **Estimated savings:** 5-15s (eliminating 14.6M useless iterations + cluster merge overhead)
- **Effort:** Medium — new sky-only list, modified iteration in `GatherSampleLight_CollectGPURays`
- **Risk:** Low — sky lights are global (PVS always visible from everywhere by definition), so per-cluster filtering gains nothing for them
- **Verdict:** ✅ **High priority** — largest single optimization opportunity

### Sub-strategy 1c: Split `BuildFacelights` into sample-gen and lighting passes

Separate `BuildFacelights` into:
1. `BuildFaceSamples` — InitLightinfo, CalcPoints, InitSampleInfo, ComputeIlluminationPointAndNormals, AllocateLightstyleSamples
2. `GatherFaceLighting` — light iteration (sky-only on GPU path, full on CPU path)

This would allow the GPU path to skip the lighting pass entirely for faces where sky lights contribute nothing (95.8% of samples are zero in current run), though the current architecture already skips via dot-product checks.

- **Estimated savings:** Unclear — depends on profiling the split. The sample generation cost (CalcPoints, GetPhongNormal, ClusterFromPoint) may be 30-40s of the 43.61s since light iteration is already minimal.
- **Effort:** High — requires careful refactoring of a monolithic function with many side effects
- **Risk:** Medium — touching the core lighting loop risks accuracy regressions
- **Verdict:** ⚠️ **Investigate but don't implement yet** — measure sample-gen cost first

---

## Strategy 2: Reduce `FinalizeAndSupersample` Overhead (11.84s)

**The problem:** [FinalizeAndSupersample](file:///e:/GitHub/vrad-rtx/src/utils/vrad/lightmap.cpp#L3465-L3498) re-creates `lightinfo_t` and `SSE_SampleInfo_t` per face via `InitLightinfo` and `InitSampleInfo`. These involve:
- Plane lookups, texinfo parsing, face winding computation
- Same work `BuildFacelights` already did for every face

Then it runs `BuildSupersampleFaceLights` (full CPU supersampling) + `BuildPatchLights`.

### Option: Cache `lightinfo_t` from `BuildFacelights`

Store the `lightinfo_t` computed in `BuildFacelights` in a per-face array, reuse it in `FinalizeAndSupersample` to skip re-computation.

- **Estimated savings:** 1-3s (InitLightinfo is not the dominant cost; supersampling itself is ~8-10s)
- **Effort:** Medium — 30,301 × sizeof(lightinfo_t) memory cost, thread-safety considerations
- **Risk:** Low
- **Verdict:** ⚠️ **Marginal** — most time is in `BuildSupersampleFaceLights` itself, which can't be cached

---

## Strategy 3: Reduce VisMatrix CPU Prep (26.04s of 28.82s)

**The problem:** VisMatrix spends 90.4% of wall time on CPU prep ([BuildVisRow](file:///e:/GitHub/vrad-rtx/src/utils/vrad/vismat.cpp#L442-L494)):
- Iterating 330 clusters × leaves × leaffaces 
- Hierarchical patch subdivision (`TestPatchToPatch` recursion)
- Form factor geometry (dot products, area checks)
- Building ray lists for GPU submission

The GPU only handles the ray tracing after the CPU has done all the geometric setup. 1.33 billion rays are traced, but CPU prep takes 26s versus 2.78s of GPU wait.

### Option: GPU-side hierarchical subdivision

Move the patch hierarchy traversal and form factor geometry to a CUDA kernel. The GPU would receive cluster→face→patch data and do the subdivision + ray generation in one kernel, then trace inline.

- **Estimated savings:** Potentially 15-20s
- **Effort:** Very High — requires reimplementing recursive `TestPatchToPatch` hierarchy in CUDA, managing linked-list-based patch structures on GPU, handling variable-depth recursion
- **Risk:** High — the brute-force GPU path was already tried and disabled (line 682-693: `BuildVisLeafs_GPU` disabled because it "creates too many transfers ~1.7 billion"). The hierarchical subdivision is what prunes this to tractable levels, and that pruning logic is deeply CPU-oriented.
- **Verdict:** ❌ **Not worth it** — the architectural mismatch between recursive hierarchy traversal and GPU execution makes this impractical. The CPU is genuinely doing work the GPU can't efficiently replicate.

---

## Strategy 4: Move Sky Lights to GPU Kernel

**The problem:** The GPU direct lighting kernel handles point/surface/spot lights. Sky/ambient lights are explicitly processed on CPU in `GatherSampleLight_CollectGPURays` using the full SSE path (`GatherSampleLightSSE`).

### Option: Add sky light support to the CUDA direct lighting kernel

Extend the OptiX `__raygen__direct_lighting` kernel to handle `emit_skylight` and `emit_skyambient` light types, with appropriate sky tracing logic (parallel rays to sky normal, hemisphere sampling for ambient).

- **Estimated savings:** Would eliminate the remaining CPU light iteration entirely (the 14.6M iterations), saving an estimated 5-10s. However, this saving is **shared with Strategy 1b** — if sky lights are iterated separately (1b), the per-light cost drops dramatically regardless.
- **Effort:** Very High — sky light tracing has different geometry (parallel rays vs point-to-point), requires hemisphere sampling for `emit_skyambient`, and involves `ComputeSunAmount` logic
- **Risk:** High — sky lights are accuracy-critical (they affect large surface areas); any GPU/CPU floating-point divergence creates visible artifacts
- **Verdict:** ❌ **Not worth the risk** vs Strategy 1b — 1b gets most of the savings for 10% of the effort, and sky lights are too few (typically 1-5) to benefit from GPU parallelism

---

## Strategy 5: Dead Code Removal (Minor Cleanup)

The GPU path carries several vestiges:
- `FlushAllThreadShadowRays()` call in [vrad.cpp:2198](file:///e:/GitHub/vrad-rtx/src/utils/vrad/vrad.cpp#L2198) — always a no-op (0.00s)
- `DeferredShadowRay_t` metadata buffers in `lightmap.cpp` — allocated but never used
- `GPUShadowRayBatch` struct in `direct_lighting_gpu_vrad.cpp` — unused since Strategy 5 (Meta-Batching) was replaced

- **Estimated savings:** ~0s (allocations are negligible)
- **Effort:** Low
- **Risk:** None
- **Verdict:** ✅ **Do it** for code hygiene

---

## Recommended Priority Order

| # | Strategy | Est. Savings | Effort | Risk | Go? |
|---|---|---|---|---|---|
| 1 | **1b: Sky-only light list** | 5-15s | Medium | Low | ✅ Yes |
| 2 | **1a: Remove dead shadow batching** | 0-1s | Low | None | ✅ Yes |
| 3 | **5: Dead code cleanup** | ~0s | Low | None | ✅ Yes |
| 4 | **2: Cache lightinfo_t** | 1-3s | Medium | Low | ⚠️ Maybe |
| 5 | **1c: Split BuildFacelights** | Unknown | High | Medium | ⚠️ Measure first |
| 6 | **3: GPU VisMatrix hierarchy** | 15-20s | Very High | High | ❌ No |
| 7 | **4: GPU sky lights** | 5-10s | Very High | High | ❌ No |

**Best-case net gain from recommended items (1b + 1a + 5):** ~5-16s, reducing GPU path from ~105s to ~89-100s.

> [!IMPORTANT]
> Strategy 1b (sky-only light list) is the clear winner. The GPU path currently iterates the **full per-cluster light list** (avg 119 lights) for every sample group, just to `continue` past 98% of them. A dedicated sky-light-only list should collapse the inner loop from ~119 iterations to ~1-5, saving the majority of the 14.6M light iterations and their associated cluster merging overhead.

## Verification Plan

### Automated Tests
Run the existing `test_vrad_optix.ps1` validation suite, which compares CPU-only vs GPU outputs:
```powershell
cd e:\GitHub\vrad-rtx\game\bin\x64
.\test_vrad_optix.ps1
```
- **Parity check:** Lightmap difference must remain ≤0.1% (currently 0.0197%)
- **Visual check:** Visual difference must remain <8% (currently 6.65%)
- **Timing check:** Direct Lighting phase wall clock should decrease; Total Elapsed should decrease
- **Light iteration stats:** "Light iterations" counter should drop from 14.6M to ~100K-500K (proportional to sky light count × face count)

### Manual Verification
After implementing Strategy 1b, compare the `Direct Lighting Statistics` block in the CUDA log:
- Confirm `GatherSampleLightSSE` calls match the number of sky/ambient lights × sample groups (not the full cluster light count)
- Confirm `FlushShadowRays: 0.00s` remains 0.00s (validating 1a)
