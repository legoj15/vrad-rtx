//========= GPU Direct Lighting - VRAD Integration ============//
//
// Purpose: Integrate GPU direct lighting with VRAD lightmap.cpp
//          Converts directlight_t to GPULight and provides batched shadow
//          tracing
//
//=============================================================================//

#include "vrad.h"

#ifdef VRAD_RTX_CUDA_SUPPORT

#include "direct_lighting_gpu.h"

//-----------------------------------------------------------------------------
// Convert directlight_t to GPULight
//-----------------------------------------------------------------------------
static GPULight ConvertDirectLight(const directlight_t *dl) {
  GPULight gpu;

  gpu.origin.x = dl->light.origin.x;
  gpu.origin.y = dl->light.origin.y;
  gpu.origin.z = dl->light.origin.z;

  gpu.intensity.x = dl->light.intensity.x;
  gpu.intensity.y = dl->light.intensity.y;
  gpu.intensity.z = dl->light.intensity.z;

  gpu.normal.x = dl->light.normal.x;
  gpu.normal.y = dl->light.normal.y;
  gpu.normal.z = dl->light.normal.z;

  gpu.type = dl->light.type;
  gpu.facenum = dl->facenum;

  gpu.constant_attn = dl->light.constant_attn;
  gpu.linear_attn = dl->light.linear_attn;
  gpu.quadratic_attn = dl->light.quadratic_attn;

  gpu.stopdot = dl->light.stopdot;
  gpu.stopdot2 = dl->light.stopdot2;
  gpu.exponent = dl->light.exponent;

  gpu.startFadeDistance = dl->m_flStartFadeDistance;
  gpu.endFadeDistance = dl->m_flEndFadeDistance;
  gpu.capDist = dl->m_flCapDist;

  return gpu;
}

//-----------------------------------------------------------------------------
// Initialize GPU direct lighting from VRAD's activelights
//-----------------------------------------------------------------------------
void InitGPUDirectLighting() {
  // Convert all active lights
  CUtlVector<GPULight> gpuLights;

  for (directlight_t *dl = activelights; dl != nullptr; dl = dl->next) {
    gpuLights.AddToTail(ConvertDirectLight(dl));
  }

  if (gpuLights.Count() > 0) {
    InitDirectLightingGPU(gpuLights.Base(), gpuLights.Count());
    Msg("GPU Direct Lighting: Initialized with %d lights\n", gpuLights.Count());
  }
}

//-----------------------------------------------------------------------------
// Shutdown GPU direct lighting
//-----------------------------------------------------------------------------
void ShutdownGPUDirectLighting() { ShutdownDirectLightingGPU(); }

//-----------------------------------------------------------------------------
// Thread-local shadow ray batch for GPU tracing
// Collects rays during GatherSampleLight, traces them all at once
//-----------------------------------------------------------------------------
struct GPUShadowRayBatch {
  static const int MAX_RAYS = 65536; // Max rays per batch

  GPUShadowRay rays[MAX_RAYS];
  int numRays;

  // For result application - store metadata for each ray
  struct RayMetadata {
    int sampleIndex;  // Which sample this ray is for
    int lightIndex;   // Which light (index into activelights)
    int bumpIndex;    // Which bump normal
    float falloffDot; // Pre-computed falloff * dot product
    float sunAmount;  // Sun amount for sky lights
  };
  RayMetadata metadata[MAX_RAYS];

  GPUShadowRayBatch() : numRays(0) {}

  void Clear() { numRays = 0; }

  bool IsFull() const { return numRays >= MAX_RAYS; }

  // Add a shadow ray to the batch
  // Returns true if added, false if batch is full
  bool AddRay(const Vector &samplePos, const Vector &lightPos, int sampleIdx,
              int lightIdx, int bumpIdx, float falloffDot, float sunAmount) {
    if (numRays >= MAX_RAYS)
      return false;

    Vector dir = lightPos - samplePos;
    float dist = dir.Length();
    if (dist < 0.001f)
      return false;

    dir /= dist; // Normalize

    GPUShadowRay &ray = rays[numRays];
    ray.origin.x = samplePos.x + dir.x * 0.1f; // Offset by epsilon
    ray.origin.y = samplePos.y + dir.y * 0.1f;
    ray.origin.z = samplePos.z + dir.z * 0.1f;
    ray.direction.x = dir.x;
    ray.direction.y = dir.y;
    ray.direction.z = dir.z;
    ray.tmax = dist - 0.2f; // Stop slightly before light
    ray.sampleIndex = sampleIdx;
    ray.lightIndex = lightIdx;
    ray.faceIndex = 0; // Will be set by caller

    RayMetadata &meta = metadata[numRays];
    meta.sampleIndex = sampleIdx;
    meta.lightIndex = lightIdx;
    meta.bumpIndex = bumpIdx;
    meta.falloffDot = falloffDot;
    meta.sunAmount = sunAmount;

    numRays++;
    return true;
  }
};

// Thread-local batches (one per thread)
static GPUShadowRayBatch *g_pThreadBatches = nullptr;
static int g_nThreadBatchCount = 0;

//-----------------------------------------------------------------------------
// Initialize per-thread batches
//-----------------------------------------------------------------------------
void InitGPUShadowBatches(int numThreads) {
  if (g_pThreadBatches) {
    delete[] g_pThreadBatches;
  }
  g_pThreadBatches = new GPUShadowRayBatch[numThreads];
  g_nThreadBatchCount = numThreads;
  Msg("GPU Direct Lighting: Allocated %d thread-local shadow ray batches\n",
      numThreads);
}

//-----------------------------------------------------------------------------
// Cleanup per-thread batches
//-----------------------------------------------------------------------------
void ShutdownGPUShadowBatches() {
  if (g_pThreadBatches) {
    delete[] g_pThreadBatches;
    g_pThreadBatches = nullptr;
  }
  g_nThreadBatchCount = 0;
}

//-----------------------------------------------------------------------------
// Get thread-local batch
//-----------------------------------------------------------------------------
GPUShadowRayBatch *GetThreadShadowBatch(int iThread) {
  if (!g_pThreadBatches || iThread < 0 || iThread >= g_nThreadBatchCount) {
    return nullptr;
  }
  return &g_pThreadBatches[iThread];
}

//-----------------------------------------------------------------------------
// Flush and trace a thread's shadow ray batch
// Returns results in the provided array
//-----------------------------------------------------------------------------
void FlushShadowBatch(int iThread, GPUShadowResult *results) {
  GPUShadowRayBatch *batch = GetThreadShadowBatch(iThread);
  if (!batch || batch->numRays == 0)
    return;

  TraceShadowBatch(batch->rays, results, batch->numRays);
  batch->Clear();
}

#endif // VRAD_RTX_CUDA_SUPPORT
