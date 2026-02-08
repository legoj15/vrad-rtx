//========= GPU Direct Lighting Implementation ============//
//
// Purpose: GPU-accelerated direct lighting using OptiX
//
//=============================================================================//

#include "direct_lighting_gpu.h"

#ifdef VRAD_RTX_CUDA_SUPPORT

#include "raytrace_optix.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>


// Static storage for GPU lights
static std::vector<GPULight> s_hostLights;
static GPULight *s_deviceLights = nullptr;
static int s_numLights = 0;
static bool s_initialized = false;

//-----------------------------------------------------------------------------
// Initialize GPU direct lighting with pre-converted lights
// Called from vrad side with already-converted GPULight array
//-----------------------------------------------------------------------------
void InitDirectLightingGPU(const GPULight *lights, int numLights) {
  if (s_initialized) {
    ShutdownDirectLightingGPU();
  }

  s_numLights = numLights;

  if (s_numLights == 0) {
    s_initialized = true;
    return;
  }

  // Copy to host vector
  s_hostLights.assign(lights, lights + numLights);

  // Allocate and upload to GPU
  size_t lightSize = s_numLights * sizeof(GPULight);
  cudaError_t err = cudaMalloc(&s_deviceLights, lightSize);
  if (err != cudaSuccess) {
    printf("InitDirectLightingGPU: cudaMalloc failed: %s\n",
           cudaGetErrorString(err));
    return;
  }

  err = cudaMemcpy(s_deviceLights, s_hostLights.data(), lightSize,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    printf("InitDirectLightingGPU: cudaMemcpy failed: %s\n",
           cudaGetErrorString(err));
    cudaFree(s_deviceLights);
    s_deviceLights = nullptr;
    return;
  }

  s_initialized = true;
  printf("InitDirectLightingGPU: Uploaded %d lights to GPU\n", s_numLights);
}

//-----------------------------------------------------------------------------
// Shutdown GPU direct lighting
//-----------------------------------------------------------------------------
void ShutdownDirectLightingGPU() {
  if (s_deviceLights) {
    cudaFree(s_deviceLights);
    s_deviceLights = nullptr;
  }
  s_hostLights.clear();
  s_numLights = 0;
  s_initialized = false;
}

//-----------------------------------------------------------------------------
// Trace shadow rays using OptiX
//-----------------------------------------------------------------------------
void TraceShadowBatch(const GPUShadowRay *rays, GPUShadowResult *results,
                      int numRays) {
  if (!s_initialized || numRays <= 0) {
    return;
  }

  // Convert GPUShadowRay to RayBatch for existing TraceBatch infrastructure
  std::vector<RayBatch> rayBatch(numRays);
  std::vector<RayResult> rayResults(numRays);

  for (int i = 0; i < numRays; i++) {
    rayBatch[i].origin = rays[i].origin;
    rayBatch[i].direction = rays[i].direction;
    rayBatch[i].tmin = 1e-4f; // Small offset to avoid self-intersection
    rayBatch[i].tmax = rays[i].tmax;
    rayBatch[i].skip_id = -1; // No skip for shadow rays
  }

  // Use existing TraceBatch
  RayTraceOptiX::TraceBatch(rayBatch.data(), rayResults.data(), numRays);

  // Convert results
  for (int i = 0; i < numRays; i++) {
    // Visible if no hit (hit_id == -1)
    results[i].visible = (rayResults[i].hit_id == -1) ? 1 : 0;
    results[i].hitT = rayResults[i].hit_t;
  }
}

//-----------------------------------------------------------------------------
// Get GPU light count
//-----------------------------------------------------------------------------
int GetGPULightCount() { return s_numLights; }

//-----------------------------------------------------------------------------
// Get GPU lights device pointer
//-----------------------------------------------------------------------------
GPULight *GetGPULights() { return s_deviceLights; }

//-----------------------------------------------------------------------------
// Get host light by index (for falloff calculations on CPU)
//-----------------------------------------------------------------------------
const GPULight *GetHostLight(int index) {
  if (index >= 0 && index < s_numLights) {
    return &s_hostLights[index];
  }
  return nullptr;
}

#endif // VRAD_RTX_CUDA_SUPPORT
