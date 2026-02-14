//========================================================================
// optix_kernels.cu - OptiX 9.1 Ray Tracing Kernels for VRAD RTX
// Compiled to PTX and loaded at runtime
//========================================================================

#include "gpu_scene_data.h"
#include "visibility_gpu.h"
#include <optix.h>
#include <optix_device.h>

// Include shared structures (same as CUDA version)
// Note: This is compiled separately, so we define the structs inline
// UPDATE: Now we use visibility_gpu.h which includes raytrace_shared.h
// So we must NOT redefine these structs.

//-----------------------------------------------------------------------------
// GPU Light Structure - must match GPULight in direct_lighting_gpu.h
// NOTE: Uses explicit float fields (not float3_t) to guarantee identical
// struct layout between MSVC (host) and NVCC (device PTX) compilation.
//-----------------------------------------------------------------------------
struct GPULight {
  float origin_x, origin_y, origin_z;
  float intensity_x, intensity_y, intensity_z;
  float normal_x, normal_y, normal_z; // For emit_surface and emit_spotlight

  int type; // emit_point=0, emit_surface=1, emit_spotlight=2, emit_skylight=3
  int facenum; // -1 for point lights, face index for surface lights

  // Attenuation
  float constant_attn;
  float linear_attn;
  float quadratic_attn;

  // Spotlight parameters
  float stopdot;  // cos(inner cone angle)
  float stopdot2; // cos(outer cone angle)
  float exponent;

  // Fade distances
  float startFadeDistance;
  float endFadeDistance;
  float capDist;
};

// Light type enums (must match emittype_t in bspfile.h)
#define EMIT_SURFACE 0
#define EMIT_POINT 1
#define EMIT_SPOTLIGHT 2
#define EMIT_SKYLIGHT 3
#define EMIT_QUAKELIGHT 4
#define EMIT_SKYAMBIENT 5

//-----------------------------------------------------------------------------
// Launch Parameters - must match host struct exactly
//-----------------------------------------------------------------------------
struct OptixLaunchParams {
  const RayBatch *rays;
  RayResult *results;
  int num_rays;
  OptixTraversableHandle traversable;
  const CUDATriangle *triangles;

  // Visibility Extension
  const int *shooter_patches;
  int num_shooters;
  const int *visible_clusters;
  int num_visible_clusters;
  GPUVisSceneData vis_scene_data;
  VisiblePair *visible_pairs;
  int *pair_count_atomic;
  int max_pairs;

  // Direct Lighting Extension (Phase 2)
  const GPUSampleData *d_samples;
  const GPULight *d_lights;
  const GPUClusterLightList *d_clusterLists;
  const int *d_clusterLightIndices;
  const GPUFaceInfo *d_faceInfos;
  GPULightOutput *d_lightOutput;
  int num_samples;
  int num_lights;
  int num_clusters;

  // Sky Light Extension (Phase 2b)
  const float3 *d_skyDirs; // Precomputed hemisphere sample directions
  int numSkyDirs;          // Number of sky sample directions (162 default)
  float sunAngularExtent;  // Area sun jitter (0 = point sun)
  int numSunSamples;       // Samples for area sun (30 default, 0 = point sun)

  // Sun Shadow Anti-aliasing
  int sunShadowSamples;  // Sub-luxel position samples (default 16)
  float sunShadowRadius; // World-space jitter radius in units (default 4.0)
};

extern "C" __constant__ OptixLaunchParams params;

//-----------------------------------------------------------------------------
// Ray payload - data passed between programs
// Now includes skip_id for self-intersection filtering
//-----------------------------------------------------------------------------
struct RayPayload {
  float hit_t;
  int hit_id;
  float3 normal;
  int skip_id; // Triangle ID to skip (passed through payload)
};

//-----------------------------------------------------------------------------
// Ray Generation Program - launches one ray per thread
//-----------------------------------------------------------------------------
extern "C" __global__ void __raygen__visibility() {
  const uint3 idx = optixGetLaunchIndex();
  const int rayIdx = idx.x;

  if (rayIdx >= params.num_rays)
    return;

  // Get ray parameters
  const RayBatch &ray = params.rays[rayIdx];

  float3 origin = make_float3(ray.origin.x, ray.origin.y, ray.origin.z);
  float3 direction =
      make_float3(ray.direction.x, ray.direction.y, ray.direction.z);

  // Initialize payload - include skip_id for filtering
  unsigned int p0 = __float_as_uint(1e30f);    // hit_t
  unsigned int p1 = (unsigned int)-1;          // hit_id
  unsigned int p2 = __float_as_uint(0.0f);     // normal.x
  unsigned int p3 = __float_as_uint(0.0f);     // normal.y
  unsigned int p4 = __float_as_uint(0.0f);     // normal.z
  unsigned int p5 = (unsigned int)ray.skip_id; // skip_id for filtering

  // Use small epsilon for tmin to avoid self-intersection at ray origin
  // 1e-4 is optimal for Source engine units - smaller values cause self-hits
  float tmin = ray.tmin;
  if (tmin < 1e-4f)
    tmin = 1e-4f;

  // Trace ray - OPTIX_RAY_FLAG_NONE already means no face culling
  optixTrace(params.traversable, origin, direction,
             tmin,     // tmin (with epsilon to avoid self-hits)
             ray.tmax, // tmax
             0.0f,     // rayTime
             OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
             0, // SBT offset
             1, // SBT stride
             0, // missSBTIndex
             p0, p1, p2, p3, p4, p5);

  // Write results
  RayResult &result = params.results[rayIdx];
  result.hit_t = __uint_as_float(p0);
  result.hit_id = (int)p1;
  result.normal.x = __uint_as_float(p2);
  result.normal.y = __uint_as_float(p3);
  result.normal.z = __uint_as_float(p4);
}

//-----------------------------------------------------------------------------
// Any Hit Program - called for every potential intersection
// Used to filter out self-intersections (skip_id)
//-----------------------------------------------------------------------------
extern "C" __global__ void __anyhit__visibility() {
  // Get the primitive (triangle) index
  const int primIdx = optixGetPrimitiveIndex();

  // Get skip_id from payload
  const int skip_id = (int)optixGetPayload_5();

  // Get triangle's actual ID
  const CUDATriangle &tri = params.triangles[primIdx];

  // If this triangle matches skip_id, reject the intersection
  if (tri.triangle_id == skip_id) {
    optixIgnoreIntersection();
    return;
  }

  // Otherwise, accept the intersection (continue to closest-hit)
}

//-----------------------------------------------------------------------------
// Closest Hit Program - called when ray hits a triangle
//-----------------------------------------------------------------------------
extern "C" __global__ void __closesthit__visibility() {
  // Get the primitive (triangle) index
  const int primIdx = optixGetPrimitiveIndex();

  // Get hit distance
  const float t = optixGetRayTmax();

  // Get triangle normal from our data
  const CUDATriangle &tri = params.triangles[primIdx];

  // Update payload with hit information
  optixSetPayload_0(__float_as_uint(t));
  optixSetPayload_1((unsigned int)primIdx);
  optixSetPayload_2(__float_as_uint(tri.nx));
  optixSetPayload_3(__float_as_uint(tri.ny));
  optixSetPayload_4(__float_as_uint(tri.nz));
}

//-----------------------------------------------------------------------------
// Miss Program - called when ray doesn't hit anything
//-----------------------------------------------------------------------------
extern "C" __global__ void __miss__visibility() {
  // No hit - set hit_id to -1
  optixSetPayload_0(__float_as_uint(1e30f));
  optixSetPayload_1((unsigned int)-1);
  optixSetPayload_2(__float_as_uint(0.0f));
  optixSetPayload_3(__float_as_uint(0.0f));
  optixSetPayload_4(__float_as_uint(0.0f));
}

//-----------------------------------------------------------------------------
// Cluster Visibility Ray Generation
//-----------------------------------------------------------------------------
extern "C" __global__ void __raygen__cluster_visibility() {
  const uint3 idx = optixGetLaunchIndex();
  const int shooterIdx = idx.x;
  const int clusterIdx = idx.y;

  if (shooterIdx >= params.num_shooters ||
      clusterIdx >= params.num_visible_clusters)
    return;

  // Get shooter patch
  int shooterPatchID = params.shooter_patches[shooterIdx];
  const GPUPatch &shooter = params.vis_scene_data.patches[shooterPatchID];

  // Get target cluster
  int targetClusterID = params.visible_clusters[clusterIdx];

  // Iterate leaves in target cluster
  // Note: This is an inner loop inside the thread.
  // Ideally we'd parallelize this more if leaves are many.
  // But for a first pass, one thread per Cluster-Shooter pair is decent
  // granularity.

  int leafStart = params.vis_scene_data.clusterLeafOffsets[targetClusterID];
  int leafEnd = params.vis_scene_data.clusterLeafOffsets[targetClusterID + 1];

  for (int li = leafStart; li < leafEnd; li++) {
    int receiverPatchID = params.vis_scene_data.clusterLeafIndices[li];
    const GPUPatch &receiver = params.vis_scene_data.patches[receiverPatchID];

    // Skip self-face (don't shadow other patches on same face)
    if (receiver.faceNumber == shooter.faceNumber)
      continue;

    // Plane & Visibility Check Logic from vismat.cpp:
    // Test 1: Receiver origin must be in front of shooter's plane
    // if (DotProduct(patch2->origin, patch->normal) > patch->planeDist +
    // PLANE_TEST_EPSILON)
    float3 rxOrigin = receiver.origin;
    float3 sxNormal = shooter.normal;
    float dot1 = rxOrigin.x * sxNormal.x + rxOrigin.y * sxNormal.y +
                 rxOrigin.z * sxNormal.z;

    // PLANE_TEST_EPSILON 0.01
    if (dot1 <= shooter.planeDist + 0.01f)
      continue;

    // Test 2: Shooter origin must be in front of receiver's plane
    // (from TestPatchToFace pre-filter logic)
    float3 sxOrigin = shooter.origin;
    float3 rxNormal = receiver.normal;
    float dot2 = sxOrigin.x * rxNormal.x + sxOrigin.y * rxNormal.y +
                 sxOrigin.z * rxNormal.z;

    if (dot2 <= receiver.planeDist + 0.01f)
      continue;

    {
      // Setup Ray
      // p1 = shooter.origin + shooter.normal
      // p2 = receiver.origin + receiver.normal (SDK 2013 behavior)

      float3 startPos = make_float3(shooter.origin.x + shooter.normal.x,
                                    shooter.origin.y + shooter.normal.y,
                                    shooter.origin.z + shooter.normal.z);

      float3 endPos = make_float3(receiver.origin.x + receiver.normal.x,
                                  receiver.origin.y + receiver.normal.y,
                                  receiver.origin.z + receiver.normal.z);

      float3 dir = make_float3(endPos.x - startPos.x, endPos.y - startPos.y,
                               endPos.z - startPos.z);
      float len = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);

      if (len < 1e-4f)
        continue;

      float3 dirNorm = make_float3(dir.x / len, dir.y / len, dir.z / len);

      // Trace Visibility Ray
      unsigned int p0, p1, p2, p3, p4, p5;
      p5 = (unsigned int)-1; // Skip ID - don't self collide, but we offset
                             // origin anyway

      // optixTrace
      optixTrace(
          params.traversable, startPos, dirNorm,
          0.0f,        // tmin
          len - 1e-4f, // tmax (fixed epsilon subtraction instead of scaling)
          0.0f, OptixVisibilityMask(255),
          OPTIX_RAY_FLAG_DISABLE_ANYHIT,    // Shadow ray? We just want
                                            // boolean visibility
          0, 1, 0, p0, p1, p2, p3, p4, p5); // Payload

      // Check result
      // Our __miss__ sets hit_id to -1. __closesthit__ sets it to primIdx.
      int hitID = (int)p1;

      // If hitID is -1 (Miss), then the path is clear -> VISIBLE
      if (hitID == -1) {
        // Record Visibility
        int idx = atomicAdd(params.pair_count_atomic, 1);
        // Check bounds? (host handles overflow check on readback, but writing
        // OOB is bad) We really should pass buffer size. For now assuming 2M
        // is enough or we risk crash if extremely visible scene. Ideally we
        // add a bounds check here.

        if (idx < params.max_pairs) {
          params.visible_pairs[idx].shooter = shooterPatchID;
          params.visible_pairs[idx].receiver = receiverPatchID;
        }
      }
    }
  }
}

//=============================================================================
// Phase 2: Direct Lighting Kernel
//
// Each thread processes ONE lightmap sample.
// For each sample, iterates over all lights visible from the sample's PVS
// cluster, computes falloff + dot product (replicating CPU
// GatherSampleStandardLightSSE math), traces an inline shadow ray for
// occlusion, and atomicAdd's the contribution to the output buffer.
//
// Only handles emit_point (0), emit_surface (1), emit_spotlight (2).
// Sky lights (3, 4) stay on CPU.
//=============================================================================

// DIST_EPSILON from Source Engine (bspfile.h)
#define GPU_DIST_EPSILON 0.03125f

//-----------------------------------------------------------------------------
// Inline shadow trace: returns 1.0 if visible, 0.0 if occluded
//-----------------------------------------------------------------------------
__forceinline__ __device__ float TraceShadowRay(float3 origin, float3 direction,
                                                float tmax) {
  // Shadow ray: we only need boolean hit/miss
  // Use OPTIX_RAY_FLAG_NONE to allow any-hit to filter skip_id
  unsigned int p0 = __float_as_uint(1e30f);
  unsigned int p1 = (unsigned int)-1;
  unsigned int p2 = 0, p3 = 0, p4 = 0;
  unsigned int p5 = (unsigned int)-1; // No skip ID for shadow rays

  optixTrace(
      params.traversable, origin, direction,
      1e-3f, // tmin: avoid self-shadow artifacts (tested: 1e-4 worsened parity)
      tmax,  // tmax: distance to light
      0.0f,  // rayTime
      OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1,
      0, // SBT offset, stride, miss index
      p0, p1, p2, p3, p4, p5);

  // Miss (hit_id == -1) means visible
  return ((int)p1 == -1) ? 1.0f : 0.0f;
}

//-----------------------------------------------------------------------------
// Inline sky visibility trace: returns 1.0 if sky is visible, 0.0 if blocked
// "Does hit sky" semantics: trace toward sky direction.
//   - Miss = ray escapes to sky → visible (1.0)
//   - Hit sky triangle (TRACE_ID_SKY) → visible (1.0)
//   - Hit solid geometry → blocked (0.0)
//-----------------------------------------------------------------------------
__forceinline__ __device__ float TraceSkyRay(float3 origin, float3 direction,
                                             float tmax) {
  unsigned int p0 = __float_as_uint(1e30f);
  unsigned int p1 = (unsigned int)-1;
  unsigned int p2 = 0, p3 = 0, p4 = 0;
  unsigned int p5 = (unsigned int)-1;

  optixTrace(params.traversable, origin, direction,
             1e-3f, // tmin
             tmax,  // tmax
             0.0f,  // rayTime
             OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1,
             0, // SBT offset, stride, miss index
             p0, p1, p2, p3, p4, p5);

  int hitPrim = (int)p1;

  // Miss → sky visible
  if (hitPrim == -1)
    return 1.0f;

  // Check if closest hit is a sky triangle
  if (params.triangles[hitPrim].triangle_id & TRACE_ID_SKY_GPU)
    return 1.0f;

  // Hit solid geometry → blocked
  return 0.0f;
}

//-----------------------------------------------------------------------------
// Direct Lighting Ray Generation
// 1D launch: one thread per lightmap sample
//-----------------------------------------------------------------------------
extern "C" __global__ void __raygen__direct_lighting() {
  const int sampleIdx = optixGetLaunchIndex().x;

  if (sampleIdx >= params.num_samples)
    return;

  // Load sample data
  const GPUSampleData &sample = params.d_samples[sampleIdx];
  float3 samplePos =
      make_float3(sample.position.x, sample.position.y, sample.position.z);
  float3 sampleNormal =
      make_float3(sample.normal.x, sample.normal.y, sample.normal.z);
  int clusterIdx = sample.clusterIndex;

  // If sample has no valid cluster, skip (can't look up lights)
  if (clusterIdx < 0 || clusterIdx >= params.num_clusters)
    return;

  // Load per-face bump info
  const GPUFaceInfo &faceInfo = params.d_faceInfos[sample.faceIndex];
  int normalCount = faceInfo.normalCount; // 1 or 4

  // Build bump normal array for this face
  // bumpNormal[0] = flat normal (from sample), bumpNormal[1..3] = bump basis
  float3 bumpNormals[4];
  bumpNormals[0] = sampleNormal;
  if (normalCount > 1) {
    bumpNormals[1] = make_float3(faceInfo.bumpNormal0_x, faceInfo.bumpNormal0_y,
                                 faceInfo.bumpNormal0_z);
    bumpNormals[2] = make_float3(faceInfo.bumpNormal1_x, faceInfo.bumpNormal1_y,
                                 faceInfo.bumpNormal1_z);
    bumpNormals[3] = make_float3(faceInfo.bumpNormal2_x, faceInfo.bumpNormal2_y,
                                 faceInfo.bumpNormal2_z);
  }

  // Look up which lights are visible from this cluster
  const GPUClusterLightList &clusterList = params.d_clusterLists[clusterIdx];
  int lightOffset = clusterList.lightOffset;
  int lightCount = clusterList.lightCount;

  // Accumulators for this sample — one per bump vector
  // Point/surface/spot lights go here.
  float accumR[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float accumG[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float accumB[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  // Separate sun/sky accumulators removed — CPU evaluates sky now.
  // Only point/surface/spot lights are accumulated on GPU.

  // Iterate over all lights visible from this cluster
  for (int li = 0; li < lightCount; li++) {
    int lightIdx = params.d_clusterLightIndices[lightOffset + li];
    if (lightIdx < 0 || lightIdx >= params.num_lights)
      continue;

    const GPULight &light = params.d_lights[lightIdx];

    // ---------------------------------------------------------------
    // Sky light types — CPU handles sun and ambient sky evaluation.
    // Skip entirely on GPU to save compute time and VRAM.
    // ---------------------------------------------------------------
    if (light.type == EMIT_SKYLIGHT || light.type == EMIT_SKYAMBIENT)
      continue;

    // ---------------------------------------------------------------
    // Compute delta = lightOrigin - samplePos
    // ---------------------------------------------------------------
    float3 lightOrigin =
        make_float3(light.origin_x, light.origin_y, light.origin_z);
    float3 src = lightOrigin;

    float3 delta = make_float3(src.x - samplePos.x, src.y - samplePos.y,
                               src.z - samplePos.z);
    float dist2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;

    if (dist2 < 1e-10f)
      continue;

    float dist = sqrtf(dist2);
    float invDist = 1.0f / dist;
    delta.x *= invDist;
    delta.y *= invDist;
    delta.z *= invDist;

    // ---------------------------------------------------------------
    // Compute dot product for flat normal: N · delta
    // ---------------------------------------------------------------
    float dot = sampleNormal.x * delta.x + sampleNormal.y * delta.y +
                sampleNormal.z * delta.z;
    dot = fmaxf(dot, 0.0f);

    // ---------------------------------------------------------------
    // Hard falloff: zero contribution if past endFadeDistance
    // ---------------------------------------------------------------
    bool hasHardFalloff = (light.endFadeDistance > light.startFadeDistance);
    if (hasHardFalloff) {
      if (dist > light.endFadeDistance)
        continue;
    }

    // Clamp distance for falloff evaluation (CPU: max(1, min(dist, capDist)))
    float falloffEvalDist = fmaxf(dist, 1.0f);
    falloffEvalDist = fminf(falloffEvalDist, light.capDist);

    // ---------------------------------------------------------------
    // Compute falloff based on light type (matches CPU SSE exactly)
    // ---------------------------------------------------------------
    float falloff = 0.0f;
    float3 shadowOrigin = src;

    switch (light.type) {
    case EMIT_POINT: {
      // falloff = 1 / (constant + linear*d + quadratic*d²)
      float denom = light.constant_attn + light.linear_attn * falloffEvalDist +
                    light.quadratic_attn * falloffEvalDist * falloffEvalDist;
      falloff = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
      break;
    }

    case EMIT_SURFACE: {
      // dot2 = -delta · lightNormal (how much light faces sample)
      float3 lightNormal =
          make_float3(light.normal_x, light.normal_y, light.normal_z);
      float dot2 = -(delta.x * lightNormal.x + delta.y * lightNormal.y +
                     delta.z * lightNormal.z);
      dot2 = fmaxf(dot2, 0.0f);

      if (dot <= 0.0f || dot2 <= 0.0f) {
        falloff = 0.0f;
        break;
      }

      // falloff = dot2 / dist²
      falloff = (dist2 > 0.0f) ? (dot2 / dist2) : 0.0f;

      // CPU offsets shadow origin along light normal by DIST_EPSILON
      shadowOrigin.x += lightNormal.x * GPU_DIST_EPSILON;
      shadowOrigin.y += lightNormal.y * GPU_DIST_EPSILON;
      shadowOrigin.z += lightNormal.z * GPU_DIST_EPSILON;
      break;
    }

    case EMIT_SPOTLIGHT: {
      float3 lightNormal =
          make_float3(light.normal_x, light.normal_y, light.normal_z);
      float dot2 = -(delta.x * lightNormal.x + delta.y * lightNormal.y +
                     delta.z * lightNormal.z);

      // Outside outer cone entirely? Skip
      if (dot2 <= light.stopdot2) {
        falloff = 0.0f;
        break;
      }

      // Zero dot if outside cone (CPU: dot = AndSIMD(inCone, dot))
      if (dot2 <= light.stopdot2)
        dot = 0.0f;

      // Point-light-style attenuation
      float denom = light.constant_attn + light.linear_attn * falloffEvalDist +
                    light.quadratic_attn * falloffEvalDist * falloffEvalDist;
      falloff = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
      falloff *= dot2;

      // Fringe interpolation: between stopdot (inner) and stopdot2 (outer)
      if (dot2 <= light.stopdot) {
        float range = light.stopdot - light.stopdot2;
        float mult = (range > 0.0f) ? ((dot2 - light.stopdot2) / range) : 0.0f;
        mult = fminf(fmaxf(mult, 0.0f), 1.0f);

        // Apply exponent (CPU uses PowSIMD which is fixed-point)
        if (light.exponent != 0.0f && light.exponent != 1.0f) {
          mult = powf(mult, light.exponent);
        }
        falloff *= mult;
      }
      break;
    }

    default:
      continue;
    } // switch

    // ---------------------------------------------------------------
    // Hard falloff fade: quintic smoothstep
    // ---------------------------------------------------------------
    if (hasHardFalloff) {
      float range = light.endFadeDistance - light.startFadeDistance;
      float t =
          (range > 0.0f) ? ((dist - light.startFadeDistance) / range) : 0.0f;
      t = fminf(fmaxf(t, 0.0f), 1.0f);
      t = 1.0f - t;
      float fade = t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
      falloff *= fade;
    }

    // Quick check: if falloff * flat-normal dot is zero, skip entirely
    float contribution = falloff * dot;
    if (contribution <= 0.0f)
      continue;

    // ---------------------------------------------------------------
    // Shadow ray: trace from sample toward light
    // ---------------------------------------------------------------
    float3 shadowDir =
        make_float3(shadowOrigin.x - samplePos.x, shadowOrigin.y - samplePos.y,
                    shadowOrigin.z - samplePos.z);
    float shadowDist =
        sqrtf(shadowDir.x * shadowDir.x + shadowDir.y * shadowDir.y +
              shadowDir.z * shadowDir.z);

    if (shadowDist < 1e-6f)
      continue;

    float invShadowDist = 1.0f / shadowDist;
    shadowDir.x *= invShadowDist;
    shadowDir.y *= invShadowDist;
    shadowDir.z *= invShadowDist;

    float visibility = TraceShadowRay(samplePos, shadowDir, shadowDist);

    if (visibility <= 0.0f)
      continue;

    // ---------------------------------------------------------------
    // Accumulate per bump vector: falloff * dot[n] * visibility * intensity
    // dot[0] is the flat normal (already computed); dot[1..3] are bump basis
    // ---------------------------------------------------------------
    float scale0 = falloff * dot * visibility;
    accumR[0] += scale0 * light.intensity_x;
    accumG[0] += scale0 * light.intensity_y;
    accumB[0] += scale0 * light.intensity_z;

    // Compute per-bump-vector contributions (only for bumpmapped faces)
    for (int n = 1; n < normalCount; n++) {
      float bDot = bumpNormals[n].x * delta.x + bumpNormals[n].y * delta.y +
                   bumpNormals[n].z * delta.z;
      bDot = fmaxf(bDot, 0.0f);
      float bScale = falloff * bDot * visibility;
      accumR[n] += bScale * light.intensity_x;
      accumG[n] += bScale * light.intensity_y;
      accumB[n] += bScale * light.intensity_z;
    }

  } // for each light

  // ---------------------------------------------------------------
  // Write results via atomicAdd
  // ---------------------------------------------------------------
  for (int n = 0; n < normalCount; n++) {
    if (accumR[n] > 0.0f || accumG[n] > 0.0f || accumB[n] > 0.0f) {
      atomicAdd(&params.d_lightOutput[sampleIdx].r[n], accumR[n]);
      atomicAdd(&params.d_lightOutput[sampleIdx].g[n], accumG[n]);
      atomicAdd(&params.d_lightOutput[sampleIdx].b[n], accumB[n]);
    }
    // Only write point/surface/spot light results.
    // Sun/sky removed — CPU handles those.
  }
}
