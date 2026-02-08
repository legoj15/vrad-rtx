//========================================================================
// optix_kernels.cu - OptiX 9.1 Ray Tracing Kernels for VRAD RTX
// Compiled to PTX and loaded at runtime
//========================================================================

#include "visibility_gpu.h"
#include <optix.h>
#include <optix_device.h>

// Include shared structures (same as CUDA version)
// Note: This is compiled separately, so we define the structs inline
// UPDATE: Now we use visibility_gpu.h which includes raytrace_shared.h
// So we must NOT redefine these structs.

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
