//========= Copyright Valve Corporation, All rights reserved. ============//
//
// Purpose:
//
// $Workfile:     $
// $Date:         $
//
//-----------------------------------------------------------------------------
// $Log: $
//
// $NoKeywords: $
//=============================================================================//

#ifndef LIGHTMAP_H
#define LIGHTMAP_H
#pragma once

// vrad.h needed for LightingValue_t definition
#include "mathlib/bumpvects.h"
#include "vrad.h"

typedef struct {
  dface_t *faces[2];
  Vector interface_normal;
  qboolean coplanar;
} edgeshare_t;

extern edgeshare_t *edgeshare;
void AllocateEdgeshare();
void FreeEdgeshare();

//==============================================

// This is incremented each time BuildFaceLights and FinalLightFace
// are called. It's used for a status bar in WorldCraft.
extern int g_iCurFace;

extern int vertexref[MAX_MAP_VERTS];
extern int *vertexface[MAX_MAP_VERTS];

struct faceneighbor_t {
  int numneighbors; // neighboring faces that share vertices
  int *neighbor;    // neighboring face list (max of 64)

  Vector *normal;    // adjusted normal per vertex
  Vector facenormal; // face normal

  bool bHasDisp; // is this surface a displacement surface???
};

extern faceneighbor_t faceneighbor[MAX_MAP_FACES];

//==============================================

struct sample_t {
  // in local luxel space
  winding_t *w;
  int s, t;
  Vector2D coord;
  Vector2D mins;
  Vector2D maxs;
  // in world units
  Vector pos;
  Vector normal;
  float area;
};

struct facelight_t {
  // irregularly shaped light sample data, clipped by face and luxel grid
  int numsamples;
  sample_t *sample;
  LightingValue_t *light[MAXLIGHTMAPS]
                        [NUM_BUMP_VECTS +
                         1]; // result of direct illumination, indexed by sample

#ifdef VRAD_RTX_CUDA_SUPPORT
  // GPU point light contributions stored separately for subtract/restore.
  // Before gradient detection, gpu_point[] is subtracted from fl.light[];
  // after supersampling, it is restored. This lets the supersampler skip
  // re-evaluating point lights (CPU handles only sky at sub-positions),
  // saving ~7 seconds.
  LightingValue_t *gpu_point[MAXLIGHTMAPS][NUM_BUMP_VECTS + 1];
#endif

  // regularly spaced lightmap grid
  int numluxels;
  Vector *luxel;        // world space position of luxel
  Vector *luxelNormals; // world space normal of luxel
  float worldAreaPerLuxel;
};

// Forward declaration (defined in vrad.h)
struct directlight_t;

extern directlight_t *activelights;
extern directlight_t *freelights;

extern facelight_t facelight[MAX_MAP_FACES];
extern int numdlights;

// Per-cluster light list globals (defined in lightmap.cpp)
extern directlight_t **g_ClusterLightFlat;
extern int *g_ClusterLightOffsets;
extern int *g_nClusterLights;
extern int g_nTotalClusterLightEntries;

//==============================================

struct lightinfo_t {
  vec_t facedist;
  Vector facenormal;

  Vector facemid; // world coordinates of center

  Vector modelorg; // for origined bmodels

  Vector luxelOrigin;
  Vector
      worldToLuxelSpace[2]; // s = (world - luxelOrigin) . worldToLuxelSpace[0],
                            // t = (world - luxelOrigin) . worldToLuxelSpace[1]
  Vector luxelToWorldSpace[2]; // world = luxelOrigin + s * luxelToWorldSpace[0]
                               // + t * luxelToWorldSpace[1]

  int facenum;
  dface_t *face;

  int isflat;
  int hasbumpmap;
};

struct SSE_SampleInfo_t {
  int m_FaceNum;
  int m_WarnFace;
  dface_t *m_pFace;
  facelight_t *m_pFaceLight;
  int m_LightmapWidth;
  int m_LightmapHeight;
  int m_LightmapSize;
  int m_NormalCount;
  int m_iThread;
  texinfo_t *m_pTexInfo;
  bool m_IsDispFace;

  int m_NumSamples;
  int m_NumSampleGroups;
  int m_Clusters[4];
  FourVectors m_Points;
  FourVectors m_PointNormals[NUM_BUMP_VECTS + 1];
};

extern void InitLightinfo(lightinfo_t *l, int facenum);

void FreeDLights();

void ExportDirectLightsToWorldLights();

// Direct Lighting diagnostic counters (defined in lightmap.cpp)
extern long g_nLightsSkippedDistance[MAX_TOOL_THREADS];
extern long g_nLightsSkippedPVS[MAX_TOOL_THREADS];
extern long g_nLightsSkippedZeroDot[MAX_TOOL_THREADS];
extern long g_nGatherSSECalls[MAX_TOOL_THREADS];
extern long g_nSSGradientQualified[MAX_TOOL_THREADS];
extern long g_nSSGradientTotal[MAX_TOOL_THREADS];

// BuildFacelights sub-phase timing (per-thread accumulators)
extern double g_flBFL_Setup[MAX_TOOL_THREADS];
extern double g_flBFL_IllumNormals[MAX_TOOL_THREADS];
extern double g_flBFL_SkyGather[MAX_TOOL_THREADS];
extern long g_nBFL_FacesProcessed[MAX_TOOL_THREADS];

#ifdef VRAD_RTX_CUDA_SUPPORT
void FinalizeAndSupersample(int iThread, int facenum);
void BuildGPUSceneData();
void ShutdownGPUSceneDataBridge();
void LaunchGPUDirectLighting();
void DownloadAndApplyGPUResults();
#endif

#endif // LIGHTMAP_H
