// ==========================================================================
// VRAD SIMD Instruction Set Benchmark
// Tests SSE2 vs SSE4.1 vs AVX vs AVX2 vs AVX-512 performance
// for computational patterns found in VRAD's light gathering pipeline.
// ==========================================================================
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <intrin.h>

// ---------------------------------------------------------------------------
// CPUID Detection
// ---------------------------------------------------------------------------
struct CPUFeatures {
  bool sse2, sse41, avx, avx2, fma3, avx512f;
  char brand[49];
};

static CPUFeatures DetectCPU() {
  CPUFeatures f{};
  int info[4];
  __cpuid(info, 0);
  int maxId = info[0];

  // Brand string
  __cpuid((int *)&f.brand[0], 0x80000002);
  __cpuid((int *)&f.brand[16], 0x80000003);
  __cpuid((int *)&f.brand[32], 0x80000004);
  f.brand[48] = 0;

  if (maxId >= 1) {
    __cpuid(info, 1);
    f.sse2 = (info[3] & (1 << 26)) != 0;
    f.sse41 = (info[2] & (1 << 19)) != 0;
    f.avx = (info[2] & (1 << 28)) != 0;
    f.fma3 = (info[2] & (1 << 12)) != 0;
  }
  if (maxId >= 7) {
    __cpuidex(info, 7, 0);
    f.avx2 = (info[1] & (1 << 5)) != 0;
    f.avx512f = (info[1] & (1 << 16)) != 0;
  }
  return f;
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
using hrclock = std::chrono::high_resolution_clock;

static double NsPerOp(hrclock::duration elapsed, int64_t ops) {
  return std::chrono::duration<double, std::nano>(elapsed).count() / ops;
}

// Prevent dead-code elimination
static volatile float g_sink;
static alignas(32) float g_sinkbuf[8];

template <typename T> __forceinline void DoNotOptimize(T val) {
  if constexpr (sizeof(T) == sizeof(__m256))
    _mm256_store_ps(g_sinkbuf, *reinterpret_cast<__m256 *>(&val));
  else if constexpr (sizeof(T) == sizeof(__m128))
    _mm_store_ps(g_sinkbuf, *reinterpret_cast<__m128 *>(&val));
  else
    g_sink = *reinterpret_cast<float *>(&val);
  _ReadWriteBarrier();
}

// ---------------------------------------------------------------------------
// Benchmark parameters
// ---------------------------------------------------------------------------
static constexpr int64_t ITERATIONS = 20'000'000;
static constexpr int64_t FLOOR_ITERS = 10'000'000;
static constexpr int64_t BATCH_SIZE = 4096;

// ---------------------------------------------------------------------------
// Kernel 1: FMA Chain (attenuation: 1 / (quad*d^2 + lin*d + const))
// This is the #1 upgrade opportunity: MaddSIMD becomes a single FMA op.
// ---------------------------------------------------------------------------
static double Bench_FMAChain_SSE2() {
  __m128 d = _mm_set_ps(1.5f, 2.0f, 3.0f, 4.0f);
  __m128 q = _mm_set1_ps(0.001f);
  __m128 lin = _mm_set1_ps(0.01f);
  __m128 con = _mm_set1_ps(1.0f);
  __m128 acc = _mm_setzero_ps();
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < ITERATIONS; i++) {
    // quad * d^2
    __m128 d2 = _mm_mul_ps(d, d);
    __m128 f = _mm_mul_ps(q, d2);
    // + lin * d
    f = _mm_add_ps(f, _mm_mul_ps(lin, d));
    // + const
    f = _mm_add_ps(f, con);
    // reciprocal via rcp + NR
    __m128 est = _mm_rcp_ps(f);
    est = _mm_sub_ps(_mm_add_ps(est, est), _mm_mul_ps(f, _mm_mul_ps(est, est)));
    acc = _mm_add_ps(acc, est);
    d = _mm_add_ps(d, _mm_set1_ps(0.0001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, ITERATIONS);
}

static double Bench_FMAChain_FMA3() {
  __m128 d = _mm_set_ps(1.5f, 2.0f, 3.0f, 4.0f);
  __m128 q = _mm_set1_ps(0.001f);
  __m128 lin = _mm_set1_ps(0.01f);
  __m128 con = _mm_set1_ps(1.0f);
  __m128 acc = _mm_setzero_ps();
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < ITERATIONS; i++) {
    __m128 d2 = _mm_mul_ps(d, d);
    __m128 f = _mm_fmadd_ps(q, d2, _mm_fmadd_ps(lin, d, con));
    __m128 est = _mm_rcp_ps(f);
    est = _mm_fnmadd_ps(f, _mm_mul_ps(est, est), _mm_add_ps(est, est));
    acc = _mm_add_ps(acc, est);
    d = _mm_add_ps(d, _mm_set1_ps(0.0001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, ITERATIONS);
}

static double Bench_FMAChain_AVX() {
  __m256 d = _mm256_set_ps(1.5f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
  __m256 q = _mm256_set1_ps(0.001f);
  __m256 lin = _mm256_set1_ps(0.01f);
  __m256 con = _mm256_set1_ps(1.0f);
  __m256 acc = _mm256_setzero_ps();
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < ITERATIONS; i++) {
    __m256 d2 = _mm256_mul_ps(d, d);
    __m256 f = _mm256_fmadd_ps(q, d2, _mm256_fmadd_ps(lin, d, con));
    __m256 est = _mm256_rcp_ps(f);
    est = _mm256_fnmadd_ps(f, _mm256_mul_ps(est, est), _mm256_add_ps(est, est));
    acc = _mm256_add_ps(acc, est);
    d = _mm256_add_ps(d, _mm256_set1_ps(0.0001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, ITERATIONS);
}

// ---------------------------------------------------------------------------
// Kernel 2: Reciprocal Sqrt + Newton-Raphson (ReciprocalSqrtSIMD)
// ---------------------------------------------------------------------------
static double Bench_RSqrtNR_SSE2() {
  __m128 a = _mm_set_ps(1.0f, 4.0f, 9.0f, 16.0f);
  __m128 acc = _mm_setzero_ps();
  __m128 half = _mm_set1_ps(0.5f);
  __m128 three = _mm_set1_ps(3.0f);
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < ITERATIONS; i++) {
    __m128 guess = _mm_rsqrt_ps(a);
    // NR: y = 0.5 * y * (3 - a*y^2)
    __m128 g2 = _mm_mul_ps(guess, guess);
    __m128 ag2 = _mm_mul_ps(a, g2);
    guess = _mm_mul_ps(half, _mm_mul_ps(guess, _mm_sub_ps(three, ag2)));
    acc = _mm_add_ps(acc, guess);
    a = _mm_add_ps(a, _mm_set1_ps(0.0001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, ITERATIONS);
}

static double Bench_RSqrtNR_FMA3() {
  __m128 a = _mm_set_ps(1.0f, 4.0f, 9.0f, 16.0f);
  __m128 acc = _mm_setzero_ps();
  __m128 half = _mm_set1_ps(0.5f);
  __m128 three = _mm_set1_ps(3.0f);
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < ITERATIONS; i++) {
    __m128 guess = _mm_rsqrt_ps(a);
    __m128 g2 = _mm_mul_ps(guess, guess);
    // 3 - a*y^2  =>  fnmadd(a, g2, three)
    guess = _mm_mul_ps(half, _mm_mul_ps(guess, _mm_fnmadd_ps(a, g2, three)));
    acc = _mm_add_ps(acc, guess);
    a = _mm_add_ps(a, _mm_set1_ps(0.0001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, ITERATIONS);
}

static double Bench_RSqrtNR_AVX() {
  __m256 a = _mm256_set_ps(1, 4, 9, 16, 25, 36, 49, 64);
  __m256 acc = _mm256_setzero_ps();
  __m256 half = _mm256_set1_ps(0.5f);
  __m256 three = _mm256_set1_ps(3.0f);
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < ITERATIONS; i++) {
    __m256 guess = _mm256_rsqrt_ps(a);
    __m256 g2 = _mm256_mul_ps(guess, guess);
    guess = _mm256_mul_ps(half,
                          _mm256_mul_ps(guess, _mm256_fnmadd_ps(a, g2, three)));
    acc = _mm256_add_ps(acc, guess);
    a = _mm256_add_ps(a, _mm256_set1_ps(0.0001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, ITERATIONS);
}

// ---------------------------------------------------------------------------
// Kernel 3: Dot Product (FourVectors::operator*)
// ---------------------------------------------------------------------------
static double Bench_Dot3_SSE2() {
  __m128 ax = _mm_set_ps(1, 2, 3, 4), ay = _mm_set_ps(5, 6, 7, 8),
         az = _mm_set_ps(9, 10, 11, 12);
  __m128 bx = _mm_set_ps(0.1f, 0.2f, 0.3f, 0.4f),
         by = _mm_set_ps(0.5f, 0.6f, 0.7f, 0.8f),
         bz = _mm_set_ps(0.9f, 1.0f, 1.1f, 1.2f);
  __m128 acc = _mm_setzero_ps();
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < ITERATIONS; i++) {
    // VRAD pattern: dot = mul(x,bx); dot = madd(y,by,dot); dot =
    // madd(z,bz,dot);
    __m128 dot = _mm_mul_ps(ax, bx);
    dot = _mm_add_ps(dot, _mm_mul_ps(ay, by)); // MaddSIMD = add(mul,c)
    dot = _mm_add_ps(dot, _mm_mul_ps(az, bz));
    acc = _mm_add_ps(acc, dot);
    ax = _mm_add_ps(ax, _mm_set1_ps(0.00001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, ITERATIONS);
}

static double Bench_Dot3_FMA3() {
  __m128 ax = _mm_set_ps(1, 2, 3, 4), ay = _mm_set_ps(5, 6, 7, 8),
         az = _mm_set_ps(9, 10, 11, 12);
  __m128 bx = _mm_set_ps(0.1f, 0.2f, 0.3f, 0.4f),
         by = _mm_set_ps(0.5f, 0.6f, 0.7f, 0.8f),
         bz = _mm_set_ps(0.9f, 1.0f, 1.1f, 1.2f);
  __m128 acc = _mm_setzero_ps();
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < ITERATIONS; i++) {
    __m128 dot = _mm_mul_ps(ax, bx);
    dot = _mm_fmadd_ps(ay, by, dot);
    dot = _mm_fmadd_ps(az, bz, dot);
    acc = _mm_add_ps(acc, dot);
    ax = _mm_add_ps(ax, _mm_set1_ps(0.00001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, ITERATIONS);
}

static double Bench_Dot3_AVX() {
  __m256 ax = _mm256_set_ps(1, 2, 3, 4, 5, 6, 7, 8);
  __m256 ay = _mm256_set_ps(9, 10, 11, 12, 13, 14, 15, 16);
  __m256 az = _mm256_set_ps(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f);
  __m256 bx = _mm256_set1_ps(0.3f), by = _mm256_set1_ps(0.6f),
         bz = _mm256_set1_ps(0.9f);
  __m256 acc = _mm256_setzero_ps();
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < ITERATIONS; i++) {
    __m256 dot = _mm256_mul_ps(ax, bx);
    dot = _mm256_fmadd_ps(ay, by, dot);
    dot = _mm256_fmadd_ps(az, bz, dot);
    acc = _mm256_add_ps(acc, dot);
    ax = _mm256_add_ps(ax, _mm256_set1_ps(0.00001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, ITERATIONS);
}

// ---------------------------------------------------------------------------
// Kernel 4: Floor/Ceil — VRAD's FloorSIMD is a complex scalar fallback
// ---------------------------------------------------------------------------
static double Bench_Floor_SSE2() {
  // VRAD's SSE2 FloorSIMD: bit-trick emulation
  __m128 magic = _mm_set1_ps(8388608.0f); // 2^23
  __m128 ones = _mm_set1_ps(1.0f);
  __m128 signmask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
  __m128 val = _mm_set_ps(1.3f, -2.7f, 0.5f, -0.1f);
  __m128 acc = _mm_setzero_ps();
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < FLOOR_ITERS; i++) {
    __m128 absval = _mm_and_ps(val, signmask);
    __m128 ival = _mm_sub_ps(_mm_add_ps(absval, magic), magic);
    // if ival > absval, ival -= 1
    __m128 mask = _mm_cmpgt_ps(ival, absval);
    ival = _mm_sub_ps(ival, _mm_and_ps(ones, mask));
    // restore sign
    __m128 signbits = _mm_xor_ps(val, absval);
    __m128 result = _mm_xor_ps(ival, signbits);
    acc = _mm_add_ps(acc, result);
    val = _mm_add_ps(val, _mm_set1_ps(0.00001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, FLOOR_ITERS);
}

static double Bench_Floor_SSE41() {
  __m128 val = _mm_set_ps(1.3f, -2.7f, 0.5f, -0.1f);
  __m128 acc = _mm_setzero_ps();
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < FLOOR_ITERS; i++) {
    __m128 result = _mm_floor_ps(val); // SSE4.1 _mm_round_ps
    acc = _mm_add_ps(acc, result);
    val = _mm_add_ps(val, _mm_set1_ps(0.00001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, FLOOR_ITERS);
}

static double Bench_Floor_AVX() {
  __m256 val =
      _mm256_set_ps(1.3f, -2.7f, 0.5f, -0.1f, 3.9f, -1.1f, 2.2f, -4.4f);
  __m256 acc = _mm256_setzero_ps();
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < FLOOR_ITERS; i++) {
    __m256 result = _mm256_floor_ps(val);
    acc = _mm256_add_ps(acc, result);
    val = _mm256_add_ps(val, _mm256_set1_ps(0.00001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, FLOOR_ITERS);
}

// ---------------------------------------------------------------------------
// Kernel 5: Masked Blend (MaskedAssign / SelectSIMD)
// ---------------------------------------------------------------------------
static double Bench_Blend_SSE2() {
  __m128 a = _mm_set_ps(1, 2, 3, 4), b = _mm_set_ps(5, 6, 7, 8);
  __m128 acc = _mm_setzero_ps();
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < ITERATIONS; i++) {
    __m128 mask = _mm_cmpgt_ps(a, _mm_set1_ps(2.5f));
    // VRAD SelectSIMD: xor(and(xor(new,old),mask), old)
    __m128 result = _mm_xor_ps(_mm_and_ps(_mm_xor_ps(b, a), mask), a);
    acc = _mm_add_ps(acc, result);
    a = _mm_add_ps(a, _mm_set1_ps(0.00001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, ITERATIONS);
}

static double Bench_Blend_SSE41() {
  __m128 a = _mm_set_ps(1, 2, 3, 4), b = _mm_set_ps(5, 6, 7, 8);
  __m128 acc = _mm_setzero_ps();
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < ITERATIONS; i++) {
    __m128 mask = _mm_cmpgt_ps(a, _mm_set1_ps(2.5f));
    __m128 result = _mm_blendv_ps(a, b, mask); // SSE4.1
    acc = _mm_add_ps(acc, result);
    a = _mm_add_ps(a, _mm_set1_ps(0.00001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, ITERATIONS);
}

// ---------------------------------------------------------------------------
// Kernel 6: Full Light Gather Pipeline Simulation
// ---------------------------------------------------------------------------
static double Bench_LightGather_SSE2() {
  // Simulates GatherSampleStandardLightSSE for emit_point:
  // delta = src - pos; dist2 = dot(delta,delta); rsqrt+NR; attenuation; dot
  // with normal; masking
  __m128 src_x = _mm_set1_ps(100.0f), src_y = _mm_set1_ps(200.0f),
         src_z = _mm_set1_ps(50.0f);
  __m128 pos_x = _mm_set_ps(10, 20, 30, 40), pos_y = _mm_set_ps(50, 60, 70, 80),
         pos_z = _mm_set_ps(5, 10, 15, 20);
  __m128 nx = _mm_set1_ps(0.0f), ny = _mm_set1_ps(0.0f), nz = _mm_set1_ps(1.0f);
  __m128 quad = _mm_set1_ps(0.001f), lin = _mm_set1_ps(0.01f),
         con = _mm_set1_ps(1.0f);
  __m128 ones = _mm_set1_ps(1.0f), zeros = _mm_setzero_ps();
  __m128 half = _mm_set1_ps(0.5f), three = _mm_set1_ps(3.0f);
  __m128 acc = _mm_setzero_ps();
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < ITERATIONS; i++) {
    // delta = src - pos
    __m128 dx = _mm_sub_ps(src_x, pos_x);
    __m128 dy = _mm_sub_ps(src_y, pos_y);
    __m128 dz = _mm_sub_ps(src_z, pos_z);
    // dist2 = dot(delta, delta)
    __m128 dist2 = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy)), _mm_mul_ps(dz, dz));
    // rsqrt + NR
    __m128 g = _mm_rsqrt_ps(dist2);
    g = _mm_mul_ps(
        half,
        _mm_mul_ps(g, _mm_sub_ps(three, _mm_mul_ps(dist2, _mm_mul_ps(g, g)))));
    // normalize delta
    dx = _mm_mul_ps(dx, g);
    dy = _mm_mul_ps(dy, g);
    dz = _mm_mul_ps(dz, g);
    // dot with normal
    __m128 dot = _mm_add_ps(_mm_add_ps(_mm_mul_ps(dx, nx), _mm_mul_ps(dy, ny)),
                            _mm_mul_ps(dz, nz));
    dot = _mm_max_ps(dot, zeros);
    // dist
    __m128 dist = _mm_sqrt_ps(dist2);
    dist = _mm_max_ps(dist, ones);
    // attenuation: 1/(q*d^2 + l*d + c)
    __m128 d2 = _mm_mul_ps(dist, dist);
    __m128 falloff = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(quad, d2), _mm_mul_ps(lin, dist)), con);
    __m128 rcp = _mm_rcp_ps(falloff);
    rcp = _mm_sub_ps(_mm_add_ps(rcp, rcp),
                     _mm_mul_ps(falloff, _mm_mul_ps(rcp, rcp)));
    acc = _mm_add_ps(acc, _mm_mul_ps(dot, rcp));
    pos_x = _mm_add_ps(pos_x, _mm_set1_ps(0.001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, ITERATIONS);
}

static double Bench_LightGather_FMA3() {
  __m128 src_x = _mm_set1_ps(100.0f), src_y = _mm_set1_ps(200.0f),
         src_z = _mm_set1_ps(50.0f);
  __m128 pos_x = _mm_set_ps(10, 20, 30, 40), pos_y = _mm_set_ps(50, 60, 70, 80),
         pos_z = _mm_set_ps(5, 10, 15, 20);
  __m128 nx = _mm_set1_ps(0.0f), ny = _mm_set1_ps(0.0f), nz = _mm_set1_ps(1.0f);
  __m128 quad = _mm_set1_ps(0.001f), lin = _mm_set1_ps(0.01f),
         con = _mm_set1_ps(1.0f);
  __m128 ones = _mm_set1_ps(1.0f), zeros = _mm_setzero_ps();
  __m128 half = _mm_set1_ps(0.5f), three = _mm_set1_ps(3.0f);
  __m128 acc = _mm_setzero_ps();
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < ITERATIONS; i++) {
    __m128 dx = _mm_sub_ps(src_x, pos_x);
    __m128 dy = _mm_sub_ps(src_y, pos_y);
    __m128 dz = _mm_sub_ps(src_z, pos_z);
    __m128 dist2 =
        _mm_fmadd_ps(dz, dz, _mm_fmadd_ps(dy, dy, _mm_mul_ps(dx, dx)));
    __m128 g = _mm_rsqrt_ps(dist2);
    g = _mm_mul_ps(
        half, _mm_mul_ps(g, _mm_fnmadd_ps(dist2, _mm_mul_ps(g, g), three)));
    dx = _mm_mul_ps(dx, g);
    dy = _mm_mul_ps(dy, g);
    dz = _mm_mul_ps(dz, g);
    __m128 dot = _mm_fmadd_ps(dz, nz, _mm_fmadd_ps(dy, ny, _mm_mul_ps(dx, nx)));
    dot = _mm_max_ps(dot, zeros);
    __m128 dist = _mm_sqrt_ps(dist2);
    dist = _mm_max_ps(dist, ones);
    __m128 d2 = _mm_mul_ps(dist, dist);
    __m128 falloff = _mm_fmadd_ps(quad, d2, _mm_fmadd_ps(lin, dist, con));
    __m128 rcp = _mm_rcp_ps(falloff);
    rcp = _mm_fnmadd_ps(falloff, _mm_mul_ps(rcp, rcp), _mm_add_ps(rcp, rcp));
    acc = _mm_fmadd_ps(dot, rcp, acc);
    pos_x = _mm_add_ps(pos_x, _mm_set1_ps(0.001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, ITERATIONS);
}

static double Bench_LightGather_AVX() {
  __m256 src_x = _mm256_set1_ps(100.0f), src_y = _mm256_set1_ps(200.0f),
         src_z = _mm256_set1_ps(50.0f);
  __m256 pos_x = _mm256_set_ps(10, 20, 30, 40, 50, 60, 70, 80);
  __m256 pos_y = _mm256_set_ps(50, 60, 70, 80, 90, 100, 110, 120);
  __m256 pos_z = _mm256_set_ps(5, 10, 15, 20, 25, 30, 35, 40);
  __m256 nx = _mm256_set1_ps(0.0f), ny = _mm256_set1_ps(0.0f),
         nz = _mm256_set1_ps(1.0f);
  __m256 quad = _mm256_set1_ps(0.001f), lin = _mm256_set1_ps(0.01f),
         con = _mm256_set1_ps(1.0f);
  __m256 ones = _mm256_set1_ps(1.0f), zeros = _mm256_setzero_ps();
  __m256 half = _mm256_set1_ps(0.5f), three = _mm256_set1_ps(3.0f);
  __m256 acc = _mm256_setzero_ps();
  auto t0 = hrclock::now();
  for (int64_t i = 0; i < ITERATIONS; i++) {
    __m256 dx = _mm256_sub_ps(src_x, pos_x);
    __m256 dy = _mm256_sub_ps(src_y, pos_y);
    __m256 dz = _mm256_sub_ps(src_z, pos_z);
    __m256 dist2 =
        _mm256_fmadd_ps(dz, dz, _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dx, dx)));
    __m256 g = _mm256_rsqrt_ps(dist2);
    g = _mm256_mul_ps(
        half,
        _mm256_mul_ps(g, _mm256_fnmadd_ps(dist2, _mm256_mul_ps(g, g), three)));
    dx = _mm256_mul_ps(dx, g);
    dy = _mm256_mul_ps(dy, g);
    dz = _mm256_mul_ps(dz, g);
    __m256 dot =
        _mm256_fmadd_ps(dz, nz, _mm256_fmadd_ps(dy, ny, _mm256_mul_ps(dx, nx)));
    dot = _mm256_max_ps(dot, zeros);
    __m256 dist = _mm256_sqrt_ps(dist2);
    dist = _mm256_max_ps(dist, ones);
    __m256 d2 = _mm256_mul_ps(dist, dist);
    __m256 falloff = _mm256_fmadd_ps(quad, d2, _mm256_fmadd_ps(lin, dist, con));
    __m256 rcp = _mm256_rcp_ps(falloff);
    rcp = _mm256_fnmadd_ps(falloff, _mm256_mul_ps(rcp, rcp),
                           _mm256_add_ps(rcp, rcp));
    acc = _mm256_fmadd_ps(dot, rcp, acc);
    pos_x = _mm256_add_ps(pos_x, _mm256_set1_ps(0.001f));
  }
  auto t1 = hrclock::now();
  DoNotOptimize(acc);
  return NsPerOp(t1 - t0, ITERATIONS);
}

// ---------------------------------------------------------------------------
// Kernel 7: Batch Vector Normalize (memory throughput + compute)
// ---------------------------------------------------------------------------
static double Bench_BatchNorm_SSE2() {
  alignas(16) float data_x[BATCH_SIZE], data_y[BATCH_SIZE], data_z[BATCH_SIZE];
  for (int i = 0; i < BATCH_SIZE; i++) {
    data_x[i] = (float)(i + 1);
    data_y[i] = (float)(i + 2);
    data_z[i] = (float)(i + 3);
  }
  __m128 half = _mm_set1_ps(0.5f), three = _mm_set1_ps(3.0f);
  auto t0 = hrclock::now();
  int64_t total = FLOOR_ITERS / BATCH_SIZE * BATCH_SIZE;
  for (int64_t iter = 0; iter < FLOOR_ITERS / BATCH_SIZE; iter++) {
    for (int i = 0; i < BATCH_SIZE; i += 4) {
      __m128 x = _mm_load_ps(&data_x[i]);
      __m128 y = _mm_load_ps(&data_y[i]);
      __m128 z = _mm_load_ps(&data_z[i]);
      __m128 len2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(x, x), _mm_mul_ps(y, y)),
                               _mm_mul_ps(z, z));
      __m128 g = _mm_rsqrt_ps(len2);
      g = _mm_mul_ps(
          half,
          _mm_mul_ps(g, _mm_sub_ps(three, _mm_mul_ps(len2, _mm_mul_ps(g, g)))));
      _mm_store_ps(&data_x[i], _mm_mul_ps(x, g));
      _mm_store_ps(&data_y[i], _mm_mul_ps(y, g));
      _mm_store_ps(&data_z[i], _mm_mul_ps(z, g));
    }
  }
  auto t1 = hrclock::now();
  g_sink = data_x[0];
  return NsPerOp(t1 - t0, total);
}

static double Bench_BatchNorm_AVX() {
  alignas(32) float data_x[BATCH_SIZE], data_y[BATCH_SIZE], data_z[BATCH_SIZE];
  for (int i = 0; i < BATCH_SIZE; i++) {
    data_x[i] = (float)(i + 1);
    data_y[i] = (float)(i + 2);
    data_z[i] = (float)(i + 3);
  }
  __m256 half = _mm256_set1_ps(0.5f), three = _mm256_set1_ps(3.0f);
  auto t0 = hrclock::now();
  int64_t total = FLOOR_ITERS / BATCH_SIZE * BATCH_SIZE;
  for (int64_t iter = 0; iter < FLOOR_ITERS / BATCH_SIZE; iter++) {
    for (int i = 0; i < BATCH_SIZE; i += 8) {
      __m256 x = _mm256_load_ps(&data_x[i]);
      __m256 y = _mm256_load_ps(&data_y[i]);
      __m256 z = _mm256_load_ps(&data_z[i]);
      __m256 len2 =
          _mm256_fmadd_ps(z, z, _mm256_fmadd_ps(y, y, _mm256_mul_ps(x, x)));
      __m256 g = _mm256_rsqrt_ps(len2);
      g = _mm256_mul_ps(
          half,
          _mm256_mul_ps(g, _mm256_fnmadd_ps(len2, _mm256_mul_ps(g, g), three)));
      _mm256_store_ps(&data_x[i], _mm256_mul_ps(x, g));
      _mm256_store_ps(&data_y[i], _mm256_mul_ps(y, g));
      _mm256_store_ps(&data_z[i], _mm256_mul_ps(z, g));
    }
  }
  auto t1 = hrclock::now();
  g_sink = data_x[0];
  return NsPerOp(t1 - t0, total);
}

// ---------------------------------------------------------------------------
// Kernel 8: Radiosity Accumulate (double precision)
// ---------------------------------------------------------------------------
static double Bench_RadAccum_Scalar() {
  alignas(32) float src[BATCH_SIZE * 3];
  for (int i = 0; i < BATCH_SIZE * 3; i++)
    src[i] = 0.001f * (i % 97 + 1);
  double acc_r = 0, acc_g = 0, acc_b = 0;
  auto t0 = hrclock::now();
  int64_t total = FLOOR_ITERS / BATCH_SIZE * BATCH_SIZE;
  for (int64_t iter = 0; iter < FLOOR_ITERS / BATCH_SIZE; iter++) {
    for (int i = 0; i < BATCH_SIZE; i++) {
      acc_r += src[i * 3 + 0];
      acc_g += src[i * 3 + 1];
      acc_b += src[i * 3 + 2];
    }
  }
  auto t1 = hrclock::now();
  g_sink = (float)(acc_r + acc_g + acc_b);
  return NsPerOp(t1 - t0, total);
}

static double Bench_RadAccum_AVX_d() {
  alignas(32) float src[BATCH_SIZE * 3];
  for (int i = 0; i < BATCH_SIZE * 3; i++)
    src[i] = 0.001f * (i % 97 + 1);
  __m256d acc_r = _mm256_setzero_pd(), acc_g = _mm256_setzero_pd(),
          acc_b = _mm256_setzero_pd();
  auto t0 = hrclock::now();
  int64_t total = FLOOR_ITERS / BATCH_SIZE * BATCH_SIZE;
  for (int64_t iter = 0; iter < FLOOR_ITERS / BATCH_SIZE; iter++) {
    for (int i = 0; i < BATCH_SIZE; i += 4) {
      // Load 4 RGB triples, convert to double, accumulate
      __m128 r4 = _mm_set_ps(src[(i + 3) * 3], src[(i + 2) * 3],
                             src[(i + 1) * 3], src[i * 3]);
      __m128 g4 = _mm_set_ps(src[(i + 3) * 3 + 1], src[(i + 2) * 3 + 1],
                             src[(i + 1) * 3 + 1], src[i * 3 + 1]);
      __m128 b4 = _mm_set_ps(src[(i + 3) * 3 + 2], src[(i + 2) * 3 + 2],
                             src[(i + 1) * 3 + 2], src[i * 3 + 2]);
      acc_r = _mm256_add_pd(acc_r, _mm256_cvtps_pd(r4));
      acc_g = _mm256_add_pd(acc_g, _mm256_cvtps_pd(g4));
      acc_b = _mm256_add_pd(acc_b, _mm256_cvtps_pd(b4));
    }
  }
  auto t1 = hrclock::now();
  alignas(32) double tmp[4];
  _mm256_store_pd(tmp, _mm256_add_pd(acc_r, _mm256_add_pd(acc_g, acc_b)));
  g_sink = (float)(tmp[0] + tmp[1] + tmp[2] + tmp[3]);
  return NsPerOp(t1 - t0, total);
}

// ---------------------------------------------------------------------------
// Results printer
// ---------------------------------------------------------------------------
struct BenchResult {
  const char *kernel;
  const char *isa;
  double nsPerOp;
  double speedup; // vs baseline
};

static void PrintResults(BenchResult *results, int count) {
  printf("\n");
  printf("%-30s %-12s %12s %10s\n", "Kernel", "ISA", "ns/op", "Speedup");
  printf("%-30s %-12s %12s %10s\n", "------------------------------",
         "------------", "------------", "----------");
  for (int i = 0; i < count; i++) {
    printf("%-30s %-12s %12.2f %9.2fx\n", results[i].kernel, results[i].isa,
           results[i].nsPerOp, results[i].speedup);
  }
  printf("\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
  CPUFeatures cpu = DetectCPU();

  printf("=============================================================\n");
  printf("  VRAD SIMD Instruction Set Benchmark\n");
  printf("=============================================================\n");
  printf("CPU: %s\n", cpu.brand);
  printf(
      "Features: SSE2=%d  SSE4.1=%d  AVX=%d  FMA3=%d  AVX2=%d  AVX-512F=%d\n",
      cpu.sse2, cpu.sse41, cpu.avx, cpu.fma3, cpu.avx2, cpu.avx512f);
  printf("Iterations per kernel: %lld (batch: %lld)\n", ITERATIONS, BATCH_SIZE);
  printf("-------------------------------------------------------------\n\n");

  BenchResult results[64];
  int n = 0;
  double baseline;

  // --- Kernel 1: FMA Chain ---
  printf("Running: FMA Chain (attenuation falloff)...\n");
  baseline = Bench_FMAChain_SSE2();
  results[n++] = {"FMA Chain (attenuation)", "SSE2", baseline, 1.0};
  if (cpu.fma3) {
    double t = Bench_FMAChain_FMA3();
    results[n++] = {"FMA Chain (attenuation)", "FMA3", t, baseline / t};
  }
  if (cpu.avx && cpu.fma3) {
    double t = Bench_FMAChain_AVX();
    results[n++] = {"FMA Chain (attenuation)", "AVX+FMA3", t, baseline / t};
  }

  // --- Kernel 2: RSqrt + NR ---
  printf("Running: Reciprocal Sqrt + NR...\n");
  baseline = Bench_RSqrtNR_SSE2();
  results[n++] = {"RSqrt + Newton-Raphson", "SSE2", baseline, 1.0};
  if (cpu.fma3) {
    double t = Bench_RSqrtNR_FMA3();
    results[n++] = {"RSqrt + Newton-Raphson", "FMA3", t, baseline / t};
  }
  if (cpu.avx && cpu.fma3) {
    double t = Bench_RSqrtNR_AVX();
    results[n++] = {"RSqrt + Newton-Raphson", "AVX+FMA3", t, baseline / t};
  }

  // --- Kernel 3: Dot Product ---
  printf("Running: 3-Component Dot Product...\n");
  baseline = Bench_Dot3_SSE2();
  results[n++] = {"Dot Product (FourVectors)", "SSE2", baseline, 1.0};
  if (cpu.fma3) {
    double t = Bench_Dot3_FMA3();
    results[n++] = {"Dot Product (FourVectors)", "FMA3", t, baseline / t};
  }
  if (cpu.avx && cpu.fma3) {
    double t = Bench_Dot3_AVX();
    results[n++] = {"Dot Product (FourVectors)", "AVX+FMA3", t, baseline / t};
  }

  // --- Kernel 4: Floor ---
  printf("Running: Floor (FloorSIMD)...\n");
  baseline = Bench_Floor_SSE2();
  results[n++] = {"Floor (FloorSIMD)", "SSE2", baseline, 1.0};
  if (cpu.sse41) {
    double t = Bench_Floor_SSE41();
    results[n++] = {"Floor (FloorSIMD)", "SSE4.1", t, baseline / t};
  }
  if (cpu.avx) {
    double t = Bench_Floor_AVX();
    results[n++] = {"Floor (FloorSIMD)", "AVX", t, baseline / t};
  }

  // --- Kernel 5: Masked Blend ---
  printf("Running: Masked Blend (SelectSIMD)...\n");
  baseline = Bench_Blend_SSE2();
  results[n++] = {"Masked Blend (SelectSIMD)", "SSE2", baseline, 1.0};
  if (cpu.sse41) {
    double t = Bench_Blend_SSE41();
    results[n++] = {"Masked Blend (SelectSIMD)", "SSE4.1", t, baseline / t};
  }

  // --- Kernel 6: Full Light Gather ---
  printf("Running: Light Gather Pipeline...\n");
  baseline = Bench_LightGather_SSE2();
  results[n++] = {"Light Gather Pipeline", "SSE2", baseline, 1.0};
  if (cpu.fma3) {
    double t = Bench_LightGather_FMA3();
    results[n++] = {"Light Gather Pipeline", "FMA3", t, baseline / t};
  }
  if (cpu.avx && cpu.fma3) {
    double t = Bench_LightGather_AVX();
    results[n++] = {"Light Gather Pipeline", "AVX+FMA3", t, baseline / t};
  }

  // --- Kernel 7: Batch Normalize ---
  printf("Running: Batch Vector Normalize...\n");
  baseline = Bench_BatchNorm_SSE2();
  results[n++] = {"Batch Vec Normalize", "SSE2", baseline, 1.0};
  if (cpu.avx && cpu.fma3) {
    double t = Bench_BatchNorm_AVX();
    results[n++] = {"Batch Vec Normalize", "AVX+FMA3", t, baseline / t};
  }

  // --- Kernel 8: Radiosity Accumulate ---
  printf("Running: Radiosity Accumulate (double)...\n");
  baseline = Bench_RadAccum_Scalar();
  results[n++] = {"Radiosity Accumulate", "Scalar dbl", baseline, 1.0};
  if (cpu.avx) {
    double t = Bench_RadAccum_AVX_d();
    results[n++] = {"Radiosity Accumulate", "AVX __m256d", t, baseline / t};
  }

  printf("\nAll benchmarks complete.\n");
  PrintResults(results, n);

  // Summary
  printf("=============================================================\n");
  printf("  Summary: VRAD currently targets SSE2 only.\n");
  printf("  Speedup >1.0x = newer ISA is faster for that workload.\n");
  printf("  AVX columns process 2x the data per iteration (8 floats).\n");
  printf("=============================================================\n");

  return 0;
}
