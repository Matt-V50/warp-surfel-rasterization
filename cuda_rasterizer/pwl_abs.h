#pragma once

// ============================================================
// Piecewise Linear Mapping (symmetric 3-segment)
//
// h(τ),  τ = |t|:
//   0                   for τ ≤ a        (plateau, g=1)
//   (τ - a) · k         for a < τ < b    (ramp)
//   1                   for τ ≥ b        (tail, g=const)
//
// where  a = p/2,  b = 1-a,  k = 1/(1-p)
//
// Outside [-1,1]:  identity (return t, so g decays as standard Gaussian)
//
// Raw parameter: point[0] = p ∈ (0, 1)
//   → a = p/2       ∈ (0, 0.5)
//   → k = 1/(1-p)   ∈ (1, 50]  (clamped)
//
// p=0:  a=0, k=1  → h=|τ|  → standard Gaussian
// p→1:  a→0.5, k→∞ → step  → flat-top super-Gaussian
// ============================================================

struct Coeffs {
    float a;     // plateau half-width = p/2
    float b;     // = 1 - a
    float k;     // ramp slope = 1/(b-a) = 1/(1-p)
};

// ============================================================
// Precompute
// ============================================================
__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    float p = point[0];
    p = fmaxf(1e-6f, fminf(1.0f - 1e-6f, p));

    float a = p * 0.5f;           // ∈ (0, 0.5)
    float b = 1.0f - a;           // ∈ (0.5, 1)
    float k = 1.0f / (1.0f - p);  // = 1/(b-a)
    k = fminf(k, 50.0f);

    Coeffs c;
    c.a = a;
    c.b = b;
    c.k = k;
    return c;
}

// ============================================================
// Forward:  returns h(t)
//
//  |t| ≤ a :  0         (plateau)
//  a < |t| < b:  (|t|-a)·k  (ramp)
//  |t| ≥ b :  1         (tail)
//  |t| > 1 :  t         (identity, standard Gaussian tail)
// ============================================================
__device__ __forceinline__
float coeffs_eval(float t, const Coeffs& c) {
    if (t < -1.0f || t > 1.0f) return t;

    float tau = fabsf(t);
    if (tau <= c.a) return 0.0f;
    if (tau >= c.b) return 1.0f;
    return (tau - c.a) * c.k;
}

// ============================================================
// Backward
//
// h = (τ - p/2) / (1-p),  τ = |t|
//
// dh/dp = (τ - 0.5) · k²
//   τ=0.5 → 0  (fixed point, no gradient)
//   τ<0.5 → <0 (increasing p shrinks h here)
//   τ>0.5 → >0 (increasing p grows h here)
//
// dh/dt = k · sign(t)    (middle ramp)
//       = 0              (plateau / tail)
//       = 1              (outside [-1,1])
// ============================================================
__device__ __forceinline__
void backward_with_doutdt(
    float t,
    const Coeffs& c,
    float grad_output,    // dL/d(output)
    float& grad_p,        // output: dL/dp
    float& dout_dt        // output: dh/dt (raw, no grad_output)
) {
    // Outside [-1,1]: identity
    if (t < -1.0f || t > 1.0f) {
        dout_dt = 1.0f;
        grad_p = 0.0f;
        return;
    }

    float tau = fabsf(t);

    // Plateau: h=0
    if (tau <= c.a) {
        dout_dt = 0.0f;
        grad_p = 0.0f;
        return;
    }

    // Tail: h=1
    if (tau >= c.b) {
        dout_dt = 0.0f;
        grad_p = 0.0f;
        return;
    }

    // Ramp: h = (τ-a)·k
    float sign_t = (t >= 0.0f) ? 1.0f : -1.0f;
    dout_dt = c.k * sign_t;

    // dh/dp = (τ - 0.5) · k²
    float dh_dp = (tau - 0.5f) * c.k * c.k;
    if (c.k >= 50.0f) dh_dp = 0.0f;   // clamped k: stop gradient
    grad_p = grad_output * dh_dp;
}

__device__ __forceinline__
float coeffs_backward_atomic(
    float t_raw,
    float rescale,
    const Coeffs& c,
    float grad_output,
    float* dL_dinter_points,
    int base_idx              // global_id * CONTROL_POINTS * 2 + d * 2
) {
    float grad_p, dout_dt;
    backward_with_doutdt(t_raw / rescale, c, grad_output, grad_p, dout_dt);
    atomicAdd(&dL_dinter_points[base_idx + 0], grad_p * rescale);
    return dout_dt;
}