#pragma once

struct Coeffs {
    float k;
};

// ============================================================
// Precompute smooth-sigmoid coefficients
// h(τ) = σ(k · logit(τ)),  logit(τ) = ln(τ/(1-τ))
// ============================================================
__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    float k = point[0];
    Coeffs c;
    c.k = k;
    return c;
}

// ============================================================
// Forward evaluation
// v = h(|x|),  h(τ) = σ(k · logit(τ))
// k=1  → h≈τ   → v≈|x|  (standard Gaussian)
// k→∞  → h→step → flat-top
// ============================================================
__device__ __forceinline__
float coeffs_eval(float x, const Coeffs& c) {
    return x;
}

// ============================================================
// Backward pass
//   grad_k:  dL/dk  = grad_output · dv/dk
//   dv_dx:   dv/dx  (raw derivative, no grad_output applied)
// ============================================================
__device__ __forceinline__
void backward_with_doutdt(
    float x,
    const Coeffs& c,
    float grad_output,    // dL/d(output)
    float& grad_k,       // output: dL/dk
    float& dv_dx         // output: dv/dx
) {
    grad_k = 0.0f;
    dv_dx = 1.0f;
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
    float grad_k, dv_dx;
    float t = t_raw / rescale;
    backward_with_doutdt(t, c, grad_output, grad_k, dv_dx);
    atomicAdd(&dL_dinter_points[base_idx + 0], grad_k * rescale);
    return dv_dx;
}