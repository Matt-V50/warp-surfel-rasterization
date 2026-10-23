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
    k = fmin(20.0f, fmaxf(1.0f, k));  // k in [1, 20]
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
    float eps = 1e-6f;
    float sign_x = (x >= 0.0f) ? 1.0f : -1.0f;
    float tau = fabsf(x);
    tau = fmaxf(eps, fminf(1.0f - eps, tau));

    float logit_val = logf(tau / (1.0f - tau));
    float z = c.k * logit_val;

    float h;
    if (z > 30.0f) {
        h = 1.0f;
    } else if (z < -30.0f) {
        h = 0.0f;
    } else {
        h = 1.0f / (1.0f + expf(-z));
    }
    return sign_x * h;
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
    float eps = 1e-6f;
    float k = c.k;
    float sign_x = (x >= 0.0f) ? 1.0f : -1.0f;
    float tau = fabsf(x);
    float tau_c = fmaxf(eps, fminf(1.0f - eps, tau));

    float logit_val = logf(tau_c / (1.0f - tau_c));
    float z = k * logit_val;

    float h;
    if (z > 30.0f) {
        h = 1.0f;
    } else if (z < -30.0f) {
        h = 0.0f;
    } else {
        h = 1.0f / (1.0f + expf(-z));
    }

    float h1h = h * (1.0f - h);   // σ'(z)

    // ---- grad_k: dv/dk = h(1-h) · logit ----
    float dv_dk = h1h * logit_val;
    grad_k = grad_output * dv_dk * sign_x;

    // ---- dv/dx = k · h(1-h) / (τ(1-τ)) · sign(x) ----
    float dh_dtau = k * h1h / (tau_c * (1.0f - tau_c));
    dv_dx = dh_dtau;
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
    dv_dx = fmaxf(1.0f, dv_dx);
    // dv_dx = fminf(1.0f, dv_dx);
    // float eps = 1e-1f;
    // if (fabsf(t) < eps) {
    //     dv_dx = 1.0f;
    // }
    // dv_dx = 1.0f;
    return dv_dx;
}