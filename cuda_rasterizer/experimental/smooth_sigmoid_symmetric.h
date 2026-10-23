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
    float a = point[0];
    float eps = 1e-6f;
    a = fmaxf(eps, fminf(1.0f - eps, a));
    float k = 1.0f / (1 - a);  // a in (0, 1) -> k in (1, ∞)
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
    float tau = fabsf(x);
    tau = fmaxf(1e-15f, fminf(1.0f - 1e-15f, tau));

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
    // return x;
    return h;
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
    float& grad_a,       // output: dL/da
    float& dv_dx         // output: dv/dx
) {
    float k = c.k;
    float tau = fabsf(x);
    float tau_c = fmaxf(1e-15f, fminf(1.0f - 1e-15f, tau));

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

    // ---- grad_a: dv/da = h(1-h) · logit ----
    float dv_dk = h1h * logit_val;
    float dk_da = k * k;
    grad_a = grad_output * dv_dk * dk_da;

    // ---- dv/dx = k · h(1-h) / (τ(1-τ)) · sign(x) ----
    if (tau_c < 1e-12f || tau_c > 1.0f - 1e-12f) {
        dv_dx = 0.0f;
    } else {
        float sign_x = (x >= 0.0f) ? 1.0f : -1.0f;
        float dh_dtau = k * h1h / (tau_c * (1.0f - tau_c));
        dv_dx = dh_dtau * sign_x;
    }
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
    float grad_a, dv_dx;
    backward_with_doutdt(t_raw / rescale, c, grad_output, grad_a, dv_dx);
    atomicAdd(&dL_dinter_points[base_idx + 0], grad_a * rescale);
    dv_dx = fmaxf(0.0f, fminf(dv_dx, 1.0f));
    return dv_dx;
}