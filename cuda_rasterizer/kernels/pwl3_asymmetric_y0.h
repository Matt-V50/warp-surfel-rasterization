#pragma once

// ============================================================
// Piecewise Linear, 5 control points, ASYMMETRIC dead-zone:
//   xs = [-1, -x0, 0, x1, 1]
//   ys = [-1,   0, 0,  0, 1]
//
// 4 segments (middle 2 are flat):
//   seg0: [-1, -x0]  slope d0 = 1/(1-x0)   (left ramp)
//   seg1: [-x0, 0]   slope 0               (dead zone)
//   seg2: [0, x1]    slope 0               (dead zone)
//   seg3: [x1, 1]    slope d3 = 1/(1-x1)   (right ramp)
//
// Learnable parameters: x0, x1  ∈ (0, 1)
//   point[0] = x0,  point[1] = x1
//
// Outside [-1, 1]: identity (f(t) = t)
// ============================================================

struct Coeffs {
    float x0, x1;
    float h0, h3;       // h0 = 1-x0,  h3 = 1-x1
    float d0, d3;       // slopes of left/right ramps
};

// ============================================================
// Precompute coefficients
// point[0] = x0 ∈ (0,1),  point[1] = x1 ∈ (0,1)
// ============================================================
__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    float x0 = point[0];
    float x1 = point[1];

    float h0 = 1.0f - x0;
    float h3 = 1.0f - x1;
    float d0 = 1.0f / h0;
    float d3 = 1.0f / h3;

    Coeffs c;
    c.x0 = x0;  c.x1 = x1;
    c.h0 = h0;  c.h3 = h3;
    c.d0 = d0;  c.d3 = d3;
    return c;
}

// ============================================================
// Forward evaluation
//   seg0: t < -x0       out = -1 + d0*(t+1)
//   dead: -x0 <= t < x1 out = 0
//   seg3: t >= x1        out = d3*(t-x1)
// ============================================================
__device__ __forceinline__
float coeffs_eval(float t, const Coeffs& c) {
    if (t < -1.0f || t > 1.0f) {
        return t;
    }

    if (t < -c.x0) {
        return -1.0f + c.d0 * (t + 1.0f);
    } else if (t < c.x1) {
        return 0.0f;
    } else {
        return c.d3 * (t - c.x1);
    }
}

// ============================================================
// Backward pass
//
// seg0 (t < -x0):
//   f = -1 + (t+1)/h0
//   s = (t+1)/h0
//   df/dt  = d0
//   df/dx0 = s * d0        [= (t+1) / h0^2]
//   df/dx1 = 0
//
// dead zone (-x0 <= t < x1):
//   f = 0
//   df/dt  = 0
//   df/dx0 = 0
//   df/dx1 = 0
//
// seg3 (t >= x1):
//   f = (t-x1)/h3
//   s = (t-x1)/h3
//   df/dt  = d3
//   df/dx0 = 0
//   df/dx1 = d3 * (s - 1)  [= (t-1) / h3^2]
// ============================================================
__device__ __forceinline__
void coeffs_backward_with_doutdt(
    float t,
    const Coeffs& c,
    float grad_output,    // dL/d(output)
    float& grad_x0,      // output: dL/dx0
    float& grad_x1,      // output: dL/dx1
    float& dout_dt        // output: df/dt (raw, no grad_output)
) {
    if (t < -1.0f || t > 1.0f) {
        dout_dt = 1.0f;
        grad_x0 = 0.0f;
        grad_x1 = 0.0f;
        return;
    }

    if (t < -c.x0) {
        float s = (t + 1.0f) / c.h0;
        dout_dt = c.d0;
        grad_x0 = grad_output * (s * c.d0);
        grad_x1 = 0.0f;
    } else if (t < c.x1) {
        dout_dt = 0.0f;
        grad_x0 = 0.0f;
        grad_x1 = 0.0f;
    } else {
        float s = (t - c.x1) / c.h3;
        dout_dt = c.d3;
        grad_x0 = 0.0f;
        grad_x1 = grad_output * (c.d3 * (s - 1.0f));
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
    float grad_x0, grad_x1, dout_dt;
    coeffs_backward_with_doutdt(t_raw / rescale, c, grad_output, grad_x0, grad_x1, dout_dt);
    atomicAdd(&dL_dinter_points[base_idx + 0], grad_x0 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 1], grad_x1 * rescale);
    return dout_dt;
}