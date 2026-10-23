#pragma once

// ============================================================
// Piecewise Linear, 5 control points, anti-symmetric:
//   xs = [-1, -x0, 0, x0, 1]
//   ys = [-1, -y0, 0, y0, 1]
//
// 4 segments, 2 unique slopes (by symmetry):
//   d_outer = (1-y0)/(1-x0)   for seg0 [-1,-x0] and seg3 [x0,1]
//   d_inner = y0/x0            for seg1 [-x0,0]  and seg2 [0,x0]
//
// Outside [-1, 1]: identity (f(t) = t)
// ============================================================

struct Coeffs {
    float x0, y0;
    float h_outer, h_inner;   // h_outer = 1-x0,  h_inner = x0
    float d_outer, d_inner;   // slopes
};

// ============================================================
// Precompute PWL coefficients
// point[0] = x0 ∈ (0,1),  point[1] = y0 ∈ (0,1)
// ============================================================
__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    float x0 = point[0];
    float y0 = point[1];

    float h_outer = 1.0f - x0;
    float h_inner = x0;
    float d_outer = (1.0f - y0) / h_outer;
    float d_inner = y0 / h_inner;

    Coeffs c;
    c.x0 = x0;  c.y0 = y0;
    c.h_outer = h_outer;  c.h_inner = h_inner;
    c.d_outer = d_outer;  c.d_inner = d_inner;
    return c;
}

// ============================================================
// Forward evaluation
//   seg0: t < -x0       out = -1 + d_outer*(t+1)
//   seg1: -x0 <= t < 0  out = -y0 + d_inner*(t+x0)
//   seg2: 0 <= t < x0   out = d_inner*t
//   seg3: t >= x0        out = y0 + d_outer*(t-x0)
// ============================================================
__device__ __forceinline__
float coeffs_eval(float t, const Coeffs& c) {
    if (t < -1.0f || t > 1.0f) {
        return t;
    }

    if (t < -c.x0) {
        return -1.0f + c.d_outer * (t + 1.0f);
    } else if (t < 0.0f) {
        return -c.y0 + c.d_inner * (t + c.x0);
    } else if (t < c.x0) {
        return c.d_inner * t;
    } else {
        return c.y0 + c.d_outer * (t - c.x0);
    }
}

// ============================================================
// Backward pass with dout/dt
//
// seg0 (t < -x0):
//   s = (t+1)/h_outer
//   dout/dt  = d_outer
//   dout/dx0 = s * d_outer
//   dout/dy0 = -s
//
// seg1 (-x0 <= t < 0):
//   s = (t+x0)/h_inner
//   dout/dt  = d_inner
//   dout/dx0 = d_inner * (1 - s)
//   dout/dy0 = s - 1
//
// seg2 (0 <= t < x0):
//   s = t / h_inner
//   dout/dt  = d_inner
//   dout/dx0 = -d_inner * s
//   dout/dy0 = s
//
// seg3 (t >= x0):
//   s = (t-x0)/h_outer
//   dout/dt  = d_outer
//   dout/dx0 = d_outer * (s - 1)
//   dout/dy0 = 1 - s
// ============================================================
__device__ __forceinline__
void backward_with_doutdt(
    float t,
    const Coeffs& c,
    float grad_output,    // dL/d(output)
    float& grad_x0,      // output: dL/dx0
    float& grad_y0,      // output: dL/dy0
    float& dout_dt        // output: dout/dt (raw, no grad_output)
) {
    if (t < -1.0f || t > 1.0f) {
        dout_dt = 1.0f;
        grad_x0 = 0.0f;
        grad_y0 = 0.0f;
        return;
    }

    if (t < -c.x0) {
        float s = (t + 1.0f) / c.h_outer;
        dout_dt = c.d_outer;
        grad_x0 = grad_output * (s * c.d_outer);
        grad_y0 = grad_output * (-s);
    } else if (t < 0.0f) {
        float s = (t + c.x0) / c.h_inner;
        dout_dt = c.d_inner;
        grad_x0 = grad_output * (c.d_inner * (1.0f - s));
        grad_y0 = grad_output * (s - 1.0f);
    } else if (t < c.x0) {
        float s = t / c.h_inner;
        dout_dt = c.d_inner;
        grad_x0 = grad_output * (-c.d_inner * s);
        grad_y0 = grad_output * s;
    } else {
        float s = (t - c.x0) / c.h_outer;
        dout_dt = c.d_outer;
        grad_x0 = grad_output * (c.d_outer * (s - 1.0f));
        grad_y0 = grad_output * (1.0f - s);
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
    float grad_x0, grad_y0, dout_dt;
    backward_with_doutdt(t_raw / rescale, c, grad_output, grad_x0, grad_y0, dout_dt);
    atomicAdd(&dL_dinter_points[base_idx + 0], grad_x0 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 1], grad_y0 * rescale);
    return dout_dt;
}