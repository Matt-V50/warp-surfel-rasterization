#pragma once

// ============================================================
// Piecewise Linear, 3 control points, y0 FIXED at 0.5 (internal: 0).
// Only x0 is a learnable parameter.
//
// Internal domain [-1, 1]:
//   xs = [-1,  X1,  1]    X1 = x0_param * 2 - 1
//   ys = [-1,   0,  1]    (fixed)
//
// Public domain [-R, R]  (RESCALER = R):
//   f(t) = eval(t / R) * R
// ============================================================

struct Coeffs {
    float x0;        // internal knot X1 = x0_param*2-1, in (-1,1)
    float h0, h1;    // segment lengths  h0=X1+1, h1=1-X1
    float d0, d1;    // segment slopes   d0=1/h0, d1=1/h1  (Y1 fixed=0)
};

// ============================================================
// coeffs_precompute
// x0_param in (0, 1)  →  internal coords in (-1, 1)
// ============================================================
__device__ __forceinline__
Coeffs coeffs_precompute(const float* point)
{
    float x0 = point[0];
    float X1 = x0 * 2.0f - 1.0f;   // (0,1) -> (-1,1)
    float h0 = X1 + 1.0f;                 // X1 - (-1)
    float h1 = 1.0f - X1;                 // 1   -  X1
    // Y1 is fixed at 0 internally  =>  slopes simplify
    float d0 = 1.0f / h0;                 // (0-(-1)) / h0
    float d1 = 1.0f / h1;                 // (1-  0)  / h1

    Coeffs c;
    c.x0 = X1;
    c.h0 = h0; c.h1 = h1;
    c.d0 = d0; c.d1 = d1;
    return c;
}

// ============================================================
// coeffs_eval
// t in internal domain [-1, 1]. Returns value in [-1, 1].
// 调用方负责 rescaling:  f(t_raw) = coeffs_eval(t_raw / R) * R
// ============================================================
__device__ __forceinline__
float coeffs_eval(float t, const Coeffs& c)
{
    if (t <= -1.0f || t >= 1.0f)
        return t;                              // identity

    float out;
    if (t < c.x0) {
        // Segment [-1, X1]:  out/R = -1 + d0*(u+1)
        out = -1.0f + c.d0 * (t + 1.0f);
    } else {
        // Segment [X1,  1]:  out/R = Y1 + d1*(u-X1),  Y1=0
        out = c.d1 * (t - c.x0);
    }
    return out;
}

__device__ __forceinline__
void backward_with_doutdt(
    float t,
    const Coeffs& c,
    float grad_output,
    float& grad_x0,    // output: dL/d(x0_param)
    float& dout_dt     // output: df/dt
)
{

    if (t <= -1.0f || t >= 1.0f) {
        dout_dt = 1.0f;
        grad_x0 = 0.0f;
        return;
    }

    if (t < c.x0) {
        // Seg 0
        float s      = (t + 1.0f) / c.h0;
        dout_dt      = c.d0;
        float dX1    = (-s / c.h0);
        grad_x0      = grad_output * dX1 * 2.0f;
    } else {
        // Seg 1
        float s      = (t - c.x0) / c.h1;
        dout_dt      = c.d1;
        float dX1    = (-(1.0f - s) / c.h1);
        grad_x0      = grad_output * dX1 * 2.0f;
    }
}

// ============================================================
// coeffs_backward_atomic
// Accumulates dL/d(x0_param) into dL_dparams[base_idx] via
// atomicAdd (thread-safe across warps).
// Returns dout_dt.
//
// Layout: dL_dparams[base_idx] holds the gradient for x0_param.
// (y0 is fixed, so no second slot is needed.)
// ============================================================
__device__ __forceinline__
float coeffs_backward_atomic(
    float t_raw,
    float rescale,          // pass PWL_RESCALER  (kept for API consistency)
    const Coeffs& c,
    float grad_output,
    float* dL_dparams,
    int   base_idx
)
{
    float grad_x0, dout_dt;
    backward_with_doutdt(t_raw / rescale, c, grad_output, grad_x0, dout_dt);
    atomicAdd(&dL_dparams[base_idx], grad_x0 * rescale);
    return dout_dt;
}