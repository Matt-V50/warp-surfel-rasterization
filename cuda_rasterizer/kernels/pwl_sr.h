#pragma once

// ============================================================
// Decoupled PWL parameterization with sigmoided points
//
// Assumption:
//   point[0] in [0, 1]  -> knot position
//   point[1] in [0, 1]  -> normalized log slope ratio
//
// Mapping:
//   u = point[0]
//   v = point[1]
//
//   p = 2u - 1
//   q = QMAX * (2v - 1)
//   R = exp(q) = d0 / d1
//
// Then:
//   h0 = p + 1
//   h1 = 1 - p
//   D  = h1 + R*h0
//
//   d0 = 2R / D
//   d1 = 2  / D
//   y0 = -1 + d0*h0
//
// Identity:
//   point[0] = 0.5
//   point[1] = 0.5
//   => p = 0, q = 0, R = 1, d0 = d1 = 1
//
// Recommended:
//   QMAX = 2.0 ~ 4.0
//   Start with QMAX = 3.0
//
// With this parameterization, point[0] and point[1] are both
// normalized [0,1] controls, so using the same Adam lr is reasonable.
// ============================================================

#ifndef PWL_EPS
#define PWL_EPS 1e-3f
#endif

#ifndef PWL_QMAX
#define PWL_QMAX 3.0f
#endif

struct Coeffs {
    float x0, y0;
    float h0, h1;
    float d0, d1;

    // Store for backward to avoid recovering R from d0/d1
    float R;
    float D;
};

// ============================================================
// Clamp helper
// ============================================================
__device__ __forceinline__
float clamp01_eps(float x) {
    return fminf(fmaxf(x, PWL_EPS), 1.0f - PWL_EPS);
}

// ============================================================
// Precompute PWL coefficients
// ============================================================
__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    // point is already sigmoided, but clamp for numerical safety.
    float u = clamp01_eps(point[0]);
    float v = clamp01_eps(point[1]);

    // knot position
    float p = 2.0f * u - 1.0f;

    // normalized log slope ratio
    float q = PWL_QMAX * (2.0f * v - 1.0f);
    float R = expf(q);

    float h0 = p + 1.0f;
    float h1 = 1.0f - p;

    float D = h1 + R * h0;

    // More stable than computing slopes through y0.
    float d0 = 2.0f * R / D;
    float d1 = 2.0f / D;

    float y0 = -1.0f + d0 * h0;

    Coeffs c;
    c.x0 = p;
    c.y0 = y0;
    c.h0 = h0;
    c.h1 = h1;
    c.d0 = d0;
    c.d1 = d1;
    c.R  = R;
    c.D  = D;
    return c;
}

// ============================================================
// Forward evaluation
// ============================================================
__device__ __forceinline__
float coeffs_eval(float t, const Coeffs& c) {
    if (t < -1.0f || t > 1.0f) {
        return t;
    }

    if (t < c.x0) {
        // Segment [-1, x0]
        return -1.0f + c.d0 * (t + 1.0f);
    } else {
        // Segment [x0, 1]
        return c.y0 + c.d1 * (t - c.x0);
    }
}

// ============================================================
// Backward pass with dout/dt
//
// Step 1:
//   Compute dL/dx0 and dL/dy0 as if x0,y0 were independent.
//
// Step 2:
//   Chain rule from:
//
//     x0 = p
//     y0 = y0(p, q)
//
//   where:
//
//     p = 2u - 1
//     q = QMAX * (2v - 1)
//
//   Derivatives:
//
//     dy0/dp = 4R / D^2
//     dy0/dq = 2R*h0*h1 / D^2
//
//   Therefore:
//
//     dL/dp = dL/dx0 + dL/dy0 * dy0/dp
//     dL/dq = dL/dy0 * dy0/dq
//
//     dL/du = dL/dp * 2
//     dL/dv = dL/dq * 2*QMAX
//
// Outputs:
//   grad_u = dL/dpoint[0]
//   grad_v = dL/dpoint[1]
// ============================================================
__device__ __forceinline__
void backward_with_doutdt(
    float t,
    const Coeffs& c,
    float grad_output,   // dL/d(output)
    float& grad_u,       // output: dL/dpoint[0]
    float& grad_v,       // output: dL/dpoint[1]
    float& dout_dt       // output: d(output)/d(t)
) {
    // Out of range: identity
    if (t < -1.0f || t > 1.0f) {
        dout_dt = 1.0f;
        grad_u = 0.0f;
        grad_v = 0.0f;
        return;
    }

    // --------------------------------------------------------
    // Step 1: dL/dx0 and dL/dy0, same as original PWL
    // --------------------------------------------------------
    float dL_dx0;
    float dL_dy0;

    if (t < c.x0) {
        // Segment [-1, x0]
        //
        // out = -1 + d0 * (t + 1)
        // s   = (t + 1) / h0
        //
        // dout/dx0 = -d0 * s
        // dout/dy0 = s
        float s = (t + 1.0f) / c.h0;

        dout_dt = c.d0;

        dL_dx0 = grad_output * (-c.d0 * s);
        dL_dy0 = grad_output * s;
    } else {
        // Segment [x0, 1]
        //
        // out = y0 + d1 * (t - x0)
        // s   = (t - x0) / h1
        //
        // dout/dx0 = d1 * (s - 1)
        // dout/dy0 = 1 - s
        float s = (t - c.x0) / c.h1;

        dout_dt = c.d1;

        dL_dx0 = grad_output * (c.d1 * (s - 1.0f));
        dL_dy0 = grad_output * (1.0f - s);
    }

    // --------------------------------------------------------
    // Step 2: chain rule from (x0, y0) to (u, v)
    // --------------------------------------------------------
    float R = c.R;
    float D = c.D;

    float inv_D_sq = 1.0f / (D * D);

    float dy0_dp = 4.0f * R * inv_D_sq;
    float dy0_dq = 2.0f * R * c.h0 * c.h1 * inv_D_sq;

    float dL_dp = dL_dx0 + dL_dy0 * dy0_dp;
    float dL_dq = dL_dy0 * dy0_dq;

    // p = 2u - 1
    // q = QMAX * (2v - 1)
    grad_u = dL_dp * 2.0f;
    grad_v = dL_dq * (2.0f * PWL_QMAX);
}

// ============================================================
// Atomic backward
//
// Assumes forward shape is something like:
//
//   t = t_raw / rescale
//   out = rescale * coeffs_eval(t, c)
//
// Then:
//
//   d(out)/d(t_raw) = dout_dt
//   dL/dparams gets an extra * rescale
// ============================================================
__device__ __forceinline__
float coeffs_backward_atomic(
    float t_raw,
    float rescale,
    const Coeffs& c,
    float grad_output,
    float* dL_dinter_points,
    int base_idx              // global_id * CONTROL_POINTS * DOF + d * DOF
) {
    float grad_u;
    float grad_v;
    float dout_dt;

    float t = t_raw / rescale;

    backward_with_doutdt(
        t,
        c,
        grad_output,
        grad_u,
        grad_v,
        dout_dt
    );

    atomicAdd(&dL_dinter_points[base_idx + 0], grad_u * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 1], grad_v * rescale);

    return dout_dt;
}