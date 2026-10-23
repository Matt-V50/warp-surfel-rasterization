#pragma once

// ============================================================
// Piecewise Linear, 5 control points, ASYMMETRIC, pq slope-ratio:
//   xs = [-1, -x0, 0, x1, 1]
//   ys = [-1, -y0, 0, y1, 1]
//
// Instead of directly learning [x0, y0, x1, y1], we use
// normalized controls:
//   point[0] = u0 in [0, 1]  -> left knot magnitude x0
//   point[1] = v0 in [0, 1]  -> left normalized log slope ratio
//   point[2] = u1 in [0, 1]  -> right knot position x1
//   point[3] = v1 in [0, 1]  -> right normalized log slope ratio
//
// Left half [-1, 0]:
//   h_ol = 1 - x0, h_il = x0
//   q0 = QMAX * (2v0 - 1)
//   R0 = exp(q0) = d_ol / d_il
//   Z0 = h_il + R0 * h_ol
//   d_ol = R0 / Z0, d_il = 1 / Z0
//   y0 = d_il * h_il = h_il / Z0
//
// Right half [0, 1]:
//   h_ir = x1, h_or = 1 - x1
//   q1 = QMAX * (2v1 - 1)
//   R1 = exp(q1) = d_ir / d_or
//   Z1 = h_or + R1 * h_ir
//   d_ir = R1 / Z1, d_or = 1 / Z1
//   y1 = d_ir * h_ir = R1 * h_ir / Z1
//
// Identity:
//   point = [0.5, 0.5, 0.5, 0.5]
//   => x0 = y0 = x1 = y1 = 0.5, all slopes = 1
//
// Outside [-1, 1]: identity (f(t) = t)
// ============================================================

#ifndef PWL_EPS
#define PWL_EPS 1e-3f
#endif

#ifndef PWL_QMAX
#define PWL_QMAX 3.0f
#endif

struct Coeffs {
    float x0, y0, x1, y1;
    float h_ol, h_il, h_ir, h_or;   // segment widths
    float d_ol, d_il, d_ir, d_or;   // slopes

    // Store slope-ratio parameters for backward.
    float R0, Z0;
    float R1, Z1;
};

// ============================================================
// Clamp helper
// ============================================================
__device__ __forceinline__
float clamp01_eps(float x) {
    return fminf(fmaxf(x, PWL_EPS), 1.0f - PWL_EPS);
}

// ============================================================
// Precompute coefficients from point = [u0, v0, u1, v1]
// ============================================================
__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    // The input point is assumed to be already sigmoided into [0, 1].
    // Clamp only for numerical safety.
    float u0 = clamp01_eps(point[0]);
    float v0 = clamp01_eps(point[1]);
    float u1 = clamp01_eps(point[2]);
    float v1 = clamp01_eps(point[3]);

    // Knot locations.
    float x0 = u0;
    float x1 = u1;

    // Segment widths.
    float h_ol = 1.0f - x0;
    float h_il = x0;
    float h_ir = x1;
    float h_or = 1.0f - x1;

    // Left slope ratio: outer-left slope / inner-left slope.
    float q0 = PWL_QMAX * (2.0f * v0 - 1.0f);
    float R0 = expf(q0);
    float Z0 = h_il + R0 * h_ol;

    float d_ol = R0 / Z0;
    float d_il = 1.0f / Z0;
    float y0   = d_il * h_il;

    // Right slope ratio: inner-right slope / outer-right slope.
    float q1 = PWL_QMAX * (2.0f * v1 - 1.0f);
    float R1 = expf(q1);
    float Z1 = h_or + R1 * h_ir;

    float d_ir = R1 / Z1;
    float d_or = 1.0f / Z1;
    float y1   = d_ir * h_ir;

    Coeffs c;
    c.x0 = x0;  c.y0 = y0;
    c.x1 = x1;  c.y1 = y1;
    c.h_ol = h_ol;  c.h_il = h_il;
    c.h_ir = h_ir;  c.h_or = h_or;
    c.d_ol = d_ol;  c.d_il = d_il;
    c.d_ir = d_ir;  c.d_or = d_or;
    c.R0 = R0;      c.Z0 = Z0;
    c.R1 = R1;      c.Z1 = Z1;
    return c;
}

// ============================================================
// Forward evaluation
//   seg0: t < -x0       out = -1  + d_ol*(t+1)
//   seg1: -x0 <= t < 0  out = -y0 + d_il*(t+x0)
//   seg2: 0 <= t < x1   out = d_ir*t
//   seg3: t >= x1       out = y1  + d_or*(t-x1)
// ============================================================
__device__ __forceinline__
float coeffs_eval(float t, const Coeffs& c) {
    if (t < -1.0f || t > 1.0f) {
        return t;
    }

    if (t < -c.x0) {
        return -1.0f + c.d_ol * (t + 1.0f);
    } else if (t < 0.0f) {
        return -c.y0 + c.d_il * (t + c.x0);
    } else if (t < c.x1) {
        return c.d_ir * t;
    } else {
        return c.y1 + c.d_or * (t - c.x1);
    }
}

// ============================================================
// Backward pass with dout/dt
//
// Step 1:
//   Compute local gradients w.r.t. [x0, y0, x1, y1] as if these
//   control-point coordinates were independent.
//
// Step 2:
//   Chain rule from induced y0/y1 to pq controls:
//
//   Left:
//     x0 = u0
//     y0 = h_il / Z0,       Z0 = h_il + R0*h_ol
//     q0 = QMAX*(2v0 - 1),  R0 = exp(q0)
//
//     dy0/dx0 =  R0 / Z0^2
//     dy0/dq0 = -R0*h_ol*h_il / Z0^2
//
//     dL/du0 = dL/dx0 + dL/dy0 * dy0/dx0
//     dL/dv0 = dL/dq0 * 2*QMAX
//
//   Right:
//     x1 = u1
//     y1 = R1*h_ir / Z1,    Z1 = h_or + R1*h_ir
//     q1 = QMAX*(2v1 - 1),  R1 = exp(q1)
//
//     dy1/dx1 =  R1 / Z1^2
//     dy1/dq1 =  R1*h_ir*h_or / Z1^2
//
//     dL/du1 = dL/dx1 + dL/dy1 * dy1/dx1
//     dL/dv1 = dL/dq1 * 2*QMAX
//
// Outputs:
//   grad_u0 = dL/dpoint[0]
//   grad_v0 = dL/dpoint[1]
//   grad_u1 = dL/dpoint[2]
//   grad_v1 = dL/dpoint[3]
// ============================================================
__device__ __forceinline__
void backward_with_doutdt(
    float t,
    const Coeffs& c,
    float grad_output,    // dL/d(output)
    float& grad_u0,       // output: dL/dpoint[0]
    float& grad_v0,       // output: dL/dpoint[1]
    float& grad_u1,       // output: dL/dpoint[2]
    float& grad_v1,       // output: dL/dpoint[3]
    float& dout_dt        // output: dout/dt (raw, no grad_output)
) {
    if (t < -1.0f || t > 1.0f) {
        dout_dt = 1.0f;
        grad_u0 = 0.0f;  grad_v0 = 0.0f;
        grad_u1 = 0.0f;  grad_v1 = 0.0f;
        return;
    }

    // --------------------------------------------------------
    // Step 1: local gradients w.r.t. independent [x0,y0,x1,y1]
    // --------------------------------------------------------
    float grad_x0 = 0.0f;
    float grad_y0 = 0.0f;
    float grad_x1 = 0.0f;
    float grad_y1 = 0.0f;

    if (t < -c.x0) {
        // Segment [-1, -x0]
        float s = (t + 1.0f) / c.h_ol;
        dout_dt = c.d_ol;
        grad_x0 = grad_output * ( c.d_ol * s);
        grad_y0 = grad_output * (-s);
    } else if (t < 0.0f) {
        // Segment [-x0, 0]
        float s = (t + c.x0) / c.h_il;
        dout_dt = c.d_il;
        grad_x0 = grad_output * (c.d_il * (1.0f - s));
        grad_y0 = grad_output * (s - 1.0f);
    } else if (t < c.x1) {
        // Segment [0, x1]
        float s = t / c.h_ir;
        dout_dt = c.d_ir;
        grad_x1 = grad_output * (-c.d_ir * s);
        grad_y1 = grad_output * s;
    } else {
        // Segment [x1, 1]
        float s = (t - c.x1) / c.h_or;
        dout_dt = c.d_or;
        grad_x1 = grad_output * (c.d_or * (s - 1.0f));
        grad_y1 = grad_output * (1.0f - s);
    }

    // --------------------------------------------------------
    // Step 2: chain rule from independent coordinates to pq controls
    // --------------------------------------------------------
    float inv_Z0_sq = 1.0f / (c.Z0 * c.Z0);
    float dy0_dx0  = c.R0 * inv_Z0_sq;
    float dy0_dq0  = -c.R0 * c.h_ol * c.h_il * inv_Z0_sq;

    float dL_dx0_total = grad_x0 + grad_y0 * dy0_dx0;
    float dL_dq0       = grad_y0 * dy0_dq0;

    float inv_Z1_sq = 1.0f / (c.Z1 * c.Z1);
    float dy1_dx1  = c.R1 * inv_Z1_sq;
    float dy1_dq1  = c.R1 * c.h_ir * c.h_or * inv_Z1_sq;

    float dL_dx1_total = grad_x1 + grad_y1 * dy1_dx1;
    float dL_dq1       = grad_y1 * dy1_dq1;

    // x0 = u0, x1 = u1
    // q0 = QMAX*(2v0 - 1), q1 = QMAX*(2v1 - 1)
    grad_u0 = dL_dx0_total;
    grad_v0 = dL_dq0 * (2.0f * PWL_QMAX);
    grad_u1 = dL_dx1_total;
    grad_v1 = dL_dq1 * (2.0f * PWL_QMAX);
}

// ============================================================
// Atomic backward
//
// Assumes forward shape is something like:
//   t = t_raw / rescale
//   out = rescale * coeffs_eval(t, c)
//
// Then:
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
    int base_idx              // global_id * CONTROL_POINTS * 4 + d * 4
) {
    float grad_u0, grad_v0, grad_u1, grad_v1, dout_dt;

    float t = t_raw / rescale;

    backward_with_doutdt(
        t,
        c,
        grad_output,
        grad_u0,
        grad_v0,
        grad_u1,
        grad_v1,
        dout_dt
    );

    atomicAdd(&dL_dinter_points[base_idx + 0], grad_u0 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 1], grad_v0 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 2], grad_u1 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 3], grad_v1 * rescale);

    return dout_dt;
}