#pragma once

// ============================================================
// Piecewise Linear, 5 control points, ASYMMETRIC:
//   xs = [-1, -x0, 0, x1, 1]
//   ys = [-1, -y0, 0, y1, 1]
//
// 4 segments, 4 independent slopes:
//   d_ol = (1-y0)/(1-x0)   seg0: [-1,  -x0]   (outer left)
//   d_il = y0/x0            seg1: [-x0,   0]   (inner left)
//   d_ir = y1/x1            seg2: [0,    x1]   (inner right)
//   d_or = (1-y1)/(1-x1)   seg3: [x1,    1]   (outer right)
//
// Outside [-1, 1]: identity (f(t) = t)
// ============================================================

struct Coeffs {
    float x0, y0, x1, y1;
    float h_ol, h_il, h_ir, h_or;   // segment widths
    float d_ol, d_il, d_ir, d_or;   // slopes
};

// ============================================================
// Precompute coefficients from point = [x0, y0, x1, y1]
// ============================================================
__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    float x0 = point[0];   // left knot position  ∈ (0,1)
    float y0 = point[1];   // left knot value      ∈ (0,1)
    float x1 = point[2];   // right knot position ∈ (0,1)
    float y1 = point[3];   // right knot value     ∈ (0,1)

    float h_ol = 1.0f - x0;   // width of seg0
    float h_il = x0;           // width of seg1
    float h_ir = x1;           // width of seg2
    float h_or = 1.0f - x1;   // width of seg3

    Coeffs c;
    c.x0 = x0;  c.y0 = y0;
    c.x1 = x1;  c.y1 = y1;
    c.h_ol = h_ol;  c.h_il = h_il;
    c.h_ir = h_ir;  c.h_or = h_or;
    c.d_ol = (1.0f - y0) / h_ol;
    c.d_il = y0 / h_il;
    c.d_ir = y1 / h_ir;
    c.d_or = (1.0f - y1) / h_or;
    return c;
}

// ============================================================
// Forward evaluation
//   seg0: t < -x0       out = -1  + d_ol*(t+1)
//   seg1: -x0 <= t < 0  out = -y0 + d_il*(t+x0)
//   seg2: 0 <= t < x1   out = d_ir*t
//   seg3: t >= x1        out = y1  + d_or*(t-x1)
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
// Backward pass
//
// seg0 (t < -x0):  s = (t+1)/h_ol
//   dout/dt  = d_ol
//   dout/dx0 = d_ol * s        dout/dy0 = -s
//   dout/dx1 = 0               dout/dy1 = 0
//
// seg1 (-x0 <= t < 0):  s = (t+x0)/h_il
//   dout/dt  = d_il
//   dout/dx0 = d_il * (1-s)    dout/dy0 = s - 1
//   dout/dx1 = 0               dout/dy1 = 0
//
// seg2 (0 <= t < x1):  s = t/h_ir
//   dout/dt  = d_ir
//   dout/dx0 = 0               dout/dy0 = 0
//   dout/dx1 = -d_ir * s       dout/dy1 = s
//
// seg3 (t >= x1):  s = (t-x1)/h_or
//   dout/dt  = d_or
//   dout/dx0 = 0               dout/dy0 = 0
//   dout/dx1 = d_or * (s-1)    dout/dy1 = 1 - s
// ============================================================
__device__ __forceinline__
void backward_with_doutdt(
    float t,
    const Coeffs& c,
    float grad_output,    // dL/d(output)
    float& grad_x0,      // output: dL/dx0
    float& grad_y0,      // output: dL/dy0
    float& grad_x1,      // output: dL/dx1
    float& grad_y1,      // output: dL/dy1
    float& dout_dt        // output: dout/dt (raw, no grad_output)
) {
    if (t < -1.0f || t > 1.0f) {
        dout_dt = 1.0f;
        grad_x0 = 0.0f;  grad_y0 = 0.0f;
        grad_x1 = 0.0f;  grad_y1 = 0.0f;
        return;
    }

    if (t < -c.x0) {
        float s = (t + 1.0f) / c.h_ol;
        dout_dt = c.d_ol;
        grad_x0 = grad_output * ( c.d_ol * s);
        grad_y0 = grad_output * (-s);
        grad_x1 = 0.0f;
        grad_y1 = 0.0f;
    } else if (t < 0.0f) {
        float s = (t + c.x0) / c.h_il;
        dout_dt = c.d_il;
        grad_x0 = grad_output * (c.d_il * (1.0f - s));
        grad_y0 = grad_output * (s - 1.0f);
        grad_x1 = 0.0f;
        grad_y1 = 0.0f;
    } else if (t < c.x1) {
        float s = t / c.h_ir;
        dout_dt = c.d_ir;
        grad_x0 = 0.0f;
        grad_y0 = 0.0f;
        grad_x1 = grad_output * (-c.d_ir * s);
        grad_y1 = grad_output * s;
    } else {
        float s = (t - c.x1) / c.h_or;
        dout_dt = c.d_or;
        grad_x0 = 0.0f;
        grad_y0 = 0.0f;
        grad_x1 = grad_output * (c.d_or * (s - 1.0f));
        grad_y1 = grad_output * (1.0f - s);
    }
}


__device__ __forceinline__
float coeffs_backward_atomic(
    float t_raw,
    float rescale,
    const Coeffs& c,
    float grad_output,
    float* dL_dinter_points,
    int base_idx              // global_id * CONTROL_POINTS * 4 + d * 4
) {
    float grad_x0, grad_y0, grad_x1, grad_y1, dout_dt;
    backward_with_doutdt(t_raw / rescale, c, grad_output,
                         grad_x0, grad_y0, grad_x1, grad_y1, dout_dt);
    atomicAdd(&dL_dinter_points[base_idx + 0], grad_x0 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 1], grad_y0 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 2], grad_x1 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 3], grad_y1 * rescale);
    return dout_dt;
}