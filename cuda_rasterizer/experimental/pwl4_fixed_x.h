#pragma once

// ============================================================
// 3-segment PWL with FIXED x control points
//   Knots: [-1, -0.5, 0.5, 1] -> [-1, Y1, Y2, 1]
//   Raw params: y0, y1  (2 per activation)
//   Internal:   Y1 = y0,  Y2 = y1
//   Caller guarantees y1 >= y0 via external clamping.
// ============================================================

struct Coeffs {
    float y0, y1;       // raw params = internal Y values
    float ds[3];        // segment slopes
};

// Fixed constants
#define PWL_X1  (-0.5f)
#define PWL_X2  ( 0.5f)
#define PWL_H0  ( 0.5f)   // X1 - (-1) = -0.5 + 1
#define PWL_H1  ( 1.0f)   // X2 - X1   =  0.5 - (-0.5)
#define PWL_H2  ( 0.5f)   // 1  - X2   =  1   - 0.5

__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    Coeffs c;
    c.y0 = point[0] * 2 - 1.0f;  // init with -0.5
    c.y1 = point[1] * 2 - 1.0f;  // init with 0.5

    c.ds[0] = (c.y0 + 1.0f) / PWL_H0;   // (-1 -> Y1) over 0.5
    c.ds[1] = (c.y1 - c.y0) / PWL_H1;   // (Y1 -> Y2) over 1.0
    c.ds[2] = (1.0f - c.y1) / PWL_H2;   // (Y2 -> 1)  over 0.5

    return c;
}

// ============================================================
// Find segment: 0, 1, or 2
// ============================================================
__device__ __forceinline__
int _find_seg(float t) {
    return (t >= PWL_X1) + (t >= PWL_X2);
}

// ============================================================
// Forward
// ============================================================
__device__ __forceinline__
float coeffs_eval(float t, const Coeffs& c) {
    if (t <= -1.0f || t >= 1.0f) return t;

    int seg = _find_seg(t);

    // Segment left endpoint & left y
    float xi = (seg == 0) ? -1.0f : (seg == 1 ? PWL_X1 : PWL_X2);
    float yi = (seg == 0) ? -1.0f : (seg == 1 ? c.y0   : c.y1);
    float h  = (seg == 1) ? PWL_H1 : ((seg == 0) ? PWL_H0 : PWL_H2);

    float s = (t - xi) / h;
    float yi1 = (seg == 0) ? c.y0 : (seg == 1 ? c.y1 : 1.0f);

    return yi + s * (yi1 - yi);
}

// ============================================================
// Backward
// ============================================================
__device__ __forceinline__
void backward_with_doutdt(
    float t,
    const Coeffs& c,
    float grad_output,
    float& grad_y0,
    float& grad_y1,
    float& dout_dt
) {
    grad_y0 = grad_y1 = 0.0f;

    if (t <= -1.0f || t >= 1.0f) {
        dout_dt = 1.0f;
        return;
    }

    int seg = _find_seg(t);

    float xi = (seg == 0) ? -1.0f : (seg == 1 ? PWL_X1 : PWL_X2);
    float h  = (seg == 1) ? PWL_H1 : ((seg == 0) ? PWL_H0 : PWL_H2);
    float yi = (seg == 0) ? -1.0f : (seg == 1 ? c.y0   : c.y1);
    float yi1 = (seg == 0) ? c.y0 : (seg == 1 ? c.y1   : 1.0f);

    float dy = yi1 - yi;
    float s  = (t - xi) / h;

    dout_dt = dy / h;

    // x knots are fixed -> no dout/dX gradients
    // Only need dout/dY1 and dout/dY2
    //
    // seg 0: out = -1 + s*(Y1 - (-1))  ->  dout/dY1 = s
    // seg 1: out = Y1 + s*(Y2 - Y1)    ->  dout/dY1 = 1-s,  dout/dY2 = s
    // seg 2: out = Y2 + s*(1 - Y2)     ->  dout/dY2 = 1-s
    //
    // Y1 = y0, Y2 = y1  (identity, no chain rule needed)

    switch (seg) {
    case 0:
        grad_y0 = s * grad_output;
        break;
    case 1:
        grad_y0 = (1.0f - s) * grad_output;
        grad_y1 = s * grad_output;
        break;
    case 2:
        grad_y1 = (1.0f - s) * grad_output;
        break;
    }
    grad_y0 *= 2.0f;
    grad_y1 *= 2.0f;
}

// ============================================================
// Atomic backward
// ============================================================
__device__ __forceinline__
float coeffs_backward_atomic(
    float t_raw,
    float rescale,
    const Coeffs& c,
    float grad_output,
    float* dL_dinter_points,
    int base_idx
) {
    float grad_y0, grad_y1, dout_dt;
    backward_with_doutdt(
        t_raw / rescale, c, grad_output,
        grad_y0, grad_y1, dout_dt
    );
    atomicAdd(&dL_dinter_points[base_idx + 0], grad_y0 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 1], grad_y1 * rescale);
    return dout_dt;
}