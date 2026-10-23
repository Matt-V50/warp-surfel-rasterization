#pragma once

// ============================================================
// 3-segment PWL: [-1, X1, X2, 1] -> [-1, Y1, Y2, 1]
// Raw params:  x0, x1, y0, y1 ∈ (0, 1)  (from torch: sigmoid + clamp)
//   torch ensures x0 <= x1, y0 <= y1 via clamp
// Internal:    X1 = 2*x0 - 1,  X2 = 2*x1 - 1
//              Y1 = 2*y0 - 1,  Y2 = 2*y1 - 1
// ============================================================

struct Coeffs {
    float X1, X2;
    float Y1, Y2;
    float hs[3];
    float ds[3];
};

__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    Coeffs c;
    c.X1 = 2.0f * point[0] - 1.0f;
    c.X2 = 2.0f * point[1] - 1.0f;
    c.Y1 = 2.0f * point[2] - 1.0f;
    c.Y2 = 2.0f * point[3] - 1.0f;
    c.hs[0] = c.X1 + 1.0f;       // X1 - (-1)
    c.hs[1] = c.X2 - c.X1;
    c.hs[2] = 1.0f - c.X2;

    c.ds[0] = (c.Y1 + 1.0f) / c.hs[0];
    c.ds[1] = (c.Y2 - c.Y1) / c.hs[1];
    c.ds[2] = (1.0f - c.Y2) / c.hs[2];

    return c;
}

// ============================================================
// Find segment: 0, 1, or 2
// ============================================================
__device__ __forceinline__
int _find_seg(float t, float X1, float X2) {
    return (t >= X1) + (t >= X2);
}

// ============================================================
// Forward
// ============================================================
__device__ __forceinline__
float coeffs_eval(float t, const Coeffs& c) {
    if (t <= -1.0f || t >= 1.0f) return t;

    int seg = _find_seg(t, c.X1, c.X2);
    float xi  = (seg == 0) ? -1.0f : (seg == 1 ? c.X1 : c.X2);
    float yi  = (seg == 0) ? -1.0f : (seg == 1 ? c.Y1 : c.Y2);
    float yi1 = (seg == 0) ? c.Y1  : (seg == 1 ? c.Y2 : 1.0f);
    float s   = (t - xi) / c.hs[seg];
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
    float& grad_x0,
    float& grad_x1,
    float& grad_y0,
    float& grad_y1,
    float& dout_dt
) {
    grad_x0 = grad_x1 = grad_y0 = grad_y1 = 0.0f;

    if (t <= -1.0f || t >= 1.0f) {
        dout_dt = 1.0f;
        return;
    }

    int seg = _find_seg(t, c.X1, c.X2);

    const float xs[] = {-1.0f, c.X1, c.X2, 1.0f};
    const float ys[] = {-1.0f, c.Y1, c.Y2, 1.0f};
    float xi  = xs[seg];
    float xi1 = xs[seg + 1];
    float yi  = ys[seg];
    float yi1 = ys[seg + 1];

    float h  = c.hs[seg];
    float s  = (t - xi) / h;
    float dy = yi1 - yi;

    dout_dt = dy / h;

    float dout_dxi  = dy * (t - xi1) / (h * h);
    float dout_dxi1 = dy * (xi - t)  / (h * h);

    float gX1 = 0.0f, gX2 = 0.0f, gY1 = 0.0f, gY2 = 0.0f;

    switch (seg) {
    case 0:
        gX1 = dout_dxi1;
        gY1 = s;
        break;
    case 1:
        gX1 = dout_dxi;   gX2 = dout_dxi1;
        gY1 = 1.0f - s;   gY2 = s;
        break;
    case 2:
        gX2 = dout_dxi;
        gY2 = 1.0f - s;
        break;
    }

    gX1 *= grad_output;  gX2 *= grad_output;
    gY1 *= grad_output;  gY2 *= grad_output;

    // Chain rule: X1 = 2*x0 - 1 → dX1/dx0 = 2
    //             X2 = 2*x1 - 1 → dX2/dx1 = 2
    //             (no cross terms)
    grad_x0 = gX1 * 2.0f;
    grad_x1 = gX2 * 2.0f;
    grad_y0 = gY1 * 2.0f;
    grad_y1 = gY2 * 2.0f;
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
    float grad_x0, grad_x1, grad_y0, grad_y1, dout_dt;
    backward_with_doutdt(
        t_raw / rescale, c, grad_output,
        grad_x0, grad_x1, grad_y0, grad_y1, dout_dt
    );
    atomicAdd(&dL_dinter_points[base_idx + 0], grad_x0 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 1], grad_x1 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 2], grad_y0 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 3], grad_y1 * rescale);
    return dout_dt;
}