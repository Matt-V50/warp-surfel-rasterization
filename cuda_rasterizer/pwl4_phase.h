#pragma once

// ============================================================
// 3-segment PWL: [-1, X0, X1, 1] -> [-1, Y0, Y1, 1]
// Raw params:  x0, x1, y0, y1 ∈ (0, 1)
// Internal:    X0 = -x0,          X1 = x1
//              Y0 = 2*y0*y1 - 1,  Y1 = 2*y1 - 1
// ============================================================

struct Coeffs {
    float x0, x1, y0, y1;  // raw params
    float X0, X1;           // internal x control points (endpoints -1, 1 are implicit)
    float Y0, Y1;           // internal y control points
    float hs[3];            // segment lengths
    float ds[3];            // segment slopes
};

__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    Coeffs c;
    c.x0 = point[0];
    c.x1 = point[1];
    c.y0 = point[2];
    c.y1 = point[3];

    c.X0 = -c.x0;
    c.X1 =  c.x1;
    c.Y0 = -c.y0;
    c.Y1 =  c.y1;

    c.hs[0] = c.X0 + 1.0f;       // X0 - (-1)
    c.hs[1] = c.X1 - c.X0;
    c.hs[2] = 1.0f - c.X1;       // 1 - X1

    c.ds[0] = (c.Y0 + 1.0f) / c.hs[0];
    c.ds[1] = (c.Y1 - c.Y0) / c.hs[1];
    c.ds[2] = (1.0f - c.Y1) / c.hs[2];

    return c;
}

// ============================================================
// Find segment: 0, 1, or 2
// ============================================================
__device__ __forceinline__
int _find_seg(float t, float X0, float X1) {
    return (t >= X0) + (t >= X1);
}

// ============================================================
// Helper: get segment endpoints from index
// ============================================================
__device__ __forceinline__
void _seg_endpoints(int seg, const Coeffs& c,
                    float& xi, float& xi1, float& yi, float& yi1) {
    // xs = {-1, X0, X1, 1},  ys = {-1, Y0, Y1, 1}
    const float xs[] = {-1.0f, c.X0, c.X1, 1.0f};
    const float ys[] = {-1.0f, c.Y0, c.Y1, 1.0f};
    xi  = xs[seg];      yi  = ys[seg];
    xi1 = xs[seg + 1];  yi1 = ys[seg + 1];
}

// ============================================================
// Forward
// ============================================================
__device__ __forceinline__
float coeffs_eval(float t, const Coeffs& c) {
    if (t <= -1.0f || t >= 1.0f) return t;

    int seg = _find_seg(t, c.X0, c.X1);
    float s = (t - (seg == 0 ? -1.0f : (seg == 1 ? c.X0 : c.X1))) / c.hs[seg];
    float yi  = (seg == 0) ? -1.0f : (seg == 1 ? c.Y0 : c.Y1);
    float yi1 = (seg == 0) ? c.Y0  : (seg == 1 ? c.Y1 : 1.0f);
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

    int seg = _find_seg(t, c.X0, c.X1);

    float xi, xi1, yi, yi1;
    _seg_endpoints(seg, c, xi, xi1, yi, yi1);

    float h  = c.hs[seg];
    float s  = (t - xi) / h;
    float dy = yi1 - yi;

    dout_dt = dy / h;

    // dout/d(internal nodes)
    float dout_dxi1 = dy * (xi  - t) / (h * h);
    float dout_dxi  = dy * (t - xi1) / (h * h);

    // Accumulate into X0, X1, Y0, Y1 gradients
    float gX0 = 0.0f, gX1 = 0.0f, gY0 = 0.0f, gY1 = 0.0f;

    switch (seg) {
    case 0:
        gX0 = dout_dxi1;
        gY0 = s;
        break;
    case 1:
        gX0 = dout_dxi;   gX1 = dout_dxi1;
        gY0 = 1.0f - s;   gY1 = s;
        break;
    case 2:
        gX1 = dout_dxi;
        gY1 = 1.0f - s;
        break;
    }

    gX0 *= grad_output;  gX1 *= grad_output;
    gY0 *= grad_output;  gY1 *= grad_output;

    // Chain rule: internal → raw
    // X0 = -x0       →  dX0/dx0 = -1
    // X1 =  x1       →  dX1/dx1 =  1
    // Y0 = -y0       →  dY0/dy0 = -1
    // Y1 =  y1       →  dY1/dy1 =  1
    grad_x0 = gX0 * (-1.0f);
    grad_x1 = gX1 * ( 1.0f);
    grad_y0 = gY0 * (-1.0f);
    grad_y1 = gY1 * ( 1.0f);
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