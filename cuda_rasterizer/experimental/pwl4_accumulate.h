#pragma once

// ============================================================
// 4 control points: [-1, X1, X2, 1] -> [-1, Y1, Y2, 1]
// Raw params:  x0, x1, x2, y0, y1, y2 > 0
// Internal:    Sx = x0+x1+x2,  Sy = y0+y1+y2
//   xs = [0, x0/Sx, (x0+x1)/Sx, 1] * 2 - 1
//   ys = [0, y0/Sy, (y0+y1)/Sy, 1] * 2 - 1
//
// Jacobian:
//   X1 = 2*x0/Sx - 1
//     dX1/dx0 =  2*(x1+x2)/Sx^2
//     dX1/dx1 = -2*x0/Sx^2
//     dX1/dx2 = -2*x0/Sx^2
//   X2 = 2*(x0+x1)/Sx - 1
//     dX2/dx0 =  2*x2/Sx^2
//     dX2/dx1 =  2*x2/Sx^2
//     dX2/dx2 = -2*(x0+x1)/Sx^2
//   (Y analogous)
// ============================================================

struct Coeffs {
    float x0, x1, x2, y0, y1, y2;  // raw params (6)
    float xs[4];                     // internal x control points
    float ys[4];                     // internal y control points
    float hs[3];                     // segment lengths
    float ds[3];                     // segment slopes
};

// ============================================================
// Find segment index
// ============================================================
__device__ __forceinline__
int _find_seg(const float* xs, float t) {
    int seg = 0;
    seg += (t >= xs[1]);
    seg += (t >= xs[2]);
    return seg;
}

// ============================================================
// Precompute coefficients
// point[0]=x0, [1]=x1, [2]=x2, [3]=y0, [4]=y1, [5]=y2
// ============================================================
__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    Coeffs c;
    c.x0 = point[0];
    c.x1 = point[1];
    c.x2 = point[2];
    c.y0 = point[3];
    c.y1 = point[4];
    c.y2 = point[5];

    float Sx = c.x0 + c.x1 + c.x2;
    float Sy = c.y0 + c.y1 + c.y2;

    c.xs[0] = -1.0f;
    c.xs[1] =  2.0f * c.x0 / Sx - 1.0f;
    c.xs[2] =  2.0f * (c.x0 + c.x1) / Sx - 1.0f;
    c.xs[3] =  1.0f;

    c.ys[0] = -1.0f;
    c.ys[1] =  2.0f * c.y0 / Sy - 1.0f;
    c.ys[2] =  2.0f * (c.y0 + c.y1) / Sy - 1.0f;
    c.ys[3] =  1.0f;

    for (int i = 0; i < 3; ++i) {
        c.hs[i] = c.xs[i + 1] - c.xs[i];
        c.ds[i] = (c.ys[i + 1] - c.ys[i]) / c.hs[i];
    }

    return c;
}

// ============================================================
// Forward evaluation
// ============================================================
__device__ __forceinline__
float coeffs_eval(float t, const Coeffs& c) {
    if (t <= c.xs[0] || t >= c.xs[3]) return t;

    int seg = _find_seg(c.xs, t);

    float s = (t - c.xs[seg]) / c.hs[seg];
    return (1.0f - s) * c.ys[seg] + s * c.ys[seg + 1];
}

// ============================================================
// Backward pass with dout/dt
// ============================================================
__device__ __forceinline__
void backward_with_doutdt(
    float t,
    const Coeffs& c,
    float grad_output,
    float& grad_x0,
    float& grad_x1,
    float& grad_x2,
    float& grad_y0,
    float& grad_y1,
    float& grad_y2,
    float& dout_dt
) {
    grad_x0 = grad_x1 = grad_x2 = 0.0f;
    grad_y0 = grad_y1 = grad_y2 = 0.0f;
    dout_dt = 0.0f;

    if (t <= c.xs[0] || t >= c.xs[3]) {
        dout_dt = 1.0f;
        return;
    }

    int seg = _find_seg(c.xs, t);

    float xi  = c.xs[seg];
    float xi1 = c.xs[seg + 1];
    float yi  = c.ys[seg];
    float yi1 = c.ys[seg + 1];
    float h   = c.hs[seg];
    float s   = (t - xi) / h;
    float dy  = yi1 - yi;

    dout_dt = dy / h;

    // Gradients w.r.t. internal control points
    float dout_dxi  = dy * (t - xi1) / (h * h);
    float dout_dxi1 = dy * (xi  - t) / (h * h);

    // seg=0: xi=xs[0](固定), xi1=X1;  yi=ys[0](固定), yi1=Y1
    // seg=1: xi=X1,          xi1=X2;  yi=Y1,          yi1=Y2
    // seg=2: xi=X2,    xi1=xs[3](固定); yi=Y2,  yi1=ys[3](固定)

    float grad_X1 = 0.0f, grad_X2 = 0.0f;
    float grad_Y1 = 0.0f, grad_Y2 = 0.0f;

    if (seg == 0) {
        grad_X1 = grad_output * dout_dxi1;
        grad_Y1 = grad_output * s;
    } else if (seg == 1) {
        grad_X1 = grad_output * dout_dxi;
        grad_X2 = grad_output * dout_dxi1;
        grad_Y1 = grad_output * (1.0f - s);
        grad_Y2 = grad_output * s;
    } else {  // seg == 2
        grad_X2 = grad_output * dout_dxi;
        grad_Y2 = grad_output * (1.0f - s);
    }

    // Chain rule: internal → raw params
    float Sx  = c.x0 + c.x1 + c.x2;
    float Sy  = c.y0 + c.y1 + c.y2;
    float Sx2 = Sx * Sx;
    float Sy2 = Sy * Sy;

    grad_x0 = grad_X1 * ( 2.0f * (c.x1 + c.x2) / Sx2) + grad_X2 * ( 2.0f * c.x2 / Sx2);
    grad_x1 = grad_X1 * (-2.0f * c.x0 / Sx2)           + grad_X2 * ( 2.0f * c.x2 / Sx2);
    grad_x2 = grad_X1 * (-2.0f * c.x0 / Sx2)           + grad_X2 * (-2.0f * (c.x0 + c.x1) / Sx2);

    grad_y0 = grad_Y1 * ( 2.0f * (c.y1 + c.y2) / Sy2) + grad_Y2 * ( 2.0f * c.y2 / Sy2);
    grad_y1 = grad_Y1 * (-2.0f * c.y0 / Sy2)           + grad_Y2 * ( 2.0f * c.y2 / Sy2);
    grad_y2 = grad_Y1 * (-2.0f * c.y0 / Sy2)           + grad_Y2 * (-2.0f * (c.y0 + c.y1) / Sy2);
}

// ============================================================
// Atomic backward (供 kernel 调用)
// base_idx: offset into dL_dinter_points
//   [base+0]=x0, [base+1]=x1, [base+2]=x2,
//   [base+3]=y0, [base+4]=y1, [base+5]=y2
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
    float grad_x0, grad_x1, grad_x2, grad_y0, grad_y1, grad_y2, dout_dt;
    backward_with_doutdt(
        t_raw / rescale, c, grad_output,
        grad_x0, grad_x1, grad_x2,
        grad_y0, grad_y1, grad_y2, dout_dt
    );
    atomicAdd(&dL_dinter_points[base_idx + 0], grad_x0 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 1], grad_x1 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 2], grad_x2 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 3], grad_y0 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 4], grad_y1 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 5], grad_y2 * rescale);
    return dout_dt;
}