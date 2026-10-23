#pragma once

struct Coeffs {
    float x0, y0;
    float h0, h1;       // segment lengths: h0 = x0+1, h1 = 1-x0
    float d0, d1;       // segment slopes
};

// ============================================================
// Precompute PWL coefficients
// Control points: [-1, x0, 1] -> [-1, y0, 1]
// ============================================================
__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    float x0 = point[0] * 2 - 1.0f;  // rescale to [-1, 1]
    float y0 = point[1] * 2 - 1.0f;
    float h0 = x0 + 1.0f;          // x0 - (-1)
    float h1 = 1.0f - x0;          // 1 - x0

    float d0 = (y0 + 1.0f) / h0;   // slope of [-1, x0]
    float d1 = (1.0f - y0) / h1;   // slope of [x0, 1]

    Coeffs c;
    c.x0 = x0; c.y0 = y0;
    c.h0 = h0; c.h1 = h1;
    c.d0 = d0; c.d1 = d1;
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
        // Segment [-1, x0]: out = -1 + d0 * (t + 1)
        return -1.0f + c.d0 * (t + 1.0f);
    } else {
        // Segment [x0, 1]: out = y0 + d1 * (t - x0)
        return c.y0 + c.d1 * (t - c.x0);
    }
}

// ============================================================
// Backward pass with dout/dt
//
// Segment 0 (t < x0):
//   s = (t+1)/h0
//   dout/dt  = d0
//   dout/dx0 = -d0 * s
//   dout/dy0 = s
//
// Segment 1 (t >= x0):
//   s = (t-x0)/h1
//   dout/dt  = d1
//   dout/dx0 = d1 * (s - 1)
//   dout/dy0 = 1 - s
// ============================================================
__device__ __forceinline__
void backward_with_doutdt(
    float t,
    const Coeffs& c,
    float grad_output,    // dL/d(output)
    float& grad_x0,      // output: dL/dx0
    float& grad_y0,      // output: dL/dy0
    float& dout_dt        // output: dout/dt
) {
    // Out of range
    if (t < -1.0f || t > 1.0f) {
        dout_dt = 1.0f;
        grad_x0 = 0.0f;
        grad_y0 = 0.0f;
        return;
    }

    if (t < c.x0) {
        // Segment [-1, x0]
        float s = (t + 1.0f) / c.h0;
        dout_dt = c.d0;
        grad_x0 = grad_output * (-c.d0 * s);
        grad_y0 = grad_output * s;
    } else {
        // Segment [x0, 1]
        float s = (t - c.x0) / c.h1;
        dout_dt = c.d1;
        grad_x0 = grad_output * (c.d1 * (s - 1.0f));
        grad_y0 = grad_output * (1.0f - s);
    }

    grad_x0 *= 2.0f;  // chain rule for rescaling x0 from [0, 1] to [-1, 1]
    grad_y0 *= 2.0f;  // chain rule for rescaling y0 from [0, 1] to [-1, 1]
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