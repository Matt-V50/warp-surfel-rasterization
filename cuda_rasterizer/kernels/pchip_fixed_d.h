#pragma once

struct Coeffs {
    float x0;
    float m0, m1, m2;   // derivatives at -1, x0, 1
    float h0, h1;        // segment lengths
};

// ============================================================
// Edge derivative (clamped to non-negative)
// ============================================================
__device__ __forceinline__
float pchip_edge_derivative(float h0, float h1, float d0, float d1) {
    float d = ((2.0f * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    return (d < 0.0f) ? 0.0f : d;
}

// ============================================================
// Precompute PCHIP coefficients
// Control points: [-1, x0, 1] -> [-1, 0, 1]   (y0 fixed to 0)
// ============================================================
__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    float x0 = point[0] * 2 - 1;
    float h0 = x0 + 1.0f;          // x0 - (-1)
    float h1 = 1.0f - x0;          // 1 - x0

    float d0 = 1.0f / h0;          // (0+1)/h0
    float d1 = 1.0f / h1;          // (1-0)/h1

    // Boundary derivatives
    float m0 = pchip_edge_derivative(h0, h1, d0, d1);
    float m2 = pchip_edge_derivative(h1, h0, d1, d0);

    // Interior derivative (d0*d1 > 0 always since h0,h1 > 0)
    float w1 = 2.0f * h1 + h0;
    float w2 = h1 + 2.0f * h0;
    float m1 = (w1 + w2) / (w1 * h0 + w2 * h1);  // = 6 / (w1*h0 + w2*h1)

    Coeffs c;
    c.x0 = x0;
    c.m0 = m0; c.m1 = m1; c.m2 = m2;
    c.h0 = h0; c.h1 = h1;
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

    float x_i, y_i, y_ip1, h_i, m_i, m_ip1;

    if (t < c.x0) {
        // Segment [-1, x0]: y goes -1 -> 0
        x_i = -1.0f; y_i = -1.0f; y_ip1 = 0.0f;
        h_i = c.h0;  m_i = c.m0;  m_ip1 = c.m1;
    } else {
        // Segment [x0, 1]: y goes 0 -> 1
        x_i = c.x0;  y_i = 0.0f;  y_ip1 = 1.0f;
        h_i = c.h1;  m_i = c.m1;  m_ip1 = c.m2;
    }

    float s  = (t - x_i) / h_i;
    float s2 = s * s;
    float s3 = s2 * s;

    return (2.0f*s3 - 3.0f*s2 + 1.0f) * y_i
         + (s3 - 2.0f*s2 + s)          * h_i * m_i
         + (-2.0f*s3 + 3.0f*s2)        * y_ip1
         + (s3 - s2)                    * h_i * m_ip1;
}

// ============================================================
// Backward pass with dout/dt
// Only grad_x0 needed (y0 is fixed at 0)
// ============================================================
__device__ __forceinline__
void backward_with_doutdt(
    float t,
    const Coeffs& c,
    float grad_output,    // dL/d(output)
    float& grad_x0,      // output: dL/dx0
    float& dout_dt        // output: dout/dt
) {
    float x0 = c.x0;
    float h0 = c.h0, h1 = c.h1;

    // Out of range
    if (t < -1.0f || t > 1.0f) {
        dout_dt = 1.0f;
        grad_x0 = 0.0f;
        return;
    }

    // d0 = 1/h0, d1 = 1/h1
    float d0 = 1.0f / h0;
    float d1 = 1.0f / h1;

    // Determine segment
    int seg;
    float x_i, y_i, y_ip1, h_i, m_i, m_ip1;

    if (t < x0) {
        seg = 0;  // [-1, x0]
        x_i = -1.0f; y_i = -1.0f; y_ip1 = 0.0f;
        h_i = h0; m_i = c.m0; m_ip1 = c.m1;
    } else {
        seg = 1;  // [x0, 1]
        x_i = x0; y_i = 0.0f; y_ip1 = 1.0f;
        h_i = h1; m_i = c.m1; m_ip1 = c.m2;
    }

    float s  = (t - x_i) / h_i;
    float s2 = s * s;
    float s3 = s2 * s;

    // Hermite basis
    float H0 = 2.0f*s3 - 3.0f*s2 + 1.0f;
    float H1 = s3 - 2.0f*s2 + s;
    float H2 = -2.0f*s3 + 3.0f*s2;
    float H3 = s3 - s2;

    // dH/ds
    float dH0_ds = 6.0f*s2 - 6.0f*s;
    float dH1_ds = 3.0f*s2 - 4.0f*s + 1.0f;
    float dH2_ds = -6.0f*s2 + 6.0f*s;
    float dH3_ds = 3.0f*s2 - 2.0f*s;

    // ===== dout/dt =====
    float dout_ds = dH0_ds * y_i + dH1_ds * h_i * m_i
                  + dH2_ds * y_ip1 + dH3_ds * h_i * m_ip1;
    dout_dt = dout_ds / h_i;

    // ===== Parameter gradients w.r.t. x0 only =====

    // dd0/dx0 = -1/h0^2,  dd1/dx0 = 1/h1^2
    float dd0_dx0 = -d0 * d0;   // -1/h0^2
    float dd1_dx0 =  d1 * d1;   //  1/h1^2

    // ----- m0 derivative -----
    float dm0_dx0;
    float m0_raw = ((2.0f*h0 + h1)*d0 - h0*d1) / (h0 + h1);
    if (m0_raw < 0.0f) {
        dm0_dx0 = 0.0f;
    } else {
        float den = h0 + h1;  // = 2.0
        // dh0/dx0 = 1, dh1/dx0 = -1
        float dnum_dx0 = d0 + (2.0f*h0 + h1)*dd0_dx0 - d1 - h0*dd1_dx0;
        dm0_dx0 = dnum_dx0 / den;
    }

    // ----- m2 derivative -----
    float dm2_dx0;
    float m2_raw = ((2.0f*h1 + h0)*d1 - h1*d0) / (h0 + h1);
    if (m2_raw < 0.0f) {
        dm2_dx0 = 0.0f;
    } else {
        float den = h0 + h1;  // = 2.0
        float dnum_dx0 = -d1 + (2.0f*h1 + h0)*dd1_dx0 + d0 - h1*dd0_dx0;
        dm2_dx0 = dnum_dx0 / den;
    }

    // ----- m1 derivative (d0*d1 > 0 always) -----
    float w1 = 2.0f*h1 + h0;
    float w2 = h1 + 2.0f*h0;
    float sum_w = w1 + w2;           // = 6
    float inv_term = w1*h0 + w2*h1;  // = w1/d0 + w2/d1

    // dw1/dx0 = -1, dw2/dx0 = 1
    float dinv_dx0 = -h0 + w1 - w1*h0*dd0_dx0
                   +  h1 - w2 - w2*h1*dd1_dx0;
    // Simplify: dinv = -h0 + w1 - w1*h0*(-d0^2) + h1 - w2 - w2*h1*(d1^2)
    // Actually let me keep the original form for clarity:
    // dinv_dx0 = dw1_dx0/d0 - w1*dd0_dx0/(d0^2) + dw2_dx0/d1 - w2*dd1_dx0/(d1^2)
    //          = -1*h0 - w1*(-1/h0^2)*h0^2 + 1*h1 - w2*(1/h1^2)*h1^2
    //          = -h0 + w1 + h1 - w2
    //          = -h0 + (2h1+h0) + h1 - (h1+2h0)
    //          = -h0 + 2h1 + h0 + h1 - h1 - 2h0
    //          = 2h1 - 2h0
    float dinv_dx0_simplified = 2.0f * (h1 - h0);

    // dm1/dx0 = (0 * inv_term - 6 * dinv) / inv_term^2  (dsum_w/dx0 = 0)
    float dm1_dx0 = -sum_w * dinv_dx0_simplified / (inv_term * inv_term);

    // ===== Final gradient by segment =====
    if (seg == 0) {
        // Segment [-1, x0]: h_i = h0, x_i = -1
        // s = (t+1)/h0,  ds/dx0 = -s/h0
        float ds_dx0 = -s / h0;
        float dout_dh  = H1 * m_i + H3 * m_ip1;
        float dout_dm0 = H1 * h0;
        float dout_dm1 = H3 * h0;

        grad_x0 = grad_output * (dout_ds * ds_dx0 + dout_dh
                                + dout_dm0 * dm0_dx0 + dout_dm1 * dm1_dx0);
    } else {
        // Segment [x0, 1]: h_i = h1, x_i = x0
        // s = (t-x0)/h1,  ds/dx0 = (t-1)/(h1^2)
        float ds_dx0 = (t - 1.0f) / (h1 * h1);
        float dout_dh  = H1 * m_i + H3 * m_ip1;
        float dout_dm1 = H1 * h1;
        float dout_dm2 = H3 * h1;

        grad_x0 = grad_output * (dout_ds * ds_dx0 - dout_dh
                                + dout_dm1 * dm1_dx0 + dout_dm2 * dm2_dx0);
    }
    grad_x0 *= 2.0f;  // Chain rule for x0 = point[0]*2-1
}


__device__ __forceinline__
float coeffs_backward_atomic(
    float t_raw,
    float rescale,
    const Coeffs& c,
    float grad_output,
    float* dL_dinter_points,
    int base_idx
) {
    float grad_x0, dout_dt;
    backward_with_doutdt(t_raw / rescale, c, grad_output, grad_x0, dout_dt);
    atomicAdd(&dL_dinter_points[base_idx], grad_x0 * rescale);
    return dout_dt;
}