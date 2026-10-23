#pragma once

// ============================================================
// PCHIP with decoupled p/q parameterization
//
// Assumption:
//   point[0] in [0, 1] -> u, controls knot position
//   point[1] in [0, 1] -> v, controls log slope ratio
//
// Mapping:
//   p = 2u - 1
//   q = PCHIP_QMAX * (2v - 1)
//   R = exp(q) = d0 / d1
//
// Recover:
//   h0 = p + 1
//   h1 = 1 - p
//   D  = h1 + R*h0
//
//   d0 = 2R / D
//   d1 = 2  / D
//   y0 = -1 + d0*h0
//
// This guarantees:
//   d0*h0 + d1*h1 = 2
//   d0 > 0, d1 > 0
//   monotone increasing PCHIP control data
//
// Identity:
//   point[0] = 0.5
//   point[1] = 0.5
//   => p = 0, q = 0, R = 1, d0 = d1 = 1
// ============================================================

#ifndef PCHIP_EPS
#define PCHIP_EPS 1e-3f
#endif

#ifndef PCHIP_QMAX
#define PCHIP_QMAX 3.0f
#endif

struct Coeffs {
    float x0, y0;

    float m0, m1, m2;   // derivatives at -1, x0, 1
    float h0, h1;       // segment lengths
    float d0, d1;       // segment slopes

    // For p/q backward
    float R;
    float D;
};

// ============================================================
// Clamp helper
// ============================================================
__device__ __forceinline__
float clamp01_eps(float x) {
    return fminf(fmaxf(x, PCHIP_EPS), 1.0f - PCHIP_EPS);
}

// ============================================================
// Edge derivative, clamped to non-negative
// ============================================================
__device__ __forceinline__
float pchip_edge_derivative(float h0, float h1, float d0, float d1) {
    float d = ((2.0f * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    return (d < 0.0f) ? 0.0f : d;
}

// ============================================================
// Precompute PCHIP coefficients
//
// p/q parameterization:
//   point[0] -> u -> p = x0
//   point[1] -> v -> q = log(d0/d1)
// ============================================================
__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    // point is assumed sigmoided, clamp only for numerical safety.
    float u = clamp01_eps(point[0]);
    float v = clamp01_eps(point[1]);

    float p = 2.0f * u - 1.0f;

    float q = PCHIP_QMAX * (2.0f * v - 1.0f);
    float R = expf(q);

    float h0 = p + 1.0f;
    float h1 = 1.0f - p;

    float D = h1 + R * h0;

    // Direct slope computation is more stable than computing through y0 first.
    float d0 = 2.0f * R / D;
    float d1 = 2.0f / D;

    float y0 = -1.0f + d0 * h0;

    // Boundary derivatives
    float m0 = pchip_edge_derivative(h0, h1, d0, d1);
    float m2 = pchip_edge_derivative(h1, h0, d1, d0);

    // Interior derivative, weighted harmonic mean.
    // With this parameterization d0,d1 > 0, so this branch should normally hold.
    float m1;
    if (d0 * d1 > 0.0f) {
        float w1 = 2.0f * h1 + h0;
        float w2 = h1 + 2.0f * h0;
        m1 = (w1 + w2) / (w1 / d0 + w2 / d1);
    } else {
        m1 = 0.0f;
    }

    Coeffs c;
    c.x0 = p;
    c.y0 = y0;

    c.m0 = m0;
    c.m1 = m1;
    c.m2 = m2;

    c.h0 = h0;
    c.h1 = h1;

    c.d0 = d0;
    c.d1 = d1;

    c.R = R;
    c.D = D;

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
        // Segment [-1, x0]
        x_i = -1.0f;
        y_i = -1.0f;
        y_ip1 = c.y0;
        h_i = c.h0;
        m_i = c.m0;
        m_ip1 = c.m1;
    } else {
        // Segment [x0, 1]
        x_i = c.x0;
        y_i = c.y0;
        y_ip1 = 1.0f;
        h_i = c.h1;
        m_i = c.m1;
        m_ip1 = c.m2;
    }

    float s  = (t - x_i) / h_i;
    float s2 = s * s;
    float s3 = s2 * s;

    return (2.0f*s3 - 3.0f*s2 + 1.0f) * y_i
         + (s3 - 2.0f*s2 + s)          * h_i * m_i
         + (-2.0f*s3 + 3.0f*s2)        * y_ip1
         + (s3 - s2)                   * h_i * m_ip1;
}

// ============================================================
// Backward pass with dout/dt
//
// Internally:
//   1. Compute dL/dx0 and dL/dy0 as if x0,y0 were independent.
//   2. Chain from:
//
//        x0 = p
//        y0 = y0(p, q)
//
//      where:
//
//        p = 2u - 1
//        q = PCHIP_QMAX * (2v - 1)
//
//      and:
//
//        dy0/dp = 4R / D^2
//        dy0/dq = 2R*h0*h1 / D^2
//
// Therefore:
//
//   dL/dp = dL/dx0 + dL/dy0 * dy0/dp
//   dL/dq = dL/dy0 * dy0/dq
//
//   dL/du = dL/dp * 2
//   dL/dv = dL/dq * 2*PCHIP_QMAX
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
    float x0 = c.x0;
    float y0 = c.y0;

    float h0 = c.h0;
    float h1 = c.h1;

    float d0 = c.d0;
    float d1 = c.d1;

    // Out of range: identity
    if (t < -1.0f || t > 1.0f) {
        dout_dt = 1.0f;
        grad_u = 0.0f;
        grad_v = 0.0f;
        return;
    }

    // --------------------------------------------------------
    // Determine segment
    // --------------------------------------------------------
    int seg;
    float x_i, y_i, y_ip1, h_i, m_i, m_ip1;

    if (t < x0) {
        seg = 0;  // [-1, x0]
        x_i = -1.0f;
        y_i = -1.0f;
        y_ip1 = y0;
        h_i = h0;
        m_i = c.m0;
        m_ip1 = c.m1;
    } else {
        seg = 1;  // [x0, 1]
        x_i = x0;
        y_i = y0;
        y_ip1 = 1.0f;
        h_i = h1;
        m_i = c.m1;
        m_ip1 = c.m2;
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

    // --------------------------------------------------------
    // dout/dt
    // --------------------------------------------------------
    float dout_ds = dH0_ds * y_i
                  + dH1_ds * h_i * m_i
                  + dH2_ds * y_ip1
                  + dH3_ds * h_i * m_ip1;

    dout_dt = dout_ds / h_i;

    // --------------------------------------------------------
    // Parameter gradients wrt independent x0,y0
    // --------------------------------------------------------

    // d0, d1 derivatives w.r.t. x0, y0
    //
    // d0 = (y0 + 1) / h0, h0 = x0 + 1
    // d1 = (1 - y0) / h1, h1 = 1 - x0
    float dd0_dx0 = -d0 / h0;
    float dd0_dy0 =  1.0f / h0;

    float dd1_dx0 =  d1 / h1;
    float dd1_dy0 = -1.0f / h1;

    // --------------------------------------------------------
    // m0 derivatives
    //
    // m0_raw = ((2*h0+h1)*d0 - h0*d1) / (h0+h1)
    // --------------------------------------------------------
    float dm0_dx0;
    float dm0_dy0;

    float m0_raw = ((2.0f*h0 + h1)*d0 - h0*d1) / (h0 + h1);

    if (m0_raw < 0.0f) {
        dm0_dx0 = 0.0f;
        dm0_dy0 = 0.0f;
    } else {
        float num = (2.0f*h0 + h1)*d0 - h0*d1;
        float den = h0 + h1;  // = 2

        // dh0/dx0 = 1
        // dh1/dx0 = -1
        //
        // d(2h0+h1)/dx0 = 1
        float dnum_dx0 = d0
                       + (2.0f*h0 + h1) * dd0_dx0
                       - d1
                       - h0 * dd1_dx0;

        float dnum_dy0 = (2.0f*h0 + h1) * dd0_dy0
                       - h0 * dd1_dy0;

        // den = h0+h1 = 2, so dden/dx0 = 0
        dm0_dx0 = dnum_dx0 / den;
        dm0_dy0 = dnum_dy0 / den;
    }

    // --------------------------------------------------------
    // m2 derivatives
    //
    // m2_raw = ((2*h1+h0)*d1 - h1*d0) / (h0+h1)
    // --------------------------------------------------------
    float dm2_dx0;
    float dm2_dy0;

    float m2_raw = ((2.0f*h1 + h0)*d1 - h1*d0) / (h0 + h1);

    if (m2_raw < 0.0f) {
        dm2_dx0 = 0.0f;
        dm2_dy0 = 0.0f;
    } else {
        float num = (2.0f*h1 + h0)*d1 - h1*d0;
        float den = h0 + h1;  // = 2

        // d(2h1+h0)/dx0 = -1
        float dnum_dx0 = -d1
                       + (2.0f*h1 + h0) * dd1_dx0
                       + d0
                       - h1 * dd0_dx0;

        float dnum_dy0 = (2.0f*h1 + h0) * dd1_dy0
                       - h1 * dd0_dy0;

        dm2_dx0 = dnum_dx0 / den;
        dm2_dy0 = dnum_dy0 / den;
    }

    // --------------------------------------------------------
    // m1 derivatives, interior weighted harmonic mean
    //
    // m1 = (w1+w2) / (w1/d0 + w2/d1)
    //
    // w1 = 2*h1 + h0
    // w2 = h1 + 2*h0
    // --------------------------------------------------------
    float dm1_dx0;
    float dm1_dy0;

    if (d0 * d1 > 0.0f) {
        float w1 = 2.0f*h1 + h0;
        float w2 = h1 + 2.0f*h0;

        float sum_w = w1 + w2;
        float inv_term = w1 / d0 + w2 / d1;

        // dw1/dx0 = 2*(-1) + 1 = -1
        // dw2/dx0 = (-1) + 2*1 = 1
        float dw1_dx0 = -1.0f;
        float dw2_dx0 =  1.0f;

        float dsum_w_dx0 = dw1_dx0 + dw2_dx0;  // = 0

        float dinv_dx0 = dw1_dx0 / d0
                       - w1 * dd0_dx0 / (d0*d0)
                       + dw2_dx0 / d1
                       - w2 * dd1_dx0 / (d1*d1);

        float dinv_dy0 = -w1 * dd0_dy0 / (d0*d0)
                       - w2 * dd1_dy0 / (d1*d1);

        dm1_dx0 = (dsum_w_dx0 * inv_term - sum_w * dinv_dx0)
                / (inv_term * inv_term);

        dm1_dy0 = -sum_w * dinv_dy0
                / (inv_term * inv_term);
    } else {
        dm1_dx0 = 0.0f;
        dm1_dy0 = 0.0f;
    }

    // --------------------------------------------------------
    // Final gradients wrt independent x0,y0
    // --------------------------------------------------------
    float dL_dx0;
    float dL_dy0;

    if (seg == 0) {
        // Segment [-1, x0]
        //
        // h_i = h0
        // x_i = -1
        // s = (t + 1) / h0
        // ds/dx0 = -s / h0
        float ds_dx0 = -s / h0;

        float dout_dh = H1 * m_i + H3 * m_ip1;  // H1*m0 + H3*m1

        float dout_dm0 = H1 * h0;
        float dout_dm1 = H3 * h0;

        // dh0/dx0 = 1
        dL_dx0 = grad_output * (
            dout_ds * ds_dx0
          + dout_dh * 1.0f
          + dout_dm0 * dm0_dx0
          + dout_dm1 * dm1_dx0
        );

        // y0 is y_ip1 on segment 0
        dL_dy0 = grad_output * (
            H2
          + dout_dm0 * dm0_dy0
          + dout_dm1 * dm1_dy0
        );
    } else {
        // Segment [x0, 1]
        //
        // h_i = h1
        // x_i = x0
        // s = (t - x0) / h1
        // ds/dx0 = (t - 1) / h1^2
        float ds_dx0 = (t - 1.0f) / (h1 * h1);

        float dout_dh = H1 * m_i + H3 * m_ip1;  // H1*m1 + H3*m2

        float dout_dm1 = H1 * h1;
        float dout_dm2 = H3 * h1;

        // dh1/dx0 = -1
        dL_dx0 = grad_output * (
            dout_ds * ds_dx0
          + dout_dh * (-1.0f)
          + dout_dm1 * dm1_dx0
          + dout_dm2 * dm2_dx0
        );

        // y0 is y_i on segment 1
        dL_dy0 = grad_output * (
            H0
          + dout_dm1 * dm1_dy0
          + dout_dm2 * dm2_dy0
        );
    }

    // --------------------------------------------------------
    // Chain rule: independent x0,y0 -> p,q -> point[0],point[1]
    // --------------------------------------------------------
    float R = c.R;
    float D = c.D;

    float inv_D_sq = 1.0f / (D * D);

    // y0 = -1 + 2R*h0 / D
    //
    // p = x0
    // h0 = p + 1
    // h1 = 1 - p
    // D = h1 + R*h0
    float dy0_dp = 4.0f * R * inv_D_sq;
    float dy0_dq = 2.0f * R * h0 * h1 * inv_D_sq;

    float dL_dp = dL_dx0 + dL_dy0 * dy0_dp;
    float dL_dq = dL_dy0 * dy0_dq;

    // p = 2u - 1
    // q = PCHIP_QMAX * (2v - 1)
    grad_u = dL_dp * 2.0f;
    grad_v = dL_dq * (2.0f * PCHIP_QMAX);
}

// ============================================================
// Atomic backward
//
// Assumes forward shape:
//
//   t = t_raw / rescale
//   out = rescale * coeffs_eval(t, c)
//
// Then parameter grads get an extra * rescale,
// while returned dout_dt is already d(out)/d(t_raw).
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