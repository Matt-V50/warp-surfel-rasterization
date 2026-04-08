/*
 * 5-point symmetric PCHIP
 * =======================
 * Control points: [-1, -x0, 0, x0, 1] -> [-1, -y0, 0, y0, 1]
 * Odd symmetry: f(-t) = -f(t), f(0) = 0
 * Outside [-1, 1]: identity (f(t) = t)
 *
 * 4 segments:
 *   seg 0: [-1,  -x0]  h_a  me -> mi
 *   seg 1: [-x0,   0]  h_b  mi -> mc
 *   seg 2: [0,    x0]  h_b  mc -> mi
 *   seg 3: [x0,    1]  h_a  mi -> me
 *
 * h_a = 1 - x0          (outer segment length)
 * h_b = x0              (inner segment length)
 * d_a = (1-y0)/(1-x0)   (outer slope)
 * d_b = y0/x0            (inner slope)
 * me  = clamped edge derivative at |t|=1
 * mi  = weighted harmonic mean at |t|=x0
 * mc  = d_b  (center derivative at t=0)
 *
 * Only 2 free parameters (x0, y0) thanks to symmetry.
 */

#pragma once

struct Coeffs {
    float x0, y0;
    float me, mi, mc;   // derivatives at |t|=1, |t|=x0, t=0
    float h_a, h_b;     // outer / inner segment lengths
    float d_a, d_b;     // outer / inner slopes
};

// ============================================================
// Edge derivative (clamped to non-negative)
// ============================================================
__device__ __forceinline__
float pchip_edge_derivative(float h_a, float h_b, float d_a, float d_b) {
    // h_a + h_b = 1, so denominator is 1
    float d = (2.0f * h_a + h_b) * d_a - h_a * d_b;
    return fmaxf(d, 0.0f);
}

// ============================================================
// Interior derivative (weighted harmonic mean, Fritsch-Carlson)
// At x0: left segment has (h_b, d_b), right has (h_a, d_a)
// ============================================================
__device__ __forceinline__
float pchip_interior_derivative(float h_a, float h_b, float d_a, float d_b) {
    if (d_a * d_b <= 0.0f) return 0.0f;
    float w1 = 2.0f * h_a + h_b;
    float w2 = h_a + 2.0f * h_b;
    return (w1 + w2) / (w1 / d_b + w2 / d_a);
}

// ============================================================
// Precompute coefficients
// ============================================================
__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    float x0 = point[0];
    float y0 = point[1];

    float h_a = 1.0f - x0;
    float h_b = x0;
    float d_a = (1.0f - y0) / h_a;
    float d_b = y0 / h_b;

    float me = pchip_edge_derivative(h_a, h_b, d_a, d_b);
    float mi = pchip_interior_derivative(h_a, h_b, d_a, d_b);
    float mc = (d_b != 0.0f) ? d_b : 0.0f;

    Coeffs c;
    c.x0 = x0;  c.y0 = y0;
    c.h_a = h_a; c.h_b = h_b;
    c.d_a = d_a; c.d_b = d_b;
    c.me = me;   c.mi = mi;   c.mc = mc;
    return c;
}

// ============================================================
// Forward evaluation
// ============================================================
__device__ __forceinline__
float coeffs_eval(float t, const Coeffs& c) {
    if (t < -1.0f || t > 1.0f) return t;

    float x_i, y_i, y_ip1, h_i, m_i, m_ip1;

    if (t < -c.x0) {
        // seg 0: [-1, -x0]
        x_i = -1.0f; y_i = -1.0f; y_ip1 = -c.y0;
        h_i = c.h_a; m_i = c.me;  m_ip1 = c.mi;
    } else if (t < 0.0f) {
        // seg 1: [-x0, 0]
        x_i = -c.x0; y_i = -c.y0; y_ip1 = 0.0f;
        h_i = c.h_b; m_i = c.mi;  m_ip1 = c.mc;
    } else if (t < c.x0) {
        // seg 2: [0, x0]
        x_i = 0.0f; y_i = 0.0f; y_ip1 = c.y0;
        h_i = c.h_b; m_i = c.mc;  m_ip1 = c.mi;
    } else {
        // seg 3: [x0, 1]
        x_i = c.x0; y_i = c.y0; y_ip1 = 1.0f;
        h_i = c.h_a; m_i = c.mi;  m_ip1 = c.me;
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
// Backward pass
// ============================================================
__device__ __forceinline__
void coeffs_backward_with_doutdt(
    float t,
    const Coeffs& c,
    float grad_output,
    float& grad_x0,
    float& grad_y0,
    float& dout_dt
) {
    if (t < -1.0f || t > 1.0f) {
        dout_dt = 1.0f;
        grad_x0 = 0.0f;
        grad_y0 = 0.0f;
        return;
    }

    float x0 = c.x0, y0 = c.y0;
    float h_a = c.h_a, h_b = c.h_b;
    float d_a = c.d_a, d_b = c.d_b;

    // ======================================================
    // Determine segment
    // ======================================================
    int seg;
    float x_i, y_i, y_ip1, h_i, m_i, m_ip1;

    if (t < -x0) {
        seg = 0;
        x_i = -1.0f; y_i = -1.0f; y_ip1 = -y0;
        h_i = h_a; m_i = c.me; m_ip1 = c.mi;
    } else if (t < 0.0f) {
        seg = 1;
        x_i = -x0; y_i = -y0; y_ip1 = 0.0f;
        h_i = h_b; m_i = c.mi; m_ip1 = c.mc;
    } else if (t < x0) {
        seg = 2;
        x_i = 0.0f; y_i = 0.0f; y_ip1 = y0;
        h_i = h_b; m_i = c.mc; m_ip1 = c.mi;
    } else {
        seg = 3;
        x_i = x0; y_i = y0; y_ip1 = 1.0f;
        h_i = h_a; m_i = c.mi; m_ip1 = c.me;
    }

    float s  = (t - x_i) / h_i;
    float s2 = s * s;
    float s3 = s2 * s;

    // Hermite basis
    float H0 =  2.0f*s3 - 3.0f*s2 + 1.0f;
    float H1 =  s3 - 2.0f*s2 + s;
    float H2 = -2.0f*s3 + 3.0f*s2;
    float H3 =  s3 - s2;

    // dH/ds
    float dH0_ds =  6.0f*s2 - 6.0f*s;
    float dH1_ds =  3.0f*s2 - 4.0f*s + 1.0f;
    float dH2_ds = -6.0f*s2 + 6.0f*s;
    float dH3_ds =  3.0f*s2 - 2.0f*s;

    // dout/dt
    float dout_ds = dH0_ds * y_i   + dH1_ds * h_i * m_i
                  + dH2_ds * y_ip1 + dH3_ds * h_i * m_ip1;
    dout_dt = dout_ds / h_i;

    // ======================================================
    // Slope derivatives w.r.t. (x0, y0)
    // d_a = (1-y0)/(1-x0),   d_b = y0/x0
    // ======================================================
    float dd_a_dx0 =  d_a / h_a;     // (1-y0)/(1-x0)^2
    float dd_a_dy0 = -1.0f / h_a;    // -1/(1-x0)
    float dd_b_dx0 = -d_b / h_b;     // -y0/x0^2
    float dd_b_dy0 =  1.0f / h_b;    // 1/x0

    // ======================================================
    // dme/d(x0,y0)  —  edge derivative at |t|=1
    // me_raw = (2*h_a + h_b)*d_a - h_a*d_b,  den = 1
    // ======================================================
    float me_raw = (2.0f*h_a + h_b)*d_a - h_a*d_b;
    float dme_dx0, dme_dy0;
    if (me_raw < 0.0f) {
        dme_dx0 = 0.0f;
        dme_dy0 = 0.0f;
    } else {
        // dh_a/dx0 = -1,  dh_b/dx0 = 1
        // d(2*h_a+h_b)/dx0 = -2+1 = -1
        // num = (2*h_a+h_b)*d_a - h_a*d_b
        dme_dx0 = -d_a + (2.0f*h_a + h_b)*dd_a_dx0
                  + d_b - h_a*dd_b_dx0;
        dme_dy0 = (2.0f*h_a + h_b)*dd_a_dy0 - h_a*dd_b_dy0;
    }

    // ======================================================
    // dmi/d(x0,y0)  —  interior derivative at |t|=x0
    // At x0: left seg (h_b, d_b), right seg (h_a, d_a)
    // w1 = 2*h_a + h_b,  w2 = h_a + 2*h_b,  w1+w2 = 3
    // mi = 3 / (w1/d_b + w2/d_a)
    // ======================================================
    float dmi_dx0, dmi_dy0;
    if (d_a * d_b <= 0.0f) {
        dmi_dx0 = 0.0f;
        dmi_dy0 = 0.0f;
    } else {
        float w1 = 2.0f*h_a + h_b;
        float w2 = h_a + 2.0f*h_b;
        float inv_term = w1/d_b + w2/d_a;

        // dw1/dx0 = -1,  dw2/dx0 = 1
        float dinv_dx0 = -1.0f/d_b - w1*dd_b_dx0/(d_b*d_b)
                        + 1.0f/d_a - w2*dd_a_dx0/(d_a*d_a);
        float dinv_dy0 = -w1*dd_b_dy0/(d_b*d_b)
                        - w2*dd_a_dy0/(d_a*d_a);

        float inv2 = inv_term * inv_term;
        dmi_dx0 = -3.0f * dinv_dx0 / inv2;
        dmi_dy0 = -3.0f * dinv_dy0 / inv2;
    }

    // ======================================================
    // dmc/d(x0,y0)  —  center derivative at t=0
    // mc = d_b = y0/x0
    // ======================================================
    float dmc_dx0, dmc_dy0;
    if (d_b != 0.0f) {
        dmc_dx0 = dd_b_dx0;   // -y0/x0^2
        dmc_dy0 = dd_b_dy0;   // 1/x0
    } else {
        dmc_dx0 = 0.0f;
        dmc_dy0 = 0.0f;
    }

    // ======================================================
    // Per-segment gradient assembly
    // ======================================================
    float dout_dh  = H1 * m_i + H3 * m_ip1;
    float dout_dml = H1 * h_i;   // dout/d(m_left)
    float dout_dmr = H3 * h_i;   // dout/d(m_right)

    float ds_dx0;

    if (seg == 0) {
        // [-1, -x0]: h_i=h_a, x_i=-1(fixed), y_i=-1(fixed), y_ip1=-y0
        // s = (t+1)/h_a,  dh_a/dx0=-1
        // ds/dx0 = (0 - s*(-1)) / h_a = s/h_a
        ds_dx0 = s / h_a;

        grad_x0 = grad_output * (dout_ds * ds_dx0
                                + dout_dh * (-1.0f)         // dh_a/dx0 = -1
                                + dout_dml * dme_dx0
                                + dout_dmr * dmi_dx0);
        grad_y0 = grad_output * (H2 * (-1.0f)              // dy_ip1/dy0 = -1
                                + dout_dml * dme_dy0
                                + dout_dmr * dmi_dy0);

    } else if (seg == 1) {
        // [-x0, 0]: h_i=h_b, x_i=-x0, y_i=-y0, y_ip1=0(fixed)
        // s = (t+x0)/h_b,  dx_i/dx0=-1, dh_b/dx0=1
        // ds/dx0 = (1 - s*1)/h_b  ... but simplifies to -t/h_b^2
        ds_dx0 = -t / (h_b * h_b);

        grad_x0 = grad_output * (dout_ds * ds_dx0
                                + dout_dh * 1.0f            // dh_b/dx0 = 1
                                + dout_dml * dmi_dx0
                                + dout_dmr * dmc_dx0);
        grad_y0 = grad_output * (H0 * (-1.0f)              // dy_i/dy0 = -1
                                + dout_dml * dmi_dy0
                                + dout_dmr * dmc_dy0);

    } else if (seg == 2) {
        // [0, x0]: h_i=h_b, x_i=0(fixed), y_i=0(fixed), y_ip1=y0
        // s = t/h_b,  dh_b/dx0=1
        // ds/dx0 = -s/h_b
        ds_dx0 = -s / h_b;

        grad_x0 = grad_output * (dout_ds * ds_dx0
                                + dout_dh * 1.0f            // dh_b/dx0 = 1
                                + dout_dml * dmc_dx0
                                + dout_dmr * dmi_dx0);
        grad_y0 = grad_output * (H2 * 1.0f                 // dy_ip1/dy0 = 1
                                + dout_dml * dmc_dy0
                                + dout_dmr * dmi_dy0);

    } else {
        // [x0, 1]: h_i=h_a, x_i=x0, y_i=y0, y_ip1=1(fixed)
        // s = (t-x0)/h_a,  dx_i/dx0=1, dh_a/dx0=-1
        // ds/dx0 = (-1 - s*(-1))/h_a = (s-1)/h_a = (t-1)/h_a^2
        ds_dx0 = (t - 1.0f) / (h_a * h_a);

        grad_x0 = grad_output * (dout_ds * ds_dx0
                                + dout_dh * (-1.0f)         // dh_a/dx0 = -1
                                + dout_dml * dmi_dx0
                                + dout_dmr * dme_dx0);
        grad_y0 = grad_output * (H0 * 1.0f                 // dy_i/dy0 = 1
                                + dout_dml * dmi_dy0
                                + dout_dmr * dme_dy0);
    }
}

// ============================================================
// Convenience: backward + atomicAdd for training loops
// Returns dout/dt for upstream chain rule
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
    float grad_x0, grad_y0, dout_dt;
    coeffs_backward_with_doutdt(
        t_raw / rescale, c, grad_output,
        grad_x0, grad_y0, dout_dt
    );
    atomicAdd(&dL_dinter_points[base_idx + 0], grad_x0 * rescale);
    atomicAdd(&dL_dinter_points[base_idx + 1], grad_y0 * rescale);
    return dout_dt;
}