
#pragma once
struct Coeffs {
    float x0, y0;
    float m0, m1, m2, m3;
    float h0, h1, h2;
    // temporary values for derivative calculations
    float d1, d2;
};

__device__ __forceinline__
float clampf(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ __forceinline__
float pchip_edge_derivative_device(float h0, float h1, float m0, float m1) {
    float d = ((2.0f * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if (d < 0.0f) {
        return 0.0f;
    }
    return d;
}

__device__ __forceinline__
Coeffs coeffs_precompute(const float* point) {
    float x0 = point[0];
    float y0 = point[1];
    float x0 = clampf(x0, 1e-6f, 1.0f - 1e-6f);
    float d0 = 1.0f;
    float d1 = y0 / x0;
    float d2 = (1.0f - y0) / (1.0f - x0);

    float h0 = 1.0f, h1 = x0, h2 = 1.0f - x0;
    
    float m0 = pchip_edge_derivative_device(h0, h1, d0, d1);
    float m3 = pchip_edge_derivative_device(h2, h1, d2, d1);
    
    float m1 = 3.0f * (x0 + 1.0f) * d1 / (d1 * (2.0f*x0 + 1.0f) + (x0 + 2.0f));
    float m2 = 3.0f * d1 * d2 / (d2 * (2.0f - x0) + d1 * (1.0f + x0));
    
    return Coeffs{x0, y0, m0, m1, m2, m3, 1.0f, x0, 1.0f - x0, d1, d2};
}

__device__ __forceinline__ 
float coeffs_eval(float t, const Coeffs& c) {
    float x_i, y_i, y_ip1, h_i, m_i, m_ip1;
    if (t < -1 || t > 1) {
        return t;
    } else if (t < 0.0f) {
        x_i = -1.0f; y_i = -1.0f; y_ip1 = 0.0f;
        h_i = c.h0; m_i = c.m0; m_ip1 = c.m1;
    } else if (t < c.x0) {
        x_i = 0.0f; y_i = 0.0f; y_ip1 = c.y0;
        h_i = c.h1; m_i = c.m1; m_ip1 = c.m2;
    } else {
        x_i = c.x0; y_i = c.y0; y_ip1 = 1.0f;
        h_i = c.h2; m_i = c.m2; m_ip1 = c.m3;
    }
    
    float s = (t - x_i) / h_i;
    float s2 = s * s;
    float s3 = s2 * s;
    
    return (2.0f*s3 - 3.0f*s2 + 1.0f) * y_i +
           (s3 - 2.0f*s2 + s) * h_i * m_i +
           (-2.0f*s3 + 3.0f*s2) * y_ip1 +
           (s3 - s2) * h_i * m_ip1;
}

__device__ __forceinline__
void backward_with_doutdt(
    float t, 
    const Coeffs& c,
    float grad_output,    // dL/d(output)
    float& grad_x0,       // 输出: dL/dx0
    float& grad_y0,       // 输出: dL/dy0
    float& dout_dt        // 输出: dout/dt (for chain rule)
) {    
    float x0 = c.x0;
    float y0 = c.y0;
    float d1 = c.d1;
    float d2 = c.d2;
    
    // Determine segment (只算一次)
    int seg;
    float x_i, y_i, y_ip1;
    float h_i, m_i, m_ip1;

    if (t < -1 || t > 1) {
        dout_dt = 1.0f;
        grad_x0 = 0.0f;
        grad_y0 = 0.0f;
        return;
    }else if (t < 0.0f) {
        seg = 0;
        x_i = -1.0f; y_i = -1.0f; y_ip1 = 0.0f;
        h_i = c.h0; m_i = c.m0; m_ip1 = c.m1;
    } else if (t < x0) {
        seg = 1;
        x_i = 0.0f; y_i = 0.0f; y_ip1 = y0;
        h_i = c.h1; m_i = c.m1; m_ip1 = c.m2;
    } else {
        seg = 2;
        x_i = x0; y_i = y0; y_ip1 = 1.0f;
        h_i = c.h2; m_i = c.m2; m_ip1 = c.m3;
    }
    
    float s = (t - x_i) / h_i;
    float s2 = s * s;
    float s3 = s2 * s;
    
    // Hermite basis functions (backward 需要)
    float H0 = 2.0f*s3 - 3.0f*s2 + 1.0f;
    float H1 = s3 - 2.0f*s2 + s;
    float H2 = -2.0f*s3 + 3.0f*s2;
    float H3 = s3 - s2;
    
    // dH/ds (两个都需要，只算一次)
    float dH0_ds = 6.0f*s2 - 6.0f*s;
    float dH1_ds = 3.0f*s2 - 4.0f*s + 1.0f;
    float dH2_ds = -6.0f*s2 + 6.0f*s;
    float dH3_ds = 3.0f*s2 - 2.0f*s;
    
    // ========== 计算 dout/dt (原 coeffs_eval_with_grad 的功能) ==========
    float dout_ds = dH0_ds * y_i + dH1_ds * h_i * m_i +
                    dH2_ds * y_ip1 + dH3_ds * h_i * m_ip1;
    dout_dt = dout_ds / h_i;
    
    // ========== 计算参数梯度 (原 pchip_backward 的功能) ==========
    // d1, d2 derivatives w.r.t. x0, y0
    float dd1_dx0 = -y0 / (x0 * x0);
    float dd1_dy0 = 1.0f / x0;
    float one_minus_x0 = 1.0f - x0;
    float dd2_dx0 = (1.0f - y0) / (one_minus_x0 * one_minus_x0);
    float dd2_dy0 = -1.0f / one_minus_x0;
    
    // m0 derivatives
    float dm0_dx0, dm0_dy0;
    float m0_raw = (2.0f + x0 - d1) / (1.0f + x0);
    if (m0_raw < 0.0f) {
        dm0_dx0 = 0.0f;
        dm0_dy0 = 0.0f;
    } else {
        float num = 2.0f + x0 - d1;
        float den = 1.0f + x0;
        float dnum_dx0 = 1.0f - dd1_dx0;
        float dnum_dy0 = -dd1_dy0;
        dm0_dx0 = (dnum_dx0 * den - num) / (den * den);
        dm0_dy0 = dnum_dy0 / den;
    }
    
    // m1 derivatives
    float num1 = 3.0f * (x0 + 1.0f) * d1;
    float den1 = d1 * (2.0f*x0 + 1.0f) + (x0 + 2.0f);
    float dnum1_dx0 = 3.0f * d1 + 3.0f * (x0 + 1.0f) * dd1_dx0;
    float dnum1_dy0 = 3.0f * (x0 + 1.0f) * dd1_dy0;
    float dden1_dx0 = dd1_dx0 * (2.0f*x0 + 1.0f) + d1 * 2.0f + 1.0f;
    float dden1_dy0 = dd1_dy0 * (2.0f*x0 + 1.0f);
    float dm1_dx0 = (dnum1_dx0 * den1 - num1 * dden1_dx0) / (den1 * den1);
    float dm1_dy0 = (dnum1_dy0 * den1 - num1 * dden1_dy0) / (den1 * den1);
    
    // m2 derivatives
    float num2 = 3.0f * d1 * d2;
    float den2 = d2 * (2.0f - x0) + d1 * (1.0f + x0);
    float dnum2_dx0 = 3.0f * (dd1_dx0 * d2 + d1 * dd2_dx0);
    float dnum2_dy0 = 3.0f * (dd1_dy0 * d2 + d1 * dd2_dy0);
    float dden2_dx0 = dd2_dx0 * (2.0f - x0) - d2 + dd1_dx0 * (1.0f + x0) + d1;
    float dden2_dy0 = dd2_dy0 * (2.0f - x0) + dd1_dy0 * (1.0f + x0);
    float dm2_dx0 = (dnum2_dx0 * den2 - num2 * dden2_dx0) / (den2 * den2);
    float dm2_dy0 = (dnum2_dy0 * den2 - num2 * dden2_dy0) / (den2 * den2);
    
    // m3 derivatives
    float dm3_dx0, dm3_dy0;
    float m3_raw = (2.0f - x0) * d2 - (1.0f - x0) * d1;
    if (m3_raw < 0.0f) {
        dm3_dx0 = 0.0f;
        dm3_dy0 = 0.0f;
    } else {
        dm3_dx0 = -d2 + (2.0f - x0) * dd2_dx0 + d1 - (1.0f - x0) * dd1_dx0;
        dm3_dy0 = (2.0f - x0) * dd2_dy0 - (1.0f - x0) * dd1_dy0;
    }
    
    // Compute final gradients based on segment
    if (seg == 0) {
        float dout_dm0 = H1 * h_i;
        float dout_dm1 = H3 * h_i;
        grad_x0 = grad_output * (dout_dm0 * dm0_dx0 + dout_dm1 * dm1_dx0);
        grad_y0 = grad_output * (dout_dm0 * dm0_dy0 + dout_dm1 * dm1_dy0);
    } else if (seg == 1) {
        float ds_dx0 = -s / x0;
        float dout_dh = H1 * m_i + H3 * m_ip1;
        float dout_dm1 = H1 * h_i;
        float dout_dm2 = H3 * h_i;
        float dout_dy_ip1 = H2;
        
        grad_x0 = grad_output * (dout_ds * ds_dx0 + dout_dh +
                                  dout_dm1 * dm1_dx0 + dout_dm2 * dm2_dx0);
        grad_y0 = grad_output * (dout_dy_ip1 + dout_dm1 * dm1_dy0 + dout_dm2 * dm2_dy0);
    } else {  // seg == 2
        float ds_dx0 = (t - 1.0f) / (h_i * h_i);
        float dout_dh = H1 * m_i + H3 * m_ip1;
        float dout_dm2 = H1 * h_i;
        float dout_dm3 = H3 * h_i;
        float dout_dy_i = H0;
        
        grad_x0 = grad_output * (dout_ds * ds_dx0 - dout_dh +
                                  dout_dm2 * dm2_dx0 + dout_dm3 * dm3_dx0);
        grad_y0 = grad_output * (dout_dy_i + dout_dm2 * dm2_dy0 + dout_dm3 * dm3_dy0);
    }
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