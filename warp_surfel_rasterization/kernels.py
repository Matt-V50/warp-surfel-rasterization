#
# Warp kernel registry, shared by setup.py (build) and __init__.py (runtime).
# Must stay importable without torch.
#
# name -> header in cuda_rasterizer/kernels/<header>.h and its degrees of freedom
# (learnable parameters per control point, i.e. inter_points.shape[-1]).
# Names follow the paper: PWL<k> has k learnable interior knots (k omitted for one),
# (c1, d1) is the interior control point, and "sr" is the slope-ratio parameterization.
#

KERNELS = {
    # Main paper
    "identity":           dict(header="identity",           dof=1),  # D(t) = t, plain 2DGS
    "pwl":                dict(header="pwl",                dof=2),  # PWL
    "pwl_sr":             dict(header="pwl_sr",             dof=2),  # PWL-sr
    "pchip":              dict(header="pchip",              dof=2),  # Pchip
    "pchip_sr":           dict(header="pchip_sr",           dof=2),  # Pchip-sr
    # Multi-knot variants (Appendix D)
    "pwl2_mult":          dict(header="pwl2_mult",          dof=4),  # PWL2-mult
    "pwl2_segment":       dict(header="pwl2_segment",       dof=4),  # PWL2-segment
    "pwl3_symmetric":     dict(header="pwl3_symmetric",     dof=2),  # PWL3-symmetric
    "pwl3_asymmetric":    dict(header="pwl3_asymmetric",    dof=4),  # PWL3-asymmetric
    "pwl3_asymmetric_y0": dict(header="pwl3_asymmetric_y0", dof=2),  # PWL3-asymmetric-y0
    # Additional variants, not reported in the paper
    "pwl_fixed_d":        dict(header="pwl_fixed_d",        dof=1),  # PWL with d1 = 0, only c1 learned
    "pchip_fixed_d":      dict(header="pchip_fixed_d",      dof=1),  # Pchip with d1 = 0, only c1 learned
    "pchip_antidiag":     dict(header="pchip_antidiag",     dof=1),  # Pchip with d1 = -c1
    "pwl_plateau":        dict(header="pwl_plateau",        dof=1),  # symmetric flat-top: plateau + linear ramp
    "pchip3_symmetric":   dict(header="pchip3_symmetric",   dof=2),  # Pchip counterpart of PWL3-symmetric
}

CONTROL_POINTS = 2
