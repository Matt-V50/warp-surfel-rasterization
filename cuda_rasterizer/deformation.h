#pragma once

// Warp kernel selection. Each kernel is compiled into its own extension module
// (see setup.py), which passes these macros on the command line:
//   WARP_KERNEL     name of the header in cuda_rasterizer/kernels/ (without .h)
//   DOF             learnable parameters per control point
//   CONTROL_POINTS  warped axes per Gaussian (2 = warp both s.x and s.y)
// RESCALE_FACTOR and cutoff are runtime arguments (GaussianRasterizationSettings).

#ifndef WARP_KERNEL
#define WARP_KERNEL pwl_sr
#endif

#ifndef DOF
#define DOF 2
#endif

#ifndef CONTROL_POINTS
#define CONTROL_POINTS 2
#endif

#define WARP_STR_(x) #x
#define WARP_STR(x) WARP_STR_(x)
#include WARP_STR(kernels/WARP_KERNEL.h)
