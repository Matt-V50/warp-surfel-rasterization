#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import runpy

ROOT = os.path.dirname(os.path.abspath(__file__))
registry = runpy.run_path(os.path.join(ROOT, "warp_surfel_rasterization", "kernels.py"))
KERNELS, CONTROL_POINTS = registry["KERNELS"], registry["CONTROL_POINTS"]

# Build a subset with e.g. WARP_KERNELS=pwl_sr,pchip_sr pip install ...
selected = os.environ.get("WARP_KERNELS", "")
selected = [k.strip() for k in selected.split(",") if k.strip()] or list(KERNELS)
unknown = set(selected) - set(KERNELS)
if unknown:
    raise ValueError(f"Unknown WARP_KERNELS {sorted(unknown)}, available: {list(KERNELS)}")


class BuildWarpExtension(BuildExtension):
    # All kernels share the same sources with different macros, so give every
    # extension its own build dir to keep the object files apart.
    def build_extension(self, ext):
        build_temp = self.build_temp
        self.build_temp = os.path.join(build_temp, ext.name)
        try:
            super().build_extension(ext)
        finally:
            self.build_temp = build_temp


def make_extension(name):
    cfg = KERNELS[name]
    macros = [f"-DWARP_KERNEL={cfg['header']}", f"-DDOF={cfg['dof']}", f"-DCONTROL_POINTS={CONTROL_POINTS}"]
    return CUDAExtension(
        name=f"warp_surfel_rasterization._C_{name}",
        sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
        extra_compile_args={
            "cxx": macros,
            "nvcc": macros + ["-I" + os.path.join(ROOT, "third_party/glm/")]})


setup(
    name="warp_surfel_rasterization",
    packages=['warp_surfel_rasterization'],
    version='0.1.0',
    ext_modules=[make_extension(name) for name in selected],
    cmdclass={
        'build_ext': BuildWarpExtension
    }
)
