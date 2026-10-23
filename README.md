# Warp Surfel Rasterization

Differentiable surfel rasterizer with learnable warped kernels, the CUDA backend of
[Warp-2DGS](https://github.com/Matt-V50/Warp-2DGS). It extends the
[2DGS rasterizer](https://github.com/hbb1/diff-surfel-rasterization): before the Gaussian is
evaluated, the splat-space coordinates of each axis pass through a learnable monotonic
deformation function D(t), so every primitive can learn an asymmetric, sharp-edged falloff.

## Install

```bash
pip install --no-build-isolation .
# or build only the kernels you need (each kernel is a separate extension, ~35 s each)
WARP_KERNELS=pwl_sr,pchip_sr pip install --no-build-isolation .
```

## Usage

```python
from warp_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer, KERNELS

settings = GaussianRasterizationSettings(
    ...,                 # same fields as diff-surfel-rasterization
    kernel="pwl_sr",     # warp kernel, see KERNELS
    rescale=3.0,         # warped domain [-rescale, rescale] in splat space
    cutoff=-1.0,         # footprint cutoff, <= 0 means rescale * sqrt(2)
)
color, radii, original_radii, allmap = GaussianRasterizer(settings)(
    inter_points=inter_points,   # (P, 2, KERNELS[kernel]["dof"]), sigmoid-activated
    means3D=..., means2D=..., opacities=..., shs=..., scales=..., rotations=...)
```

`original_radii` is the unwarped 3-sigma screen radius, used for densification statistics.

## Kernels

| Kernel | DOF | Paper |
|---|---|---|
| `identity` | 1 | D(t) = t, plain 2DGS |
| `pwl`, `pwl_sr` | 2 | PWL, PWL-sr |
| `pchip`, `pchip_sr` | 2 | Pchip, Pchip-sr |
| `pwl2_mult`, `pwl2_segment` | 4 | PWL2-mult, PWL2-segment (Appendix D) |
| `pwl3_symmetric` | 2 | PWL3-symmetric (Appendix D) |
| `pwl3_asymmetric` | 4 | PWL3-asymmetric (Appendix D) |
| `pwl3_asymmetric_y0` | 2 | PWL3-asymmetric-y0 (Appendix D) |
| `pwl_fixed_d`, `pchip_fixed_d` | 1 | not in the paper: d1 = 0, only c1 learned |
| `pchip_antidiag` | 1 | not in the paper: Pchip with d1 = -c1 |
| `pwl_plateau` | 1 | not in the paper: symmetric flat-top (plateau + linear ramp) |
| `pchip3_symmetric` | 2 | not in the paper: Pchip counterpart of PWL3-symmetric |

Kernels live in `cuda_rasterizer/kernels/<name>.h` and are registered in
`warp_surfel_rasterization/kernels.py`. `cuda_rasterizer/experimental/` keeps variants that are
not built by default (e.g. PWL2-clamp / -accumulate / -fixed-x / -quadrant from Figure 9).

## Citation

```bibtex
@inproceedings{li2026warp2dgs,
  title     = {Warp-2DGS: Learnable Warped Kernels for 2D Gaussian Splatting},
  author    = {Li, Huibin and Yeum, Chul Min},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

This rasterizer builds on 2DGS and 3DGS; please also cite them:

```bibtex
@inproceedings{Huang2DGS2024,
  title     = {2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
  author    = {Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
  publisher = {Association for Computing Machinery},
  booktitle = {SIGGRAPH 2024 Conference Papers},
  year      = {2024},
  doi       = {10.1145/3641519.3657428}
}

@Article{kerbl3Dgaussians,
  author  = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title   = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal = {ACM Transactions on Graphics},
  number  = {4},
  volume  = {42},
  month   = {July},
  year    = {2023},
  url     = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

## License

Released under the Inria / MPII non-commercial license inherited from 3DGS and 2DGS, see
[LICENSE.md](LICENSE.md).
