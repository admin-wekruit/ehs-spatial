# SAM3D mesh runtime build inputs

No runtime image has been built or approved. The source patch is locally checked;
the complete Linux dependency artifact lock and redistribution audit remain open.

The official source is fixed at
`facebookresearch/sam-3d-objects@f91db411c50efee93d8db7aeb323885650f6f722`.
`scripts/prepare_sam3d_mesh_source.py` verifies five upstream file hashes before
applying `modal_apps/sam3d_mesh_only.patch` and writes a hash-bound build receipt.
The patch defers optional Gaussian package exports, renderer imports and mesh
postprocessing dependencies. It retains the original source and license headers.

Local preparation on 2026-09-15 produced receipt SHA-256
`05117f70c333b66e4842c613ffaa3c3ce928ec2503f79819eb86e5d84e54b54b`.
Four CPU tests passed with Torch 2.5.1, trimesh 4.10.0 and NumPy 1.26.4. They check
actual upstream mesh postprocessing, exact vertices/colors/topology, deferred
exports, and a conservative 71-module eager-import closure from the pointmap
pipeline and mesh decoder. The original source is the negative control for
Gaussian imports. These checks are not a Linux CUDA import trace or inference
fixture, and do not pass any release or license gate.

## Verifiable base candidates

Registry manifests were fetched without downloading image layers:

| Official Linux amd64 image | Verified manifest digest |
| --- | --- |
| `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel` | `sha256:e8e63dd7baca894ba11fe1ba48a52a550793c8974f89b533d697784dd20a4dc0` |
| `nvidia/cuda:12.1.1-devel-ubuntu22.04` | `sha256:327c9e046fbf662275be0934742f7e5412f9b24402ee90bf4d649c1a21707912` |

[PyTorch registry metadata](https://hub.docker.com/v2/repositories/pytorch/pytorch/tags/2.5.1-cuda12.1-cudnn9-devel)
and [NVIDIA registry metadata](https://hub.docker.com/v2/repositories/nvidia/cuda/tags/12.1.1-devel-ubuntu22.04)
provide the source records. Python's exact version in the PyTorch image has not
been verified. The pinned upstream
[Conda environment](https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/f91db411c50efee93d8db7aeb323885650f6f722/environments/default.yml)
specifies Python 3.11.0, CUDA 12.1.1 and GCC 12.4.0.

## Remaining build prerequisites

- Resolve and hash the complete Linux package graph, including transitive and
  build dependencies. Upstream requirements pin many direct versions but do not
  lock artifacts, `hatchling`, or `hatch-requirements-txt`.
- Fix compiler, Torch ABI and CUDA architecture targets for native extensions.
  Pin the selected FlashAttention wheel or complete source build, the Kaolin
  wheel, PyTorch3D build configuration, and the fetched Hydra patch bytes.
- Audit the exact redistributed file inventory and notices. The unused Gaussian
  renderer and `representations/gaussian/general_utils.py` retain INRIA
  non-commercial headers; deferring imports does not clear redistribution.
  Kaolin also ships separately licensed `non_commercial` components.
- Obtain approved checkpoint access for the deployed account. Public source
  access does not grant access to the gated model repository.
- Build and record the final immutable image digest, then run the frozen admin
  runtime validation with the existing paid-call reservation. Record actual
  external-pointmap/no-internal-depth and official native-pose evidence before
  changing runtime release evidence; quality validation remains separate.

A Dockerfile is intentionally absent until these inputs can be fixed without
claiming an unresolved dependency graph is reproducible.
