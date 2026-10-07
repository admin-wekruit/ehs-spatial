# Third-party notices: panoptes-workcell-geometry (docker/geometry.Dockerfile)

In the image at `/licences/THIRD_PARTY_NOTICES.md`. Compiled 2026-10-06 from a scan of the built image (every installed
distribution's metadata and licence files, the shared libraries bundled in the wheels, `torch.__config__.show()`, the Debian
package list and its copyright files): research-notes/geometry-backbone-ab-2026-10-06/geometry_image_scan.json. This is an
inventory, not legal advice.

**No CUDA or NVIDIA component.** torch and torchvision are the CPU builds (`torch 2.5.1+cpu`, `USE_CUDA=0`), there are no
`nvidia-*` wheels and no triton, and the base is not a CUDA image. The NVIDIA EULA does not apply to this image.

**Not in this image, on purpose.** pycolmap (its PyPI wheel statically links GPL-2.0-or-later SuiteSparse), vggt (VGGT licence),
plyfile (GPL-3.0), Pi3 / Pi3X (CC BY-NC), DA3-LARGE-1.1 (licence disputed, treated as non-commercial), and romatch's
`estimate_pose` / `estimate_pose_uncalibrated` (lineage through LGPL-2.1 DenseMatching to Magic Leap code; deleted, see below).

## Copyleft and special-licence parts, and what they ask of a distributor

| Part | Where in the image | Licence | Note |
|---|---|---|---|
| libgfortran | `numpy.libs/`, `scipy.libs/`, `opencv_python_headless.libs/` | GPL-3.0 with the GCC Runtime Library Exception 3.1 | The exception lets the code linked against it be distributed under any terms. Keep the licence and exception texts (in numpy's and scipy's `LICENSE.txt`). Source: gcc.gnu.org |
| libgomp (OpenMP runtime) | `torch/lib/libgomp-a34b3233.so.1` | GPL-3.0 with the GCC Runtime Library Exception 3.1 | Same as libgfortran |
| libquadmath | `numpy.libs/`, `scipy.libs/`, `opencv_python_headless.libs/` | LGPL-2.1-or-later | A separate shared library, so a user can replace it. Keep the licence text (numpy / scipy `LICENSE.txt`) and give or offer its source (gcc.gnu.org, the GCC version the wheels name) |
| FFmpeg: libavcodec, libavformat, libavutil, libswresample, libswscale | `opencv_python_headless.libs/` | LGPL-2.1-or-later (as built for opencv-python) | Shared libraries, replaceable. The route does no video I/O. Notice: `cv2/LICENSE-3RD-PARTY.txt` |
| Intel IPP ICV | statically linked into `cv2/cv2.abi3.so` | Intel Simplified Software License (October 2022) | Redistribution allowed under its terms; text in `cv2/LICENSE-3RD-PARTY.txt` |
| Intel oneAPI MKL 2024.2 | statically linked into `torch/lib/libtorch_cpu.so` (`BLAS_INFO=mkl`) | Intel Simplified Software License | Redistribution allowed under its terms. The torch wheel's `LICENSE` concatenates its third-party texts |
| OpenSSL 1.1 (libcrypto, libssl) | `opencv_python_headless.libs/` | OpenSSL + SSLeay licences (advertising clause) | Text in `cv2/LICENSE-3RD-PARTY.txt`. The route makes no network connection |
| certifi, tqdm | site-packages | MPL-2.0 (tqdm: MPL-2.0 AND MIT) | File-level copyleft: unmodified here; source is the wheel itself |
| Debian 12 userland (bash, coreutils, dash, grep, sed, tar, util-linux, perl-base, libc6, libgcc-s1, libstdc++6, ...) | the base image | GPL-2/3, LGPL-2.1/3 and permissive, per package | 105 packages, unmodified Debian bookworm. The route runs none of them as a library (CPython links libc / libgcc / libstdc++ / libssl3, which are LGPL or GCC-exception or Apache-2.0). When shipping the image, give or offer the corresponding Debian source packages (snapshot.debian.org has every version; `dpkg-query -W` lists them). Each package's terms: `/usr/share/doc/<package>/copyright` |
| GNU readline | Debian `libreadline8`, linked by CPython's `readline` extension | GPL-3.0-or-later | Imported only by an interactive Python prompt; the stage never loads it |

## Python distributions (`/opt/geometry`, docker/geometry-requirements.txt, every wheel pinned by SHA-256)

| Distribution | Version | Licence | Bundled native libraries (scan) |
|---|---|---|---|
| numpy | 1.26.4 | BSD-3-Clause | OpenBLAS 0.3.23 (BSD-3-Clause), libgfortran, libquadmath (table above); `LICENSE.txt` lists lapack-lite, tempita, dragon4, libdivide |
| scipy | 1.14.1 | BSD-3-Clause | scipy-openblas (BSD-3-Clause), libgfortran, libquadmath; source notices for ARPACK, Qhull, pocketfft, uarray in the package |
| opencv-python-headless | 4.10.0.84 | Apache-2.0 | FFmpeg (LGPL-2.1+), OpenSSL 1.1, libvpx (BSD-3-Clause), libpng (libpng), OpenBLAS (BSD-3-Clause), libgfortran, libquadmath, Intel IPP ICV (static); `cv2/LICENSE-3RD-PARTY.txt` |
| torch | 2.5.1+cpu | BSD-3-Clause | libgomp; static: Intel oneAPI MKL 2024.2, oneDNN (MKL-DNN) 3.5.3 (Apache-2.0), XNNPACK, NNPACK, fbgemm, sleef, pthreadpool, cpuinfo, pybind11, protobuf, gloo, kineto and the rest listed in `torch-2.5.1+cpu.dist-info/LICENSE` (permissive) |
| torchvision | 0.20.1+cpu | BSD-3-Clause | libjpeg-turbo (IJG + BSD-3-Clause + zlib), libpng, libwebp (BSD-3-Clause), zlib |
| pillow | 11.0.0 | MIT-CMU | FreeType (FTL, chosen from its FTL / GPL-2.0 dual licence), HarfBuzz (MIT), libjpeg-turbo, Little CMS (MIT), liblzma (0BSD / public domain), OpenJPEG (BSD-2-Clause), libpng, libtiff (libtiff), libwebp + sharpyuv (BSD-3-Clause), Brotli (MIT), libxcb + libXau (MIT); `pillow-11.0.0.dist-info/LICENSE` |
| kornia | 0.7.4 | Apache-2.0 | |
| kornia_rs | 0.2.0 | Apache-2.0 | one Rust extension; the wheel ships no notices for its statically linked crates (not audited) |
| einops 0.8.0, loguru 0.7.2, filelock 4.0.12, PyYAML 6.0.3, urllib3 2.8.0, charset-normalizer 3.5.2, setuptools 65.5.1, pip 24.0 | | MIT | |
| huggingface-hub 0.36.0, hf-xet 1.7.0, requests 2.34.2 | | Apache-2.0 | hf-xet: one Rust extension (crate notices not audited) |
| packaging 26.3 | | Apache-2.0 OR BSD-2-Clause | |
| Jinja2 3.1.6, MarkupSafe 3.0.4, idna 3.10, fsspec 2026.9.0, networkx 3.6.1, sympy 1.13.1, mpmath 1.3.0 | | BSD-3-Clause | |
| typing_extensions 4.12.2 | | PSF-2.0 | |
| certifi 2024.8.30 | | MPL-2.0 | |
| tqdm 4.67.1 | | MPL-2.0 AND MIT | |

Each distribution's own licence files are in `/opt/geometry/lib/python3.11/site-packages/<name>-<version>.dist-info/`.

## Vendored source

- **romatch** (Parskatt/RoMa @ `77f8d68803526dcddfd9b7a46bc76125bdc25f15`, archive SHA-256 `7b5264cd…10cb1e6`) at `/vendor/romatch`,
  MIT (`/vendor/romatch/LICENSE`, Copyright (c) 2023 Johan Edstedt). Modified: lines 28-76 of `romatch/utils/utils.py`
  (`estimate_pose`, `estimate_pose_uncalibrated`, "Code taken from PruneTruong/DenseMatching") and their two names in
  `romatch/utils/__init__.py` deleted, `romatch/benchmarks/` (their only callers) deleted. RoMa's README: "All our code except
  DINOv2 is MIT license".
- **DINOv2 model code inside romatch** (`romatch/models/transformer/`, "Copyright (c) Meta Platforms ... licensed under the
  license found in the LICENSE file"): this relies on DINOv2's Apache-2.0 licence (facebookresearch/dinov2 @ `7764ea0` LICENSE,
  mirrored next to the weights). DINOv2 was CC BY-NC 4.0 before Meta relicensed it in 2023; whether RoMa's copy differs from the
  Apache-2.0 upstream code was not checked. Parts adapted from timm (Apache-2.0) and DINO (Apache-2.0) per the file headers.
- **Panoptes kit files** (`/workcell/scripts/onprem`, `/workcell/modal_apps`): this repository's own terms.

## Base image

`python:3.11.10-slim-bookworm@sha256:840e180ebcc6e5c8efab209c43f5e40fd2af98cb49db5c7103c90539c56bb30e`: CPython 3.11.10
(PSF-2.0, `/usr/local/lib/python3.11/LICENSE.txt`) on Debian 12 (bookworm), 105 Debian packages, nothing added by apt. See the
Debian row in the first table.

## Model weights (never in the image; mounted at run time from scripts/onprem/fetch_weights_geometry.py --cache DIR)

| Weights | Used by | Licence | SHA-256 pin | Source |
|---|---|---|---|---|
| RoMa v1 outdoor `roma_outdoor.pth` | this image | MIT, inferred: the release asset carries no licence; the Parskatt/storage repository that hosts it is MIT (its LICENSE is mirrored). Trained on MegaDepth; training-data provenance not audited | `c7a45c80…9e5cc8ba` | github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth |
| DINOv2 ViT-L/14 `dinov2_vitl14_pretrain.pth` | this image (RoMa's encoder) | Apache-2.0 (DINOv2 README: code and model weights; LICENSE mirrored). The same server hosts FAIR non-commercial models: mirror this file only | `d5383ea8…62cbf428` | dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth |
| DA3-BASE `model.safetensors` + `config.json` | the start stage (docker/da3.Dockerfile) | Apache-2.0 (Hugging Face card and the official Depth-Anything-3 README agree) | `e01067dc…bdae78b5` | huggingface.co/depth-anything/DA3-BASE @ f4a6c9b |
| MoGe-3 `model.pt` (alternate start) | the MoGe stage | MIT (card and MoGe LICENSE) | `9b41b7b9…fb7925` | huggingface.co/Ruicheng/moge-3-vitl @ 184008f |

Full pins: scripts/onprem/fetch_weights_geometry.py (`FILES`, `HF`); the stage refuses a RoMa file whose SHA-256 is not its pin.
