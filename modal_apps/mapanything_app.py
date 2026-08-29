"""MapAnything multiview reconstruction on Modal — scaffold.

Contract (must match the Replicate response the pipeline already reads,
see ehs_spatial/backends.py):

    MapAnything().run.remote(input: dict) -> dict
      input:  {"inputs": [dataURI, ...], **flags}   (replicate schema)
      output: replicate MapAnything response schema (per-frame pts3d /
              valid masks / camera_to_world / intrinsics as URLs or bytes)

Not yet deployed: facebook/map-anything weights are Apache-2.0 but the
packaging (multi-view batching, memory profile on L4 vs A100) needs a
sizing pass before we commit GPU dollars. GEOMETRY_BACKEND=replicate stays
the default until this is smoke-tested; the router in
ehs_spatial/providers/map_anything.py already points at
("mapanything-inference", "MapAnything"), so deploying this file is the
only step left when we flip.
"""

import modal

app = modal.App("mapanything-inference")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "huggingface_hub",
        # ponytail: pin once smoke-tested; scaffold stays loose
        "mapanything @ git+https://github.com/facebookresearch/map-anything",
    )
)

weights = modal.Volume.from_name("mapanything-hf-cache", create_if_missing=True)


@app.cls(
    image=image,
    gpu="A100",
    volumes={"/cache": weights},
    scaledown_window=120,
    timeout=600,
)
class MapAnything:
    @modal.enter()
    def load(self):
        raise NotImplementedError(
            "scaffold — port the Replicate MapAnything handler here"
        )

    @modal.method()
    def run(self, input: dict) -> dict:
        raise NotImplementedError
