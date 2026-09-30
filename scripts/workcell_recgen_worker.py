"""Run the existing internal RecGen inference on selected photo views in its pinned venv."""

import json
import sys
import time

import numpy as np

from fast_report import recgen_fast, x7


def main(source, target, groups):
    z = np.load(source)
    started = time.monotonic()
    pipeline = x7.load_recgen()
    loaded = time.monotonic() - started
    records = {}
    for name, indices in groups:
        views = [{"rgb": z[f"v{i}_rgb"], "depth": z[f"v{i}_depth"],
                  "mask": z[f"v{i}_mask"], "camera_intrinsics": z[f"v{i}_K"]}
                 for i in indices]
        result = recgen_fast.run(pipeline, views, 42, recgen_fast.setting(**recgen_fast.FAST))
        np.savez_compressed(f"{target}-{name}.npz", vertices=result["vertices"],
                            faces=result["faces"], colors=result["colors"])
        records[name] = {"views": indices, "generationSeconds": result["seconds"],
                         "faces": result["faces_n"], "stages": result["stages"]}
    print(json.dumps({"modelLoadSeconds": loaded, "models": records}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         [(name, [int(i) for i in indices.split(",")])
          for name, indices in (part.split(":", 1) for part in sys.argv[3].split(";"))])
