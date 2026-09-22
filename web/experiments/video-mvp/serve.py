"""Serve existing experiment artifacts with HTTP Range support for video seeking."""
import argparse
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles


class Revalidated(StaticFiles):
    """Artifacts change in place between runs: without a cache header the browser reused a stale page and manifest."""
    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"  # keep the copy, but ask (ETag) before using it; Range and 304 still work
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--port", type=int, default=8799)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("directory must be an existing artifact directory")
    app = Starlette(routes=[Mount("/", app=Revalidated(directory=args.directory, html=True))])
    uvicorn.run(app, host="127.0.0.1", port=args.port)
