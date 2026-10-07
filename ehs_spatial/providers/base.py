import os
import sys
from pathlib import Path


class ProviderError(RuntimeError):
    def __init__(self, provider: str, operation: str, original_message: str) -> None:
        self.provider = provider
        self.operation = operation
        self.original_message = original_message
        super().__init__(f"{provider} {operation} failed: {original_message}")


def serving_root() -> Path:
    """This checkout (panoptes-serving): PANOPTES_SERVING, else the package's parent."""
    return Path(os.environ.get("PANOPTES_SERVING") or Path(__file__).resolve().parents[2])


def workcell_root() -> Path:
    """The ehs-spatial checkout (platform + workcell checks + on-prem kit): PANOPTES_WORKCELL,
    else ../ehs-spatial next to this repo (env.sh's default)."""
    root = Path(os.environ.get("PANOPTES_WORKCELL") or serving_root().parent / "ehs-spatial")
    if not (root / "scripts/onprem/run_stage.py").exists():
        raise ProviderError("onprem", "workcell", f"{root} has no scripts/onprem/run_stage.py: set PANOPTES_WORKCELL")
    return root


def onprem():
    """The on-prem runner (ehs-spatial/scripts/onprem/run_stage.py) with its prologue done, for the `local` backends:
    the modal stub first on sys.path, torch.hub pinned, the weights mirror (WEIGHTS, as env.sh names it) linked.
    When this process IS run_stage.py (env.sh's MODAL_RUN), its own module is reused and nothing is redone."""
    main = sys.modules.get("__main__")
    if Path(getattr(main, "__file__", "") or "").name == "run_stage.py":
        return main
    if "run_stage" in sys.modules:
        return sys.modules["run_stage"]
    sys.path.insert(0, str(workcell_root() / "scripts/onprem"))
    import run_stage

    os.environ["PANOPTES_ONPREM"] = "1"
    run_stage.use_stub()
    run_stage.pin_torch_hub()
    if os.environ.get("WEIGHTS"):
        os.environ["HF_HUB_OFFLINE"] = "1"
        run_stage.link_weights(Path(os.environ["WEIGHTS"]).resolve())
    return run_stage
