"""Import the production build code from its git-archive snapshots.

``code/cae8640`` holds ``packages/`` at the commit that built the US national
release (populace-us-2024-buildp-sparse-rmloss100-cae8640); ``code/67ee2ddf3``
holds the commit that built the UK national release. Only donor loaders and
module constants are used: no calibration runs, so ``torch`` (imported at
module load by the calibration package) is replaced by an inert stub instead of
being installed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent


def _stub(name: str) -> None:
    if name in sys.modules:
        return
    module = mock.MagicMock(name=name)
    module.__spec__ = None
    module.__path__ = []
    # Libraries probe these with issubclass(); they must be real classes.
    module.Tensor = type("Tensor", (), {})
    module.Module = type("Module", (), {})
    module.Optimizer = type("Optimizer", (), {})
    sys.modules[name] = module


def us(commit: str = "cae8640") -> None:
    """Put populace-* at ``commit`` on sys.path.

    ``cae8640`` built the US national release; ``6d06410bb`` built its base
    pool (base sha ``5d8a3162``), whose stages include the PUF tax-detail
    chain and the CPS-trained PUF-support fits. One process uses one commit.
    """
    chosen = [p for p in sys.path if "/packages/populace-build/src" in p]
    if chosen:
        if f"/code/{commit}/" not in chosen[0] and commit != "cae8640":
            raise RuntimeError(f"populace already on sys.path from {chosen[0]}")
        commit = chosen[0].split("/code/")[1].split("/")[0]
    for package in ("populace-build", "populace-fit", "populace-frame", "populace-data",
                    "populace-calibrate"):
        path = str(ROOT / "code" / commit / "packages" / package / "src")
        if path not in sys.path:
            sys.path.insert(0, path)
    for name in ("torch", "torch.nn", "torch.optim", "torch.nn.functional"):
        _stub(name)


def uk() -> None:
    """Put microcosm-build/data/calibrate/diagnostics at 67ee2ddf3 on sys.path.

    microcosm-fit/frame/graph come from the venv, installed at the same commit.
    """
    for package in ("microcosm-build", "microcosm-data", "microcosm-calibrate",
                    "microcosm-diagnostics"):
        path = str(ROOT / "code/67ee2ddf3/packages" / package / "src")
        if path not in sys.path:
            sys.path.insert(0, path)
    for name in ("torch", "torch.nn", "torch.optim", "torch.nn.functional"):
        _stub(name)
    _stub_torch_version()


def _stub_torch_version() -> None:
    """Answer ``importlib.metadata.version("torch")`` for the stubbed torch.

    ``microcosm.build.spec_engine.seeds`` reads torch's installed version at
    import to label the calibration seed site (``seeds.py:826-838``, the
    ``torch_calibration_reseed`` record's ``rng_version`` string). The UK
    country spec loads that module whenever a uk_runtime stage module is
    imported (``uk_runtime/source_runtime.py:29-32``). No QRF stage reads the
    label, so the stub version only lets the import finish.
    """
    from importlib import metadata

    if getattr(metadata.version, "_torch_stub", False):
        return
    real = metadata.version

    def version(name: str) -> str:
        if name == "torch":
            return "0+stub"
        return real(name)

    version._torch_stub = True
    metadata.version = version
