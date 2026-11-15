"""Import the unchanged GeoLife research modules without eager Torch imports.

The historical research/__init__.py eagerly exports sequence-model classes.
Member B needs the statistical-feature/RF modules only. This helper registers
the package search path without executing that convenience initializer; the
individual research modules are imported normally and remain unchanged.

Example in a script at the extracted package root:
    from pathlib import Path
    import sys
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "tools"))
    from member_b_bootstrap import bootstrap_research
    bootstrap_research(root)
    from research.deployment_protocol import load_config, degrade_window
"""
from __future__ import annotations

from importlib.machinery import ModuleSpec
from pathlib import Path
import sys
from types import ModuleType


def bootstrap_research(root: str | Path) -> ModuleType:
    """Register research's module path, preserving any already imported package.

    This delays only package initialization. It does not change files, replace
    numerical implementations, or stub Torch. Existing packages are left alone;
    a package from another checkout is rejected to prevent mixing two datasets.
    """
    root = Path(root).resolve()
    package = root / "project" / "research"
    if not (package / "__init__.py").is_file():
        raise FileNotFoundError(f"Missing research package: {package}")
    existing = sys.modules.get("research")
    if existing is not None:
        locations = [Path(value).resolve() for value in getattr(existing, "__path__", [])]
        if package not in locations:
            raise RuntimeError("research is already imported from a different checkout; use a fresh Python process")
        return existing
    project = str(root / "project")
    if project not in sys.path:
        sys.path.insert(0, project)
    module = ModuleType("research")
    module.__file__ = str(package / "__init__.py")
    module.__package__ = "research"
    module.__path__ = [str(package)]
    module.__spec__ = ModuleSpec("research", loader=None, origin=module.__file__, is_package=True)
    module.__spec__.submodule_search_locations = module.__path__
    module.__bootstrap_delayed_init__ = True
    sys.modules["research"] = module
    return module

