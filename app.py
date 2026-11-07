"""
app.py - Root Streamlit entry point.
Strictly isolates sys.path and sys.modules to this project directory and sets __file__
correctly before executing web/app.py on every Streamlit script rerun.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_ROOT_STR = str(_ROOT)

# Remove parent directory if present so Python never imports an older sibling folder
sys.path = [p for p in sys.path if Path(p).resolve() != _ROOT.parent]
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

# Evict any cached modules loaded from outside _ROOT
for _mod_name in list(sys.modules.keys()):
    if _mod_name.split(".")[0] in ("config", "data", "engine", "web", "hk_quant_system"):
        _mod_obj = sys.modules.get(_mod_name)
        _mod_file = getattr(_mod_obj, "__file__", None)
        if _mod_file and not str(Path(_mod_file).resolve()).startswith(_ROOT_STR):
            del sys.modules[_mod_name]

_WEB_APP_FILE = _ROOT / "web" / "app.py"
_exec_globals = dict(globals())
_exec_globals["__file__"] = str(_WEB_APP_FILE)
with open(_WEB_APP_FILE, "rb") as _f:
    _code = compile(_f.read(), str(_WEB_APP_FILE), "exec")
exec(_code, _exec_globals)
