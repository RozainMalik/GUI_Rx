from importlib import import_module

_MOD = {"screenspot": "screenspot", "sspro": "sspro", "osworld_g": "osworld_g", "osworld_g_refined": "osworld_g",
        "ui_vision": "ui_vision", "mmbench_gui": "mmbench_gui", "medspot": "medspot"}


def load(loader, root):
    mod = import_module(f"benchev.benchmarks.{_MOD[loader]}")
    if loader == "osworld_g_refined":
        return mod.load(root, refined=True)
    return mod.load(root)
