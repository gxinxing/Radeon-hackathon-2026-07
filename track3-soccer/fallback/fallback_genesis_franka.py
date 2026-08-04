"""Minimal Genesis Franka fallback, matching the remote GPU verification run."""
import json
import time
import traceback
from pathlib import Path

import genesis as gs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)
result = {"example": "Genesis Franka official-style minimal scene", "started_at": time.time()}
try:
    gs.init(backend=gs.gpu)
    scene = gs.Scene(show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()
    for _ in range(5):
        scene.step()
    result.update(status="passed", backend="gpu", steps=5, entities=2)
except Exception as exc:
    result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
(OUT / "run.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
