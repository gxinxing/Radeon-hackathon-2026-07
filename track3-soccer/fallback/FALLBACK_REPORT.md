# Genesis Franka Fallback Report

## Status: PASSED

Remote AMD GPU run completed with `backend=gpu`, 5 physics steps, and 2 entities (plane + Franka).

## Reproduce

```bash
cd /workspace/amd-physical-ai-soccer
/opt/venv/bin/python fallback_genesis_franka.py
```

Machine-readable evidence is in `artifacts/run.json`. The remote run also produced `fallback_artifacts/status_card.png`.

## Limitations

This validates the Genesis/AMD GPU execution path only. It does not validate soccer, 3v3 balance, kicking, scoring, or RL training, and is not Track 3 acceptance evidence.
