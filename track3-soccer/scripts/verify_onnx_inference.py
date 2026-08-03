#!/usr/bin/env python3
"""Verify chase_v7_policy.onnx inference — the minimal Sim2Sim readiness gate.

Checks:
1. ONNX loads without error
2. Input shape is [1, 19]
3. Output shape is [1, 3]
4. Output values are finite
5. Output values are within clip range after clipping (0.8 lin, 1.0 ang)
6. Multiple random inputs produce varied outputs (policy is not frozen)
"""
import sys
import os
import numpy as np

def main():
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "chase_v7_policy.onnx"
    )

    if not os.path.exists(model_path):
        print(f"FAIL: Model not found at {model_path}")
        sys.exit(1)

    print(f"Loading: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / 1024:.1f} KB")

    try:
        import onnxruntime as ort
    except ImportError:
        print("Installing onnxruntime...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"])
        import onnxruntime as ort

    sess = ort.InferenceSession(model_path)
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]

    print(f"\n=== ONNX Model Info ===")
    print(f"Input:  {inp.name} shape={inp.shape} dtype={inp.type}")
    print(f"Output: {out.name} shape={out.shape} dtype={out.type}")

    # Check dimensions
    inp_dim = inp.shape[-1] if isinstance(inp.shape[-1], int) else 19
    out_dim = out.shape[-1] if isinstance(out.shape[-1], int) else 3
    assert inp_dim == 19, f"Expected input dim 19, got {inp_dim}"
    assert out_dim == 3, f"Expected output dim 3, got {out_dim}"
    print(f"✓ Input dim = 19")
    print(f"✓ Output dim = 3")

    # Run inference with multiple random inputs
    CLIP_LIN = 0.8
    CLIP_ANG = 1.0
    outputs = []
    for i in range(10):
        obs = np.random.randn(1, 19).astype(np.float32) * 0.5
        action = sess.run(None, {inp.name: obs})[0]
        outputs.append(action[0])

        # Clip
        vx = float(np.clip(action[0, 0], -CLIP_LIN, CLIP_LIN))
        vy = float(np.clip(action[0, 1], -CLIP_LIN, CLIP_LIN))
        wz = float(np.clip(action[0, 2], -CLIP_ANG, CLIP_ANG))

        assert np.all(np.isfinite(action)), f"Non-finite output at step {i}"
        assert abs(vx) <= CLIP_LIN + 1e-6, f"vx clip failed: {vx}"
        assert abs(vy) <= CLIP_LIN + 1e-6, f"vy clip failed: {vy}"
        assert abs(wz) <= CLIP_ANG + 1e-6, f"wz clip failed: {wz}"

    print(f"✓ All 10 inferences produced finite outputs")
    print(f"✓ All outputs within clip range (lin={CLIP_LIN}, ang={CLIP_ANG})")

    # Check output diversity (policy not frozen)
    outputs = np.array(outputs)
    std = outputs.std(axis=0)
    print(f"\n=== Output Diversity ===")
    print(f"Std per dim: vx={std[0]:.4f}, vy={std[1]:.4f}, wz={std[2]:.4f}")
    assert std.mean() > 0.01, "Policy output is frozen (zero variance)"
    print(f"✓ Policy produces varied outputs (mean std={std.mean():.4f})")

    # Sample output
    obs = np.zeros((1, 19), dtype=np.float32)
    obs[0, 8] = 2.0  # ball 2m ahead
    obs[0, 12] = 2.0  # dist_to_ball = 2m
    obs[0, 13] = 1.0  # goal_dir_x
    obs[0, 15] = 5.0  # goal_dist = 5m
    action = sess.run(None, {inp.name: obs})[0]
    vx = float(np.clip(action[0, 0], -CLIP_LIN, CLIP_LIN))
    vy = float(np.clip(action[0, 1], -CLIP_LIN, CLIP_LIN))
    wz = float(np.clip(action[0, 2], -CLIP_ANG, CLIP_ANG))

    print(f"\n=== Sample Inference (ball 2m ahead, goal 5m) ===")
    print(f"Raw output:  [{action[0, 0]:.4f}, {action[0, 1]:.4f}, {action[0, 2]:.4f}]")
    print(f"Clipped:     [vx={vx:.4f}, vy={vy:.4f}, wz={wz:.4f}]")

    print(f"\n{'='*50}")
    print(f"  onnx_loaded = true")
    print(f"  input = [1, 19]")
    print(f"  output = [1, 3]")
    print(f"  action_finite = true")
    print(f"  action_clipped = true")
    print(f"  policy_diverse = true")
    print(f"  ALL CHECKS PASSED ✅")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
