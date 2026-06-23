"""
Stage 0 VERIFICATION HARNESS  (provided — do NOT edit this to make it pass).

Checks YOUR engine.Value against PyTorch autograd on many random expressions.
PASS condition: worst forward Δ and worst gradient Δ < 1e-6 over all trials.

Run:  python test_engine.py
"""
import sys
import random

try:
    import torch
except ImportError:
    print("PyTorch not installed. Run:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(2)

try:
    from engine import Value
except Exception as e:  # noqa: BLE001
    print("Could not import Value from engine.py:", repr(e))
    sys.exit(1)

torch.set_default_dtype(torch.double)

N_TRIALS = 200
TOL = 1e-6


def expr(a, b, c, d):
    """One expression, evaluated identically for engine.Value and torch.Tensor.

    Uses only method-style ops that BOTH types implement, and is shaped to keep
    every gradient non-trivial (no saturating final squash)."""
    e = (a * b + c).tanh()           # *, +, tanh
    f = (a / (b * b + 1.0)).relu()   # /, **(via b*b), +scalar, relu
    g = (e * 0.5 + d * 0.3).exp()    # *scalar, +, exp on a bounded argument
    h = e * f - c + g                # mix
    return h + (a ** 3) * 0.1 + (-b)  # **int, *scalar, unary neg


def run_trial(seed):
    rng = random.Random(seed)
    raw = [rng.uniform(-2.0, 2.0) for _ in range(4)]

    vs = [Value(x) for x in raw]
    out_v = expr(*vs)
    out_v.backward()

    ts = [torch.tensor(x, requires_grad=True) for x in raw]
    out_t = expr(*ts)
    out_t.backward()

    fwd_diff = abs(out_v.data - out_t.item())
    grad_diff = max(abs(v.grad - t.grad.item()) for v, t in zip(vs, ts))
    return fwd_diff, grad_diff


def main():
    worst_fwd = worst_grad = 0.0
    try:
        for s in range(N_TRIALS):
            fd, gd = run_trial(s)
            worst_fwd = max(worst_fwd, fd)
            worst_grad = max(worst_grad, gd)
    except NotImplementedError as e:
        print("engine.py not implemented yet:", e)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001
        print("Your engine raised an error during the check:", repr(e))
        print("(Fix it, then re-run. This is the debugging that builds the skill.)")
        sys.exit(1)

    print(f"trials           : {N_TRIALS}")
    print(f"worst forward Δ  : {worst_fwd:.2e}")
    print(f"worst gradient Δ : {worst_grad:.2e}")
    print(f"tolerance        : {TOL:.0e}")
    if worst_fwd < TOL and worst_grad < TOL:
        print("\nPASS  Stage 0 engine verified — your gradients match PyTorch.")
        sys.exit(0)
    print("\nFAIL  Not within tolerance yet. Keep debugging backward().")
    sys.exit(1)


if __name__ == "__main__":
    main()
