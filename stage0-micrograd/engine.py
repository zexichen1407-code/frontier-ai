"""
Stage 0 — engine.py : a scalar-valued reverse-mode autograd engine (micrograd).

>>> YOU implement this file. AI-OFF. <<<
(Watching the Karpathy lecture / reading 3b1b / googling a derivative = allowed.
 Having an AI write or fix this code for you = the failure mode this whole project exists to kill.)

Spec — match karpathy/micrograd so the Zero-to-Hero lecture maps 1:1:

  Value(data)        wraps a single python float
  .data              the scalar value (float)
  .grad              d(final_output)/d(self); starts at 0.0, filled by backward()

  operators to support (each returns a new Value, and wires up the backward closure):
      +   -   *   /   **(int/float exponent)   unary -      (with reflected __radd__/__rmul__)
  nonlinearities:
      .relu()        .tanh()        .exp()

  .backward()        topological sort of the graph, set output's grad = 1.0,
                     then walk in reverse calling each node's local _backward().

Verify with:  python test_engine.py
Done when:    worst forward Δ and worst gradient Δ vs PyTorch are both < 1e-6.
"""


class Value:
    def __init__(self, data, _children=(), _op=""):
        raise NotImplementedError(
            "Stage 0: implement Value yourself, AI-off. Start with __init__ "
            "(store data, grad=0.0, _backward=lambda:None, _prev=set(_children))."
        )

    # Implement these yourself (delete this comment as you go):
    #   __add__, __radd__, __mul__, __rmul__, __pow__, __neg__, __sub__, __truediv__
    #   relu, tanh, exp
    #   backward
    #   __repr__  (handy for debugging)
