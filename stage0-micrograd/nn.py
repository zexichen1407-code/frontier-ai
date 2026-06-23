"""
Stage 0 — nn.py : a tiny neural-net library built on YOUR engine.Value.

>>> YOU implement this, AFTER engine.py passes test_engine.py. AI-OFF. <<<

Spec — matches the second half of the Karpathy micrograd lecture:

  Neuron(nin)        nin weight Values + 1 bias Value (init random in [-1,1])
                     __call__(x: list[Value|float]) -> Value      # act(w·x + b)
                     parameters() -> list[Value]

  Layer(nin, nout)   nout Neurons
                     __call__(x) -> list[Value]  (or a single Value when nout == 1)
                     parameters() -> list[Value]

  MLP(nin, nouts)    layers of sizes zip([nin]+nouts, nouts)
                     __call__(x) -> Value | list[Value]
                     parameters() -> list[Value]

You will use MLP in train.py to fit two-moons.
"""


class Neuron:
    def __init__(self, nin):
        raise NotImplementedError("Stage 0: implement Neuron yourself (AI-off).")


class Layer:
    def __init__(self, nin, nout):
        raise NotImplementedError("Stage 0: implement Layer yourself (AI-off).")


class MLP:
    def __init__(self, nin, nouts):
        raise NotImplementedError("Stage 0: implement MLP yourself (AI-off).")
