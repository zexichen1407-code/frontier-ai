"""
Stage 0 — train.py : train YOUR MLP on two-moons to > 95% accuracy.

>>> YOU write this, AI-OFF. <<<

Available to you (already provided):
    from engine import Value
    from nn import MLP
    from data import load_two_moons, accuracy, plot_decision_boundary

Steps you implement yourself:
    1. X, y = load_two_moons()                  # X: list[[x1,x2]], y: list[0/1]
    2. model = MLP(2, [16, 16, 1])
    3. training loop (e.g. 100 epochs):
         - forward every point -> a score Value
         - loss = some differentiable loss (hinge or binary cross-entropy) + small L2
         - zero every p.grad in model.parameters()
         - loss.backward()
         - SGD update: p.data -= lr * p.grad   for p in model.parameters()
         - every ~10 epochs, print epoch, loss, train accuracy
    4. at the end: print final accuracy; optionally plot_decision_boundary(...)

Done when:  final accuracy > 0.95   (print it clearly).
"""

raise SystemExit(
    "Stage 0: write train.py yourself (AI-off). Delete this line when you start."
)
