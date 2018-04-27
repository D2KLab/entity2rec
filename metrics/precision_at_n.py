"""
Precision at N
"""

from . import MetricItem


class PrecisionAtN(MetricItem):
    def __init__(self, k=10, cutoff=0.5):
        super(PrecisionAtN, self).__init__()
        self.k = k
        self.cutoff = cutoff

    def evaluate(self, qid, targets, items=None):
        hits = 0.0

        for i in range(self.k):
            if targets[i] >= self.cutoff:  # relevant item
                hits += 1

        return (hits / self.k) if hits > 0 else 0.0
