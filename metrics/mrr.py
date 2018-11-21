"""
Mean Reciprocal Rank
"""

from . import MetricItem


class MRR(MetricItem):
    def __init__(self, k=10, cutoff=0.5):
        super(MRR, self).__init__()
        self.k = k
        self.cutoff = cutoff

    def evaluate(self, qid, targets, items=None):
        n_targets = len(targets)
        mrr = 0.
        for i in range(n_targets):
            if targets[i] >= self.cutoff:  # first relevant item
                    pos = i + 1  # start counting from 1 to N
                    mrr = 1. / pos
                    break
        return mrr if mrr > 0 else 0.0
