"""
Recall at N
"""

from . import MetricItem


class RecallAtN(MetricItem):
    def __init__(self, k=10, cutoff=0.5):
        super(RecallAtN, self).__init__()
        self.k = k
        self.cutoff = cutoff

    def evaluate(self, qid, targets, items=None):
        n_targets = len(targets)
        num_rel = 0
        total_retrieved = 0.0
        for i in range(n_targets):
            if targets[i] >= self.cutoff:  # relevant item
                num_rel += 1
                if i < self.k:  # relevant item in the first k positions
                    total_retrieved += 1
        return (total_retrieved / num_rel) if num_rel > 0 else 0.0
