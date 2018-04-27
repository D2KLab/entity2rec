from . import MetricItem


class Serendipity(MetricItem):
    def __init__(self, sorted_items, k=10, cutoff=0.5):
        super(Serendipity, self).__init__()
        self.top_items = sorted_items[:k]
        self.k = k
        self.cutoff = cutoff

    def evaluate(self, qid, targets, items=None):
        if items is None:
            return 0.0

        hits = 0.0

        for i in range(self.k):
            if targets[i] >= self.cutoff and items[i] not in self.top_items:  # relevant item
                hits += 1

        return (hits / self.k) if hits > 0 else 0.0
