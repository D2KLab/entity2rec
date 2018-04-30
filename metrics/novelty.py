from . import MetricItem
import math


class Novelty(MetricItem):
    def __init__(self, items_rated, k=10):
        super(Novelty, self).__init__()
        self.k = k

        self.num_ratings = 0
        self.items_count = {}

        for items_list in items_rated.values():
            for item in items_list:
                self.num_ratings += 1
                if item in self.items_count:
                    self.items_count[item] += 1
                else:
                    self.items_count[item] = 1

    def evaluate(self, qid, targets, items=None):
        novelty = 0.0

        for i in range(self.k):
            try:
                seen = self.items_count[items[i]] / self.num_ratings
                novelty -= math.log2(seen)
            except KeyError:
                # log(0) = 0
                pass

        return novelty / self.k
