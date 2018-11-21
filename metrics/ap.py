from . import MetricItem
import pyltr.metrics


class AP(MetricItem):
    def __init__(self, k=10, cutoff=0.5):
        super(AP, self).__init__()
        self.metric = pyltr.metrics.AP(k=k, cutoff=cutoff)

    def evaluate(self, qid, targets, items=None):
        return self.metric.evaluate(qid, targets)
