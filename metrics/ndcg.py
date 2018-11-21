from . import MetricItem
import pyltr.metrics


class NDCG(MetricItem):
    def __init__(self, k=10, gain_type='exp2'):
        super(NDCG, self).__init__()
        self.metric = pyltr.metrics.NDCG(k=k, gain_type=gain_type)

    def evaluate(self, qid, targets, items=None):
        return self.metric.evaluate(qid, targets)
