import numpy as np

from pyltr.metrics import Metric
from pyltr.metrics._metrics import check_qids, get_groups
from pyltr.util.sort import get_sorted_y_positions


def get_sorted_y(y, y_pred, items, check=True):
    positions = get_sorted_y_positions(y, y_pred, check=check)
    return y[positions], items[positions]


class MetricItem(Metric):

    def evaluate(self, qid, targets, items=None):
        raise NotImplementedError()

    def calc_mean(self, qids, targets, preds, items=None):
        if items is None:
            return Metric.calc_mean(self, qids, targets, preds)
        check_qids(qids)
        query_groups = get_groups(qids)
        return np.mean([self.evaluate_preds(qid, targets[a:b], preds[a:b], items=items[a:b])
                        for qid, a, b in query_groups])

    def calc_mean_var(self, qids, targets, preds, items=None):
        if items is None:
            return Metric.calc_mean(self, qids, targets, preds)
        check_qids(qids)
        query_groups = get_groups(qids)
        score_list = [self.evaluate_preds(qid, targets[a:b], preds[a:b], items=items[a:b])
                        for qid, a, b in query_groups]
        return np.var(score_list)/len(score_list)

    def evaluate_preds(self, qid, targets, preds, items=None):
        if items is None:
            return Metric.evaluate_preds(self, qid, targets, preds)
        sorted_y = get_sorted_y(targets, preds, items)
        return self.evaluate(qid, sorted_y[0], sorted_y[1])
