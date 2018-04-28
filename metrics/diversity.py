from . import MetricItem
import itertools
import numpy as np
import scipy.spatial.distance as distance


class Diversity(MetricItem):
    def __init__(self, items_rated_by_user, k=10):
        super(Diversity, self).__init__()
        self.metric = CosineSimilarity(items_rated_by_user)
        self.k = k

    def evaluate(self, qid, targets, items=None):
        diversity = 0.0

        for items in itertools.combinations(items[:self.k], 2):
            diversity += (1 - self.metric.similarity(items[0], items[1]))

        return diversity / (self.k * (self.k - 1) * 0.5)


class CosineSimilarity:

    def __init__(self, items_rated_by_user):
        # Create items index and users index
        self.items_index = {}
        self.users_index = {}
        current_item_id = 0
        current_user_id = 0

        for user, items_list in items_rated_by_user.items():
            self.users_index[user] = current_user_id
            current_user_id += 1
            for item in items_list:
                if item not in self.items_index:
                    self.items_index[item] = current_item_id
                    current_item_id += 1

        # Create an item user matrix
        self.matrix = np.full((current_item_id, current_user_id), 0)

        for user, items_list in items_rated_by_user.items():
            user_index = self.users_index[user]
            for item in items_list:
                item_index = self.items_index[item]
                self.matrix[item_index, user_index] += 1

    def similarity(self, item_i, item_j):
        try:
            index_i = self.items_index[item_i]
            index_j = self.items_index[item_j]
        except KeyError:
            return 0
        array_i = self.matrix[index_i]
        array_j = self.matrix[index_j]
        if array_i.sum() == 0 or array_j.sum() == 0:
            return 0
        else:
            return 1 - distance.cosine(array_i, array_j)
