import codecs
import collections
from metrics import precision_at_n, mrr, recall_at_n
from joblib import Parallel, delayed
import pyltr
import numpy as np
from random import shuffle


class Evaluator(object):

    def __init__(self, implicit=False, threshold=4, all_unrated_items=True):

        self.implicit = implicit

        self.all_unrated_items = all_unrated_items

        self.threshold = threshold

        self.training = None

        self.validation = None

        self.test = None

        self.model = None

        self.metrics = None

    def _parse_data(self, training, test, validation=None):

        """
        Reads the data, generates the set of all items and defines the metrics
        :param training: training set
        :param test: test set
        :param validation: validation set (optional)
        """

        self.training = training

        self.validation = validation

        self.test = test

        self._get_items_liked_by_user()  # defines the dictionary of items liked by each user in the training set

        self._get_all_items()  # define all the items that can be used as candidates for the recommandations

        self._define_metrics()

    def _get_items_liked_by_user(self):

        self.all_train_items = []

        self.items_liked_by_user_dict = collections.defaultdict(list)

        self.items_ratings_by_user_test = {}

        self.items_rated_by_user_train = collections.defaultdict(list)

        with codecs.open(self.training, 'r', encoding='utf-8') as train:

            for line in train:

                line = line.split(' ')

                u = line[0]

                item = line[1]

                relevance = int(line[2])

                self.items_rated_by_user_train[u].append(item)

                if self.implicit is False and relevance >= self.threshold:  # only relevant items are used to compute the similarity

                    self.items_liked_by_user_dict[u].append(item)

                elif self.implicit and relevance == 1:

                    self.items_liked_by_user_dict[u].append(item)

                self.all_train_items.append(item)

        self.all_train_items = list(set(self.all_train_items))  # remove duplicates

    def _get_all_items(self):

        self.all_items = []

        with codecs.open(self.test, 'r', encoding='utf-8') as test:

            test_items = []

            for line in test:

                line = line.split(' ')

                u = line[0]

                item = line[1]

                relevance = int(line[2])

                test_items.append(item)

                self.items_ratings_by_user_test[(u, item)] = relevance

            self.all_items = list(set(self.all_train_items + test_items))  # merge lists and remove duplicates

            del self.all_train_items

        if self.validation:

            self.items_ratings_by_user_val = {}

            with codecs.open(self.validation, 'r', encoding='utf-8') as val:

                val_items = []

                for line in val:
                    line = line.split(' ')

                    u = line[0]

                    item = line[1]

                    relevance = int(line[2])

                    val_items.append(item)

                    self.items_ratings_by_user_val[(u, item)] = relevance

                self.all_items = list(set(self.all_items + val_items))  # merge lists and remove duplicates

    def _define_metrics(self):

        M = len(self.all_items)

        self.metrics = {
            'P@5': precision_at_n.PrecisionAtN(k=5),  # P@5
            'P@10': precision_at_n.PrecisionAtN(k=10),  # P@10
            'MAP': pyltr.metrics.AP(k=M),  # MAP
            'R@5': recall_at_n.RecallAtN(k=5),
            'R@10': recall_at_n.RecallAtN(k=10),
            'NDCG': pyltr.metrics.NDCG(k=M, gain_type='identity'),  # NDCG
            'MRR': mrr.MRR(k=M)  # MRR
        }

    def parse_users_items_rel(self, line):

        line = line.split(' ')

        user = line[0]  # user29

        user_id = int(user.strip('user'))  # 29

        item = line[1]  # http://dbpedia.org/resource/The_Golden_Child

        relevance = int(line[2])  # 5

        # binarization of the relevance values

        if self.implicit is False:

            relevance = 1 if relevance >= self.threshold else 0

        return user, user_id, item, relevance

    def get_candidates(self, user):

        if self.all_unrated_items:

            rated_items_train = self.items_rated_by_user_train[user]

            candidate_items = [item for item in self.all_items if
                               item not in rated_items_train]  # all unrated items in the train

        else:

            candidate_items = self.all_items

        return candidate_items

    def get_relevance(self, user, item, val_set=False):

        if val_set:

            feedback = self.items_ratings_by_user_val

        else:

            feedback = self.items_ratings_by_user_test

        try:
            relevance = int(feedback[(user, item)])  # get the relevance score if it's in the test

            if self.implicit is False:
                relevance = 1 if relevance >= self.threshold else 0

        except KeyError:

            relevance = 0  # unrated items are assumed to be negative

        return relevance

    def features(self, recommender, training, test, validation=None, n_users=False):

        # reads .dat format
        self._parse_data(training, test, validation=validation)

        if self.validation:

            user_item_features = Parallel(n_jobs=3, backend='threading')(delayed(self._compute_features)
                                  (data, recommender, negative_candidates, n_users, val_set)
                                  for data, recommender, negative_candidates, val_set in [(training, recommender, False, False),
                                                                              (test, recommender, True, False),
                                                                              (validation, recommender, True, True)])

            x_train, y_train, qids_train = user_item_features[0]

            x_test, y_test, qids_test = user_item_features[1]

            x_val, y_val, qids_val = user_item_features[2]

        else:

            user_item_features = Parallel(n_jobs=2, backend='threading')(delayed(self._compute_features)
                                  (data, negative_candidates, n_users)
                                  for data, negative_candidates in [(training, False), (test, True)])

            x_train, y_train, qids_train = user_item_features[0]

            x_test, y_test, qids_test = user_item_features[1]

            x_val, y_val, qids_val = None, None, None

        return x_train, y_train, qids_train, x_test, y_test, qids_test, x_val, y_val, qids_val

    def _compute_features(self, data, recommender, negative_candidates=False, n_users=False, val_set=False):

        TX = []
        Ty = []
        Tqids = []

        if n_users:

            users_list = list(self.items_rated_by_user_train.keys())[0:n_users]

            check_user = True

        else:

            users_list = list(self.items_rated_by_user_train.keys())

            check_user = False

        if self.implicit:  # only positive feedback is available, need to generate false candidataes

            negative_candidates = True

        if negative_candidates:  # generate the features also for negative candidates

            for user in users_list:

                print(user)

                user_id = int(user.strip('user'))

                candidate_items = self.get_candidates(user)

                shuffle(candidate_items)  # relevant and non relevant items are shuffled

                for item in candidate_items:

                    items_liked_by_user = self.items_liked_by_user_dict[user]

                    features = recommender.compute_user_item_features(user, item, items_liked_by_user)

                    TX.append(features)

                    relevance = self.get_relevance(user, item, val_set=val_set)

                    Ty.append(relevance)

                    Tqids.append(user_id)

        else:  # only generate features for data in the training set

            with codecs.open(data, 'r', encoding='utf-8') as data_file:

                for line in data_file:

                    user, user_id, item, relevance = self.parse_users_items_rel(line)

                    if check_user:

                        if str(user_id) in users_list:  # only compute features for a sample of users

                            Tqids.append(user_id)

                            items_liked_by_user = self.items_liked_by_user_dict[user]

                            features = recommender.compute_user_item_features(user, item, items_liked_by_user)

                            TX.append(features)

                            Ty.append(relevance)

                    else:

                        Tqids.append(user_id)

                        items_liked_by_user = self.items_liked_by_user_dict[user]

                        features = recommender.compute_user_item_features(user, item, items_liked_by_user)

                        TX.append(features)

                        Ty.append(relevance)

        return np.asarray(TX), np.asarray(Ty), np.asarray(Tqids)

    def evaluate(self, recommender, x_test, y_test, qids_test):

        if recommender.model and self.metrics:

            preds = recommender.predict(x_test)

            for name, metric in self.metrics.items():

                if name != 'fit':
                    print('%s-----%f\n' % (name, metric.calc_mean(qids_test, y_test, preds)))

        else:

            raise ValueError('Fit the model before you evaluate')

    def evaluate_heuristics(self, x_test, y_test, qids_test):

        preds_average = list(map(lambda x: np.mean(x), x_test))  # average of the relatedness scores

        preds_max = list(map(lambda x: np.max(x), x_test))  # max of the relatedness scores

        preds_min = list(map(lambda x: np.min(x), x_test))  # min of the relatedness scores

        print('Average:')

        for name, metric in self.metrics.items():

            if name != 'fit':
                print('%s-----%f\n' % (name, metric.calc_mean(qids_test, y_test, preds_average)))

        print('Min:')

        for name, metric in self.metrics.items():

            if name != 'fit':
                print('%s-----%f\n' % (name, metric.calc_mean(qids_test, y_test, preds_min)))

        print('Max:')

        for name, metric in self.metrics.items():

            if name != 'fit':
                print('%s-----%f\n' % (name, metric.calc_mean(qids_test, y_test, preds_max)))