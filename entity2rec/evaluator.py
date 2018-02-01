import codecs
import collections
from metrics import precision_at_n, mrr, recall_at_n
from joblib import Parallel, delayed
import pyltr
import numpy as np
from random import shuffle
from sklearn import preprocessing


def parse_line(line):

    line = line.split(' ')

    u = line[0]

    item = line[1]

    relevance = int(line[2])

    return u, item, relevance


class Evaluator(object):

    def __init__(self, implicit=False, threshold=4, all_unrated_items=True):

        """
        Evaluates a recommender system using ranking metrics
        :param implicit: whether it is binary feedback or has to converted
        :param threshold: threshold to convert rating in binary feedback
        :param all_unrated_items: whether using the allunrated items eval protocol
        """

        self.implicit = implicit

        self.all_unrated_items = all_unrated_items  # evalua

        self.threshold = threshold  # threshold to convert ratings into positive implicit feedback

        self.model = None  # model object to train

        self.metrics = {}  # defines the metrics to be evaluated

        self.feedback = {}  # save users feedback in a dictionary for train, val and test

    def _parse_data(self, training, test, validation=None):

        """
        Reads the data, generates the set of all items and defines the metrics
        :param training: training set
        :param test: test set
        :param validation: validation set (optional)
        """

        self.all_items = []

        self.items_liked_by_user_dict = collections.defaultdict(list)

        self.items_rated_by_user_train = collections.defaultdict(list)

        with codecs.open(training, 'r', encoding='utf-8') as train:

            all_train_items = []

            for line in train:

                u, item, relevance = parse_line(line)

                self.items_rated_by_user_train[u].append(item)

                self.feedback[(u, item, 'train')] = relevance

                if self.implicit is False and relevance >= self.threshold:  # only relevant items are used to compute the similarity

                    self.items_liked_by_user_dict[u].append(item)

                elif self.implicit and relevance == 1:

                    self.items_liked_by_user_dict[u].append(item)

                all_train_items.append(item)

        with codecs.open(test, 'r', encoding='utf-8') as test:

            test_items = []

            for line in test:

                u, item, relevance = parse_line(line)

                test_items.append(item)

                self.feedback[(u, item, 'test')] = relevance

            self.all_items = list(set(all_train_items + test_items))  # merge lists and remove duplicates

        if validation:

            self.items_ratings_by_user_val = {}

            with codecs.open(validation, 'r', encoding='utf-8') as val:

                val_items = []

                for line in val:

                    u, item, relevance = parse_line(line)

                    val_items.append(item)

                    self.feedback[(u, item, 'val')] = relevance

                self.all_items = list(set(self.all_items + val_items))  # merge lists and remove duplicates

        self._define_metrics()

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

    def get_candidates(self, user, data, num_negative_candidates=100):

        rated_items_train = self.items_rated_by_user_train[user]

        unrated_items = [item for item in self.all_items if
                               item not in rated_items_train]

        if self.all_unrated_items and data != 'train':

            candidate_items = unrated_items  # all unrated items in the train

        else:  # for training set features, need to include training items

            if self.implicit:  # sample negative items randomly

                negative_candidates = list(np.random.choice(unrated_items, num_negative_candidates, replace=False))

                candidate_items = negative_candidates + rated_items_train

            else:  # use positive and negative feedback from the training set

                items_rated_by_user = self.items_rated_by_user_train[user]

                candidate_items = items_rated_by_user

        return candidate_items

    def get_relevance(self, user, item, data):

        try:

            relevance = int(self.feedback[(user, item, data)])  # get the relevance score if it's in the data

            if self.implicit is False:

                relevance = 1 if relevance >= self.threshold else 0

        except KeyError:

            relevance = 0  # unrated items are assumed to be negative

        return relevance

    def features(self, recommender, training, test, validation=None, n_users=False, n_jobs=4, supervised=True):

        if n_users:
            assert (n_users >= n_jobs), "Number of users cannot be lower than number of workers"

        # reads .dat format
        self._parse_data(training, test, validation=validation)

        if n_users:  # select a sub-sample of users

            users_list = list(self.items_rated_by_user_train.keys())[0:n_users]

        else:  # select all users

            users_list = list(self.items_rated_by_user_train.keys())

        def chunkify(lst, n):

            return [lst[i::n] for i in range(n)]

        users_list_chunks = chunkify(users_list, n_jobs)

        if validation:

            print('Compute features for testing')
            x_test, y_test, qids_test = self._compute_features_parallel('test', recommender, users_list_chunks, n_jobs)

            x_test = preprocessing.scale(x_test)

            if supervised:

                print('Compute features for training')
                x_train, y_train, qids_train = self._compute_features_parallel('train', recommender, users_list_chunks, n_jobs)

                x_train = preprocessing.scale(x_train)

                print('Compute features for validation')
                x_val, y_val, qids_val = self._compute_features_parallel('val', recommender, users_list_chunks, n_jobs)

                x_val = preprocessing.scale(x_val)

            else:

                x_train, y_train, qids_train = None, None, None

                x_val, y_val, qids_val = None, None, None

        else:

            if supervised:

                print('Compute features for training')
                x_train, y_train, qids_train = self._compute_features_parallel('train', recommender, users_list_chunks, n_jobs)

                x_train = preprocessing.scale(x_train)

            else:

                x_train, y_train, qids_train = None, None, None

            print('Compute features for testing')
            x_test, y_test, qids_test = self._compute_features_parallel('test', recommender, users_list_chunks, n_jobs)

            x_test = preprocessing.scale(x_test)

            x_val, y_val, qids_val = None, None, None

        return x_train, y_train, qids_train, x_test, y_test, qids_test, x_val, y_val, qids_val

    def _compute_features_parallel(self, data, recommender, users_list_chunks, n_jobs):

        user_item_features = Parallel(n_jobs=n_jobs)(delayed(self._compute_features)
                                                                          (data, recommender, users_list)
                                                                          for users_list in users_list_chunks)

        x_chunks = [user_item_features[i][0] for i in range(n_jobs)]

        y_chunks = [user_item_features[i][1] for i in range(n_jobs)]

        qids_chunks = [user_item_features[i][2] for i in range(n_jobs)]

        x = np.concatenate(x_chunks, axis=0)

        y = np.concatenate(y_chunks, axis=0)

        qids = np.concatenate(qids_chunks, axis=0)

        return x, y, qids

    def _compute_features(self, data, recommender, users_list):

        TX = []
        Ty = []
        Tqids = []

        for user in users_list:

            print(user)

            user_id = int(user.strip('user'))

            candidate_items = self.get_candidates(user, data)

            shuffle(candidate_items)  # relevant and non relevant items are shuffled

            for item in candidate_items:

                items_liked_by_user = self.items_liked_by_user_dict[user]

                features = recommender.compute_user_item_features(user, item, items_liked_by_user)

                TX.append(features)

                relevance = self.get_relevance(user, item, data)

                Ty.append(relevance)

                Tqids.append(user_id)

        return np.asarray(TX), np.asarray(Ty), np.asarray(Tqids)

    def evaluate(self, recommender, x_test, y_test, qids_test, verbose=True):

        scores = []

        if self.metrics:

            preds = recommender.predict(x_test)

            for name, metric in self.metrics.items():

                if name != 'fit':

                    score = metric.calc_mean(qids_test, y_test, preds)

                    scores.append((name, score))

                    if verbose:

                        print('%s-----%f\n' % (name, score))

        return scores

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
