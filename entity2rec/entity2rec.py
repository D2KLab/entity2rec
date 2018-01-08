from __future__ import print_function
import os
import codecs
import collections
import numpy as np
from entity2vec import Entity2Vec
from entity2rel import Entity2Rel
import time
from random import shuffle
import pyltr
import sys
sys.path.append('.')
from metrics import precision_at_n, mrr, recall_at_n
from joblib import Parallel, delayed


class Entity2Rec(Entity2Vec, Entity2Rel):

    """Computes a set of relatedness scores between user-item pairs from a set of property-specific Knowledge Graph
    embeddings and user feedback and feeds them into a learning to rank algorithm"""

    def __init__(self, dataset, is_directed=False, preprocessing=True, is_weighted=False,
                 p=1, q=4, walk_length=10,
                 num_walks=500, dimensions=500, window_size=10,
                 workers=8, iterations=5, config='config/properties.json',
                 implicit=False, feedback_file=False, all_unrated_items=True, threshold=4):

        Entity2Vec.__init__(self, is_directed, preprocessing, is_weighted, p, q, walk_length, num_walks, dimensions,
                            window_size, workers, iterations, config, dataset, feedback_file)

        Entity2Rel.__init__(self)  # binary format embeddings

        self.implicit = implicit

        self.all_unrated_items = all_unrated_items

        self.threshold = threshold

        self.define_properties()

        # initializing object variables that will be assigned later

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

    def _set_embedding_files(self):

        """
        Sets the list of embedding files
        """

        for prop in self.properties:
            prop_short = prop
            if '/' in prop:
                prop_short = prop.split('/')[-1]

            self.add_embedding(u'emb/%s/%s/num%s_p%d_q%d_l%s_d%s_iter%d_winsize%d.emd' % (
                self.dataset, prop_short, self.num_walks, int(self.p), int(self.q), self.walk_length, self.dimensions,
                self.iter,
                self.window_size))

    def _get_items_liked_by_user(self):

        self.all_train_items = []

        self.items_liked_by_user_dict = collections.defaultdict(list)

        self.items_ratings_by_user_test = {}

        self.items_rated_by_user_train = collections.defaultdict(list)

        with codecs.open(self.training, 'r', encoding='utf-8') as train:

            for line in train:

                print(line)

                line = line.split(' ')

                u = line[0]

                item = line[1]

                relevance = int(line[2])

                self.items_rated_by_user_train[u].append(item)

                self.items_ratings_by_user_test[(u, item)] = relevance  # independently from the rating

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

    def collab_similarity(self, user, item):

        # feedback property

        return self.relatedness_score_by_position(user, item, -1)

    def content_similarities(self, user, item):

        # all other properties

        items_liked_by_user = self.items_liked_by_user_dict[user]

        sims = []

        for past_item in items_liked_by_user:
            sims.append(self.relatedness_scores(past_item, item,
                                                -1))  # append a list of property-specific scores, skip feedback

        if len(sims) == 0:
            sims = 0.5 * np.ones(len(self.properties) - 1)
            return sims

        return np.mean(sims, axis=0)  # return a list of averages of property-specific scores

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

    def compute_scores(self, user, item):

        collab_score = self.collab_similarity(user, item)

        content_scores = self.content_similarities(user, item)

        return collab_score, content_scores

    def write_line(self, user, user_id, item, relevance, file):

        file.write('%d qid:%d' % (relevance, user_id))

        count = 1

        collab_score, content_scores = self.compute_scores(user, item)

        file.write(' %d:%f' % (count, collab_score))

        count += 1

        l = len(content_scores)

        for content_score in content_scores:

            if count == l + 1:  # last score, end of line

                file.write(' %d:%f # %s\n' % (count, content_score, item))

            else:

                file.write(' %d:%f' % (count, content_score))

                count += 1

    def get_candidates(self, user):

        if self.all_unrated_items:

            rated_items_train = self.items_rated_by_user_train[user]

            candidate_items = [item for item in self.all_items if
                               item not in rated_items_train]  # all unrated items in the train

        else:

            candidate_items = self.all_items

        return candidate_items

    def feature_generator(self, run_all=False):

        # run entity2vec to create the embeddings
        if run_all:
            print('Running entity2vec to generate property-specific embeddings...')
            self.e2v_walks_learn()  # run entity2vec

        # reads the embedding files
        self._set_embedding_files()

        # write training set

        start_time = time.time()

        train_name = (self.training.split('/')[-1]).split('.')[0]

        feature_path = 'features/%s/p%d_q%d/' % (self.dataset, int(self.p), int(self.q))

        try:

            os.makedirs(feature_path)

        except:

            pass

        feature_file = feature_path + '%s_p%d_q%d.svm' % (train_name, int(self.p), int(self.q))

        with codecs.open(feature_file, 'w', encoding='utf-8') as train_write:

            with codecs.open(self.training, 'r', encoding='utf-8') as training:
                for i, line in enumerate(training):
                    user, user_id, item, relevance = self.parse_users_items_rel(line)

                    print(user)

                    self.write_line(user, user_id, item, relevance, train_write)

        print('finished writing training')

        print("--- %s seconds ---" % (time.time() - start_time))

        # write test set

        test_name = (self.test.split('/')[-1]).split('.')[0]

        feature_file = feature_path + '%s_p%d_q%d.svm' % (test_name, int(self.p), int(self.q))

        with codecs.open(feature_file, 'w', encoding='utf-8') as test_write:

            for user in self.items_rated_by_user_train.keys():

                # write some candidate items

                print(user)

                user_id = int(user.strip('user'))

                candidate_items = self.get_candidates(user)

                shuffle(candidate_items)  # relevant and non relevant items are shuffled

                for item in candidate_items:

                    try:
                        rel = int(self.items_ratings_by_user_test[
                                      (user, item)])  # get the relevance score if it's in the test

                        if self.implicit is False:
                            rel = 1 if rel >= self.threshold else 0

                    except KeyError:
                        rel = 0  # unrated items are assumed to be negative

                    self.write_line(user, user_id, item, rel, test_write)

        print('finished writing test')

        print("--- %s seconds ---" % (time.time() - start_time))

        if self.validation:  # write validation set

            val_name = (self.validation.split('/')[-1]).split('.')[0]

            feature_file = feature_path + '%s_p%d_q%d.svm' % (val_name, int(self.p), int(self.q))

            with codecs.open(feature_file, 'w', encoding='utf-8') as val_write:

                for user in self.items_rated_by_user_train.keys():

                    # write some candidate items

                    print(user)

                    user_id = int(user.strip('user'))

                    candidate_items = self.get_candidates(user)

                    shuffle(candidate_items)  # relevant and non relevant items are shuffled

                    for item in candidate_items:

                        try:
                            rel = int(self.items_ratings_by_user_test[
                                          (user, item)])  # get the relevance score if it's in the test

                            if self.implicit is False:
                                rel = 1 if rel >= self.threshold else 0

                        except KeyError:
                            rel = 0  # unrated items are assumed to be negative

                        self.write_line(user, user_id, item, rel, val_write)

            print('finished writing validation')

            print("--- %s seconds ---" % (time.time() - start_time))

    def read_features(self):

        """
        Reads features from .SVM files
        """

        with open('features/%s/p%s_q%s/train_p%s_q%s.svm' % (self.dataset, int(self.p), int(self.q), int(self.p),
                                                             int(self.q))) as trainfile:

            x_train, y_train, qids_train, _ = pyltr.data.letor.read_dataset(trainfile)

        with open('features/%s/p%s_q%s/test_p%s_q%s.svm' % (self.dataset,int(self.p), int(self.q), int(self.p),
                                                            int(self.q))) as testfile:

            x_test, y_test, qids_test, _ = pyltr.data.letor.read_dataset(testfile)

        if self.validation:

            with open('features/%s/p%s_q%s/val_p%s_q%s.svm' % (self.dataset, int(self.p), int(self.q), int(self.p),
                                                               int(self.q))) as valfile:

                x_val, y_val, qids_val, _ = pyltr.data.letor.read_dataset(valfile)

        else:

            x_val, y_val, qids_val = None, None, None

        return x_train, y_train, qids_train, x_test, y_test, qids_test, x_val, y_val, qids_val

    def _compute_user_item_features(self, user, item):

        try:
            relevance = int(self.items_ratings_by_user_test[
                                (user, item)])  # get the relevance score if it's in the test

            if self.implicit is False:
                relevance = 1 if relevance >= self.threshold else 0

        except KeyError:
            relevance = 0  # unrated items are assumed to be negative

        collab_score, content_scores = self.compute_scores(user, item)

        features = [collab_score] + list(content_scores)

        return features, relevance

    def _compute_features(self, data, test=False):

        TX = []
        Ty = []
        Tqids = []

        if self.implicit:  # only positive feedback is available, need to generate false candidataes
            test = True

        if test:  # generate the features also for negative candidates

            for user in self.items_rated_by_user_train.keys():

                print(user)

                user_id = int(user.strip('user'))

                candidate_items = self.get_candidates(user)

                shuffle(candidate_items)  # relevant and non relevant items are shuffled

                for item in candidate_items:

                    features, relevance = self._compute_user_item_features(user, item)

                    TX.append(features)

                    Ty.append(relevance)

                    Tqids.append(user_id)

        else:  # only generate features for data in the training set

            with codecs.open(data, 'r', encoding='utf-8') as data_file:

                for line in data_file:

                    user, user_id, item, relevance = self.parse_users_items_rel(line)

                    Tqids.append(user_id)

                    collab_score, content_scores = self.compute_scores(user, item)

                    features = [collab_score] + list(content_scores)

                    TX.append(features)

                    Ty.append(relevance)

        return np.asarray(TX), np.asarray(Ty), np.asarray(Tqids)

    def features(self, training, test, validation=None, run_all=False):

        # reads .dat format
        self._parse_data(training, test, validation=validation)

        # run entity2vec to create the embeddings
        if run_all:
            print('Running entity2vec to generate property-specific embeddings...')
            self.e2v_walks_learn()  # run entity2vec

        # reads the embedding files
        self._set_embedding_files()

        if self.validation:

            user_item_features = Parallel(n_jobs=3, backend='threading')(delayed(self._compute_features)
                                  (data, test)
                                  for data, test in [(training, False), (test, True), (validation, True)])

            x_train, y_train, qids_train = user_item_features[0]

            x_test, y_test, qids_test = user_item_features[1]

            x_val, y_val, qids_val = user_item_features[2]

        else:

            user_item_features = Parallel(n_jobs=2, backend='threading')(delayed(self._compute_features)
                                  (data, test)
                                  for data, test in [(training, False), (test, True)])

            x_train, y_train, qids_train = user_item_features[0]

            x_test, y_test, qids_test = user_item_features[1]

            x_val, y_val, qids_val = None, None, None

        return x_train, y_train, qids_train, x_test, y_test, qids_test, x_val, y_val, qids_val

    def fit(self, x_train, y_train, qids_train, x_val=None, y_val=None, qids_val=None, optimize='AP', N=10):

        # choose the metric to optimize during the fit process

        if optimize == 'NDCG':

            self.metrics['fit'] = pyltr.metrics.NDCG(k=N)

        elif optimize == 'P':

            self.metrics['fit'] = precision_at_n.PrecisionAtN(k=N)

        elif optimize == 'MRR':

            self.metrics['fit'] = mrr.MRR(k=N)

        elif optimize == 'AP':

            self.metrics['fit'] = pyltr.metrics.AP(k=N)

        else:

            raise ValueError('Metric not implemented')

        self.model = pyltr.models.LambdaMART(
            metric=self.metrics['fit'],
            n_estimators=1000,
            learning_rate=0.02,
            max_features=0.5,
            query_subsample=0.5,
            max_leaf_nodes=10,
            min_samples_leaf=64,
            verbose=1
        )

        # Only needed if you want to perform validation (early stopping & trimming)

        if self.validation:

            monitor = pyltr.models.monitors.ValidationMonitor(
                x_val, y_val, qids_val, metric=self.metrics['fit'], stop_after=250)

            self.model.fit(x_train, y_train, qids_train, monitor=monitor)

        else:

            self.model.fit(x_train, y_train, qids_train)

    def evaluate(self, x_test, y_test, qids_test):

        if self.model and self.metrics:

            preds = self.model.predict(x_test)

            print(len(qids_test))

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
