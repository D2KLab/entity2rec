from __future__ import print_function
import os
import codecs
import collections
import numpy as np
from entity2vec import Entity2Vec
from entity2rel import Entity2Rel
import argparse
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

    def __init__(self, is_directed, preprocessing, is_weighted, p, q, walk_length, num_walks, dimensions, window_size,
                 workers, iterations, config, sparql, dataset, entities, default_graph, implicit, entity_class,
                 feedback_file, all_unrated_items=False):

        Entity2Vec.__init__(self, is_directed, preprocessing, is_weighted, p, q, walk_length, num_walks, dimensions,
                            window_size, workers, iterations, config, sparql, dataset, entities, default_graph,
                            entity_class, feedback_file)

        Entity2Rel.__init__(self)  # binary format embeddings

        self.implicit = implicit

        self.all_unrated_items = all_unrated_items

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

    def _get_embedding_files(self):

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

                line = line.split(' ')

                u = line[0]

                item = line[1]

                relevance = int(line[2])

                self.items_rated_by_user_train[u].append(item)

                self.items_ratings_by_user_test[(u, item)] = relevance  # independently from the rating

                if self.implicit is False and relevance >= 4:  # only relevant items are used to compute the similarity, rel = 5 in a previous work

                    self.items_liked_by_user_dict[u].append(item)

                elif self.implicit and relevance == 1:

                    self.items_liked_by_user_dict[u].append(item)

                self.all_train_items.append(item)

        self.all_train_items = list(set(self.all_train_items))  # remove duplicates

    def _get_all_items(self):

        self.all_items = []

        if self.entities != "all":  # if it has been provided a list of items as an external file, read from it

            del self.all_train_items  # free memory space

            with codecs.open(self.entities, 'r', encoding='utf-8') as items:

                for item in items:
                    item = item.strip('\n')

                    self.all_items.append(item)

        else:  # otherwise join the items from the train, validation and test set

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
            relevance = 1 if relevance >= 4 else 0

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

        # get candidates according to the all items protocol
        # use as candidates all the the items that are not in the training set

        if self.all_unrated_items:
            rated_items_train = self.items_rated_by_user_train[user]  # both in the train and in the test

            candidate_items = [item for item in self.all_items if
                               item not in rated_items_train]  # all unrated items in the train

        else:

            candidate_items = self.all_items

        return candidate_items

    def feature_generator(self, run_all=False):

        if run_all:
            super(Entity2Rec, self).run()  # run entity2vec

        self._get_embedding_files()

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
                            rel = 1 if rel >= 4 else 0

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
                                rel = 1 if rel >= 4 else 0

                        except KeyError:
                            rel = 0  # unrated items are assumed to be negative

                        self.write_line(user, user_id, item, rel, val_write)

            print('finished writing validation')

            print("--- %s seconds ---" % (time.time() - start_time))

    def _compute_features(self, data, test=False):

        TX = []
        Ty = []
        Tqids = []

        if self.implicit:
            test = False

        if test:  # generate the features also for negative candidates

            for user in self.items_rated_by_user_train.keys():

                print(user)

                user_id = int(user.strip('user'))

                candidate_items = self.get_candidates(user)

                shuffle(candidate_items)  # relevant and non relevant items are shuffled

                for item in candidate_items:

                    try:
                        relevance = int(self.items_ratings_by_user_test[
                                            (user, item)])  # get the relevance score if it's in the test

                        if self.implicit is False:
                            relevance = 1 if relevance >= 4 else 0

                    except KeyError:
                        relevance = 0  # unrated items are assumed to be negative

                    collab_score, content_scores = self.compute_scores(user, item)

                    features = [collab_score] + list(content_scores)

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

        if run_all:
            print('Running entity2vec to generate property-specific embeddings...')
            super(Entity2Rec, self).run()  # run entity2vec

        # reads the embedding files
        self._get_embedding_files()

        # reads .dat format
        self._parse_data(training, test, validation=validation)

        if self.validation:

            feat = Parallel(n_jobs=3)(delayed(self._compute_features)
                                      (data, test=is_test)
                                      for data, is_test in [(training, False), (test, True), (validation, True)])

            x_train, y_train, qids_train = feat[0]

            x_test, y_test, qids_test = feat[1]

            x_val, y_val, qids_val = feat[2]

        else:

            feat = Parallel(n_jobs=2)(delayed(self._compute_features)
                                      (data, test=is_test)
                                      for data, is_test in [(training, False), (test, True)])

            x_train, y_train, qids_train = feat[0]

            x_test, y_test, qids_test = feat[1]

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

            for name, metric in self.metrics.items():

                if name != 'fit':
                    print('%s-----%f\n' % (name, metric.calc_mean(qids_test, y_test, preds)))

        else:

            raise ValueError('Fit the model before you evaluate')

    @staticmethod
    def parse_args():

        """
        Parses the entity2rec arguments.
        """

        parser = argparse.ArgumentParser(description="Run entity2rec.")

        parser.add_argument('--walk_length', type=int, default=10,
                            help='Length of walk per source. Default is 10.')

        parser.add_argument('--num_walks', type=int, default=500,
                            help='Number of walks per source. Default is 40.')

        parser.add_argument('--p', type=float, default=1,
                            help='Return hyperparameter. Default is 1.')

        parser.add_argument('--q', type=float, default=1,
                            help='Inout hyperparameter. Default is 1.')

        parser.add_argument('--weighted', dest='weighted', action='store_true',
                            help='Boolean specifying (un)weighted. Default is unweighted.')
        parser.add_argument('--unweighted', dest='unweighted', action='store_false')
        parser.set_defaults(weighted=False)

        parser.add_argument('--directed', dest='directed', action='store_true',
                            help='Graph is (un)directed. Default is directed.')
        parser.set_defaults(directed=False)

        parser.add_argument('--no_preprocessing', dest='preprocessing', action='store_false',
                            help='Whether preprocess all transition probabilities or compute on the fly')
        parser.set_defaults(preprocessing=True)

        parser.add_argument('--dimensions', type=int, default=500,
                            help='Number of dimensions. Default is 128.')

        parser.add_argument('--window-size', type=int, default=10,
                            help='Context size for optimization. Default is 10.')

        parser.add_argument('--iter', default=5, type=int,
                            help='Number of epochs in SGD')

        parser.add_argument('--workers', type=int, default=8,
                            help='Number of parallel workers. Default is 8.')

        parser.add_argument('--config_file', nargs='?', default='config/properties.json',
                            help='Path to configuration file')

        parser.add_argument('--dataset', nargs='?', default='movielens_1m',
                            help='Dataset')

        parser.add_argument('--sparql', dest='sparql',
                            help='Whether downloading the graphs from a sparql endpoint')
        parser.set_defaults(sparql=False)

        parser.add_argument('--entities', dest='entities', default="all",
                            help='A specific list of entities for which the embeddings have to be computed')

        parser.add_argument('--default_graph', dest='default_graph', default=False,
                            help='Default graph to query when using a Sparql endpoint')

        parser.add_argument('--train', dest='train', help='train', default=False)

        parser.add_argument('--test', dest='test', help='test')

        parser.add_argument('--validation', dest='validation', default=False, help='validation')

        parser.add_argument('--run_all', dest='run_all', action='store_true', default=False,
                            help='If computing also the embeddings')

        parser.add_argument('--implicit', dest='implicit', action='store_true', default=False,
                            help='Implicit feedback with boolean values')

        parser.add_argument('--entity_class', dest='entity_class', help='entity class', default=False)

        parser.add_argument('--feedback_file', dest='feedback_file', default=False,
                            help='Path to a DAT file that contains all the couples user-item')

        parser.add_argument('--all_unrated_items', dest='all_unrated_items', action='store_true', default=False,
                            help='Whether removing the rated items of the training set from the candidates')

        parser.add_argument('--write_features', dest='write_features', action='store_true', default=False,
                            help='Writes the features to file')

        parser.add_argument('--metric', dest='metric', default='AP',
                            help='Metric to optimize in the training')

        parser.add_argument('--N', dest='N', type=int, default=10,
                            help='Cutoff to estimate metric')

        return parser.parse_args()


if __name__ == '__main__':

    start_time = time.time()

    print('Starting entity2rec...')

    args = Entity2Rec.parse_args()

    rec = Entity2Rec(args.directed, args.preprocessing, args.weighted, args.p, args.q, args.walk_length, args.num_walks,
                     args.dimensions, args.window_size, args.workers, args.iter, args.config_file, args.sparql,
                     args.dataset,
                     args.entities, args.default_graph, args.implicit, args.entity_class, args.feedback_file,
                     all_unrated_items=args.all_unrated_items)

    if args.write_features:

        rec.feature_generator()  # writes features to file with SVM format

    else:

        x_train, y_train, qids_train, x_test, y_test, qids_test, x_val, y_val, qids_val = rec.features(args.train,
                                                                                                       args.test,
                                                                                                       validation=args.validation)
        t2 = time.time() - start_time
        print('Finished computing features after %s seconds' % t2)
        print('Starting to fit the model...')

        rec.fit(x_train, y_train, qids_train,
                x_val=x_val, y_val=y_val, qids_val=qids_val, optimize=args.metric, N=args.N)  # train the model

        print('Finished fitting the model after %s seconds' % (time.time() - t2 - start_time))

        rec.evaluate(x_test, y_test, qids_test)  # evaluates the model on the test set

    print("--- %s seconds ---" % (time.time() - start_time))
