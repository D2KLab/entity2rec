import random
import codecs
import collections
import sys
sys.path.append('.')
import metrics
from joblib import Parallel, delayed
import pyltr
import numpy as np
from random import shuffle
from collections import Counter, defaultdict
import os
import pickle

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

        self.all_items = None  # all items

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

        self.users_liking_an_item_dict = collections.defaultdict(list)

        self.pop_items = Counter()

        with codecs.open(training, 'r', encoding='utf-8') as train:

            all_train_items = []

            for line in train:

                u, item, relevance = parse_line(line)

                self.items_rated_by_user_train[u].append(item)

                self.feedback[(u, item, 'train')] = relevance

                if self.implicit is False and relevance >= self.threshold:  # only relevant items are used to compute the similarity

                    self.items_liked_by_user_dict[u].append(item)

                    self.pop_items[item] += 1

                    self.users_liking_an_item_dict[item].append(u)

                elif self.implicit and relevance == 1:

                    self.items_liked_by_user_dict[u].append(item)

                    self.pop_items[item] += 1

                    self.users_liking_an_item_dict[item].append(u)

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

        self.top_N_items = [item for item, count in self.pop_items.most_common(100)]

    def _define_metrics(self, M):

        self.metrics = {
            'P@5': metrics.PrecisionAtN(k=5),  # P@5
            'P@10': metrics.PrecisionAtN(k=10),  # P@10
            'MAP': metrics.AP(k=M),  # MAP
            'R@5': metrics.RecallAtN(k=5),
            'R@10': metrics.RecallAtN(k=10),
            'NDCG': metrics.NDCG(k=M, gain_type='identity'),  # NDCG
            'MRR': metrics.MRR(k=M),  # MRR
            'SER@5': metrics.Serendipity(self.top_N_items, k=5),  # Serendipity@5
            'SER@10': metrics.Serendipity(self.top_N_items, k=10),  # Serendipity@10
            'NOV@5': metrics.Novelty(self.items_rated_by_user_train, k=5),  # Novelty@5
            'NOV@10': metrics.Novelty(self.items_rated_by_user_train, k=10),  # Novelty@10
            'DIV@5': metrics.Diversity(self.items_liked_by_user_dict, k=5),  # Diversity@5
            'DIV@10': metrics.Diversity(self.items_liked_by_user_dict, k=10)  # Diversity@10
        }

    def get_candidates(self, user, data, num_negative_candidates=100):

        random.seed(1)

        rated_items_train = self.items_rated_by_user_train[user]

        unrated_items = [item for item in self.all_items if
                               item not in rated_items_train]

        unrated_items = sorted(unrated_items)

        if self.all_unrated_items:

            candidate_items = unrated_items

        else:

            candidate_items = self.all_items

        # candidate items for the training set
        if data == 'train':

            if self.implicit:  # sample negative items randomly

                negative_candidates = list(random.sample(candidate_items, num_negative_candidates))

                candidate_items = negative_candidates + rated_items_train

            else:  # use positive and negative feedback from the training set

                items_rated_by_user = self.items_rated_by_user_train[user]

                candidate_items = items_rated_by_user


        shuffle(candidate_items)  # relevant and non relevant items are shuffled

        candidate_items = sorted(candidate_items)  # sorting to ensure reproducibility

        return candidate_items

    def get_relevance(self, user, item, data):

        try:

            relevance = int(self.feedback[(user, item, data)])  # get the relevance score if it's in the data

            if self.implicit is False:

                relevance = 1 if relevance >= self.threshold else 0

        except KeyError:

            relevance = 0  # unrated items are assumed to be negative

        return relevance

    def features(self, recommender, training, test, validation=None, n_users=False, n_jobs=4,
                 supervised=True, max_n_feedback=False):

        # reads .dat format
        self._parse_data(training, test, validation=validation)

        users_list = self._define_user_list(n_users, max_n_feedback, n_jobs)

        def chunkify(lst, n):

            return [lst[i::n] for i in range(n)]

        users_list_chunks = chunkify(users_list, n_jobs)

        if validation:

            print('Compute features for testing')
            x_test, y_test, qids_test, items_test = self._compute_features_parallel('test', recommender,
                                                                                    users_list_chunks, n_jobs, users_list)

            if supervised:

                print('Compute features for training')
                x_train, y_train, qids_train, items_train = self._compute_features_parallel('train', recommender,
                                                                                            users_list_chunks, n_jobs, users_list)

                print('Compute features for validation')
                x_val, y_val, qids_val, items_val = self._compute_features_parallel('val', recommender,
                                                                                    users_list_chunks, n_jobs, users_list)

            else:

                x_train, y_train, qids_train, items_train = None, None, None, None

                x_val, y_val, qids_val, items_val = None, None, None, None

        else:

            if supervised:

                print('Compute features for training')
                x_train, y_train, qids_train, items_train = self._compute_features_parallel('train', recommender,
                                                                                            users_list_chunks, n_jobs, users_list)

            else:

                x_train, y_train, qids_train, items_train = None, None, None, None

            print('Compute features for testing')

            x_test, y_test, qids_test, items_test = self._compute_features_parallel('test', recommender,
                                                                                    users_list_chunks, n_jobs, users_list)

            x_val, y_val, qids_val, items_val = None, None, None, None

        return x_train, y_train, qids_train, items_train,\
               x_test, y_test, qids_test, items_test,\
               x_val, y_val, qids_val, items_val

    def _compute_features_parallel(self, data, recommender, users_list_chunks, n_jobs, users_list):

        if n_jobs > 1: # parallel

            user_item_features = Parallel(n_jobs=n_jobs)(delayed(self._compute_features)
                                                                              (data, recommender, users_list)
                                                                              for users_list in users_list_chunks)

            x_chunks = [user_item_features[i][0] for i in range(n_jobs)]

            y_chunks = [user_item_features[i][1] for i in range(n_jobs)]

            qids_chunks = [user_item_features[i][2] for i in range(n_jobs)]

            items_chunks = [user_item_features[i][3] for i in range(n_jobs)]

            x = np.concatenate(x_chunks, axis=0)

            y = np.concatenate(y_chunks, axis=0)

            qids = np.concatenate(qids_chunks, axis=0)

            items = np.concatenate(items_chunks, axis=0)

        else: # sequential

            x, y, qids, items = self._compute_features(data, recommender, users_list)

        return x, y, qids, items

    def _compute_features(self, data, recommender, users_list):

        TX = []
        Ty = []
        Tqids = []
        Titems = []

        for user in users_list:

            print(user)

            user_id = int(user.strip('user'))

            candidate_items = self.get_candidates(user, data)

            for item in candidate_items:

                items_liked_by_user = self.items_liked_by_user_dict[user]

                users_liking_the_item = self.users_liking_an_item_dict[item]

                features = recommender.compute_user_item_features(user, item, items_liked_by_user, users_liking_the_item)

                TX.append(features)

                relevance = self.get_relevance(user, item, data)

                Ty.append(relevance)

                Tqids.append(user_id)

                Titems.append(item)

        return np.asarray(TX), np.asarray(Ty), np.asarray(Tqids), np.asarray(Titems)

    def evaluate(self, recommender, x_test, y_test, qids_test, items_test, verbose=True,
                 write_to_file='results.csv',
                 baseline=False):

        if '/' in write_to_file:

            path_split = write_to_file.split('/')

            path = '.'

            for p in path_split[:-1]:
                path = path + '/' + p

            try:
                os.makedirs(path)

            except FileExistsError:
                pass

        if not self.all_items:  # reading the features from file

            M = len(list(set(items_test)))

        else:

            M = len(self.all_items)

        self._define_metrics(M)

        scores = {}

        if baseline:
            strategies = {'algorithm':  recommender.predict(x_test, qids_test)}

        else:
            strategies = {'l2r': recommender.predict(x_test, qids_test),
                          'avg': list(map(lambda x: np.mean(x), x_test)),
                          'min': list(map(lambda x: np.min(x), x_test)),
                          'max': list(map(lambda x: np.max(x), x_test))}

        if self.metrics:

            if verbose:
                print('\n')
                print('Strategy-----Metric-----Mean-----Var\n')

            with open(write_to_file, 'w') as file_write:

                file_write.write('strategy,')

                len_metrics = len(self.metrics.items())

                for i, (metric_name, metric) in enumerate(self.metrics.items()):

                    if i < len_metrics - 1:

                        file_write.write('%s,' % metric_name)

                    else:

                        file_write.write('%s\n' % metric_name)

                for strategy_name, preds in strategies.items():

                    file_write.write('%s,' % strategy_name)

                    for i, (metric_name, metric) in enumerate(self.metrics.items()):

                        if metric_name != 'fit':

                            score = metric.calc_mean(qids_test, y_test, preds, items=items_test)

                            var = metric.calc_mean_var(qids_test, y_test, preds, items=items_test)

                            scores[(strategy_name, metric_name)] = (score, var)

                            if verbose:

                                print('%s-----%s-----%.4f+-%.4f\n' % (strategy_name, metric_name, score, var))

                            if i < len_metrics - 1:

                                file_write.write('%.4f+-%.4f,' % (score, var))
                            else:

                                file_write.write('%.4f+-%.4f\n' % (score, var))

        return scores

    @staticmethod
    def read_features(train, test, val=None):

        with open(train) as train_file:

            x_train, y_train, qids_train, items_train = pyltr.data.letor.read_dataset(train_file)

        print('finished reading train')

        with open(test) as test_file:

            x_test, y_test, qids_test, items_test = pyltr.data.letor.read_dataset(test_file)

        print('finished reading test')


        x_val, y_val, qids_val, items_val = None, None, None, None

        if val:

            with open(val) as val_file:

                x_val, y_val, qids_val, items_val = pyltr.data.letor.read_dataset(val_file)

            print('finished reading val')


        return x_train, y_train, qids_train, items_train,\
               x_test, y_test, qids_test, items_test,\
               x_val, y_val, qids_val, items_val

    @staticmethod
    def write_features_to_file(data, qids, x, y, items):

        with open('%s.svm' % data, 'w') as feature_file:

            for i, x_data in enumerate(x):

                rel = y[i]

                item = items[i]

                feature_file.write('%d qid:%d ' % (rel, qids[i]))

                length = len(x_data)

                for j, f in enumerate(x_data):

                    if j < length - 1:

                        feature_file.write('%d:%f ' % (j+1, f))

                    else:

                        feature_file.write('%d:%f # %s\n' % (j+1, f, item))

    def write_candidates(self,training, test, users_folder, candidates_folder, index_file, data='test', validation=None):

        # reads dat format
        self._parse_data(training, test, validation=validation)

        users_list = list(sorted(self.items_rated_by_user_train.keys()))

        index_dict = {}

        with open(index_file) as index_read:

            for line in index_read:

                line_split = line.strip('\n').split(' ')

                index = line_split[0]

                item = line_split[1]

                index_dict[item] = index

        for user in users_list:

            print(user)

            user_id = int(user.strip('user'))

            with open('%s/%s.txt' %(users_folder, user), 'w') as user_file:

                user_file.write('%s\n' %user)

                with open('%s/%s.txt' %(candidates_folder, user),'w') as candidates_file:

                    candidate_items = self.get_candidates(user, data)

                    for item in candidate_items:

                        index = index_dict[item]

                        candidates_file.write('%s\n' %index)

    def _define_user_list(self, n_users, max_n_feedback, n_jobs):

        if max_n_feedback:

            users_list = [user for user in self.items_rated_by_user_train.keys()
                          if len(self.items_liked_by_user_dict[user]) <= max_n_feedback]

            users_list = sorted(users_list)

        else:

            users_list = list(sorted(self.items_rated_by_user_train.keys()))

        if n_users:  # select a sub-sample of users

            users_list = users_list[0:n_users]

        else:  # select all users

            pass

        assert (len(users_list) >= n_jobs), "Number of users cannot be lower than number of workers"

        return users_list

    def compute_item_to_item_similarity(self, recommender, training, test, dataset, validation=None, n_users=False, n_jobs=4,
                                        supervised=False, max_n_feedback=False, property_specif_emb=True):

        
        if not property_specif_emb:
            # compute features
            x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test, \
            x_val, y_val, qids_val, items_val = self.features(recommender, training, test, supervised=supervised,
                                                              validation=validation, n_users=n_users,
                                                              n_jobs=n_jobs, max_n_feedback=max_n_feedback)


            item_index = {item: index for index, item in enumerate(items_test)}

            # user-item relatedness matrix
            predictions = recommender.predict(x_test, qids_test)

            # user-item relatedness dictionary
            user_item_relatedness = {}

            for i, user_id in enumerate(qids_test):

                item = items_test[i]

                user_item_relatedness[str(user_id), item] = predictions[i]

            # create item-to-item dictionary

            W = {}

            for seed_item in item_index.keys():

                d = {}

                for candidate_item in item_index.keys():

                    try:

                        users_liking_item = self.users_liking_an_item_dict[seed_item]

                    # item not in the training set
                    except KeyError:

                        continue

                    d[candidate_item] = np.mean([user_item_relatedness[u_id, candidate_item] for u_id in users_liking_item], axis=0)

                # normalize the scores

                tot_scores = sum(d.values())

                for key, value in d.items():

                    d[key] = value/tot_scores

                # given the seed, returns a dictionary with item-score
                W[seed_item] = d

            with open('item_index', 'wb') as f:

                pickle.dump(item_index, f, pickle.HIGHEST_PROTOCOL)

            with open('item_to_item_matrix', 'wb') as f:

                pickle.dump(W, f, pickle.HIGHEST_PROTOCOL)

            item_to_item_similarity_dict = {}

            for seed, d in W.items():

                c = {}

                ranks = sorted(d, key=lambda x: d[x])

                print(seed)

                for key, value in d.items():

                    c[key] = ranks.index(key)  # replace scores with ranking

                print(c)

                item_to_item_similarity_dict[seed] = c

            with open('datasets/%s/item_to_item_ranking' % dataset, 'wb') as f:

                pickle.dump(item_to_item_similarity_dict, f, pickle.HIGHEST_PROTOCOL)

        else:

            # defines all items
            self._parse_data(training, test, validation=validation)

            items = self.all_items

            W = defaultdict(dict)

            for i1 in items:

                print(i1)

                for i2 in items:

                    W[i1][i2] = np.mean(recommender.collab_similarities(i1,i2))

            with open('datasets/%s/item_to_item_similarity_%s' % (dataset, recommender.name), 'wb') as f:

                pickle.dump(W, f, pickle.HIGHEST_PROTOCOL)









