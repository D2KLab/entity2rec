import time
import numpy as np
from evaluator import Evaluator
from parse_args import parse_args
import pandas as pd
from scipy.spatial.distance import euclidean


class TransRecommender(object):

    def __init__(self, dataset, method="TransE"):

        self.dataset = dataset
        self.method = method

        self.entity2id = self._parse_ind_file('datasets/%s/KB2E/entity2id.txt' % self.dataset)
        self.relation2id = self._parse_ind_file('datasets/%s/KB2E/relation2id.txt' % self.dataset)

        self.entity_emb_matrix = self._parse_emb_file('datasets/%s/KB2E/%s/entity2vec.bern' % (self.dataset, method))
        self.relation_emb_matrix = self._parse_emb_file('datasets/%s/KB2E/%s/relation2vec.bern' % (self.dataset, method))

        self.entity_emb_dict = self._build_emb_dictionary(self.entity_emb_matrix, self.entity2id)
        self.relation_emb_dict = self._build_emb_dictionary(self.relation_emb_matrix, self.relation2id)

        if method == "TransH":

            self.norm_matrix = self._parse_emb_file('datasets/%s/KB2E/%s/A.bern' % (self.dataset, method))

            index = [i for i in self.relation2id.keys() if self.relation2id[i] == 'feedback']

            self.norm_feedback = np.array(self.norm_matrix[index]).reshape((100,))

        if method == "TransR":

            # matrix containing rel*size*size elements

            self.M = self._parse_emb_file('datasets/%s/KB2E/%s/A.bern' % (self.dataset, method))

            index_feedback = [i for i in self.relation2id.keys() if self.relation2id[i] == 'feedback'][0]

            data = self.M

            size_emb = 100

            data = data[index_feedback*size_emb:index_feedback*size_emb+size_emb, :]

            self.M = data

    @staticmethod
    def _parse_ind_file(file):

        ind_dict = {}

        with open(file) as read_file:

            for line in read_file:

                line_split = line.strip('\n').split('\t')

                name = line_split[0]

                int_id = int(line_split[1])

                ind_dict[int_id] = name

        return ind_dict

    @staticmethod
    def _parse_emb_file(file):

        data = pd.read_table(file, header=None)

        data = data.values[:, :-1]  # drop last column of nan values

        return data

    @staticmethod
    def _build_emb_dictionary(emb_table, emb_index):

        emb_dictionary = {}

        for i, values in enumerate(emb_table):

            name = emb_index[i]

            emb_dictionary[name] = values

        return emb_dictionary

    def compute_user_item_features(self, user, item, items_liked_by_user, users_liking_the_item):

        try:

            emb_user = self.entity_emb_dict[user]

        except KeyError:

            emb_user = np.zeros(100)

        try:

            emb_item = self.entity_emb_dict[item]

        except KeyError:

            emb_item = np.zeros(100)

        emb_feedback = self.relation_emb_dict['feedback']

        if self.method == "TransH":

            # project user on feedback relation

            emb_user = emb_user - np.dot(np.dot(self.norm_feedback.T, emb_user), self.norm_feedback)

            emb_item = emb_item - np.dot(np.dot(self.norm_feedback.T, emb_item), self.norm_feedback)

        elif self.method == "TransR":

            # project user and item in relation space

            emb_user = np.matmul(emb_user, self.M)

            emb_item = np.matmul(emb_item, self.M)

        features = [-euclidean(emb_user + emb_feedback, emb_item)]

        return features

    def fit(self, x_train, y_train, qids_train):

        return 0

    def predict(self, x_test, qids_test):

        preds = x_test

        return preds


if __name__ == '__main__':

    np.random.seed(1)  # fixed seed for reproducibility

    start_time = time.time()

    print('Starting trans_recommender...')

    args = parse_args()

    # initialize trans recommender
    trans_rec = TransRecommender(args.dataset, method="TransH")

    # initialize evaluator

    evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

    # compute features
    x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test, x_val, y_val, qids_val, items_val = evaluat.features(trans_rec, args.train, args.test, validation=False, n_users=args.num_users,
                                              n_jobs=args.workers, supervised=False)

    print('Finished computing features after %s seconds' % (time.time() - start_time))

    evaluat.evaluate(trans_rec, x_test, y_test, qids_test, items_test)  # evaluates the recommender on the test set

    print("--- %s seconds ---" % (time.time() - start_time))
