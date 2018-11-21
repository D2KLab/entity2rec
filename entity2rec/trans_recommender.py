import time
import numpy as np
from evaluator import Evaluator
import pandas as pd
from scipy.spatial.distance import euclidean
import argparse
import subprocess
import os


class TransRecommender(object):

    def __init__(self, dataset, dimensions=100, learning_rate=0.001, method="TransE"):

        self.dataset = dataset
        self.method = method
        self.dimensions = dimensions
        self.learning_rate = learning_rate

        self.entity2id = self._parse_ind_file('benchmarks/KB2E/data/%s/entity2id.txt' % self.dataset)
        self.relation2id = self._parse_ind_file('benchmarks/KB2E/data/%s/relation2id.txt' % self.dataset)

        self.entity_emb_matrix = self._parse_emb_file('benchmarks/KB2E/%s/entity2vec_d%d_lr%.3f.bern' % (method,
                                                                                                         self.dimensions,
                                                                                                         self.learning_rate))
        self.relation_emb_matrix = self._parse_emb_file('benchmarks/KB2E/%s/relation2vec_d%d_lr%.3f.bern' % (method,
                                                                                                         self.dimensions,
                                                                                                         self.learning_rate))

        self.entity_emb_dict = self._build_emb_dictionary(self.entity_emb_matrix, self.entity2id)
        self.relation_emb_dict = self._build_emb_dictionary(self.relation_emb_matrix, self.relation2id)

        if method == "TransH":

            self.norm_matrix = self._parse_emb_file('benchmarks/KB2E/%s/A_d%d_lr%.3f.bern' % (method, self.dimensions,
                                                                                              self.learning_rate))

            index = [i for i in self.relation2id.keys() if self.relation2id[i] == 'feedback']

            self.norm_feedback = np.array(self.norm_matrix[index]).reshape((self.dimensions,))

        if method == "TransR":

            # matrix containing rel*size*size elements

            self.M = self._parse_emb_file('benchmarks/KB2E/%s/A_d%d_lr%.3f.bern' % (method, self.dimensions,self.learning_rate))

            index_feedback = [i for i in self.relation2id.keys() if self.relation2id[i] == 'feedback'][0]

            data = self.M

            size_emb = self.dimensions

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

            emb_user = np.zeros(self.dimensions)

        try:

            emb_item = self.entity_emb_dict[item]

        except KeyError:

            emb_item = np.zeros(self.dimensions)

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

    @staticmethod
    def parse_args():

        parser = argparse.ArgumentParser(description="Run translational recommender")

        parser.add_argument('--dimensions', type=int, default=100,
                            help='Number of dimensions. Default is 100.')

        parser.add_argument('--workers', type=int, default=8,
                            help='Number of parallel workers. Default is 8.')

        parser.add_argument('--config_file', nargs='?', default='config/properties.json',
                            help='Path to configuration file')

        parser.add_argument('--dataset', nargs='?', default='Movielens1M',
                            help='Dataset')

        parser.add_argument('--train', dest='train', help='train', default=None)

        parser.add_argument('--test', dest='test', help='test', default=None)

        parser.add_argument('--validation', dest='validation', default=None, help='validation')

        parser.add_argument('--run_all', dest='run_all', action='store_true', default=False,
                            help='If computing also the embeddings')

        parser.add_argument('--implicit', dest='implicit', action='store_true', default=False,
                            help='Implicit feedback with boolean values')

        parser.add_argument('--all_items', dest='all_unrated_items', action='store_false', default=True,
                            help='Whether keeping the rated items of the training set as candidates. '
                                 'Default is AllUnratedItems')

        parser.add_argument('--N', dest='N', type=int, default=None,
                            help='Cutoff to estimate metric')

        parser.add_argument('--threshold', dest='threshold', default=4, type=int,
                            help='Threshold to convert ratings into binary feedback')

        parser.add_argument('--num_users', dest='num_users', type=int, default=False,
                            help='Sample of users for evaluation')

        parser.add_argument('--max_n_feedback', dest='max_n_feedback', type=int, default=False,
                            help='Only select users with less than max_n_feedback for training and evaluation')

        parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.001,
                            help='Learning rate')

        return parser.parse_args()

    @staticmethod
    def create_knowledge_graph(dataset):

        folder = 'datasets/%s/graphs' % dataset

        entities = []

        relations = []

        with open('benchmarks/KB2E/data/%s/train.txt' % dataset, 'w') as write_kg:

            for file in os.listdir(folder):

                if 'edgelist' in file:

                    prop_name = file.replace('.edgelist', '')

                    print(prop_name)

                    with open('%s/%s' % (folder, file), 'r') as edgelist_read:

                        for edge in edgelist_read:
                            edge_split = edge.strip('\n').split(' ')

                            left_edge = edge_split[0]

                            right_edge = edge_split[1]

                            write_kg.write('%s\t%s\t%s\n' % (left_edge, right_edge, prop_name))

                            entities.append(left_edge)

                            entities.append(right_edge)

                            relations.append(prop_name)

        # create index

        entities = list(set(entities))

        with open('benchmarks/KB2E/data/%s/entity2id.txt' % dataset, 'w') as entity2id:

            for i, entity in enumerate(entities):
                entity2id.write('%s\t%d\n' % (entity, i))

        relations = list(set(relations))

        with open('benchmarks/KB2E/data/%s/relation2id.txt' % dataset, 'w') as relation2id:

            for i, relation in enumerate(relations):
                relation2id.write('%s\t%d\n' % (relation, i))


if __name__ == '__main__':

    np.random.seed(1)  # fixed seed for reproducibility

    start_time = time.time()

    print('Starting trans_recommender...')

    args = TransRecommender.parse_args()

    if not args.train:
        args.train = 'datasets/' + args.dataset + '/train.dat'

    if not args.test:
        args.test = 'datasets/' + args.dataset + '/test.dat'

    if not args.validation:
        args.validation = 'datasets/' + args.dataset + '/val.dat'
    # initialize evaluator

    if args.dataset == 'LastFM':
        implicit = True

    else:
        implicit = args.implicit

    if args.dataset == 'LibraryThing':
        threshold = 8
    else:
        threshold = args.threshold

    for method in ["TransE", "TransH", "TransR"]:

        print(method)

        if args.run_all:

            create_index = False

            if create_index:

                TransRecommender.create_knowledge_graph(args.dataset)

            print('Training the %s algorithm' % method)
            print('dataset: %s, size: %s, lr: %.3f' %(args.dataset, args.dimensions, args.learning_rate))

            if not os.path.isfile("benchmarks/KB2E/%s/entity2vec_d%d_lr%.3f.bern" % (method, args.dimensions, args.learning_rate)):

                subprocess.check_output(["./Train_%s" % method, "%s" % args.dataset, "-size", "%d" % args.dimensions,
                                         "-rate", "%.3f" % args.learning_rate], cwd="benchmarks/KB2E/%s" % method)
                subprocess.check_output(["mv", "entity2vec.bern", "entity2vec_d%d_lr%.3f.bern" %(args.dimensions, args.learning_rate)],
                                        cwd="benchmarks/KB2E/%s" % method)
                subprocess.check_output(["mv", "relation2vec.bern", "relation2vec_d%d_lr%.3f.bern" % (args.dimensions, args.learning_rate)],
                                        cwd="benchmarks/KB2E/%s" % method)

                if method is not "TransE":
                    subprocess.check_output(["mv", "A.bern", "A_d%d_lr%.3f.bern" % (args.dimensions, args.learning_rate)],
                                        cwd="benchmarks/KB2E/%s" % method)

            else:

                print("embeddings already exist")

        # initialize trans recommender
        trans_rec = TransRecommender(args.dataset, dimensions=args.dimensions, learning_rate=args.learning_rate,
                                     method=method)

        # initialize evaluator

        evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

        # compute features
        x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test, x_val, y_val, qids_val, items_val = evaluat.features(trans_rec, args.train, args.test, validation=False, n_users=args.num_users,
                                                  n_jobs=args.workers, supervised=False)

        print('Finished computing features after %s seconds' % (time.time() - start_time))

        evaluat.evaluate(trans_rec, x_test, y_test, qids_test, items_test,
                         write_to_file="results/%s/translational/%s.csv" % (args.dataset, method),
                         baseline=True)  # evaluates the recommender on the test set

        print("--- %s seconds ---" % (time.time() - start_time))
