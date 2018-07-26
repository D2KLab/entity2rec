import time
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from evaluator import Evaluator
from parse_args import parse_args
from sklearn.cluster import KMeans
from node2vec import Node2Vec
import os


class Node2VecRecommender(Node2Vec):

    def __init__(self, dataset, p=1, q=4, walk_length=100,
                 num_walks=50, dimensions=200, window_size=30, workers=8, iterations=5):

        Node2Vec.__init__(self, False, True, False, p, q, walk_length, num_walks, dimensions, window_size,
                          workers, iterations)

        self.dataset = dataset

        file = 'num%d_p%d_q%d_l%d_d%d_iter%d_winsize%d.emd' % (num_walks, p, q,
                                                               walk_length, dimensions,
                                                               iterations, window_size)

        self.path = 'datasets/%s/node2vec/' % self.dataset + file

        if file not in os.listdir('datasets/%s/node2vec/' % self.dataset):

            self.run('datasets/%s/node2vec/altogether.edgelist' % self.dataset,
                             self.path)

        self.node2vec_model = KeyedVectors.load_word2vec_format(self.path, binary=True)

    def compute_user_item_features(self, user, item, items_liked_by_user, users_liking_the_item):

        try:

            features = [self.node2vec_model.similarity(user, item)]  # user item relatedness from node2vec

        except KeyError:  # do not have user item pair in embedding

            features = [0.]

        return features

    def fit(self, x_train, y_train, qids_train):

        return 0

    def predict(self, x_test, qids_test):

        preds = x_test

        return preds

    def cluster_users(self, n_clusters, users):

        user_to_cluster = {}

        X = []

        for user in users:

            print(user)

            try:

                X.append(self.node2vec_model[user])

            except KeyError:

                X.append(np.zeros(200))

        X = np.asarray(X)

        kmeans = KMeans(n_clusters=n_clusters, random_state=1).fit(X)

        cluster_labels = kmeans.labels_

        for i, user in enumerate(users):

            user_to_cluster[user] = cluster_labels[i]

        return user_to_cluster


if __name__ == '__main__':

    np.random.seed(1)  # fixed seed for reproducibility

    start_time = time.time()

    print('Starting node2vec recommender...')

    args = parse_args()

    # default settings

    if not args.train:
        args.train = 'datasets/' + args.dataset + '/train.dat'

    if not args.test:
        args.test = 'datasets/' + args.dataset + '/test.dat'

    if not args.validation:
        args.validation = 'datasets/' + args.dataset + '/val.dat'

    if args.dataset == 'LastFM':

        implicit = True

    else:

        implicit = args.implicit

    if args.dataset == 'LibraryThing':

        threshold = 8

    else:

        threshold = args.threshold

    # initialize node2vec recommender
    node2vec_rec = Node2VecRecommender(args.dataset, p=args.p, q=args.q,
                                       walk_length=args.walk_length, num_walks=args.num_walks,
                                       dimensions=args.dimensions, window_size=args.window_size,
                                       iterations=args.iter)

    # initialize evaluator

    evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

    # compute e2rec features
    x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test, \
    x_val, y_val, qids_val, items_val = evaluat.features(node2vec_rec, args.train, args.test,
                                                         validation=False, n_users=args.num_users,
                                                         n_jobs=args.workers, supervised=False)

    print('Finished computing features after %s seconds' % (time.time() - start_time))

    evaluat.evaluate(node2vec_rec, x_test, y_test, qids_test, items_test,
                     write_to_file='results/%s/node2vec/num%d_p%d_q%d_l%d_d%d_iter%d_winsize%d.csv'
                                   % (args.dataset, args.num_walks, args.p, args.q, args.walk_length,
                                      args.dimensions, args.iter, args.window_size),
                     baseline=True)  # evaluates the recommender on the test set

    print("--- %s seconds ---" % (time.time() - start_time))
