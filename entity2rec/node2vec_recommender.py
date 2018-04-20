import time
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from evaluator import Evaluator
from parse_args import parse_args
from sklearn.cluster import KMeans

class Node2VecRecommender(object):

    def __init__(self, dataset, p=1, q=4, walk_length=10,
                 num_walks=500, dimensions=500, window_size=10, iterations=5):

        self.node2vec_model = KeyedVectors.load_word2vec_format(
            'datasets/Movielens1M/node2vec/node2vec_recommender_default.emd'
            , binary=True)

    def compute_user_item_features(self, user, item, items_liked_by_user, users_liking_the_item):

        try:

            features = [self.node2vec_model.similarity(user, item)]  # user item relatedness from node2vec

        except KeyError:  # do not have user item pair in embedding

            features = [0.]

        return features

    def fit(self, x_train, y_train, qids_train):

        return 0

    def predict(self, x_test):

        preds = x_test

        return preds

    def cluster_users(self, n_clusters, users):

        user_to_cluster = {}

        X = []

        for user in users:

            X.append(self.node2vec_model[user])

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

    # initialize entity2rec recommender
    node2vec_rec = Node2VecRecommender(args.dataset, p=args.p, q=args.q,
                                       walk_length=args.walk_length, num_walks=args.num_walks,
                                       dimensions=args.dimensions, window_size=args.window_size,
                                       iterations=args.iter)

    # initialize evaluator

    evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

    # compute e2rec features
    x_train, y_train, qids_train, x_test, y_test, qids_test, \
    x_val, y_val, qids_val = evaluat.features(node2vec_rec, args.train, args.test,
                                              validation=False, n_users=args.num_users,
                                              n_jobs=args.workers, supervised=False)

    print('Finished computing features after %s seconds' % (time.time() - start_time))

    evaluat.evaluate(node2vec_rec, x_test, y_test, qids_test)  # evaluates the recommender on the test set

    print("--- %s seconds ---" % (time.time() - start_time))
