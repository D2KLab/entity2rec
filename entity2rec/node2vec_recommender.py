from entity2rec import Entity2Rec
import time
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from evaluator import Evaluator
from parse_args import parse_args


class Node2VecRecommender(Entity2Rec):

    def __init__(self, dataset, p=1, q=4, walk_length=10,
                            num_walks=500, dimensions=500, window_size=10, iterations=5):

        Entity2Rec.__init__(self, dataset, p=p, q=q, walk_length=walk_length,
                            num_walks=num_walks, dimensions=dimensions,
                            window_size=window_size, iterations=iterations)

        self.node2vec_model = KeyedVectors.load_word2vec_format(
            'emb/%s/feedback/num%d_p%d_q%d_l%d_d%d_iter%d_winsize%d.emd'
            % (self.dataset, num_walks, p, q, walk_length, dimensions, iterations, window_size), binary=True)

    def compute_user_item_features(self, user, item, items_liked_by_user):

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

