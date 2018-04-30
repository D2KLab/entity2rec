import time
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from evaluator import Evaluator
from parse_args import parse_args


class MMLRecommender(object):

    def __init__(self, dataset):

        self.mml_model = self._read_scores('benchmarks/MyMediaLite-3.11/scores.txt')

    def _read_scores(self, file):
 
        model = {}

        with open(file) as file_read:

            for line in file_read:

                line_split = line.strip('\n').split(' ')

                user = line_split[0]
                item = line_split[1]
                score = line_split[2]

                model[(user,item)] = float(score)

        return model

    def compute_user_item_features(self, user, item, items_liked_by_user, users_liking_the_item):

        try:

            features = [self.mml_model[(user, item)]]  # user item relatedness from mml model

        except KeyError:  # do not have user item pair in embedding

            features = [0.]

        return features

    def fit(self, x_train, y_train, qids_train):

        return 0

    def predict(self, x_test, qids_test):

        preds = x_test

        return preds


if __name__ == '__main__':

    np.random.seed(1)  # fixed seed for reproducibility

    start_time = time.time()

    print('Starting MyMediaLite recommender...')

    args = parse_args()

    # initialize MyMediaLite recommender
    mml_rec = MMLRecommender(args.dataset)

    # initialize evaluator

    evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

    # compute e2rec features
    x_train, y_train, qids_train, x_test, y_test, qids_test, \
    x_val, y_val, qids_val = evaluat.features(mml_rec, args.train, args.test,
                                              validation=False, n_users=args.num_users,
                                              n_jobs=args.workers, supervised=False)

    print('Finished computing features after %s seconds' % (time.time() - start_time))

    evaluat.evaluate(mml_rec, x_test, y_test, qids_test)  # evaluates the recommender on the test set

    print("--- %s seconds ---" % (time.time() - start_time))
