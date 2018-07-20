import time
import numpy as np
from evaluator import Evaluator
import argparse


class MMLRecommender(object):

    def __init__(self, dataset, recommender):

        self.mml_model = self._read_scores('benchmarks/MyMediaLite-3.11/%s_scores.txt' % recommender)

    def _read_scores(self, file):
        
        model = {}

        with open(file) as file_read:

            for line in file_read:

                line_split = line.strip('\n').split(' ')

                user = line_split[0]
                item = line_split[1]
                score = line_split[2]

                model[(user, item)] = float(score)

        return model

    def compute_user_item_features(self, user, item, items_liked_by_user, users_liking_the_item):

        try:

            features = [self.mml_model[(user, item)]]  # user item relatedness from node2vec

        except KeyError:  # do not have user item pair in embedding

            features = [0.]

        return features

    def fit(self, x_train, y_train, qids_train):

        return 0

    def predict(self, x_test, qids_test):

        preds = x_test

        return preds


def parse_args():

    """
    Parses the entity2rec arguments.
    """

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--config_file', nargs='?', default='config/properties.json',
                        help='Path to configuration file')

    parser.add_argument('--dataset', nargs='?', default='Movielens1M',
                        help='Dataset')

    parser.add_argument('--train', dest='train', help='train', default=None)

    parser.add_argument('--test', dest='test', help='test', default=None)

    parser.add_argument('--validation', dest='validation', default=None, help='validation')

    parser.add_argument('--implicit', dest='implicit', action='store_true', default=False,
                        help='Implicit feedback with boolean values')

    parser.add_argument('--all_items', dest='all_unrated_items', action='store_false', default=True,
                        help='Whether keeping the rated items of the training set as candidates. '
                             'Default is AllUnratedItems')
    
    parser.add_argument('--threshold', dest='threshold', default=4, type=int,
                        help='Threshold to convert ratings into binary feedback')

    parser.add_argument('--recommender', dest='recommender', help="which recommender to use")

    return parser.parse_args()


if __name__ == '__main__':

    np.random.seed(1)  # fixed seed for reproducibility

    start_time = time.time()

    print('Starting MyMediaLite recommender...')

    args = parse_args()

    # initialize MyMediaLite recommender
    mml_rec = MMLRecommender(args.dataset, args.recommender)

    # initialize evaluator

    evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

    # compute e2rec features
    x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test,\
    x_val, y_val, qids_val, items_val = evaluat.features(mml_rec, args.train, args.test,
                                              validation=False, 
                                              n_jobs=args.workers, supervised=False)

    print('Finished computing features after %s seconds' % (time.time() - start_time))

    scores = evaluat.evaluate(mml_rec, x_test, y_test, qids_test, items_test, verbose=False)  # evaluates the recommender on the test set

    scores_ = [np.round(score, decimals=6) for metric, score in scores]

    print(scores_)

    print("--- %s seconds ---" % (time.time() - start_time))
