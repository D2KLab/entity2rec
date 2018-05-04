import time
import numpy as np
from evaluator import Evaluator
from parse_args import parse_args
from surprise import SVD, KNNBaseline, NMF, KNNWithMeans
from surprise import Reader
from surprise import Dataset
import os
import sys


class SurpriseRecommender:

    def __init__(self, algorithm, dataset, train):

        self.algorithm = algorithm

        self.train = train

        self.dataset = dataset

        self.item_to_ind = {}

        self.user_to_ind = {}

        self.model = self._learn_model_surprise()

    def _learn_model_surprise(self):

        file_path = os.path.expanduser(self.train)

        reader = Reader(line_format='user item rating timestamp', sep=' ')

        data = Dataset.load_from_file(file_path, reader=reader)

        algo = self.algorithm

        trainset = data.build_full_trainset()

        algo.train(trainset)

        return algo

    def compute_user_item_features(self, user, item, items_liked_by_user, users_liking_the_item):

        features = [self.model.predict(user, item)[3]]

        return features

    def fit(self, x_train, y_train, qids_train):

        return 0

    def predict(self, x_test, qids_test):

        preds = x_test

        return preds


if __name__ == '__main__':

    np.random.seed(1)  # fixed seed for reproducibility

    start_time = time.time()

    rec = sys.argv[1]

    print('Starting %s recommender...' %rec)

    args = parse_args()

    if not args.train:
        args.train = 'datasets/'+args.dataset+'/train.dat'

    if not args.test:
        args.test = 'datasets/'+args.dataset+'/test.dat'

    if not args.validation:
        args.validation = 'datasets/'+args.dataset+'/val.dat'

    sim_options = {'name': 'cosine',
                
                   'user_based': False  # compute  similarities between items
                   }

    if rec == 'ItemKNN':
        algorithm = KNNBaseline(sim_options=sim_options)

    elif rec == 'SVD':
        algorithm = SVD()

    elif rec == 'NMF':
        algorithm = NMF()

    else:
        raise ValueError("Choose between ItemKNN, SVD or NMF")

    itemrec = SurpriseRecommender(algorithm, args.dataset, args.train)

    # initialize evaluator

    evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

    # compute features
    x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test, \
    x_val, y_val, qids_val, items_val = evaluat.features(itemrec, args.train, args.test,
                                              validation=False, n_users=args.num_users,
                                              n_jobs=args.workers, supervised=False)

    print('Finished computing features after %s seconds' % (time.time() - start_time))

    scores = evaluat.evaluate(itemrec, x_test, y_test, qids_test, verbose=False)  # evaluates the recommender on the test set

    print(scores)

    print("--- %s seconds ---" % (time.time() - start_time))
