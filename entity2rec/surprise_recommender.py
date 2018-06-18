import time
import numpy as np
from evaluator import Evaluator
from surprise import SVD, KNNBaseline, NMF
from surprise import Reader
from surprise import Dataset
import os
import argparse


class SurpriseRecommender:
    def __init__(self, algorithm, dataset, train, implicit, threshold):
        self.algorithm = algorithm

        self.train = train

        self.dataset = dataset

        self.item_to_ind = {}

        self.user_to_ind = {}

        self.implicit = implicit

        self.threshold = threshold

        self.model = self._learn_model_surprise()


    def _learn_model_surprise(self):

        file_path = os.path.expanduser(self.train)

        if self.implicit:

            rating_scale = (0,1)

        else:
            rating_scale = (1, (self.threshold*5)//4)

        reader = Reader(line_format='user item rating timestamp', sep=' ', rating_scale=rating_scale)

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

        parser.add_argument('--num_users', dest='num_users', type=int, default=False,
                            help='Sample of users for evaluation')

        return parser.parse_args()


if __name__ == '__main__':

    np.random.seed(1)  # fixed seed for reproducibility

    start_time = time.time()

    args = SurpriseRecommender.parse_args()

    rec = args.recommender

    print('Starting surprise recommender...')

    if not args.train:
        args.train = 'datasets/' + args.dataset + '/train.dat'

    if not args.test:
        args.test = 'datasets/' + args.dataset + '/test.dat'

    if not args.validation:
        args.validation = 'datasets/' + args.dataset + '/val.dat'

    sim_options = {'name': 'cosine',

                   'user_based': False  # compute  similarities between items
                   }

    if rec == 'ItemKNN':
        algorithm = [KNNBaseline(sim_options=sim_options)]
        name = ['ItemKNN']

    elif rec == 'SVD':
        algorithm = [SVD()]
        name = ['SVD']

    elif rec == 'NMF':
        algorithm = [NMF()]
        name = ['NMF']

    else:
        algorithm = [KNNBaseline(sim_options=sim_options), SVD(), NMF()]
        name = ['ItemKNN', 'SVD', 'NMF']

    # initialize evaluator

    if args.dataset == 'LastFM':
        implicit = True

    else:
        implicit = args.implicit

    if args.dataset == 'LibraryThing':
        threshold = 8
    else:
        threshold = args.threshold

    evaluat = Evaluator(implicit=implicit, threshold=threshold, all_unrated_items=args.all_unrated_items)

    for i, alg in enumerate(algorithm):
        print(name[i])

        itemrec = SurpriseRecommender(alg, args.dataset, args.train, implicit, threshold)

        # compute features
        x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test, x_val, y_val, qids_val, items_val = evaluat.features(
            itemrec, args.train, args.test,
            validation=False,
            n_jobs=args.workers, supervised=False, n_users=args.num_users)

        print('Finished computing features after %s seconds' % (time.time() - start_time))

        scores = evaluat.evaluate(itemrec, x_test, y_test, qids_test, items_test,
                                  verbose=False)  # evaluates the recommender on the test set

        print(scores)

        print("--- %s seconds ---" % (time.time() - start_time))
