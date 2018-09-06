from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from entity2rec import Property
import json
import codecs
from collections import defaultdict
from evaluator import Evaluator
import time
import argparse
import sys
from operator import itemgetter
import random


class FMRec:

    def __init__(self, dataset, training, test, config='config/properties.json', lr=0.001, num_factors=10, num_iter=100, threshold=4, implicit=False):

        self.dataset = dataset

        self.config_file = config

        self.properties = []

        self.implicit = implicit

        if self.implicit:

            self.threshold = 0.5

        else:

            self.threshold = threshold

        self._set_properties()

        self._read_item_attributes()

        print('finished reading item attributes')

        self.model = pylibfm.FM(num_factors=num_factors, num_iter=num_iter, verbose=True,
                                task="classification", initial_learning_rate=lr,
                                learning_rate_schedule="optimal")

        self.x_train, self.y_train, self.train_users, self.train_items = self._load_data(training)

        self.x_test, self.y_test, self.test_users, self.test_items = self._load_data(test)

        if self.implicit:  # need to generate negative candidates for training

            num_negative_candidates = 100

            all_items = self.train_items.union(self.test_items)

            unrated_items = [item for item in all_items if
                               item not in self.train_items]

            unrated_items = sorted(unrated_items)

            for user in self.train_users:

                negative_candidates = list(random.sample(unrated_items, num_negative_candidates))

                for item in negative_candidates:

                    self.x_train.append(self._fetch_attributes(user, item))

                    self.y_train.append(0.)

            for user in self.test_users:

                negative_candidates = list(random.sample(unrated_items, num_negative_candidates))

                for item in negative_candidates:

                    self.x_test.append(self._fetch_attributes(user, item))

                    self.y_test.append(0.)

        print('finished reading data')

        self.vectorizer = DictVectorizer()

        self.x_train = self.vectorizer.fit_transform(self.x_train)

        self.x_test = self.vectorizer.transform(self.x_test)

        print('finished transforming data')

        self.model.fit(self.x_train, self.y_train)  # fit the model

        print('finished fitting model')

    def _set_properties(self):

        with codecs.open(self.config_file, 'r', encoding='utf-8') as config_read:

            property_file = json.loads(config_read.read())

            for typology in property_file[self.dataset]:

                for property_name in property_file[self.dataset][typology]:

                    self.properties.append(Property(property_name, typology))

    def _fetch_attributes(self, user, item):

        # create a dictionary with user item interactions and item attributes

        d = {'user_id': user, 'item_id': item}

        attribute_dict = self.item_attributes[item]

        for prop_name, attribute in attribute_dict.items():

            d[prop_name] = attribute

        return d

    def _read_item_attributes(self):

        self.item_attributes = defaultdict(dict)  # dict of dict containing item attributes

        for prop in self.properties:  # iterate in content based properties

            prop_name = prop.name

            if prop_name == 'feedback':  # no need for feedback data here

                pass

            if 'feedback_' in prop_name:  # no need for hybrid graphs

                prop_name = prop_name.replace('feedback_', '')

            with open('datasets/%s/graphs/%s.edgelist' % (self.dataset, prop_name)) as edgelist:

                for line in edgelist:

                    line_split = line.strip('\n').split(' ')

                    item = line_split[0]

                    attribute = line_split[1]

                    self.item_attributes[item][prop_name] = attribute

    def _load_data(self, data):

        X = []

        y = []

        users = set()

        items = set()

        with open(data) as data_file:

            for line in data_file:

                line_split = line.strip('\n').split(' ')

                user = line_split[0]

                item = line_split[1]

                rating = line_split[2]

                # create a dictionary with user item interactions and item attributes

                d = self._fetch_attributes(user, item)

                X.append(d)

                if int(rating) >= self.threshold:

                    rating = 1

                else:
                    rating = 0

                y.append(float(rating))

                users.add(user)

                items.add(item)

        return X, y, users, items

    def compute_user_item_features(self, user, item, items_liked_by_user, users_liking_the_item):

        try:

            d = self._fetch_attributes(user, item)

            score = self.model.predict(self.vectorizer.transform(d))[0]

            features = [score]  # user item relatedness from fm model

        except KeyError:  # do not have user item pair in embedding

            features = [0.]

        return features

    def fit(self, x_train, y_train, qids_train):

        return 0

    def predict(self, x_test, qids_test):

        preds = x_test

        return preds

    @staticmethod
    def parse_args():

        parser = argparse.ArgumentParser(description="Run entity2rec.")

        parser.add_argument('--dimensions', type=int, default=200,
                            help='Number of dimensions. Default is 200.')

        parser.add_argument('--iter', default=5, type=int,
                            help='Number of epochs in SGD')

        parser.add_argument('--workers', type=int, default=8,
                            help='Number of parallel workers. Default is 8.')

        parser.add_argument('--config_file', nargs='?', default='config/properties.json',
                            help='Path to configuration file')

        parser.add_argument('--dataset', nargs='?', default='Movielens1M',
                            help='Dataset')

        parser.add_argument('--train', dest='train', help='train', default=None)

        parser.add_argument('--test', dest='test', help='test', default=None)

        parser.add_argument('--validation', dest='validation', default=None, help='validation')

        parser.add_argument('--all_items', dest='all_unrated_items', action='store_false', default=True,
                            help='Whether keeping the rated items of the training set as candidates. '
                                 'Default is AllUnratedItems')
        parser.add_argument('--implicit', dest='implicit', action='store_true', default=False,
                            help='Implicit feedback with boolean values')

        parser.add_argument('--write_features', dest='write_features', action='store_true', default=False,
                            help='Writes the features to file')

        parser.add_argument('--read_features', dest='read_features', action='store_true', default=False,
                            help='Reads the features from a file')

        parser.add_argument('--threshold', dest='threshold', default=4, type=int,
                            help='Threshold to convert ratings into binary feedback')

        parser.add_argument('--num_users', dest='num_users', type=int, default=False,
                            help='Sample of users for evaluation')

        parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                            help='Starting value for the learning rate')

        parser.add_argument('--hyper_opt', dest='hyper_opt', default=False, action='store_true',
                    help='Sample of users for evaluation')

        return parser.parse_args()


if __name__ == '__main__':

    start_time = time.time()

    print('Starting FM...')

    args = FMRec.parse_args()

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

    evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

    search_hyper = args.hyper_opt

    lr = args.lr

    n_factors = args.dimensions

    n_iters = args.iter

    if search_hyper:

        results = {}

        print('start hyper params optimization')

        lr_values = [0.01, 0.001, 0.0001]

        n_factor_values = [10, 50]
        n_iters_values = [50, 100]

        for lr in lr_values:

            for n_factors in n_factor_values:

                for n_iters in n_iters_values:

                    print('lr:%.4f,n_factors:%d,epochs:%d\n' %(lr, n_factors, n_iters))

                    fm_rec = FMRec(args.dataset, args.train, args.validation,
                                                             lr=lr,
                                                             num_factors=n_factors,
                                                             num_iter=n_iters,
                                                             threshold=args.threshold,
                                                             implicit=args.implicit)

                    x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test, \
                    x_val, y_val, qids_val, items_val = evaluat.features(fm_rec, args.train, args.validation,
                                                             n_jobs=args.workers,
                                                             n_users=args.num_users)

                    print('Finished computing features after %s seconds' % (time.time() - start_time))

                    scores = evaluat.evaluate(fm_rec, x_test, y_test, qids_test, items_test, baseline=True)

                    results[(lr,n_factors,n_iters)] = scores[('algorithm', 'P@5')]

        lr, n_factors, n_iters = max(results.items(), key=itemgetter(1))[0]

        print('optimal parameters are:\n')
        print('lr: ', lr,'n_factors: ', n_factors,'n_iters: ', n_iters)
        print('evaluating the model on the test set usign optimal parameters...')

    fm_rec = FMRec(args.dataset, args.train, args.test, lr=lr, num_factors=n_factors, num_iter=n_iters, threshold=args.threshold, implicit=args.implicit)

    x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test, \
    x_val, y_val, qids_val, items_val = evaluat.features(fm_rec, args.train, args.test,
                                                         n_jobs=args.workers,
                                                         n_users=args.num_users)

    print('Finished computing features after %s seconds' % (time.time() - start_time))

    if args.write_features:

        evaluat.write_features_to_file('train', qids_train, x_train, y_train, items_train)

        evaluat.write_features_to_file('test', qids_test, x_test, y_test, items_test)

    evaluat.evaluate(fm_rec, x_test, y_test, qids_test, items_test, baseline=True)  # evaluates the model on the test set

    print("--- %s seconds ---" % (time.time() - start_time))







