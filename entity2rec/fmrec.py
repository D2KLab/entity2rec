from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from entity2rec import Property
import json
import codecs
from collections import defaultdict
from parse_args import parse_args
from evaluator import Evaluator
import time


class FMRec:

    def __init__(self, dataset, training, test, config='config/properties.json'):

        self.dataset = dataset

        self.config_file = config

        self.properties = []

        self._set_properties()

        self._read_item_attributes()

        print('finished reading item attributes')

        self.model = pylibfm.FM(num_factors=10, num_iter=100, verbose=True,
                                task="regression", initial_learning_rate=0.001,
                                learning_rate_schedule="optimal")

        self.x_train, self.y_train, self.train_users, self.train_items = self._load_data(training)

        self.x_test, self.y_test, self.test_users, self.test_items = self._load_data(test)

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

        d = {'user': user, 'item': item}

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


if __name__ == '__main__':

    start_time = time.time()

    print('Starting FM...')

    args = parse_args()

    fm_rec = FMRec(args.dataset, args.train, args.test)

    evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

    x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test, \
    x_val, y_val, qids_val, items_val = evaluat.features(fm_rec, args.train, args.test, validation=args.validation,
                                                         n_jobs=args.workers,
                                                         n_users=args.num_users)

    print('Finished computing features after %s seconds' % (time.time() - start_time))

    if args.write_features:

        evaluat.write_features_to_file('train', qids_train, x_train, y_train, items_train)

        evaluat.write_features_to_file('test', qids_test, x_test, y_test, items_test)

    evaluat.evaluate(fm_rec, x_test, y_test, qids_test, items_test)  # evaluates the model on the test set

    print("--- %s seconds ---" % (time.time() - start_time))







