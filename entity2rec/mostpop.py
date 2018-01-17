from entity2rec import Entity2Rec
import codecs
from collections import Counter
import time
import argparse
import numpy as np
from random import shuffle
from joblib import Parallel, delayed


def parse_args():

    """
    Parses the entity2rec arguments.
    """

    parser = argparse.ArgumentParser(description="Run entity2rec.")

    parser.add_argument('--walk_length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num_walks', type=int, default=500,
                        help='Number of walks per source. Default is 40.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is directed.')
    parser.set_defaults(directed=False)

    parser.add_argument('--no_preprocessing', dest='preprocessing', action='store_false',
                        help='Whether preprocess all transition probabilities or compute on the fly')
    parser.set_defaults(preprocessing=True)

    parser.add_argument('--dimensions', type=int, default=500,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=5, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--config_file', nargs='?', default='config/properties.json',
                        help='Path to configuration file')

    parser.add_argument('--dataset', nargs='?', default='Movielens1M',
                        help='Dataset')

    parser.add_argument('--sparql', dest='sparql', default=False,
                        help='Whether downloading the graphs from a sparql endpoint')

    parser.add_argument('--entities', dest='entities', default=False,
                        help='A specific list of entities for which the embeddings have to be computed')

    parser.add_argument('--default_graph', dest='default_graph', default=False,
                        help='Default graph to query when using a Sparql endpoint')

    parser.add_argument('--train', dest='train', help='train', default=False)

    parser.add_argument('--test', dest='test', help='test')

    parser.add_argument('--validation', dest='validation', default=False, help='validation')

    parser.add_argument('--run_all', dest='run_all', action='store_true', default=False,
                        help='If computing also the embeddings')

    parser.add_argument('--implicit', dest='implicit', action='store_true', default=False,
                        help='Implicit feedback with boolean values')

    parser.add_argument('--entity_class', dest='entity_class', help='entity class', default=False)

    parser.add_argument('--feedback_file', dest='feedback_file', default=False,
                        help='Path to a DAT file that contains all the couples user-item')

    parser.add_argument('--all_unrated_items', dest='all_unrated_items', action='store_true', default=False,
                        help='Whether removing the rated items of the training set from the candidates')

    parser.add_argument('--write_features', dest='write_features', action='store_true', default=False,
                        help='Writes the features to file')

    parser.add_argument('--read_features', dest='read_features', action='store_true', default=False,
                        help='Reads the features from a file')

    parser.add_argument('--metric', dest='metric', default='AP',
                        help='Metric to optimize in the training')

    parser.add_argument('--N', dest='N', type=int, default=10,
                        help='Cutoff to estimate metric')

    parser.add_argument('--threshold', dest='threshold', default=4,
                        help='Threshold to convert ratings into binary feedback')

    return parser.parse_args()


def compute_most_pop_dict(data, threshold):

    pop_dict = Counter()

    with codecs.open(data, 'r', encoding='utf-8') as read_train:

        for line in read_train:

            line = line.split(' ')

            item = line[1]

            rel = line[2]

            if int(rel) >= threshold:

                pop_dict[item] += 1

    # normalize

    pop_dict_tot = float(sum(pop_dict.values()))

    for key in pop_dict.keys():

        pop_dict[key] /= pop_dict_tot

    return pop_dict


class MostPop(Entity2Rec):

    def __init__(self, dataset):

        Entity2Rec.__init__(self, dataset)

        self.pop_dict = {}

    def _compute_user_item_features(self, user, item):

        try:

            relevance = int(self.items_ratings_by_user_test[
                                (user, item)])  # get the relevance score if it's in the test

            if self.implicit is False:

                relevance = 1 if relevance >= self.threshold else 0

        except KeyError:

            relevance = 0  # unrated items are assumed to be negative

        try:

            features = [np.float32(self.pop_dict[item])]  # simply the weighted popularity

        except KeyError:

            features = [np.mean([self.pop_dict.values()], dtype=float)]

        return features, relevance

    def _compute_features(self, data, test=False):

        TX = []
        Ty = []
        Tqids = []

        if self.implicit:  # only positive feedback is available, need to generate false candidataes
            test = True

        if test:  # generate the features also for negative candidates

            for user in self.items_rated_by_user_train.keys():

                print(user)

                user_id = int(user.strip('user'))

                candidate_items = self.get_candidates(user)

                shuffle(candidate_items)  # relevant and non relevant items are shuffled

                for item in candidate_items:

                    features, relevance = self._compute_user_item_features(user, item)

                    TX.append(features)

                    Ty.append(relevance)

                    Tqids.append(user_id)

        else:  # only generate features for data in the training set

            with codecs.open(data, 'r', encoding='utf-8') as data_file:

                for line in data_file:

                    user, user_id, item, relevance = self.parse_users_items_rel(line)

                    Tqids.append(user_id)

                    features, relevance = self._compute_user_item_features(user, item)

                    TX.append(features)

                    Ty.append(relevance)

        return np.asarray(TX), np.asarray(Ty), np.asarray(Tqids)

    def features(self, training, test, validation=None):

        self.pop_dict = compute_most_pop_dict(training, threshold=self.threshold)

        # reads .dat format
        self._parse_data(training, test, validation=validation)

        if self.validation:

            user_item_features = Parallel(n_jobs=3)(delayed(self._compute_features)
                                  (data, test)
                                  for data, test in [(training, False), (test, True), (validation, True)])

            x_train, y_train, qids_train = user_item_features[0]

            x_test, y_test, qids_test = user_item_features[1]

            x_val, y_val, qids_val = user_item_features[2]

        else:

            user_item_features = Parallel(n_jobs=2)(delayed(self._compute_features)
                                  (data, test)
                                  for data, test in [(training, False), (test, True)])

            x_train, y_train, qids_train = user_item_features[0]

            x_test, y_test, qids_test = user_item_features[1]

            x_val, y_val, qids_val = None, None, None

        return x_train, y_train, qids_train, x_test, y_test, qids_test, x_val, y_val, qids_val

    def evaluate(self, x_test, y_test, qids_test):

        preds = x_test

        for name, metric in self.metrics.items():

            if name != 'fit':
                print('%s-----%f\n' % (name, metric.calc_mean(qids_test, y_test, preds)))


if __name__ == '__main__':

    start_time = time.time()

    print('Starting MostPop...')

    args = parse_args()

    rec = MostPop(args.dataset)

    x_train, y_train, qids_train, x_test, y_test, qids_test, x_val, y_val, qids_val = rec.features(args.train, args.test,
                                                                                                   validation=args.validation)

    print('Finished computing features after %s seconds' % (time.time() - start_time))

    rec.evaluate(x_test, y_test, qids_test)  # evaluates the model on the test set

    print("--- %s seconds ---" % (time.time() - start_time))
