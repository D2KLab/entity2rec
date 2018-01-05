from entity2rec import Entity2Rec
import time
import argparse
import numpy as np
from collections import defaultdict
import networkx as nx
from mostpop2rec import compute_most_pop_dict
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


class Freq2Rec(Entity2Rec):

    def __init__(self, dataset):

        Entity2Rec.__init__(self, dataset)

        self.items_property_values = defaultdict(list)

        self.pop_dict = {}

        self._read_items_property()

    def _read_items_property(self):

        for property in self.properties:

            G = nx.read_edgelist('datasets/%s/graphs/%s.edgelist' %(self.dataset, property), create_using=nx.DiGraph(), edgetype=str)

            for item, value in G.edges():

                self.items_property_values[(item, property)].append(value)

    def collab_similarity(self, user, item):

        # feedback property

        return self.pop_dict[item]

    def content_similarities(self, user, item):

        sims = []

        for property in self.properties:

            sims.append(self.content_similarity(user,item,property))

        return sims

    def content_similarity(self, user, item, property):

        """
        Average fraction of overlapping values according to a specific property
        between target item and items liked by user in the past
        """

        items_liked_by_user = self.items_liked_by_user_dict[user]

        sims = []

        for past_item in items_liked_by_user:

            past_values = self.items_property_values[past_item, property]

            item_values = self.items_property_values[item, property]

            common_values = list(set(past_values) & set(item_values))
            print(past_values, item_values, common_values)
            try:

                frac_common_values = float(len(common_values)) / min(len(item_values), len(past_values))

            except ZeroDivisionError:
                frac_common_values = 0.

            sims.append(frac_common_values)  # append a list of property-specific scores, skip feedback

        return np.mean(sims, axis=0)  # return a list of averages of property-specific scores

    def features(self, training, test, validation=None):

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


if __name__ == '__main__':

    start_time = time.time()

    print('Starting MostPop2rec...')

    args = parse_args()

    rec = Freq2Rec(args.dataset)

    rec.pop_dict = compute_most_pop_dict(args.train, threshold=args.threshold)

    x_train, y_train, qids_train, x_test, y_test, qids_test, x_val, y_val, qids_val = rec.features(args.train, args.test,
                                                                                                   validation=args.validation)

    print('Finished computing features after %s seconds' % (time.time() - start_time))
    print('Starting to fit the model...')

    rec.fit(x_train, y_train, qids_train, x_val=x_val, y_val=y_val, qids_val=qids_val)  # train the model

    print('Finished fitting the model after %s seconds' % (time.time() - start_time))

    rec.evaluate(x_test, y_test, qids_test)  # evaluates the model on the test set

    print("--- %s seconds ---" % (time.time() - start_time))
