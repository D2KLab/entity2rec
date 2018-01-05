from entity2rec import Entity2Rec
import time
import argparse
import numpy as np


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


class HeuristicCombinator(Entity2Rec):

    def __init__(self, dataset, num_walks=500, iterations=15, p=1, q=4):

        Entity2Rec.__init__(self, dataset, num_walks=num_walks, iterations=iterations, p=p, q=q)

    def features(self, training, test, validation=None, run_all=False):

        # reads .dat format
        self._parse_data(training, test, validation=validation)

        # run entity2vec to create the embeddings
        if run_all:
            print('Running entity2vec to generate property-specific embeddings...')
            self.e2v_walks_learn()  # run entity2vec

        # reads the embedding files
        self._set_embedding_files()

        x_test, y_test, qids_test = self._compute_features(test, True)

        return x_test, y_test, qids_test

    def evaluate(self, x_test, y_test, qids_test):

        preds_average = list(map(lambda x: np.mean(x), x_test))  # average of the relatedness scores

        preds_max = list(map(lambda x: np.max(x), x_test))  # max of the relatedness scores

        preds_min = list(map(lambda x: np.min(x), x_test))  # min of the relatedness scores

        print('Average:')

        for name, metric in self.metrics.items():

            if name != 'fit':

                print('%s-----%f\n' % (name, metric.calc_mean(qids_test, y_test, preds_average)))

        print('Min:')

        for name, metric in self.metrics.items():

            if name != 'fit':

                print('%s-----%f\n' % (name, metric.calc_mean(qids_test, y_test, preds_min)))

        print('Max:')

        for name, metric in self.metrics.items():

            if name != 'fit':

                print('%s-----%f\n' % (name, metric.calc_mean(qids_test, y_test, preds_max)))


if __name__ == '__main__':

    start_time = time.time()

    print('Starting Heurstic combinators baselines...')

    args = parse_args()

    rec = HeuristicCombinator(args.dataset, num_walks=args.num_walks,
                              iterations=args.iter, p=args.p, q=args.q)

    x_test, y_test, qids_test = rec.features(args.train, args.test)

    print('Finished fitting the model after %s seconds' % (time.time() - start_time))

    rec.evaluate(x_test, y_test, qids_test)  # evaluates the model on the test set

    print("--- %s seconds ---" % (time.time() - start_time))
