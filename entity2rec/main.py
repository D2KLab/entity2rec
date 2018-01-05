from entity2rec import Entity2Rec
from heuristic_combinator import HeuristicCombinator
import time
import argparse


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

    parser.add_argument('--train', dest='train', help='train')

    parser.add_argument('--test', dest='test', help='test')

    parser.add_argument('--validation', dest='validation', default=False, help='validation')

    parser.add_argument('--run_all', dest='run_all', action='store_true', default=False,
                        help='If computing also the embeddings')

    parser.add_argument('--implicit', dest='implicit', action='store_true', default=False,
                        help='Implicit feedback with boolean values')

    parser.add_argument('--feedback_file', dest='feedback_file', default=False,
                        help='Path to a DAT file that contains all the couples user-item')

    parser.add_argument('--all_items', dest='all_unrated_items', action='store_false', default=True,
                        help='Whether keeping the rated items of the training set as candidates. '
                             'Default is AllUnratedItems')

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


start_time = time.time()

print('Starting entity2rec...')

args = parse_args()

rec = Entity2Rec(args.dataset, p=args.p, q=args.q,
                 implicit=args.implicit, feedback_file=args.feedback_file, walk_length=args.walk_length,
                 num_walks=args.num_walks, dimensions=args.dimensions, window_size=args.window_size,
                 workers=args.workers, iterations=args.iter,
                 all_unrated_items=args.all_unrated_items, threshold=args.threshold)

heuristic_rec = HeuristicCombinator(args.dataset, p=args.p, q=args.q,
                 walk_length=args.walk_length,
                 num_walks=args.num_walks, dimensions=args.dimensions, window_size=args.window_size,
                 iterations=args.iter,
                 all_unrated_items=args.all_unrated_items, threshold=args.threshold)

if args.write_features:

    rec.feature_generator(run_all=args.run_all)  # writes features to file with SVM format

else:

    if args.read_features:  # reads features from SVM format

        x_train, y_train, qids_train, x_test, y_test, qids_test, x_val, y_val, qids_val = rec.read_features()

    else:
        x_train, y_train, qids_train, x_test, y_test, qids_test,\
        x_val, y_val, qids_val = rec.features(args.train, args.test,
                                              validation=args.validation, run_all=args.run_all)

        print('Finished computing features after %s seconds' % (time.time() - start_time))
        print('Starting to fit the model...')

    rec.fit(x_train, y_train, qids_train,
            x_val=x_val, y_val=y_val, qids_val=qids_val, optimize=args.metric, N=args.N)  # train the model

    print('Finished fitting the model after %s seconds' % (time.time() - start_time))

    rec.evaluate(x_test, y_test, qids_test)  # evaluates the model on the test set

    rec.evaluate_heuristics(x_test, y_test, qids_test)  # evaluates the heuristics on the test set

print("--- %s seconds ---" % (time.time() - start_time))
