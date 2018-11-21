import argparse


def parse_args():

    """
    Parses the entity2rec arguments.
    """

    parser = argparse.ArgumentParser(description="Run entity2rec.")

    parser.add_argument('--walk_length', type=int, default=100,
                        help='Length of walk per source')

    parser.add_argument('--num_walks', type=int, default=50,
                        help='Number of walks per source. Default is 40.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='In-out hyperparameter. Default is 1.')

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

    parser.add_argument('--dimensions', type=int, default=200,
                        help='Number of dimensions. Default is 200.')

    parser.add_argument('--window_size', type=int, default=30,
                        help='Context size for optimization. Default is 30.')

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

    parser.add_argument('--run_all', dest='run_all', action='store_true', default=False,
                        help='If computing also the embeddings')

    parser.add_argument('--implicit', dest='implicit', action='store_true', default=False,
                        help='Implicit feedback with boolean values')

    parser.add_argument('--feedback_file', dest='feedback_file', default=False,
                        help='Path to a DAT file that contains all the couples user-item')

    parser.add_argument('--all_items', dest='all_unrated_items', action='store_false', default=True,
                        help='Whether keeping the rated items of the training set as candidates. '
                             'Default is AllUnratedItems')

    parser.add_argument('--collab_only', dest='collab_only', action='store_true', default=False,
                        help='Only use collab filtering')

    parser.add_argument('--content_only', dest='content_only', action='store_true', default=False,
                        help='Only use content filtering')

    parser.add_argument('--write_features', dest='write_features', action='store_true', default=False,
                        help='Writes the features to file')

    parser.add_argument('--read_features', dest='read_features', action='store_true', default=False,
                        help='Reads the features from a file')

    parser.add_argument('--metric', dest='metric', default='P',
                        help='Metric to optimize in the training')

    parser.add_argument('--N', dest='N', type=int, default=5,
                        help='Cutoff to estimate metric')

    parser.add_argument('--threshold', dest='threshold', default=4, type=int,
                        help='Threshold to convert ratings into binary feedback')

    parser.add_argument('--num_users', dest='num_users', type=int, default=False,
                        help='Sample of users for evaluation')

    parser.add_argument('--max_n_feedback', dest='max_n_feedback', type=int, default=False,
                        help='Only select users with less than max_n_feedback for training and evaluation')

    parser.add_argument('--user_clusters', dest='user_clusters', type=int, default=False,
                        help='Cluster users and fit several models')

    return parser.parse_args()
