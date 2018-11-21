from __future__ import print_function
import json
from os.path import isfile, join
from os import makedirs
import argparse
from node2vec import Node2Vec
import time
import shutil


class Entity2Vec(Node2Vec):

    """Generates a set of property-specific entity embeddings from a Knowledge Graph"""

    def __init__(self, is_directed, preprocessing, is_weighted, p, q, walk_length, num_walks, dimensions, window_size,
                 workers, iterations, feedback_file):

        Node2Vec.__init__(self, is_directed, preprocessing, is_weighted, p, q, walk_length, num_walks, dimensions,
                          window_size, workers, iterations)

        self.feedback_file = feedback_file

    def e2v_walks_learn(self, properties_names, dataset):

        n = self.num_walks

        p = int(self.p)

        q = int(self.q)

        l = self.walk_length

        d = self.dimensions

        it = self.iter

        win = self.window_size

        try:

            makedirs('emb/%s' % dataset)

        except:
            pass

        # copy define feedback_file, if declared
        if self.feedback_file:
            print('Copying feedback file %s' % self.feedback_file)
            shutil.copy2(self.feedback_file, "datasets/%s/graphs/feedback.edgelist" % dataset)

        # iterate through properties

        for prop_name in properties_names:

            print(prop_name)

            prop_short = prop_name

            if '/' in prop_name:

                prop_short = prop_name.split('/')[-1]

            graph = "datasets/%s/graphs/%s.edgelist" % (dataset, prop_short)

            try:
                makedirs('emb/%s/%s' % (dataset, prop_short))

            except:
                pass

            emb_output = "emb/%s/%s/num%d_p%d_q%d_l%d_d%d_iter%d_winsize%d.emd" % (dataset,
                                                                                   prop_short, n, p, q, l, d, it, win)

            if not isfile(emb_output):  # check if embedding file already exists

                print('running with', graph)

                super(Entity2Vec, self).run(graph, emb_output)  # call the run function defined in parent class node2vec

            else:

                print('Embedding file already exist, going to next property...')

                continue

    @staticmethod
    def parse_args():

        """
        Parses the entity2vec arguments.
        """

        parser = argparse.ArgumentParser(description="Run entity2vec.")

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

        parser.add_argument('--dataset', nargs='?', default='movielens_1m',
                            help='Dataset')

        parser.add_argument('--feedback_file', dest='feedback_file', default=False,
                            help='Path to a DAT file that contains all the couples user-item')

        return parser.parse_args()


if __name__ == '__main__':

    start_time = time.time()

    args = Entity2Vec.parse_args()

    print('Parameters:\n')

    print('walk length = %d\n' % args.walk_length)

    print('number of walks per entity = %d\n' % args.num_walks)

    print('p = %s\n' % args.p)

    print('q = %s\n' % args.q)

    print('weighted = %s\n' % args.weighted)

    print('directed = %s\n' % args.directed)

    print('no_preprocessing = %s\n' % args.preprocessing)

    print('dimensions = %s\n' % args.dimensions)

    print('iterations = %s\n' % args.iter)

    print('window size = %s\n' % args.window_size)

    print('workers = %s\n' % args.workers)

    print('config_file = %s\n' % args.config_file)

    print('dataset = %s\n' % args.dataset)

    print('feedback file = %s\n' % args.feedback_file)

    e2v = Entity2Vec(args.directed, args.preprocessing, args.weighted, args.p, args.q, args.walk_length, args.num_walks,
                     args.dimensions, args.window_size, args.workers, args.iter, args.config_file,
                     args.dataset, args.feedback_file)

    e2v.e2v_walks_learn()

    print("--- %s seconds ---" % (time.time() - start_time))
