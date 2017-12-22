from __future__ import print_function
import json
from os.path import isfile, join
from os import makedirs
from os import listdir
import argparse
from node2vec import Node2Vec
import time
import codecs
from sparql import Sparql
import shutil


class Entity2Vec(Node2Vec):

    """Generates a set of property-specific entity embeddings from a Knowledge Graph"""

    def __init__(self, is_directed, preprocessing, is_weighted, p, q, walk_length, num_walks, dimensions, window_size,
                 workers, iterations, config, sparql, dataset, entities, default_graph, entity_class, feedback_file):

        Node2Vec.__init__(self, is_directed, preprocessing, is_weighted, p, q, walk_length, num_walks, dimensions,
                          window_size, workers, iterations)

        self.config_file = config

        self.sparql = sparql

        self.default_graph = default_graph

        self.dataset = dataset

        self.entities = entities

        self.entity_class = entity_class

        self.feedback_file = feedback_file

    def define_properties(self, entities=False):

        with codecs.open(self.config_file, 'r', encoding='utf-8') as config_read:

            property_file = json.loads(config_read.read())

        try:

            self.properties = [i for i in property_file[self.dataset]]
            self.properties.append('feedback')

        except KeyError:  # no list of properties specified in the config file

            if self.sparql:  # get all the properties from the sparql endpoint

                sparql_query = Sparql(entities, self.config_file, self.dataset, self.sparql, self.default_graph)

                self.properties = sparql_query.properties

                self.properties.append('feedback')  # add the feedback property that is not defined in the graph

            else:  # get everything you have in the folder

                path_to_graphs = 'datasets/%s/graphs' % self.dataset

                onlyfiles = [f for f in listdir(path_to_graphs) if isfile(join(path_to_graphs, f))]

                self.properties = [file.replace('.edgelist', '') for file in onlyfiles]

                if 'feedback' in self.properties:  # feedback property always the last one of the list
                    self.properties.remove('feedback')
                    self.properties.append('feedback')

    def e2v_walks_learn(self):

        n = self.num_walks

        p = int(self.p)

        q = int(self.q)

        l = self.walk_length

        d = self.dimensions

        it = self.iter

        win = self.window_size

        try:

            makedirs('emb/%s' % self.dataset)

        except:
            pass

        # copy define feedback_file, if declared
        if self.feedback_file:
            print('Copying feedback file %s' % self.feedback_file)
            shutil.copy2(self.feedback_file, "datasets/%s/graphs/feedback.edgelist" % self.dataset)

        # iterate through properties

        for prop_name in self.properties:

            print(prop_name)

            prop_short = prop_name

            if '/' in prop_name:
                prop_short = prop_name.split('/')[-1]

            graph = "datasets/%s/graphs/%s.edgelist" % (self.dataset, prop_short)

            try:
                makedirs('emb/%s/%s' % (self.dataset, prop_short))

            except:
                pass

            emb_output = "emb/%s/%s/num%d_p%d_q%d_l%d_d%d_iter%d_winsize%d.emd" % (self.dataset,
                                                                                   prop_short, n, p, q, l, d, it, win)

            print('running with', graph)

            super(Entity2Vec, self).run(graph, emb_output)  # call the run function defined in parent class node2vec

    # generate node2vec walks and learn embeddings for each property
    def run(self):

        if self.sparql:
            sparql_query = Sparql(self.entities, self.config_file, self.dataset, self.sparql, self.default_graph)

            sparql_query.get_property_graphs()

        self.e2v_walks_learn()  # run node2vec for each property-specific graph

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

        parser.add_argument('--sparql', dest='sparql',
                            help='Whether downloading the graphs from a sparql endpoint')
        parser.set_defaults(sparql=False)

        parser.add_argument('--entities', dest='entities', default="all",
                            help='A specific list of entities for which the embeddings have to be computed')

        parser.add_argument('--default_graph', dest='default_graph', default=False,
                            help='Default graph to query when using a Sparql endpoint')

        parser.add_argument('--entity_class', dest='entity_class', help='entity class', default=False)

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

    print('sparql endpoint = %s\n' % args.sparql)

    print('dataset = %s\n' % args.dataset)

    print('entities = %s\n' % args.entities)

    print('default graph = %s\n' % args.default_graph)

    print('entity class = %s\n' % args.entity_class)

    print('feedback file = %s\n' % args.feedback_file)

    e2v = Entity2Vec(args.directed, args.preprocessing, args.weighted, args.p, args.q, args.walk_length, args.num_walks,
                     args.dimensions, args.window_size, args.workers, args.iter, args.config_file, args.sparql,
                     args.dataset, args.entities, args.default_graph, args.entity_class, args.feedback_file)

    e2v.run()

    print("--- %s seconds ---" % (time.time() - start_time))
