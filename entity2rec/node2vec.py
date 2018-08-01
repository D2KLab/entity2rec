from __future__ import print_function
import numpy as np
import networkx as nx
import random
import time
import argparse
from gensim.models import Word2Vec


class Node2Vec(object):
    def __init__(self, is_directed, preprocessing, is_weighted, p, q, walk_length, num_walks, dimensions, window_size,
                 workers, iterations):

        # graph properties
        self.is_directed = is_directed
        self.preprocessing = preprocessing
        self.is_weighted = is_weighted
        self.G = None
        self.alias_nodes = None
        self.alias_edges = None

        # walk properties
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.num_walks = num_walks

        # learning properties
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = workers
        self.iter = iterations

    def read_graph(self, nx_g):

        if self.is_weighted:

            self.G = nx.read_edgelist(nx_g, data=(('weight', float),), create_using=nx.DiGraph(), edgetype=str)

        else:

            self.G = nx.read_edgelist(nx_g, create_using=nx.DiGraph(), edgetype=str)

            for edge in self.G.edges():
                self.G[edge[0]][edge[1]]['weight'] = 1

        if not self.is_directed:
            self.G = self.G.to_undirected()

    def node2vec_walk(self, start_node):
        """
        Simulate a random walk starting from start node.
        """

        G = self.G

        walk = [start_node]

        while len(walk) < self.walk_length:

            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))

            if len(cur_nbrs) > 0:

                if self.preprocessing:

                    alias_nodes = self.alias_nodes
                    alias_edges = self.alias_edges

                    if len(walk) == 1:  # first step of the walk, no previous node

                        walk.append(cur_nbrs[Node2Vec.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])

                    else:
                        prev = walk[-2]

                        next_node = cur_nbrs[Node2Vec.alias_draw(alias_edges[(prev, cur)][0],
                                                                 alias_edges[(prev, cur)][1])]

                        walk.append(next_node)

                else:

                    p = self.p
                    q = self.q
                    G = self.G

                    unnormalized_probs = []

                    if len(walk) == 1:  # first step of the walk, no previous node

                        for dst_nbr in cur_nbrs:
                            unnormalized_probs.append(G[cur][dst_nbr]['weight'])

                        norm_const = sum(unnormalized_probs)

                        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

                        next_node = cur_nbrs[np.random.multinomial(1, normalized_probs).argmax()]

                        walk.append(next_node)

                    else:

                        prev = walk[-2]

                        for dst_nbr in cur_nbrs:

                            if dst_nbr == prev:

                                unnormalized_probs.append(G[cur][dst_nbr]['weight'] / p)

                            elif G.has_edge(dst_nbr, prev):
                                unnormalized_probs.append(G[cur][dst_nbr]['weight'])
                            else:
                                unnormalized_probs.append(G[cur][dst_nbr]['weight'] / q)

                        norm_const = sum(unnormalized_probs)
                        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

                        next_node = cur_nbrs[np.random.multinomial(1, normalized_probs).argmax()]

                        walk.append(next_node)
            else:
                break

        return walk

    def learn_embeddings(self, output):
        """
        Learn embeddings by optimizing the Skipgram objective using SGD.
        """

        walks = self._simulate_walks()  # simulate random walks

        model = Word2Vec(walks, size=self.dimensions, window=self.window_size, min_count=0,
                         workers=self.workers, iter=self.iter, negative=25, sg=1)

        print("defined model using w2v")

        model.wv.save_word2vec_format(output, binary=True)

        # free memory
        del walks
        self.alias_nodes = None
        self.alias_edges = None
        self.G = None

        print("saved model in word2vec binary format")

        return

    def get_alias_edge(self, src, dst):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight'] / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return Node2Vec.alias_setup(normalized_probs)

    def _simulate_walks(self):

        """
        Simulate random walks from each node.
        """
        G = self.G
        nodes = list(G.nodes())

        walks = []

        print('Walk iteration:')

        for walk_iter in range(self.num_walks):

            print(str(walk_iter + 1), '/', str(self.num_walks))
            random.shuffle(nodes)

            c = 1

            for node in nodes:

                if c % 10001 == 0:
                    print('Processed %d nodes' % c)

                c += 1

                walks.append(self.node2vec_walk(start_node=node))

        return walks

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """

        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = Node2Vec.alias_setup(normalized_probs)

        alias_edges = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return

    @staticmethod
    def alias_setup(probs):
        """
        Compute utility lists for non-uniform sampling from discrete distributions.
        Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        for details
        """
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    @staticmethod
    def alias_draw(j, q):
        """
        Draw sample from a non-uniform discrete distribution using alias sampling.
        """
        K = len(j)

        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return j[kk]

    @staticmethod
    def parse_args():
        '''
        Parses the node2vec arguments.
        '''
        parser = argparse.ArgumentParser(description="Run node2vec.")

        parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
                            help='Input graph path')

        parser.add_argument('--output', nargs='?', default='walks.txt.gz',
                            help='emb file name')

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
        parser.set_defaults(weighted=False)

        parser.add_argument('--directed', dest='directed', action='store_true',
                            help='Graph is (un)directed. Default is directed.')
        parser.set_defaults(directed=False)

        parser.add_argument('--no_preprocessing', dest='preprocessing', action='store_false',
                            help='Whether preprocess all transition probabilities or compute on the fly')
        parser.set_defaults(preprocessing=True)

        parser.add_argument('--dimensions', type=int, default=500,
                            help='Number of dimensions. Default is 128.')

        parser.add_argument('--window-size', type=int, default=5,
                            help='Context size for optimization. Default is 10.')

        parser.add_argument('--iter', default=5, type=int,
                            help='Number of epochs in SGD')

        parser.add_argument('--workers', type=int, default=8,
                            help='Number of parallel workers. Default is 8.')

        return parser.parse_args()

    def run(self, input_graph, output):

        self.read_graph(input_graph)

        print('read G')

        if self.preprocessing:
            self.preprocess_transition_probs()
            print('preprocessed')

        self.learn_embeddings(output)


if __name__ == '__main__':
    start_time = time.time()

    args = Node2Vec.parse_args()

    print('Parameters:\n')

    print('input = %s\n' % args.input)

    print('output = %s\n' % args.output)

    print('walk length = %d\n' % args.walk_length)

    print('number of walks per entity = %d\n' % args.num_walks)

    print('p = %s\n' % args.p)

    print('q = %s\n' % args.q)

    print('weighted = %s\n' % args.weighted)

    print('directed = %s\n' % args.directed)

    print('preprocessing = %s\n' % args.preprocessing)

    print('dimensions = %s\n' % args.dimensions)

    print('iterations = %s\n' % args.iter)

    print('window size = %s\n' % args.window_size)

    print('workers = %s\n' % args.workers)

    node2vec_graph = Node2Vec(args.directed, args.preprocessing, args.weighted, args.p, args.q, args.walk_length,
                              args.num_walks, args.dimensions, args.window_size, args.workers, args.iter)

    node2vec_graph.run(args.input, args.output)

    print("--- %s seconds ---" % (time.time() - start_time))
