from __future__ import print_function
import optparse
import os
import codecs
import collections
import numpy as np
from gensim.models import Word2Vec
from pandas import read_json
from entity2vec import entity2vec
from entity2rel import entity2rel
import argparse
import time
from random import shuffle

###############################################################################################################################################
## Computes a set of relatedness scores between user-item pairs from a set of property-specific Knowledge Graph embeddings and user feedback ##
###############################################################################################################################################

class entity2rec(entity2vec, entity2rel):

    def __init__(self, is_directed, preprocessing, is_weighted, p, q, walk_length, num_walks, dimensions, window_size, workers, iterations, config, sparql, dataset, entities, default_graph, training, test, implicit, entity_class, feedback_file):

        entity2vec.__init__(self, is_directed, preprocessing, is_weighted, p, q, walk_length, num_walks, dimensions, window_size, workers, iterations, config, sparql, dataset, entities, default_graph, entity_class, feedback_file)

        entity2rel.__init__(self, True) #binary format embeddings

        self.training = training

        self.test = test

        self.implicit = implicit

        self._get_items_liked_by_user() #defines the dictionary of items liked by each user in the training set

        self._get_all_items() #define all the items that can be used as candidates for the recommandations


    def _get_embedding_files(self):

        for prop in self.properties:
            prop_short = prop
            if '/' in prop:
                prop_short = prop.split('/')[-1]

            self.add_embedding(u'emb/%s/%s/num%s_p%d_q%d_l%s_d%s_iter%d_winsize%d.emd' % (
            self.dataset, prop_short, self.num_walks, int(self.p), int(self.q), self.walk_length, self.dimensions, self.iter,
            self.window_size))

    def _get_items_liked_by_user(self):

        self.all_train_items = []

        self.items_liked_by_user_dict = collections.defaultdict(list)

        self.items_ratings_by_user_test = {}

        self.items_rated_by_user_train = collections.defaultdict(list)

        with codecs.open(self.training,'r', encoding='utf-8') as train:

            for line in train:

                line = line.split(' ')

                u = line[0]

                item = line[1]

                relevance = int(line[2])

                self.items_rated_by_user_train[u].append(item)

                self.items_ratings_by_user_test[(u,item)] = relevance #independently from the rating

                if self.implicit == False and relevance >= 4: #only relevant items are used to compute the similarity, rel = 5 in a previous work

                    self.items_liked_by_user_dict[u].append(item)

                elif self.implicit == True and relevance == 1:

                    self.items_liked_by_user_dict[u].append(item)

                self.all_train_items.append(item)

        self.all_train_items = list(set(self.all_train_items)) #remove duplicates


    def _get_all_items(self):

        self.all_items = []

        if self.entities != "all": #if it has been provided a list of items as an external file, read from it

            del self.all_train_items #free memory space

            with codecs.open(self.entities, 'r', encoding='utf-8') as items:

                for item in items:

                    item = item.strip('\n')

                    self.all_items.append(item)

        else: #otherwise join the items from the train and test set

            with codecs.open(self.test,'r', encoding='utf-8') as test:

                test_items = []

                for line in test:

                    line = line.split(' ')

                    u = line[0]

                    item = line[1]

                    relevance = int(line[2])

                    test_items.append(item)

                    self.items_ratings_by_user_test[(u,item)] = relevance

                self.all_items = list(set(self.all_train_items+test_items)) #merge lists and remove duplicates

                del self.all_train_items


    def collab_similarity(self, user, item):

        #all other properties

        return self.relatedness_score_by_position(user, item, -1)

    def content_similarities(self, user, item):
        
        #all other properties

        items_liked_by_user = self.items_liked_by_user_dict[user]

        sims = []
        
        for past_item in items_liked_by_user:

            sims.append(self.relatedness_scores(past_item,item, -1)) #append a list of property-specific scores, skip feedback
        
        if len(sims) == 0:
            sims = 0.5*np.ones(len(self.properties) - 1)
            return sims

        return np.mean(sims, axis = 0) #return a list of averages of property-specific scores


    @staticmethod
    def parse_user_id(user):

        return int(user.strip('user')) #29

    def parse_users_items_rel(self,line):

            line = line.split(' ')

            user = line[0] #user29

            user_id = entity2rec.parse_user_id(user) #29

            item = line[1] #http://dbpedia.org/resource/The_Golden_Child

            relevance = int(line[2]) #5

            #binarization of the relevance values
            if self.implicit == False:
                relevance = 1 if relevance >= 4 else 0

            return (user, user_id, item, relevance)


    def write_line(self,user, user_id, item, relevance, file):

        file.write('%d qid:%d' %(relevance,user_id))

        count = 1

        collab_score = self.collab_similarity(user, item)

        file.write(' %d:%f' %(count,collab_score))

        count += 1

        content_scores = self.content_similarities(user, item)

        l = len(content_scores)

        for content_score in content_scores:

            if count == l + 1: #last score, end of line

                file.write(' %d:%f # %s\n' %(count,content_score,item))

            else:

                file.write(' %d:%f' %(count,content_score))

                count += 1



    def get_candidates(self,user):

        #get candidates according to the all unrated items protocol
        #use as candidates all the the items that are not in the training set

        rated_items_train = self.items_rated_by_user_train[user] #both in the train and in the test

        candidate_items = [item for item in self.all_items if item not in rated_items_train] #all unrated items in the train

        return candidate_items


    def feature_generator(self):

        #write training set

        start_time = time.time()

        train_name = ((self.training).split('/')[-1]).split('.')[0]

        feature_path = 'features/%s/p%d_q%d/' %(self.dataset,int(self.p), int(self.q))

        try:
            os.makedirs(feature_path)
        except:
            pass

        feature_file = feature_path + '%s_p%d_q%d.svm' %(train_name, int(self.p), int(self.q))

        with codecs.open(feature_file,'w', encoding='utf-8') as train_write:

            with codecs.open(self.training,'r', encoding='utf-8') as training:

                for i, line in enumerate(training):

                    user, user_id, item, relevance = self.parse_users_items_rel(line)

                    print(user)

                    self.write_line(user, user_id, item, relevance, train_write)

        print('finished writing training')

        print("--- %s seconds ---" % (time.time() - start_time))

        #write test set

        test_name = ((self.test).split('/')[-1]).split('.')[0]

        feature_file = feature_path + '%s_p%d_q%d.svm' %(test_name, int(self.p), int(self.q))

        with codecs.open(feature_file, 'w', encoding='utf-8') as test_write:

            for user in self.items_rated_by_user_train.keys():

                #write some candidate items

                print(user)

                user_id = entity2rec.parse_user_id(user)

                candidate_items = self.get_candidates(user)

                shuffle(candidate_items) #relevant and non relevant items are shuffled

                for item in candidate_items:

                    try:
                        rel = int(self.items_ratings_by_user_test[(user,item)]) #get the relevance score if it's in the test

                        if self.implicit == False:
                            rel = 1 if rel >= 4 else 0

                    except KeyError:
                        rel = 0 #unrated items are assumed to be negative

                    self.write_line(user, user_id, item, rel, test_write)

        print('finished writing test')

        print("--- %s seconds ---" % (time.time() - start_time))


    def run(self, run_all):

        if run_all:
            super(entity2rec, self).run()
            self._get_embedding_files()

        else:
            self._get_embedding_files()
            self.feature_generator()


    @staticmethod
    def parse_args():

        '''
        Parses the entity2vec arguments.
        '''

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

        parser.add_argument('--no_preprocessing', dest = 'preprocessing', action='store_false',
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

        parser.add_argument('--sparql', dest = 'sparql',
                            help='Whether downloading the graphs from a sparql endpoint')
        parser.set_defaults(sparql=False)

        parser.add_argument('--entities', dest = 'entities', default = "all",
                            help='A specific list of entities for which the embeddings have to be computed')


        parser.add_argument('--default_graph', dest = 'default_graph', default = False,
                            help='Default graph to query when using a Sparql endpoint')

        parser.add_argument('--train', dest='train', help='train', default = False)

        parser.add_argument('--test', dest='test', help='test')

        parser.add_argument('--run_all', dest='run_all', default = False, help='If computing also the embeddings')

        parser.add_argument('--implicit', dest='implicit', default = False, help='Implicit feedback with boolean values')

        parser.add_argument('--entity_class', dest = 'entity_class', help = 'entity class', default = False)

        parser.add_argument('--feedback_file', dest = 'feedback_file', default = False,
                            help='Path to a DAT file that contains all the couples user-item')

        return parser.parse_args()


if __name__ == '__main__':


    start_time = time.time()

    args = entity2rec.parse_args()

    rec = entity2rec(args.directed, args.preprocessing, args.weighted, args.p, args.q, args.walk_length, args.num_walks, args.dimensions, args.window_size, args.workers, args.iter, args.config_file, args.sparql, args.dataset, args.entities, args.default_graph, args.train, args.test, args.implicit, args.entity_class, args.feedback_file)

    rec.run(args.run_all)

    print("--- %s seconds ---" % (time.time() - start_time))