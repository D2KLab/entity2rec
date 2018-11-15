import time
import numpy as np
from evaluator import Evaluator
from surprise import SVD, KNNBaseline, NMF
from surprise import Reader
from surprise import Dataset
import os
from parse_args import parse_args
import random
import numpy as np
import turicreate as tc

class ItemKNNSimilarity:

    name = 'ItemKNNturi'

    def __init__(self, dataset, implicit):

        self.dataset = dataset

        data = tc.SFrame.read_csv('datasets/'+'%s/FM/' %self.dataset
                                  +'train.dat', delimiter=' ')

        if implicit:

            self.model = tc.item_similarity_recommender.create(data,
                        user_id='user_id', item_id='item_id')

        else:

            self.model = tc.item_similarity_recommender.create(data,
                        user_id='user_id', item_id='item_id', target='rating')
        
        similarities = self.model.get_similar_items()
        
        self.sim_matrix = {}

        for s in similarities:

            self.sim_matrix[s['item_id'], s['similar']] = s['score']

    def collab_similarities(self, item_1, item_2):

        try:

            s = self.sim_matrix[item_1, item_2]

        except KeyError: 

            s = 0.

        return s


if __name__ == '__main__':

    random.seed(1)  # fixed seed for reproducibility

    start_time = time.time()

    print('Starting ItemKNNSimilarity...')

    args = parse_args()

    # default settings

    if not args.train:
        args.train = 'datasets/'+args.dataset+'/train.dat'

    if not args.test:
        args.test = 'datasets/'+args.dataset+'/test.dat'

    if not args.validation:
        args.validation = 'datasets/'+args.dataset+'/val.dat'

    if args.dataset == 'LastFM':

        implicit = True

    else:

        implicit = args.implicit

    if args.dataset == 'LibraryThing':

        threshold = 8

    else:

        threshold = args.threshold

    ItemKNNSim = ItemKNNSimilarity(args.dataset, implicit)

    # initialize evaluator

    evaluat = Evaluator(implicit=implicit, threshold=args.threshold, all_unrated_items=False)


    evaluat.compute_item_to_item_similarity(ItemKNNSim, args.train, args.test, args.dataset, validation=args.validation,
                                            n_users=args.num_users, n_jobs=args.workers, max_n_feedback=args.max_n_feedback)


    print("--- %s seconds ---" % (time.time() - start_time))
