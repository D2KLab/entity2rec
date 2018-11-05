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

class ItemKNNSimilarity:

    name = 'ItemKNNSimilarity'

    def __init__(self, dataset, train, implicit, threshold):

        sim_options = {'name': 'cosine',

                       'user_based': False  # compute  similarities between items
                       }

        self.algorithm = KNNBaseline(sim_options=sim_options)

        self.train = train

        self.dataset = dataset

        self.item_to_ind = {}

        self.user_to_ind = {}

        self.implicit = implicit

        self.threshold = threshold

        self.model = self.learn_model_surprise()

        self.sim_matrix = self.model.compute_similarities()

        self.avg_s = np.mean(self.sim_matrix)

    def learn_model_surprise(self):

        file_path = os.path.expanduser(self.train)

        if self.implicit:

            rating_scale = (0,1)

        else:
            rating_scale = (1, (self.threshold*5)//4)

        reader = Reader(line_format='user item rating timestamp', sep=' ', rating_scale=rating_scale)

        data = Dataset.load_from_file(file_path, reader=reader)

        algo = self.algorithm

        self.trainset = data.build_full_trainset()

        algo.train(self.trainset)

        return algo

    def collab_similarities(self, item_1, item_2):

        try:

            item_1_ind = self.trainset.to_inner_iid(item_1)

            item_2_ind = self.trainset.to_inner_iid(item_2)

            s = self.sim_matrix[item_1_ind, item_2_ind]

        except ValueError:  # missing item in the training set

            s = self.avg_s

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

    ItemKNNSim = ItemKNNSimilarity(args.dataset, args.train, implicit, threshold)

    # initialize evaluator

    evaluat = Evaluator(implicit=implicit, threshold=args.threshold, all_unrated_items=False)


    evaluat.compute_item_to_item_similarity(ItemKNNSim, args.train, args.test, args.dataset, validation=args.validation,
                                            n_users=args.num_users, n_jobs=args.workers, max_n_feedback=args.max_n_feedback)


    print("--- %s seconds ---" % (time.time() - start_time))
