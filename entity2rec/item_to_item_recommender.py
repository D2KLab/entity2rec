import time
import numpy as np
import pickle
import argparse
from parse_args import parse_args
from evaluator import Evaluator


class ItemToItemRecommender:

    def __init__(self, algorithm, dataset):

        with open('datasets/'+dataset+'/item_to_item_similarity_'+algorithm, 'rb') as f1:

            self.model = pickle.load(f1)

    def compute_user_item_features(self, user, item, items_liked_by_user, users_liking_the_item):
        
        # randomly choose one item

        if len(items_liked_by_user) > 0:

            seed_item = np.random.choice(items_liked_by_user, 1)[0]            

            try:

                features = [self.model[seed_item][item]]
            except KeyError:
                features = [0.]

        else:
            features = [0.]

        return features

    def fit(self, x_train, y_train, qids_train):

        return 0

    def predict(self, x_test, qids_test):

        preds = x_test

        return preds


if __name__ == '__main__':

    np.random.seed(1)  # fixed seed for reproducibility

    start_time = time.time()

    args = parse_args()

    print('Starting item to item recommender..')

    if not args.train:
        args.train = 'datasets/' + args.dataset + '/train.dat'

    if not args.test:
        args.test = 'datasets/' + args.dataset + '/test.dat'

    if not args.validation:
        args.validation = 'datasets/' + args.dataset + '/val.dat'

    rec = "Entity2Rec"

    # initialize evaluator

    if args.dataset == 'LastFM':
        implicit = True

    else:
        implicit = args.implicit

    if args.dataset == 'LibraryThing':
        threshold = 8
    else:
        threshold = args.threshold

    evaluat = Evaluator(implicit=implicit, threshold=threshold, all_unrated_items=args.all_unrated_items)

    itemrec = ItemToItemRecommender(rec, args.dataset)

    # compute features
    x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test, x_val, y_val, qids_val, items_val = evaluat.features(
        itemrec, args.train, args.test,
        validation=False,
        n_jobs=args.workers, supervised=False, n_users=args.num_users)

    print('Finished computing features after %s seconds' % (time.time() - start_time))

    scores = evaluat.evaluate(itemrec, x_test, y_test, qids_test, items_test,
                              write_to_file="results/%s/item_to_item_similarity/%s" % (args.dataset, rec))

    print(scores)

    print("--- %s seconds ---" % (time.time() - start_time))
