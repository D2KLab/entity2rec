import turicreate as tc
import json
from evaluator import Evaluator
import time
import argparse


class TuriRankingFM:

    def __init__(self, dataset, implicit, dbpedia=False):

        self.dataset = dataset

        data = tc.SFrame.read_csv('datasets/'+'%s/FM/' %self.dataset
                                  +'train.dat', delimiter=' ')

        if dbpedia:

            items_data = tc.SFrame.read_csv('datasets/DB2Vec.txt', delimiter=' ', na_values='NAN')

        else:

            items_data = tc.SFrame.read_csv('datasets/'+'%s/FM/' %self.dataset
                                      +'items.dat', delimiter=' ', na_values='NAN')

        if implicit:

            self.model = tc.ranking_factorization_recommender.create(data,
                        user_id='user_id', item_id='item_id', item_data=items_data, random_seed=1)

        else:

            self.model = tc.ranking_factorization_recommender.create(data,
                        user_id='user_id', item_id='item_id', target='rating', item_data=items_data, random_seed=1)


    def compute_user_item_features(self, user, item, items_liked_by_user, users_liking_the_item):

        try:

            features = self.model.recommend(users=[user], items=item, k=len(item))  # user item relatedness from fm model

        except KeyError:  # do not have user item pair in embedding

            features = [0.]

        return features

    def fit(self, x_train, y_train, qids_train):

        return 0

    def predict(self, x_test, qids_test):

        preds = x_test

        return preds

    @staticmethod
    def parse_args():

        parser = argparse.ArgumentParser(description="Run entity2rec.")

        parser.add_argument('--dimensions', type=int, default=200,
                            help='Number of dimensions. Default is 200.')

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

        parser.add_argument('--all_items', dest='all_unrated_items', action='store_false', default=True,
                            help='Whether keeping the rated items of the training set as candidates. '
                                 'Default is AllUnratedItems')
        parser.add_argument('--implicit', dest='implicit', action='store_true', default=False,
                            help='Implicit feedback with boolean values')

        parser.add_argument('--write_features', dest='write_features', action='store_true', default=False,
                            help='Writes the features to file')

        parser.add_argument('--read_features', dest='read_features', action='store_true', default=False,
                            help='Reads the features from a file')

        parser.add_argument('--threshold', dest='threshold', default=4, type=int,
                            help='Threshold to convert ratings into binary feedback')

        parser.add_argument('--num_users', dest='num_users', type=int, default=False,
                            help='Sample of users for evaluation')

        parser.add_argument('--lr', dest='lr', type=float, default=0.001,
                            help='Starting value for the learning rate')

        parser.add_argument('--dbpedia', dest='dbpedia', default=False, action='store_true',
                    help='Use dbpedia embeddings as item data')

        return parser.parse_args()


if __name__ == '__main__':

    start_time = time.time()

    print('Starting FM...')

    args = TuriRankingFM.parse_args()

    if not args.train:
        args.train = 'datasets/' + args.dataset + '/train.dat'

    if not args.test:
        args.test = 'datasets/' + args.dataset + '/test.dat'

    if not args.validation:
        args.validation = 'datasets/' + args.dataset + '/val.dat'
    # initialize evaluator

    if args.dataset == 'LastFM':
        implicit = True

    else:
        implicit = args.implicit

    if args.dataset == 'LibraryThing':
        threshold = 8
    else:
        threshold = args.threshold

    evaluat = Evaluator(implicit=implicit, threshold=args.threshold,
                        all_unrated_items=args.all_unrated_items)

    fm_rec = TuriRankingFM(args.dataset, implicit, dbpedia=args.dbpedia)

    x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test, \
    x_val, y_val, qids_val, items_val = evaluat.features(fm_rec, args.train, args.test,
                                                         n_jobs=1,
                                                         n_users=args.num_users)

    print('Finished computing features after %s seconds' % (time.time() - start_time))

    if args.write_features:

        evaluat.write_features_to_file('train', qids_train, x_train, y_train, items_train)

        evaluat.write_features_to_file('test', qids_test, x_test, y_test, items_test)

    evaluat.evaluate(fm_rec, x_test, y_test, qids_test, items_test, baseline=True)  # evaluates the model on the test set

    print("--- %s seconds ---" % (time.time() - start_time))







