import time
import numpy as np
from evaluator import Evaluator
import argparse
import subprocess
import os
import sys

class MMLRecommender(object):

    def __init__(self, recommender):

        self.mml_model = self._read_scores('benchmarks/MyMediaLite-3.11/%s_scores.txt' % recommender)

    def _read_scores(self, file):
        
        model = {}

        with open(file) as file_read:

            for line in file_read:

                line_split = line.strip('\n').split(' ')

                user = line_split[0]
                item = line_split[1]
                score = line_split[2]

                model[(user, item)] = np.float16(score)

        return model

    def compute_user_item_features(self, user, item, items_liked_by_user, users_liking_the_item):

        try:

            features = [self.mml_model[(user, item)]]  # user item relatedness from node2vec

        except KeyError:  # do not have user item pair in embedding

            features = [0.]

        return features

    def fit(self, x_train, y_train, qids_train):

        return 0

    def predict(self, x_test, qids_test):

        return x_test

    @staticmethod
    def data_preprocessing(dataset):

        all_data = 'datasets/%s/all.dat' % dataset
        train_data = 'datasets/%s/train.dat' % dataset
        val_data = 'datasets/%s/val.dat' % dataset
        test_data = 'datasets/%s/test.dat' % dataset

        items_list = []

        with open(all_data) as all_file:

            for line in all_file:
                line_split = line.strip('\n').split(' ')

                items_list.append(line_split[1])

        items_list = sorted(list(set(items_list)))  # remove duplicates, sort for reproducibility

        item_index = {i: item for i, item in enumerate(items_list)}  # create index

        with open('benchmarks/MyMediaLite-3.11/item_index_%s.txt' % dataset, 'w') as item_index_file:

            for index, item in item_index.items():
                item_index_file.write('%d %s\n' % (index, item))

        index_item = {item: i for i, item in item_index.items()}

        def convert_to_mml(train, train_mml, index_item):

            with open(train) as train_file:
                with open(train_mml, 'w') as train_mml_file:
                    for line in train_file:
                        line_split = line.strip('\n').split(' ')

                        user = line_split[0]

                        item = line_split[1]

                        rating = line_split[2]

                        timestamp = line_split[3]

                        index = index_item[item]

                        train_mml_file.write('%s %d %s %s\n' % (user, index, rating, timestamp))

        convert_to_mml(train_data, "benchmarks/MyMediaLite-3.11/data/%s/train.mml" % dataset, index_item)

        convert_to_mml(val_data, "benchmarks/MyMediaLite-3.11/data/%s/val.mml" % dataset, index_item)

        convert_to_mml(test_data, "benchmarks/MyMediaLite-3.11/data/%s/test.mml" % dataset, index_item)

    @staticmethod
    def prediction_parser(recommender, dataset):

        prediction_file = 'benchmarks/MyMediaLite-3.11/%s_ranked_predictions.txt' % recommender
        scores = 'benchmarks/MyMediaLite-3.11/%s_scores.txt' % recommender
        index_file = 'benchmarks/MyMediaLite-3.11/item_index_%s.txt' % dataset

        index = dict()

        with open(index_file) as index_file_read:

            for line in index_file_read:
                line_split = line.strip('\n').split(' ')

                ind = line_split[0]

                item = line_split[1]

                index[ind] = item

        with open(prediction_file) as prediction_file_read:

            with open(scores, 'w') as score_file:

                for line in prediction_file_read:

                    line_split = line.strip('\n').split('\t')

                    user = line_split[0]

                    item_score_pairs = line_split[1].replace('[', '').replace(']', '').split(',')

                    for item_score in item_score_pairs:

                        item_score_split = item_score.split(':')

                        item_ind = item_score_split[0]

                        item = index[item_ind]

                        score = item_score_split[1]

                        score_file.write('%s %s %s\n' % (user, item, score))


def parse_args():

    """
    Parses the entity2rec arguments.
    """

    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--config_file', nargs='?', default='config/properties.json',
                        help='Path to configuration file')

    parser.add_argument('--dataset', nargs='?', default='Movielens1M',
                        help='Dataset')

    parser.add_argument('--train', dest='train', help='train', default=None)

    parser.add_argument('--test', dest='test', help='test', default=None)

    parser.add_argument('--validation', dest='validation', default=None, help='validation')

    parser.add_argument('--implicit', dest='implicit', action='store_true', default=False,
                        help='Implicit feedback with boolean values')

    parser.add_argument('--all_items', dest='all_unrated_items', action='store_false', default=True,
                        help='Whether keeping the rated items of the training set as candidates. '
                             'Default is AllUnratedItems')
    
    parser.add_argument('--threshold', dest='threshold', default=4, type=int,
                        help='Threshold to convert ratings into binary feedback')

    parser.add_argument('--recommender', dest='recommender', help="which recommender to use")

    parser.add_argument('--first_time', dest='first_time', default=False, action='store_true')

    parser.add_argument('--num_users', dest='num_users', default=False, type=int)

    return parser.parse_args()


if __name__ == '__main__':

    np.random.seed(1)  # fixed seed for reproducibility

    start_time = time.time()

    print('Starting MyMediaLite recommender...')

    args = parse_args()

    if args.recommender == 'all':

        recommenders = ['MostPopular', 'SoftMarginRankingMF', 'BPRMF', 'LeastSquareSLIM', 'ItemKNN', 'WRMF',
                        'MultiCoreBPRMF', 'WeightedBPRMF', 'BPRSLIM']

    else:

        recommenders = [args.recommender]

    # default settings

    if not args.train:
        args.train = 'datasets/' + args.dataset + '/train.dat'

    if not args.test:
        args.test = 'datasets/' + args.dataset + '/test.dat'

    if not args.validation:
        args.validation = 'datasets/' + args.dataset + '/val.dat'

    if args.dataset == 'LastFM':

        implicit = True

    else:

        implicit = args.implicit

    if args.dataset == 'LibraryThing':

        threshold = 8

    else:

        threshold = args.threshold

    # initialize evaluator

    evaluat = Evaluator(implicit=implicit, threshold=threshold, all_unrated_items=args.all_unrated_items)
    
    if args.first_time:

        # remove previous results if any for a fresh start
        os.system("rm benchmarks/MyMediaLite-3.11/predictions/%s/*.txt" % args.dataset)
        os.system("rm benchmarks/MyMediaLite-3.11/models/%s/*" % args.dataset)

        #  create mml compatible data and index
        MMLRecommender.data_preprocessing(args.dataset)

        # write candidates to file

        evaluat.write_candidates(args.train, args.test, 'benchmarks/MyMediaLite-3.11/users/%s' % args.dataset,
                                 'benchmarks/MyMediaLite-3.11/candidates/%s' % args.dataset,
                                 'benchmarks/MyMediaLite-3.11/item_index_%s.txt' % args.dataset)
    
    for recommender in recommenders:
        
        print('%s' % recommender)

        if '%s' % recommender not in 'benchmarks/MyMediaLite-3.11/models/%s' % args.dataset:

            # train mymedialite model and save it to file
            subprocess.check_output(["./benchmarks/MyMediaLite-3.11/bin/item_recommendation",
                                     "--training-file=benchmarks/MyMediaLite-3.11/data/%s/train.mml" % args.dataset,
                                     "--test-file=benchmarks/MyMediaLite-3.11/data/%s/test.mml" % args.dataset,
                                     "--recommender=%s" % recommender,
                                     "--save-model=benchmarks/MyMediaLite-3.11/models/%s/%s" % (args.dataset, recommender),
                                     "--rating-threshold=%s" % threshold])

        # generate predictions and save them to file
        for file in os.listdir('benchmarks/MyMediaLite-3.11/users/%s' % args.dataset):

            print(file)

            if "%s_%s" % (recommender, file) not in os.listdir('benchmarks/MyMediaLite-3.11/predictions/%s' % args.dataset):

                subprocess.check_output(["./benchmarks/MyMediaLite-3.11/bin/item_recommendation",
                                         "--training-file=benchmarks/MyMediaLite-3.11/data/%s/train.mml" % args.dataset,
                                         "--test-file=benchmarks/MyMediaLite-3.11/data/%s/test.mml" % args.dataset,
                                         "--recommender=%s" % recommender,
                                         "--load-model=benchmarks/MyMediaLite-3.11/models/%s/%s" % (args.dataset, recommender),
                                         "--rating-threshold=%s" % threshold,
                                         "--prediction-file=benchmarks/MyMediaLite-3.11/predictions/%s/%s_%s" % (args.dataset, recommender, file),
                                         "--candidate-items=benchmarks/MyMediaLite-3.11/candidates/%s/%s" % (args.dataset, file),
                                         "--test-users=benchmarks/MyMediaLite-3.11/users/%s/%s" % (args.dataset, file)])

            else:
                print('file already exists')

        os.system("cat benchmarks/MyMediaLite-3.11/predictions/%s/%s_* > benchmarks/MyMediaLite-3.11/%s_ranked_predictions.txt" % (args.dataset, recommender, recommender))
        
        # parse the output
        MMLRecommender.prediction_parser(recommender, args.dataset)
        
        # initialize MyMediaLite recommender
        mml_rec = MMLRecommender(recommender)


        # compute e2rec features, enforce one worker because of a bug in joblib
        x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test,\
        x_val, y_val, qids_val, items_val = evaluat.features(mml_rec, args.train, args.test,
                                                             validation=False,
                                                             n_jobs=1,
                                                             supervised=False,
                                                             n_users=args.num_users)

        print('Finished computing features after %s seconds' % (time.time() - start_time))

        scores = evaluat.evaluate(mml_rec, x_test, y_test, qids_test, items_test,
                                  write_to_file="results/%s/mml/%s.csv" % (args.dataset, recommender),
                                  baseline=True)

        print("--- %s seconds ---" % (time.time() - start_time))
