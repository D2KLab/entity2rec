import codecs
from collections import Counter
import time
import numpy as np
from evaluator import Evaluator
from parse_args import parse_args


def compute_most_pop_dict(data, threshold, implicit=False):

    if implicit:
        threshold = 0.5

    pop_dict = Counter()

    with codecs.open(data, 'r', encoding='utf-8') as read_train:

        for line in read_train:

            line = line.split(' ')

            item = line[1]

            rel = line[2]

            if int(rel) >= threshold:

                pop_dict[item] += 1

    # normalize

    pop_dict_tot = float(sum(pop_dict.values()))

    for key in pop_dict.keys():

        pop_dict[key] /= pop_dict_tot

    return pop_dict


class MostPop(object):

    def __init__(self, training_set, threshold, implicit=False):

        self.pop_dict = compute_most_pop_dict(training_set, threshold, implicit=implicit)

        self.model = True

    def compute_user_item_features(self, user, item, items_liked_by_user, users_liking_the_item):

        try:

            features = [np.float32(self.pop_dict[item])]  # simply the weighted popularity

        except KeyError:

            features = [np.mean([self.pop_dict.values()], dtype=float)]

        return features

    def predict(self, x_test, qids_test):

        return x_test


if __name__ == '__main__':

    start_time = time.time()

    print('Starting MostPop...')

    args = parse_args()

    mostpop_rec = MostPop(args.train, args.threshold, implicit=args.implicit)

    evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

    x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test, \
    x_val, y_val, qids_val, items_val = evaluat.features(mostpop_rec, args.train, args.test, validation=args.validation, n_jobs=args.workers,
                                              n_users=args.num_users)

    print('Finished computing features after %s seconds' % (time.time() - start_time))

    if args.write_features:

        evaluat.write_features_to_file('train', qids_train, x_train, y_train, items_train)

        evaluat.write_features_to_file('test', qids_test, x_test, y_test, items_test)

    evaluat.evaluate(mostpop_rec, x_test, y_test, qids_test, items_test)  # evaluates the model on the test set

    print("--- %s seconds ---" % (time.time() - start_time))
