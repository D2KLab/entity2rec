from entity2rec import Entity2Rec
from evaluator import Evaluator
import time
from parse_args import parse_args
import numpy as np
import random

random.seed(1)  # fixed seed for reproducibility

start_time = time.time()

print('Starting entity2rec...')

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


# initialize entity2rec recommender
e2rec = Entity2Rec(args.dataset, run_all=args.run_all, p=args.p, q=args.q,
                   walk_length=args.walk_length,
                   num_walks=args.num_walks, dimensions=args.dimensions, window_size=args.window_size,
                   workers=args.workers, iterations=args.iter)

# initialize evaluator

evaluat = Evaluator(implicit=implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

# compute e2rec features
x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test,\
    x_val, y_val, qids_val, items_val = evaluat.features(e2rec, args.train, args.test,
                                          validation=False, supervised=False,
                                          n_users=args.num_users, n_jobs=args.workers)

print('Finished computing features after %s seconds' % (time.time() - start_time))

for i, prop in enumerate(e2rec.properties):

    print(prop.name)

    test_feat = x_test[:, i]
    test_feat = test_feat.reshape((-1, 1))

    scores = evaluat.evaluate(e2rec, test_feat, y_test, qids_test, items_test,
                              verbose=False, write_to_file='results/%s/%s.csv' % (args.dataset,
                              prop.name))  # evaluates the recommender on the test set

