import random
from entity2rec import Entity2Rec
from evaluator import Evaluator
import time
from parse_args import parse_args

random.seed(1)  # fixed seed for reproducibility

start_time = time.time()

print('Starting entity2rec...')

args = parse_args()

if not args.train:
    args.train = 'datasets/'+args.dataset+'/train.dat'

if not args.test:
    args.test = 'datasets/'+args.dataset+'/test.dat'

if not args.validation:
    args.validation = 'datasets/'+args.dataset+'/val.dat'


# initialize entity2rec recommender
e2rec = Entity2Rec(args.dataset, run_all=args.run_all, p=args.p, q=args.q,
                   feedback_file=args.feedback_file, walk_length=args.walk_length,
                   num_walks=args.num_walks, dimensions=args.dimensions, window_size=args.window_size,
                   workers=args.workers, iterations=args.iter, collab_only=args.collab_only,
                   content_only=args.content_only)

# initialize evaluator

evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

if not args.read_features:
    # compute e2rec features
    x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test,\
    x_val, y_val, qids_val, items_val = evaluat.features(e2rec, args.train, args.test,
                                                         validation=False, n_users=args.num_users,
                                                         n_jobs=args.workers, max_n_feedback=args.max_n_feedback)

else:

    x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test,\
    x_val, y_val, qids_val, items_val = evaluat.read_features('train.svm', 'test.svm', val='val.svm')


print('Finished computing features after %s seconds' % (time.time() - start_time))

print(e2rec.recommend(1, qids_test, x_test, items_test))

print("--- %s seconds ---" % (time.time() - start_time))

