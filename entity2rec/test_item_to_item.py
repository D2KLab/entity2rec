import random
from entity2rec import Entity2Rec
from evaluator import Evaluator
import time
from parse_args import parse_args


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

# initialize recommender

e2rec = Entity2Rec(args.dataset, run_all=args.run_all, p=args.p, q=args.q,
                   feedback_file=args.feedback_file, walk_length=args.walk_length,
                   num_walks=args.num_walks, dimensions=args.dimensions, window_size=args.window_size,
                   workers=args.workers, iterations=args.iter, collab_only=args.collab_only,
                   content_only=args.content_only)

# initialize evaluator

evaluat = Evaluator(implicit=implicit, threshold=args.threshold, all_unrated_items=False)


evaluat.compute_item_to_item_similarity(e2rec, args.train, args.test, args.dataset, validation=args.validation,
                                        n_users=args.num_users, n_jobs=args.workers, max_n_feedback=args.max_n_feedback)
