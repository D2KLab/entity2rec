import sys
import random
sys.path.append('.')
from evaluator import Evaluator
from parse_args import parse_args
import os

random.seed(1)  # fixed seed for reproducibility

args = parse_args()

if not args.train:
    args.train = 'datasets/'+args.dataset+'/train.dat'

if not args.test:
    args.test = 'datasets/'+args.dataset+'/test.dat'

if not args.validation:
    args.validation = 'datasets/'+args.dataset+'/val.dat'

# initialize evaluator

evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

users_folder = 'benchmarks/MyMediaLite-3.11/users/%s' %args.dataset

try:
    os.mkdir(users_folder)

except:
    pass

candidates_folder = 'benchmarks/MyMediaLite-3.11/candidates/%s' %args.dataset

try:
    os.mkdir(candidates_folder)

except:
    pass

index_file = 'benchmarks/MyMediaLite-3.11/item_index_%s.txt' %args.dataset

evaluat.write_candidates(args.train, args.test, users_folder, candidates_folder, index_file, validation=args.validation)
