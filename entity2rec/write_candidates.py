import sys
import random
sys.path.append('.')
from evaluator import Evaluator
from parse_args import parse_args

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

users_folder = 'benchmarks/MyMediaLite-3.11/users'

candidates_folder = 'benchmarks/MyMediaLite-3.11/candidates'

evaluat.write_candidates(args.train,args.test, users_folder, candidates_folder, validation=args.validation)
