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

index_file = 'benchmarks/MyMediaLite-3.11/item_index_%s' %args.dataset


def get_item(line):
    line_split = line.strip('\n').split(' ')
    return line_split[1]

with open('benchmarks/MyMediaLite-3.11/item_index_%s' %args.dataset, 'w') as index_file_write:

    items = []

    with open(args.train, encoding='utf-8') as train_read:
        with open(args.validation, encoding='utf-8') as val_read:
            with open(args.test, encoding='utf-8') as test_read:

                for line in train_read:
                    items.append(get_item(line))

                for line in val_read:
                    items.append(get_item(line))

                for line in test_read:
                    items.append(get_item(line))

    items = list(set(items))

    index_dict = {item:i for i, item in enumerate(items)}

    for i, item in enumerate(items):

        index_file_write.write('%d %s\n' %(i, item))

evaluat.write_candidates(args.train, args.test, users_folder, candidates_folder, index_dict, validation=args.validation)
