import random
from entity2rec import Entity2Rec
from evaluator import Evaluator
import time
import pickle
from parse_args import parse_args
from collections import defaultdict
import heapq

start_time = time.time()

args = parse_args()

with open('datasets/'+args.dataset+'/item_to_item_similarity', 'rb') as f1:
  item_to_item_similarity_dict = pickle.load(f1)  # seed -> {item: score} 

live_time = time.time()

N = 5

while True:

    dicts = []
    seeds = set()

    while True:

      print('Please enter seed item. The more seeds you provide, the better the recommendations. Enter stop when you want to stop.')

      seed = input()

      if seed == 'stop':

        break

      if len(item_to_item_similarity_dict[seed]) > 0:

        dicts.append(item_to_item_similarity_dict[seed])
        seeds.add(seed)

      else:

        print('Seed item not found. Enter another seed.')

    rec_time = time.time()

    final_dict = defaultdict(int)

    l = len(dicts)

    for d in dicts:

      for key, value in d.items():

          final_dict[key] += value/l

    # remove seeds from the candidates
    candidates = [i for i in final_dict.keys() if i not in seeds]

    recs = heapq.nlargest(N, candidates, key=lambda x: final_dict[x])

    print(recs)

    print('total time')
    print("--- %s seconds ---" % (time.time() - start_time))

    print('total live time')
    print("--- %s seconds ---" % (time.time() - live_time))

    print('total rec time')
    print("--- %s seconds ---" % (time.time() - rec_time))