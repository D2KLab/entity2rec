from pymongo import MongoClient
import numpy as np
from collections import Counter, defaultdict
from pyltr.metrics._metrics import check_qids, get_groups
from scipy import stats

# metrics are computed session-wise

def p_at_k(results,k=5):

    feedback = [i['feedback'] for i in results]

    return sum(feedback)/k

def ser_at_k(results, most_pop_dict, k=5):

    most_pop_k = sorted(most_pop_dict.keys(), key=lambda x: most_pop_dict[x], reverse=True)[0:k]

    feedback = []

    for i in results:

        if i['uri'] in most_pop_k:

            feedback.append(0)

        else:
            feedback.append(i['feedback'])

    return sum(feedback)/k

def nov_at_k(results, most_pop_dict, k=5):

    uris = [i['uri'] for i in results]

    novelties = [-np.log2(most_pop_dict[i]) for i in uris]

    return sum(novelties)/k

# define popularity dictionary

dataset = 'LibraryThing'

pop_dict = Counter()

with open('datasets/'+dataset+'/all.dat') as all_ratings:

    for line in all_ratings:
        
        line_split = line.strip('\n').split(' ')

        item = line_split[1]

        pop_dict[item]+=1

# normalization

tot_pop = sum(pop_dict.values())

for item, pop in pop_dict.items():

    pop_dict[item] = pop/tot_pop

# connect to DB

mongodb_port = 27027

client = MongoClient('localhost', mongodb_port)

entity2rec = client.entity2rec

seed = entity2rec.seed
feedback = entity2rec.feedback
discard = entity2rec.discard

# query the DB

feedback_entity2rec = []
feedback_itemknn = []

qids_entity2rec = []
qids_itemknn = []

for post in feedback.find({"algorithm": "entity2rec"}):

    qids_entity2rec.append(post['user_id'])

    feedback_entity2rec.append(post)

for post in feedback.find({"algorithm": "itemknn"}):

    qids_itemknn.append(post['user_id'])

    feedback_itemknn.append(post)

# get user_id - seed mapping

seed_user_entity2rec = {}
seed_user_itemknn = {}

for post in seed.find():  # every user_id - seed pair
    
    user_id = post['user_id']

    if user_id in qids_entity2rec:
        print('user %s has received entity2rec' %user_id)
        seed_user_entity2rec[user_id] = post['seed']

    elif user_id in qids_itemknn:
        print('user %s has received itemknn' %user_id)
        seed_user_itemknn[user_id] = post['seed']

    else:
        print('user %s has not rated recommendations' %user_id)
        continue
        
# entity2rec scores

query_groups_entity2rec = get_groups(qids_entity2rec)

num_sessions_entity2rec = len(seed_user_entity2rec)

entity2rec_scores = defaultdict(list)

entity2rec_precision_by_popularity = defaultdict(list)

entity2rec_serendipity_by_popularity = defaultdict(list)

entity2rec_novelty_by_popularity_entity2rec = defaultdict(list)

for qid, a, b in query_groups_entity2rec:  # iterate through the different sessions

    entity2rec_scores['precision'].append(p_at_k(feedback_entity2rec[a:b]))
    entity2rec_scores['serendipity'].append(ser_at_k(feedback_entity2rec[a:b], pop_dict))
    entity2rec_scores['novelty'].append(nov_at_k(feedback_entity2rec[a:b], pop_dict))

    try:
        seed = seed_user_entity2rec[qid]

        seed_pop = pop_dict[seed]

        entity2rec_precision_by_popularity[seed_pop].append(p_at_k(feedback_entity2rec[a:b]))

        entity2rec_serendipity_by_popularity[seed_pop].append(ser_at_k(feedback_entity2rec[a:b], pop_dict))

        entity2rec_novelty_by_popularity_entity2rec[seed_pop].append(nov_at_k(feedback_entity2rec[a:b], pop_dict))

    except KeyError:
        continue

print('entity2rec: P@5 ', np.mean(entity2rec_scores['precision']), '+-', np.std(entity2rec_scores['precision']/np.sqrt(num_sessions_entity2rec)))
print('entity2rec: SER@5 ', np.mean(entity2rec_scores['serendipity']), '+-', np.std(entity2rec_scores['serendipity']/np.sqrt(num_sessions_entity2rec)))
print('entity2rec: NOV@5 ', np.mean(entity2rec_scores['novelty']), '+-', np.std(entity2rec_scores['novelty']/np.sqrt(num_sessions_entity2rec)))

# itemknn scores

query_groups_itemknn = get_groups(qids_itemknn)

num_sessions_itemknn = len(seed_user_entity2rec)

itemknn_scores = defaultdict(list)

itemknn_precision_by_popularity = defaultdict(list)

itemknn_serendipity_by_popularity = defaultdict(list)

itemknn_novelty_by_popularity_entity2rec = defaultdict(list)

for qid, a, b in query_groups_itemknn:

    itemknn_scores['precision'].append(p_at_k(feedback_itemknn[a:b]))
    itemknn_scores['serendipity'].append(ser_at_k(feedback_itemknn[a:b], pop_dict))
    itemknn_scores['novelty'].append(nov_at_k(feedback_itemknn[a:b], pop_dict))

    try:
        seed = seed_user_itemknn[qid]

        seed_pop = pop_dict[seed]

        itemknn_precision_by_popularity[seed_pop].append(p_at_k(feedback_entity2rec[a:b]))

        itemknn_serendipity_by_popularity[seed_pop].append(ser_at_k(feedback_entity2rec[a:b], pop_dict))

        itemknn_novelty_by_popularity_entity2rec[seed_pop].append(nov_at_k(feedback_entity2rec[a:b], pop_dict))

    except KeyError:
        continue

print('itemknn: P@5 ', np.mean(itemknn_scores['precision']), '+-', np.std(itemknn_scores['precision'])/np.sqrt(num_sessions_itemknn))
print('itemknn: SER@5 ', np.mean(itemknn_scores['serendipity']), '+-', np.std(itemknn_scores['serendipity'])/np.sqrt(num_sessions_itemknn))
print('itemknn: NOV@5 ', np.mean(itemknn_scores['novelty']), '+-', np.std(itemknn_scores['novelty'])/np.sqrt(num_sessions_itemknn))

# Welch's t-student test

print('T-test precision:')
print(stats.ttest_ind(entity2rec_scores['precision'],itemknn_scores['precision'], equal_var=False))

print('T-test serendipity:')
print(stats.ttest_ind(entity2rec_scores['serendipity'],itemknn_scores['serendipity'], equal_var=False))

print('T-test novelty:')
print(stats.ttest_ind(entity2rec_scores['novelty'],itemknn_scores['novelty'], equal_var=False))

# plot scores as a function of seed popularity

with open('plots/entity2rec_precision_by_popularity.csv', 'w') as entity2rec_precision_by_popularity_file:

    for pop, precisions in entity2rec_precision_by_popularity.items():

        p = np.mean(precisions)

        entity2rec_precision_by_popularity_file.write("%.6f,%.6f\n" %(pop,p))


with open('plots/itemknn_precision_by_popularity.csv', 'w') as itemknn_precision_by_popularity_file:

    for pop, precisions in itemknn_precision_by_popularity.items():

        p = np.mean(precisions)

        itemknn_precision_by_popularity_file.write("%.6f,%.6f\n" %(pop,p))