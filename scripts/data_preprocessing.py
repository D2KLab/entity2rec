from collections import defaultdict
import numpy as np
import codecs

"""
It creates a train, validation and test file starting from the DBpedia mappings file and the feedback file
The splits are computed per-user, using 70% train, 10% val, 20% test
Users with less than 10 ratings are discarded
"""


def sparsity(user, item, interactions):

    return 100*(1 - interactions/(user*item))


def get_real_length(l):

    return len(list(set(l)))

# reproducible results

np.random.seed(1)


dataset = 'LibraryThing'

# different datasets use different separators

if dataset == 'Movielens1M':

    sep = '::'

elif dataset == 'LastFM':

    sep = '\t'

elif dataset == 'LibraryThing':

    sep = ' '

else:

    raise ValueError('Choose one of the three supported datasets')

mappings_file = 'datasets/%s/original/mappings.tsv' % dataset

feedback_file = 'datasets/%s/original/feedback.txt' % dataset

feedback_dbpedia_file = 'datasets/%s/all.dat' % dataset

train = 'datasets/%s/train.dat' % dataset

f_train = 0.7

val = 'datasets/%s/val.dat' % dataset

f_val = 0.1

test = 'datasets/%s/test.dat' % dataset

f_test = 0.2

mappings = {}

feedback = defaultdict(list)

# read the mappings into a dictionary

with open(mappings_file) as mappings_read:

    for line in mappings_read:

        line = line.strip('\n')

        line_split = line.split('\t')

        item_id = line_split[0]

        item_uri = line_split[2]

        mappings[item_id] = item_uri

# read the feedback into a dictionary

with codecs.open(feedback_file, 'r', encoding='latin-1') as feedback_read:

    for i, line in enumerate(feedback_read):

        if i > 0:  # skip headers

            line = line.strip('\n')

            line_split = line.split(sep)

            user_id = line_split[0]

            item_id = line_split[1]

            if dataset == 'LastFM':  # implicit

                rating = 1

            else:

                rating = int(line_split[2])

            if dataset == 'Movielens1M':

                timestamp = int(line_split[3])
            
            else:
                timestamp = -1

            try:

                item_uri = mappings[item_id]
                
                feedback[user_id].append((item_uri, rating, timestamp))

            except KeyError:

                print(item_id)
                continue

# remove duplicates

for user in feedback.keys():

    items = list(set(feedback[user]))

    feedback[user] = items

# write the feedback on file

l_all = 0
l_train = 0
l_val = 0
l_test = 0

all_items = []
train_items = []
val_items = []
test_items = []

all_users = []
train_users = []
val_users = []
test_users = []

with open(feedback_dbpedia_file,'w') as feedback_write,\
     open(train, 'w') as train_write,\
     open(val, 'w') as val_write,\
     open(test, 'w') as test_write:

    for user_id in feedback.keys():

        item_uris = feedback[user_id]

        l = len(item_uris)

        if l >= 10:  # need at least 10 feedback to create the splits

            np.random.shuffle(item_uris)

            n_train = np.round(f_train*l)

            n_val = np.round(f_val*l)

            n_test = l - n_train - n_val

            for i, (item_uri, rating, timestamp) in enumerate(item_uris):

                l_all += 1

                feedback_write.write('%s %s %d %d\n' %(user_id, item_uri, rating, timestamp))

                all_items.append(item_uri)

                all_users.append(user_id)

                if i < n_train:

                    train_write.write('%s %s %d %d\n' %(user_id, item_uri, rating, timestamp))

                    # compute stats

                    l_train += 1

                    train_items.append(item_uri)

                    train_users.append(user_id)

                elif i >= n_train and i < n_train + n_val:

                    val_write.write('%s %s %d %d\n' %(user_id, item_uri, rating, timestamp))

                    # compute stats

                    l_val += 1

                    val_items.append(item_uri)

                    val_users.append(user_id)

                else:

                    test_write.write('%s %s %d %d\n' %(user_id, item_uri, rating, timestamp))

                    # compute stats

                    l_test += 1

                    test_items.append(item_uri)

                    test_users.append(user_id)

n_items_train = get_real_length(train_items)

n_users_train = get_real_length(train_users)

sparsity_train = sparsity(n_users_train, n_items_train, l_train)

n_items_val = get_real_length(val_items)

n_users_val = get_real_length(val_users)

sparsity_val = sparsity(n_users_val, n_items_val, l_val)

n_items_test = get_real_length(test_items)

n_users_test = get_real_length(test_users)

sparsity_test = sparsity(n_users_test, n_items_test, l_test)

n_items_all = get_real_length(all_items)

n_users_all = get_real_length(all_users)

sparsity_all = sparsity(n_users_all, n_items_all, l_all)

# check

assert n_users_all == n_users_test == n_users_train == n_users_val, 'number of users must be the same'

assert l_train+l_val+l_test == l_all, 'n_all == n_train + n_val + n_test'

# print stats

print('\n')
print('stats for %s:\n' %dataset)
print('dataset,size,users,items,sparsity')
print('all,%d,%d,%d,%f' %(l_all, n_users_all, n_items_all, sparsity_all))
print('train,%d,%d,%d,%f' %(l_train, n_users_train, n_items_train, sparsity_train))
print('val,%d,%d,%d,%f' %(l_val, n_users_val, n_items_val, sparsity_val))
print('test,%d,%d,%d,%f' %(l_test, n_users_test, n_items_test, sparsity_test))
