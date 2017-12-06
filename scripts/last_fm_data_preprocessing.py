from collections import defaultdict
import numpy as np

mappings_file = 'datasets/LastFM/original/MappingLastfm2DBpedia-1.2.tsv'

feedback_file = 'datasets/LastFM/original/user_artists.dat'

feedback_dbpedia_file = 'datasets/LastFM/all.dat'

train = 'datasets/LastFM/train.dat'

f_train = 0.8

val = 'datasets/LastFM/val.dat'

f_val = 0.04

test = 'datasets/LastFM/test.dat'

f_test = 0.16

mappings = {}

feedback = defaultdict(list)

# read the mappings into a dictionary

with open(mappings_file) as mappings_read:

    for line in mappings_read:

        line = line.strip('\n')

        line_split = line.split('\t')

        artist_lastfm_id = line_split[0]

        artist_uri = line_split[2]

        mappings[artist_lastfm_id] = artist_uri

# read the feedback into a dictionary

with open(feedback_file) as feedback_read:

    for i, line in enumerate(feedback_read):

        if i > 0: # skip headers

            line = line.strip('\n')

            line_split = line.split('\t')

            user_id = line_split[0]

            artist_lastfm_id = line_split[1]

            try:

                artist_uri = mappings[artist_lastfm_id]

                feedback[user_id].append(artist_uri)

            except KeyError:

                print(artist_lastfm_id)
                continue

# write the feedback on file

with open(feedback_dbpedia_file,'w') as feedback_write,\
     open(train, 'w') as train_write,\
     open(val,'w') as val_write,\
     open(test,'w') as test_write:

    for user_id in feedback.keys():

        artist_uris = feedback[user_id]

        l = len(artist_uris)

        if l > 3:  # need at least 3 feedback to create the splits

            np.random.shuffle(artist_uris)

            n_train = np.round(f_train*l)

            n_val = np.round(f_val*l)

            n_test = l - n_train - n_val

            for i, artist_uri in enumerate(artist_uris):

                if i < n_train:

                    print(i)

                    train_write.write('%s %s 1 -1\n' %(user_id, artist_uri))

                elif i >= n_train and i < n_train + n_val:

                    val_write.write('%s %s 1 -1\n' %(user_id, artist_uri))

                else:

                    test_write.write('%s %s 1 -1\n' %(user_id, artist_uri))

                feedback_write.write('%s %s 1 -1\n' %(user_id, artist_uri))






