from entity2rec.sparql import Sparql
import time
import pickle
from collections import defaultdict
import heapq
import logging
from flask import Flask
from flask import request
import json
from pymongo import MongoClient
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
start_time = time.time()

version_api = '0.1'

dataset = 'LastFM'

item_to_item_similarity_dict = {}

#@app.before_first_request
def load_model():

    # open item to item similarity matrix and read into dictionary
    with open('datasets/'+dataset+'/item_to_item_matrix', 'rb') as f1:
        global item_to_item_similarity_dict
        item_to_item_similarity_dict = pickle.load(f1)  # seed -> {item: score}

@app.before_first_request
def read_item_metadata():

    # reads list of item in the dataset
    global items
    items = set()

    with open('datasets/'+dataset+'/all.dat') as all_ratings:

        for line in all_ratings:
            line_split = line.strip('\n').split(' ')
            items.add(line_split[1])

    print(items)
    # reads items metadata from sparql endpoint and keeps them in memory
    global item_metadata
    item_metadata = {}

    for item in items:

        metadata = Sparql.get_item_metadata(item)

        print(metadata)

        if metadata:  # skip items with missing metadata

            item_metadata[item] = metadata

    global num_items
    num_items = len(item_metadata)
    print(num_items)

@app.route('/entity2rec/' + version_api + "/onboarding", methods=['GET'])
def onboarding():

    out = {}

    out['user_id'] = time.time()  # FIXME

    number_of_samples = 1000

    if num_items < number_of_samples:

        number_of_samples = num_items

    for sample in random.sample(item_metadata.items(), number_of_samples):

        out[sample[0]] = sample[1]

    out_json = json.dumps(out, indent=4, sort_keys=True)

    return out_json

@app.route('/entity2rec/' + version_api + "/recs", methods=['POST'])
def recommend():

    logger.info("Launch of the entity2rec recommendation REST API")
    content = request.get_json(silent=True)

    seed = None
    N = 5

    try:
        seed=content['seed']
        user_id=content['user_id']
    except KeyError:
        raise ValueError('Please provide a seed item and a user_id.')

    rec_time = time.time()

    # retrieve similarity values for the seed item

    d = item_to_item_similarity_dict[seed]

    # remove seed from candidate items

    candidates = [i for i in item_metadata.keys() if i != seed]

    recs = heapq.nlargest(N, candidates, key=lambda x: d[x])

    out = {}

    out['recs'] = []

    for r in recs:

        out['recs'].append({r: item_metadata[r]})

    out['user_id'] = user_id

    out_json = json.dumps(out, indent=4, sort_keys=True)

    logger.info('total rec time')
    logger.info("--- %s seconds ---" % (time.time() - rec_time))

    return out_json

@app.route('/entity2rec/' + version_api + "/feedback", methods=['POST'])
def feedback():

    content = request.get_json(silent=True)

    try:
        uri=content['uri']
        user_id=content['user_id']
        feedback=content['feedback']
        position=content['position']

    except KeyError:
        raise ValueError('Please provide a uri, user_id,feedback and position of the item.')

    content['timestamp'] = time.time()

    connection = MongoClient('localhost', 27017)
    db = connection.db
    collection = db.feedback

    collection.save(content)

    return 'ok'

if __name__ == '__main__':

    app.run(debug=True)