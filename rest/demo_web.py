from entity2rec.sparql import Sparql
import time
import pickle
from collections import defaultdict
import heapq
import logging
from flask import Flask
from flask import request
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
start_time = time.time()

version_api = '0.1'

dataset = 'item_to_item_test'

item_to_item_similarity_dict = {}

@app.before_first_request
def load_model():

    with open('datasets/'+dataset+'/item_to_item_matrix', 'rb') as f1:
        global item_to_item_similarity_dict
        item_to_item_similarity_dict = pickle.load(f1)  # seed -> {item: score} 

@app.route('/entity2rec/' + version_api + "/recs", methods=['POST'])
def recommend():

    logger.info("Launch of the entity2rec recommendation REST API")
    content = request.get_json(silent=True)

    seed = None
    N = 5

    try:
        seed=content['seed']
    except KeyError:
        raise ValueError('Please provide a seed item.')

    rec_time = time.time()

    # retrieve similarity values for the seed item

    d = item_to_item_similarity_dict[seed]

    # remove seed from candidate items

    candidates = [i for i in d.keys() if i != seed]

    recs = heapq.nlargest(N, candidates, key=lambda x: d[x])

    out = {}

    out['user_id'] = time.time()  # FIXME

    out['recs'] = []

    for r in recs:

        out['recs'].append({r: Sparql.get_item_metadata(r)})

    out_json = json.dumps(out, indent=4, sort_keys=True)

    logger.info('total rec time')
    logger.info("--- %s seconds ---" % (time.time() - rec_time))

    return out_json


if __name__ == '__main__':

    app.run(debug=True)