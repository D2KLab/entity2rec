import time
import pickle
from collections import defaultdict, Counter
import heapq
import logging
from flask import Flask
from flask import request
import json
from pymongo import MongoClient
import random
from flask_cors import CORS
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

start_time = time.time()

version_api = '0.1'

dataset = 'LibraryThing'

testing = False

@app.before_first_request
def load_model():

    print('loading model')

    # open item to item similarity matrix and read into dictionary
    with open('datasets/'+dataset+'/item_to_item_similarity_Entity2Rec', 'rb') as f1:
        global item_to_item_similarity_dict_entity2rec
        item_to_item_similarity_dict_entity2rec = pickle.load(f1)  # seed -> {item: score}

    if testing:

    # open item to item similarity matrix and read into dictionary
        with open('datasets/'+dataset+'/item_to_item_similarity_ItemKNN', 'rb') as f2:
            global item_to_item_similarity_dict_itemknn
            item_to_item_similarity_dict_itemknn = pickle.load(f2)  # seed -> {item: score}

@app.before_first_request
def read_item_metadata():

    # reads list of item in the dataset
    global items
    items = []

    # item popularity
    global pop_dict
    pop_dict = Counter()

    with open('datasets/'+dataset+'/all.dat') as all_ratings:

        for line in all_ratings:
            line_split = line.strip('\n').split(' ')

            item = line_split[1]

            pop_dict[item] +=1

    global probs
    probs = []
    tot_sum = sum(pop_dict.values())

    for key, value in pop_dict.items():

        items.append(key)
        probs.append(value/tot_sum)

    # reads items metadata from sparql endpoint and keeps them in memory
    global item_metadata
    item_metadata = {}
    
    for item in items:

        metadata = get_item_metadata(item)

        if metadata:  # skip items with missing metadata

            item_metadata[item] = metadata

        else:

            del pop_dict[item]  # remove item

        logger.info("%s\n" %item)

    global num_items
    num_items = len(item_metadata)

@app.route('/entity2rec/' + version_api + "/onboarding", methods=['GET'])
def onboarding():

    out = {}

    out['user_id'] = time.time()  # FIXME

    global item_to_item_similarity_dict
    global algorithm


    if testing:
        # A/B testing
        if random.random() >= 0.5:
            item_to_item_similarity_dict = item_to_item_similarity_dict_entity2rec
            algorithm = 'entity2rec'

        else:
            item_to_item_similarity_dict = item_to_item_similarity_dict_itemknn
            algorithm = 'itemknn'

    else:
        item_to_item_similarity_dict = item_to_item_similarity_dict_entity2rec
        algorithm = 'entity2rec'

    item_to_item_similarity_dict = item_to_item_similarity_dict_entity2rec
    algorithm = 'entity2rec'

    number_of_samples = 100

    if num_items < number_of_samples:

        number_of_samples = num_items

    for sampled_item in np.random.choice(items, number_of_samples, p=probs):

        out[sampled_item] = item_metadata[sampled_item]

    out_json = json.dumps(out, indent=4)

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

    d = item_to_item_similarity_dict_entity2rec[seed]

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

    content['algorithm'] = algorithm

    connection = MongoClient('localhost', 27027)
    db = connection.db
    collection = db.feedback

    collection.save(content)

    return 'ok\n'


def get_item_metadata(uri):

    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    sparql.setQuery("""select ?labelo ?labelp ?labels ?description ?abstract ?thumbnail ?homepage
                    where {

                    OPTIONAL {
                      <%s> <http://dbpedia.org/ontology/label> ?labelo .
                      FILTER(lang(?labelo) = 'en' )
                     }

                    OPTIONAL {
                      <%s> <http://dbpedia.org/property/label> ?labelp .
                      FILTER(lang(?labelp) = 'en' )
                    }

                    OPTIONAL {
                      <%s> <http://www.w3.org/2000/01/rdf-schema#label> ?labels.
                      FILTER(lang(?labels) = 'en' )
                    }

                    OPTIONAL {
                    <%s> <http://purl.org/dc/terms/description> ?description .
                    FILTER (lang(?description) = 'en')
                    }
                    OPTIONAL {
                    <%s> <http://dbpedia.org/ontology/thumbnail> ?thumbnail .
                    }
                    OPTIONAL{
                    <%s> <http://xmlns.com/foaf/0.1/homepage> ?homepage .
                    }
                    OPTIONAL {
                      <%s> <http://dbpedia.org/ontology/abstract> ?abstract .
                      FILTER (lang(?abstract) = 'en')
                    }

                    } """ % (uri, uri, uri, uri, uri, uri, uri))
    

    sparql.setReturnFormat(JSON)

    try:  # check whether it does not return an empty list

        result_raw = sparql.query().convert()['results']['bindings'][0]

        result = {}

        for key, value in result_raw.items():

            result[key] = value['value']

        c = 0

        try:

            result['label'] = result['labels']

        except KeyError:
            c+=1
            pass

        try:

            result['label'] = result['labelp']

        except KeyError:
            c+=1
            pass

        try:

            result['label'] = result['labelo']

        except KeyError:
            c+=1
            pass

        # at least one label must be there
        if c == 3: 
            result = None

        # either abstract or description must be there
        if 'abstract' not in result.keys() and 'description' not in result.keys():
            result = None

        # if thumbnail is not there, scrape google
        if 'thumbnail' not in result.keys():

            out=subprocess.check_output(["googleimagesdownload", "--keywords", "\"%s book\"" % result['label'], "--print_urls", "-l", "1"])

            url = out.decode('utf-8').split('\n')[4].replace('Image URL: ','')

            result['thumbnail'] = url
    except:

        result = None

    return result


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5888, debug=True)