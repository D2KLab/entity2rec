# entity2rec

Implementation of the entity recommendation algorithm described in [entity2rec: Learning User-Item Relatedness from Knowledge Graphs for Top-N Item Recommendation](https://enricopal.github.io/enricopal.github.io/publications/entity2rec.pdf). Compute user and item embeddings from a Knowledge Graph encompassing both user feedback information and item information. It is based on property-specific entity embeddings, which are obtained via entity2vec (https://github.com/MultimediaSemantics/entity2vec). Slides can be found on [Slideshare]( https://www.slideshare.net/EnricoPalumbo2/entity2rec-recsys). 
The main difference between the current implementation and what is reported in the paper is the evaluation protocol, which now ranks all the items for each user, and the use of hybrid property-specific subgraphs to compute the property-specific knowledge graph embeddings.

The command:
`python entity2rec/main.py --dataset LibraryThing --run_all`
will run entity2rec on the LibraryThing dataset. The first time it will generate the embeddings files and save them into emb/LibraryThing/. Afterwards, it will check whether the embeddings files already exist in the folder. Then, it computes property-specific relatedness scores and evaluates recommendations using a set of possible aggregation functions (LambdaMart, average, max and min) on a set of relevant metrics.

The configuration of properties.json is used to select the properties. By default, it will use hybrid property-specific subgraphs (feedback + content property). To use collaborative-content subgraphs, the user should replace the content of config/properties.json with that of config/properties_collaborative_content_example.json.

## Requirements

- Python 2.7 or above

If you are using `pip`:

        pip install -r requirements.txt

## Our Publications

* Palumbo E., Rizzo G., Troncy R. (2017) [entity2rec: Learning User-Item Relatedness from Knowledge Graphs for Top-N Item Recommendation](https://enricopal.github.io/enricopal.github.io/publications/entity2rec.pdf). In 11th ACM Conference on Recommender Systems (RecSys) , Como, Italy, 

* Palumbo E., Monti D., Rizzo G., Troncy R., Baralis E. (2020) [entity2rec: Property-specific Knowledge Graph Embeddings for Recommender Systems](https://www.sciencedirect.com/science/article/pii/S0957417420300610), Expert Systems with Applications 
