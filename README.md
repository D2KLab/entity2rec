# entity2rec

Implementation of the entity recommendation algorithm described in [entity2rec: Learning User-Item Relatedness from Knowledge Graphs for Top-N Item Recommendation](https://enricopal.github.io/enricopal.github.io/publications/Palumbo_Rizzo-RecSys2017.pdf). Compute user and item embeddings from a Knowledge Graph encompassing both user feedback information and Linked Open Data information. It is based on property-specific entity embeddings, which are obtained via entity2vec (https://github.com/MultimediaSemantics/entity2vec). 

For a usage example, see the [Wiki](https://github.com/MultimediaSemantics/entity2rec/wiki) section.

## Requirements

- Python 2.7 or above
- numpy
- gensim
- networkx
- pandas
- SPARQL Wrapper

If you are using `pip`:

        pip install gensim networkx pandas SPARQLWrapper

## Our Publications

* Palumbo E., Rizzo G., Troncy R. (2017) [entity2rec: Learning User-Item Relatedness from Knowledge Graphs for Top-N Item Recommendation](https://enricopal.github.io/enricopal.github.io/publications/Palumbo_Rizzo-RecSys2017.pdf). In 11th ACM Conference on Recommender Systems (RecSys) , Como, Italy, 
