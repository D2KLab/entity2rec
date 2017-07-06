# entity2rec

Implementation of the entity recommendation algorithm described in "entity2rec: Learning User-Item Relatedness from Knowledge Graphs for Top-N Item Recommendation".
Compute user and item embeddings from a Knowledge Graph encompassing both user feedback information (`movielens_1m/graphs/feedback.edgelist`) and Linked Open Data information (`movielens_1m/graphs/dbpedia_property.edgelist`) on the Movielens 1M dataset. It is based on property-specific entity embeddings, which can computed for the first time calling _entity2rec_ using the command line argument `--run_all`. This will run entity2vec and compute property-specific embeddings using node2vec (https://github.com/MultimediaSemantics/entity2vec). It adopts by default the _AllItems_ candidate generation for testing, which means that features are computed for each user-item pair that is not appearing in the training set. Thus, for each user, all items in the database can be ranked to obtain top-N item recommendation.

    python src/entity2rec.py --dataset my_dataset --train training_set.dat --test test_set.dat

The command accepts all the params of _entity2vec_ and, in addition:

|option          | default                |description |
|----------------|------------------------|------------|
|`train`         | null **(Required)**    | Path of the train set in DAT format (see below for syntax) |
|`test`          | null **(Required)**    | Path of the test set in DAT format (see below for syntax)  |
|`run_all`       | false                  | If `true`, it runs _entity2vec_ to compute the embeddings before the recommendation task (in this case, it is suggested to add also the related command line arguments (https://github.com/MultimediaSemantics/entity2vec)). Otherwise, it expects that the embeddings are in the `emb\` folder |
|`implicit`      | false                  | If `true`, it expects that the ratings are binary values (0/1) instead of a range of scores |


The training and test set have the format:

    user_id item_id rating timestamp

where the `user_id` should be an integer, possibly preceded by the string `user` (i.e. `13` or `user13`).

As an output, it will generate a set of property-specific relatedness scores in the SVM format inside the folder features/my_dataset:

1 qid:1 1:0.186378 2:0.000000 3:0.329318 4:0.420169 5:0.000000 6:0.407551 7:0.000000 8:0.355113 9:0.198874 10:0.146273 11:0.354844 # http://dbpedia.org/resource/The_Secret_Garden_(1993_film)

This file can be used as input of https://sourceforge.net/p/lemur/wiki/RankLib/ to learn the global relatedness model.

`cd ranking
java -jar RankLib-2.1-patched.jar -train ../features/your_dataset/train.svm -ranker $ranker -metric2t your_metric -tvs 0.9 -test ../features/your_dataset/test.svm 

## Requirements

- Python 2.7 or above
- numpy
- gensim
- networkx
- pandas
- SPARQL Wrapper

If you are using `pip`:


        pip install gensim networkx pandas SPARQLWrapper
