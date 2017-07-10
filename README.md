# entity2rec

Implementation of the entity recommendation algorithm described in "entity2rec: Learning User-Item Relatedness from Knowledge Graphs for Top-N Item Recommendation". Compute user and item embeddings from a Knowledge Graph encompassing both user feedback information (`movielens_1m/graphs/feedback.edgelist`) and Linked Open Data information (`movielens_1m/graphs/dbpedia_property.edgelist`) on the Movielens 1M dataset. It is based on property-specific entity embeddings, which are obtained via entity2vec (https://github.com/MultimediaSemantics/entity2vec). You can obtain the property-specific embeddings following the indications provided in the entity2vec repository. If you are already familiar with the procedure, you can simply add `--run_all` when you run entity2rec for the first time to compute the embeddings. It adopts by default the _AllItems_ candidate generation for testing, which means that features are computed for each user-item pair that is not appearing in the training set. Thus, for each user, all items in the database can be ranked to obtain top-N item recommendation.

Before starting:

`mkdir datasets/your_dataset`

`mkdir datasets/your_dataset/graphs`

Move your train and test files inside datasets/your_dataset:

`mv train.dat datasets/your_dataset/train.dat`

`mv test.dat datasets/your_dataset/test.dat`

Create a file containing user feedback of the training set in datasets/your_dataset/graphs/feedback.edgelist:

`cut -f1,2 datasets/your_dataset/train.dat -d' ' > datasets/your_dataset/graphs/feedback.edgelist`

Then run:

 ` python src/entity2rec.py --dataset my_dataset --train datasets/my_dataset/training_set.dat --test datasets/my_dataset/test_set.dat --run_all`

The command accepts all the params of _entity2vec_ and, in addition:

|option          | default                |description |
|----------------|------------------------|------------|
|`train`         | null **(Required)**    | Path of the train set |
|`test`          | null **(Required)**    | Path of the test set |
|`run_all`       | false                  | If `true`, it runs _entity2vec_ to compute the embeddings before the recommendation task (in this case, it is suggested to add also the related command line arguments (https://github.com/MultimediaSemantics/entity2vec)). Otherwise, it expects that the embeddings are in the `emb/` folder. Note that this needs to be done only the first time. |
|`implicit`      | false                  | If `true`, it expects that the ratings are binary values (0/1) instead of a range of scores |


The training and test set have the format:

    user_id item_id rating timestamp

where the `user_id` should be an integer, possibly preceded by the string `user` (i.e. `13` or `user13`).

As an output, entity2rec will generate a set of property-specific relatedness scores in the SVM format inside the folder features/my_dataset:

1 qid:1 1:0.186378 2:0.000000 3:0.329318 4:0.420169 5:0.000000 6:0.407551 7:0.000000 8:0.355113 9:0.198874 10:0.146273 11:0.354844 # http://dbpedia.org/resource/The_Secret_Garden_(1993_film)

This file can be used as input of https://sourceforge.net/p/lemur/wiki/RankLib/ to learn the global relatedness model.

`cd ranking`

`java -jar RankLib-2.1-patched.jar -train ../features/your_dataset/train.svm -ranker 6 -metric2t P@10 -tvs 0.9 -test ../features/your_dataset/test.svm`

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

* Palumbo E., Rizzo G., Troncy R. (2017) entity2rec: Learning User-Item Relatedness from Knowledge Graphs for Top-N Item Recommendation. In 11th ACM Conference on Recommender Systems (RecSys) , Como, Italy
