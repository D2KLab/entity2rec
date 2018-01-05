#! /bin/bash

dataset="LastFM"

# create feedback graph
cut -f1,2 datasets/$dataset/train.dat -d' ' > datasets/$dataset/graphs/feedback.edgelist

# download property-specific subgraphs from knowledge graph
python -u entity2rec/sparql.py --endpoint http://dbpedia.org/sparql --dataset $dataset

# run entity2rec
python -u entity2rec/main.py --dataset $dataset --train datasets/$dataset/train.dat --validation datasets/$dataset/val.dat --test datasets/$dataset/test.dat --run_all
