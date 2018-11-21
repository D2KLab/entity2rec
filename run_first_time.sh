#! /bin/bash

dataset="LibraryThing";
threshold=8;  # put 0.5 if you have implicit feedback
implicit="False";

# create feedback graph
cat datasets/$dataset/train.dat | awk -v thresh=$threshold -F' ' '$3 >= thresh' | cut -f1,2 -d' ' > datasets/$dataset/graphs/feedback.edgelist

# download property-specific subgraphs from knowledge graph
python -u entity2rec/sparql.py --endpoint http://dbpedia.org/sparql --dataset $dataset

# run entity2rec

if [ $implicit = "False" ]; then

    python -u entity2rec/main.py --dataset $dataset --train datasets/$dataset/train.dat --validation datasets/$dataset/val.dat --test datasets/$dataset/test.dat --run_all --threshold $threshold
    
else

    python -u entity2rec/main.py --dataset $dataset --train datasets/$dataset/train.dat --validation datasets/$dataset/val.dat --test datasets/$dataset/test.dat --run_all --threshold $threshold --implicit

fi