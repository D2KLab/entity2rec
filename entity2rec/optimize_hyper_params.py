import time
import numpy as np
from entity2rec import Entity2Rec
from evaluator import Evaluator
from parse_args import parse_args
import os

np.random.seed(1)  # fixed seed for reproducibility

start_time = time.time()

print('Starting entity2rec recommender...')

args = parse_args()

if not args.train:
    args.train = 'datasets/'+args.dataset+'/train.dat'

if not args.test:
    args.test = 'datasets/'+args.dataset+'/val.dat'

if not args.validation:
    args.validation = 'datasets/'+args.dataset+'/val.dat'

d_v = [200]

p_v = [1, 4]

q_v = [1, 4]

c_v = [30, 50]

walks_v = [100]

length_v = [100]

scores = {}

for d in d_v:

    for p in p_v:

        for q in q_v:

            for c in c_v:

                for walks in walks_v:

                    for length in length_v:

                        # run node2vec to generate the embedding
                        e2rec = Entity2Rec(args.dataset, run_all=True, p=p, q=q,
                                           feedback_file=args.feedback_file, walk_length=length,
                                           num_walks=walks, dimensions=d, window_size=c,
                                           workers=args.workers, iterations=args.iter, collab_only=args.collab_only,
                                           content_only=args.content_only)

                        # initialize evaluator

                        evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)


                        # compute e2rec features
                        x_train, y_train, qids_train, items_train, x_test, y_test, qids_test, items_test,\
                        x_val, y_val, qids_val, items_val = evaluat.features(e2rec, args.train, args.test,
                                                                             validation=args.validation, n_users=args.num_users,
                                                                             n_jobs=args.workers, max_n_feedback=args.max_n_feedback)


                        # fit e2rec on features
                        e2rec.fit(x_train, y_train, qids_train,
                                  x_val=x_val, y_val=y_val, qids_val=qids_val, optimize=args.metric, N=args.N)  # train the model

                        print('Finished computing features after %s seconds' % (time.time() - start_time))

                        print('p:%.2f,q:%.2f,c:%d,d:%d,walks:%d,length:%d,\n' % (p, q, c, d, walks, length))

                        scores[(p, q, c, d, walks, length)] = evaluat.evaluate(e2rec, x_test, y_test, qids_test, items_test)  # evaluates the recommender on the test set

                        print("--- %s seconds ---" % (time.time() - start_time))

path = 'results/%s/entity2rec/' % args.dataset

os.makedirs(path, exist_ok=True)

with open(path+'hyper_params_opt.csv', 'w') as hyper_opt_write:

    hyper_opt_write.write('p,q,c,d,walks,length,P@5,P@10,MAP,R@5,R@10,NDCG,MRR,SER@5,SER@10\n')

    for (p, q, c, d, walks, length), results in scores.items():

        hyper_opt_write.write('%.2f,%.2f,%d,%d,%d,%d,' % (p, q, c, d, walks, length))

        for (strategy_name, metric_name), (mean, var) in results.items():
            if strategy_name == 'l2r':
                hyper_opt_write.write('%.4f,' % mean)

        hyper_opt_write.write('\n')
