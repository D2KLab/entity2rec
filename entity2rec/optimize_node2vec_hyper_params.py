import time
import numpy as np
from evaluator import Evaluator
from parse_args import parse_args
from node2vec_recommender import Node2VecRecommender
from node2vec import Node2Vec
import sys

sys.path.append('.')

np.random.seed(1)  # fixed seed for reproducibility

start_time = time.time()

print('Starting node2vec recommender...')

args = parse_args()

d_v = [200, 500]

p_v = [0.25, 1, 4]

q_v = [0.25, 1, 4]

c_v = [10, 15]

scores = {}

for d in d_v:

    for p in p_v:

        for q in q_v:

            for c in c_v:

                # run node2vec to generate the embedding

                node2vec_graph = Node2Vec(args.directed, args.preprocessing, args.weighted, p, q,
                                          args.walk_length,
                                          args.num_walks, d, c, args.workers, args.iter)

                node2vec_graph.run('datasets/%s/graphs/feedback.edgelist' % args.dataset,
                                   'emb/%s/feedback/num%d_p%d_q%d_l%d_d%d_iter%d_winsize%d.emd'
                                   % (args.dataset, args.num_walks, p, q,
                                      args.walk_length, d, args.iter, c))

                # initialize entity2rec recommender
                node2vec_rec = Node2VecRecommender(args.dataset, p=p, q=q,
                                                   walk_length=args.walk_length, num_walks=args.num_walks,
                                                   dimensions=d, window_size=c,
                                                   iterations=args.iter)

                # initialize evaluator

                evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold,
                                    all_unrated_items=args.all_unrated_items)

                # compute e2rec features
                x_train, y_train, qids_train, x_test, y_test, qids_test, \
                x_val, y_val, qids_val = evaluat.features(node2vec_rec, args.train, args.test,
                                                          validation=False, n_users=args.num_users,
                                                          n_jobs=args.workers, supervised=False)

                print('Finished computing features after %s seconds' % (time.time() - start_time))

                print('c:%d,d:%d,p:%.2f,q:%.2f\n' % (c, d, p, q))

                scores[(p, q, c, d)] = evaluat.evaluate(node2vec_rec, x_test, y_test, qids_test)  # evaluates the recommender on the test set

                print("--- %s seconds ---" % (time.time() - start_time))

with open('node2vec_hyper_opt.csv', 'w') as hyper_opt_write:

    hyper_opt_write.write('p,q,c,d,P@5,P@10,MAP,R@5,R@10,NDCG,MRR\n')

    for (p, q, c, d), scores in scores.items():

        hyper_opt_write.write('%.2f,%.2f,%d,%d,' % (p, q, c, d))

        for name, score in scores[0:-1]:
            hyper_opt_write.write('%f,' % score)

        hyper_opt_write.write('%f\n' % scores[-1][1])
