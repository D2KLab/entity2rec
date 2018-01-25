from entity2rec import Entity2Rec
from evaluator import Evaluator
import time
from parse_args import parse_args
import numpy as np

np.random.seed(1)  # fixed seed for reproducibility

start_time = time.time()

print('Starting entity2rec...')

args = parse_args()

# initialize entity2rec recommender
e2rec = Entity2Rec(args.dataset, run_all=args.run_all, p=args.p, q=args.q,
                   feedback_file=args.feedback_file, walk_length=args.walk_length,
                   num_walks=args.num_walks, dimensions=args.dimensions, window_size=args.window_size,
                   workers=args.workers, iterations=args.iter, collab_only=args.collab_only,
                   content_only=args.content_only)

# initialize evaluator

evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

# compute e2rec features
x_train, y_train, qids_train, x_test, y_test, qids_test,\
x_val, y_val, qids_val = evaluat.features(e2rec, args.train, args.test,
                                          validation=args.validation, n_users=args.num_users)

print('Finished computing features after %s seconds' % (time.time() - start_time))
print('Starting to fit the model...')

with open('feature_evaluation_%s_p%d_q%d.csv' %(args.dataset, args.p, args.q), 'w') as feature_eval_file:

    feature_eval_file.write('feature,P@5,P@10,MAP,R@5,R@10,NDCG,MRR\n')  # write header

    feature_eval_file.write('all,')

    e2rec.fit(x_train, y_train, qids_train,
              x_val=x_val, y_val=y_val, qids_val=qids_val, optimize=args.metric, N=args.N)  # train the model

    scores = evaluat.evaluate(e2rec, x_test, y_test, qids_test,
                              verbose=False)  # evaluates the recommender on the test set

    for name, score in scores[0:-1]:
        feature_eval_file.write('%f,' % score)

    feature_eval_file.write('%f\n' % scores[-1][1])

    feature_names = e2rec.properties[0: -1]  # feedback at the end

    feature_names.insert(0, 'feedback')  # feedback at the beginning

    for i, feature_name in enumerate(feature_names):

        print(feature_name)

        feature_eval_file.write('%s,' % feature_name)

        # fit e2rec on one feature at the time

        train_feat = x_train[:, i]
        train_feat = train_feat.reshape((-1, 1))

        val_feat = x_val[:, i]
        val_feat = val_feat.reshape((-1, 1))

        test_feat = x_test[:, i]
        test_feat = test_feat.reshape((-1, 1))

        e2rec.fit(train_feat, y_train, qids_train,
                  x_val=val_feat, y_val=y_val, qids_val=qids_val, optimize=args.metric, N=args.N)  # train the model

        scores = evaluat.evaluate(e2rec, test_feat, y_test, qids_test,
                                  verbose=False)  # evaluates the recommender on the test set

        for name, score in scores[0:-1]:

            feature_eval_file.write('%f,' % score)

        feature_eval_file.write('%f\n' % scores[-1][1])
