from __future__ import print_function
from entity2rec import Entity2Rec
from evaluator import Evaluator
import sys
sys.path.append('.')
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR
import time
from parse_args import parse_args
from mostpop import MostPop

# evaluates LambdaMart vs other supervised classifiers on entity2rec features

start_time = time.time()

print('Starting entity2rec...')

args = parse_args()

# initialize entity2rec recommender
e2rec = Entity2Rec(args.dataset, run_all=args.run_all, p=args.p, q=args.q,
                   feedback_file=args.feedback_file, walk_length=args.walk_length,
                   num_walks=args.num_walks, dimensions=args.dimensions, window_size=args.window_size,
                   workers=args.workers, iterations=args.iter)

# initialize evaluator

evaluat = Evaluator(implicit=args.implicit, threshold=args.threshold, all_unrated_items=args.all_unrated_items)

# compute e2rec features
x_train, y_train, qids_train, x_test, y_test, qids_test, \
x_val, y_val, qids_val = evaluat.features(e2rec, args.train, args.test,
                                          validation=args.validation, n_users=args.num_users)

print('Finished computing features after %s seconds' % (time.time() - start_time))
print('Starting to fit the model...')

# fit e2rec on features

lin = LinearRegression()

lin.fit(x_train, y_train)

logistic = LogisticRegression()

logistic.fit(x_train, y_train)  # train the model

svm = SVR()

svm.fit(x_train, y_train)

print('linear')
evaluat.evaluate(lin, x_test, y_test, qids_test)  # evaluates the recommender on the test set

print('logistic')
evaluat.evaluate(logistic, x_test, y_test, qids_test)  # evaluates the recommender on the test set

print('svm')
evaluat.evaluate(svm, x_test, y_test, qids_test)  # evaluates the recommender on the test set

print('heuristics')
evaluat.evaluate_heuristics(x_test, y_test, qids_test)  # evaluates the heuristics on the test set

print('most pop')
mostpop_rec = MostPop(args.train, args.threshold)

x_train, y_train, qids_train, x_test, y_test, qids_test,\
    x_val, y_val, qids_val = evaluat.features(mostpop_rec, args.train, args.test, validation=args.validation,
                                              n_users=args.num_users)

evaluat.evaluate(mostpop_rec, x_test, y_test, qids_test)  # evaluates the model on the test set

