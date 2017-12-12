import pyltr

with open('features/movielens_1m/p1_q1/train_p1_q1.svm') as trainfile, \
        open('features/movielens_1m/p1_q1/test_p1_q1.svm') as evalfile:
    TX, Ty, Tqids, _ = pyltr.data.letor.read_dataset(trainfile)
    EX, Ey, Eqids, _ = pyltr.data.letor.read_dataset(evalfile)

metric = pyltr.metrics.NDCG(k=10)

model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.02,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
)

model.fit(TX, Ty, Tqids)

Epred = model.predict(EX)
print('Random ranking:', metric.calc_mean_random(Eqids, Ey))
print('Our model:', metric.calc_mean(Eqids, Ey, Epred))