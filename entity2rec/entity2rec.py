from __future__ import print_function
import numpy as np
from entity2vec import Entity2Vec
from entity2rel import Entity2Rel
import pyltr
import sys
sys.path.append('.')
from metrics import precision_at_n, mrr, recall_at_n


class Entity2Rec(Entity2Vec, Entity2Rel):

    """Computes a set of relatedness scores between user-item pairs from a set of property-specific Knowledge Graph
    embeddings and user feedback and feeds them into a learning to rank algorithm"""

    def __init__(self, dataset, run_all=False,
                 is_directed=False, preprocessing=True, is_weighted=False,
                 p=1, q=4, walk_length=10,
                 num_walks=500, dimensions=500, window_size=10,
                 workers=8, iterations=5, config='config/properties.json',
                 feedback_file=False):

        Entity2Vec.__init__(self, is_directed, preprocessing, is_weighted, p, q, walk_length, num_walks, dimensions,
                            window_size, workers, iterations, config, dataset, feedback_file)

        Entity2Rel.__init__(self)  # binary format embeddings

        self.define_properties()

        # run entity2vec to create the embeddings
        if run_all:
            print('Running entity2vec to generate property-specific embeddings...')
            self.e2v_walks_learn()  # run entity2vec

        # reads the embedding files
        self._set_embedding_files()

        # initialize model to None
        self.model = None

    def _set_embedding_files(self):

        """
        Sets the list of embedding files
        """

        for prop in self.properties:
            prop_short = prop
            if '/' in prop:
                prop_short = prop.split('/')[-1]

            self.add_embedding(u'emb/%s/%s/num%s_p%d_q%d_l%s_d%s_iter%d_winsize%d.emd' % (
                self.dataset, prop_short, self.num_walks, int(self.p), int(self.q), self.walk_length, self.dimensions,
                self.iter,
                self.window_size))

    def collab_similarity(self, user, item):

        # feedback property

        return self.relatedness_score_by_position(user, item, -1)

    def content_similarities(self, user, item, items_liked_by_user):

        # all other properties

        sims = []

        for past_item in items_liked_by_user:
            sims.append(self.relatedness_scores(past_item, item,
                                                -1))  # append a list of property-specific scores, skip feedback

        if len(sims) == 0:  # no content properties for the item
            sims = 0.5 * np.ones(len(self.properties) - 1)
            return sims

        return np.mean(sims, axis=0)  # return a list of averages of property-specific scores

    def _compute_scores(self, user, item, items_liked_by_user):

        collab_score = self.collab_similarity(user, item)

        content_scores = self.content_similarities(user, item, items_liked_by_user)

        return collab_score, content_scores

    def compute_user_item_features(self, user, item, items_liked_by_user):

        collab_score, content_scores = self._compute_scores(user, item, items_liked_by_user)

        features = [collab_score] + list(content_scores)

        return features

    def fit(self, x_train, y_train, qids_train, x_val=None, y_val=None, qids_val=None,
            optimize='AP', N=10, lr=0.1, n_estimators=100, max_depth=3,
            max_features=None):

        # choose the metric to optimize during the fit process

        if optimize == 'NDCG':

            fit_metric = pyltr.metrics.NDCG(k=N)

        elif optimize == 'P':

            fit_metric = precision_at_n.PrecisionAtN(k=N)

        elif optimize == 'MRR':

            fit_metric = mrr.MRR(k=N)

        elif optimize == 'AP':

            fit_metric = pyltr.metrics.AP(k=N)

        else:

            raise ValueError('Metric not implemented')

        self.model = pyltr.models.LambdaMART(
            metric=fit_metric,
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=max_depth,
            max_features=max_features,
            verbose=1
        )

        # Only needed if you want to perform validation (early stopping & trimming)

        if x_val is not None and y_val is not None and qids_val is not None:

            monitor = pyltr.models.monitors.ValidationMonitor(
                x_val, y_val, qids_val, metric=fit_metric)

            self.model.fit(x_train, y_train, qids_train, monitor=monitor)

        else:

            self.model.fit(x_train, y_train, qids_train)

    def predict(self, x_test):

        return self.model.predict(x_test)


