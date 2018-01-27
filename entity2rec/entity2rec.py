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
                 feedback_file=False, collab_only=False, content_only=False):

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

        # whether using only collab or content features

        self.collab_only = collab_only

        self.content_only = content_only

    def _set_embedding_files(self):

        """
        Sets the list of embedding files
        """

        for prop in self.properties:
            prop_short = prop
            if '/' in prop:
                prop_short = prop.split('/')[-1]

            self.add_embedding(prop, u'emb/%s/%s/num%s_p%d_q%d_l%s_d%s_iter%d_winsize%d.emd' % (
                self.dataset, prop_short, self.num_walks, int(self.p), int(self.q), self.walk_length, self.dimensions,
                self.iter,
                self.window_size))

    def collab_similarity(self, user, item):

        # feedback property

        return self.relatedness_score('feedback', user, item)

    def content_similarities(self, user, item, items_liked_by_user):

        # all other properties

        sims = []

        if not items_liked_by_user:  # no past positive feedback

            sims = np.zeros(len(self.properties)-1)

        else:

            for prop in self.properties[0:-1]:  # append a list of property-specific scores, skip feedback

                sims_prop = []

                for past_item in items_liked_by_user:

                    sims_prop.append(self.relatedness_score(prop, past_item, item))

                s = np.mean(sims_prop)

                sims.append(s)

        return sims

    def _compute_scores(self, user, item, items_liked_by_user):

        collab_score = self.collab_similarity(user, item)

        content_scores = self.content_similarities(user, item, items_liked_by_user)

        return collab_score, content_scores

    def compute_user_item_features(self, user, item, items_liked_by_user):

        collab_score, content_scores = self._compute_scores(user, item, items_liked_by_user)

        if self.collab_only is False and self.content_only is False:

            features = [collab_score] + list(content_scores)

        elif self.collab_only is True and self.content_only is False:

            features = [collab_score]

        elif self.content_only is True and self.collab_only is False:

            features = list(content_scores)

        else:

            raise ValueError('Cannot be both collab only and content only')

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


