from __future__ import print_function
import codecs
import json
import numpy as np
import joblib
from entity2vec import Entity2Vec
from entity2rel import Entity2Rel
import pyltr
import sys
sys.path.append('.')
from metrics import precision_at_n, mrr, recall_at_n
from collections import defaultdict
from sklearn import preprocessing


class Property:

    def __init__(self, name, typology):

        self.name = name
        self._typology = typology

    @property
    def typology(self):

        return self._typology

    @typology.setter
    def typology(self, value):

        if value != 'collaborative' and value != 'content' and value != 'social':

            raise ValueError('Type of property can be: collaborative, content or social')

        else:

            self._typology = value


class Entity2Rec(Entity2Vec, Entity2Rel):

    """Computes a set of relatedness scores between user-item pairs from a set of property-specific Knowledge Graph
    embeddings and user feedback and feeds them into a learning to rank algorithm"""

    def __init__(self, dataset, run_all=False,
                 is_directed=False, preprocessing=True, is_weighted=False,
                 p=1, q=4, walk_length=10,
                 num_walks=500, dimensions=500, window_size=10,
                 workers=8, iterations=5, config='config/properties.json',
                 feedback_file=False, collab_only=False, content_only=False,
                 social_only=False):

        Entity2Vec.__init__(self, is_directed, preprocessing, is_weighted, p, q, walk_length, num_walks, dimensions,
                            window_size, workers, iterations, feedback_file)

        Entity2Rel.__init__(self)

        self.config_file = config

        self.dataset = dataset

        self.properties = []

        self._set_properties()

        # run entity2vec to create the embeddings
        if run_all:

            print('Running entity2vec to generate property-specific embeddings...')

            properties_names = []

            for prop in self.properties:

                properties_names.append(prop.name)

            self.e2v_walks_learn(properties_names, dataset)  # run entity2vec

        # reads the embedding files
        self._set_embedding_files()

        # initialize model to None
        self.model = None 

        # whether using only collab or content features

        self.collab_only = collab_only

        self.content_only = content_only

        self.social_only = social_only

        # initialize cluster models

        self.models = {}
        self.user_to_cluster = None

    def _set_properties(self):

        with codecs.open(self.config_file, 'r', encoding='utf-8') as config_read:

            property_file = json.loads(config_read.read())

            for typology in property_file[self.dataset]:

                for property_name in property_file[self.dataset][typology]:

                    self.properties.append(Property(property_name, typology))

    def _set_embedding_files(self):

        """
        Creates the dictionary of embedding files
        """

        for prop in self.properties:
            prop_name = prop.name
            prop_short = prop_name
            if '/' in prop_name:
                prop_short = prop_name.split('/')[-1]

            self.add_embedding(prop_name, u'emb/%s/%s/num%s_p%d_q%d_l%s_d%s_iter%d_winsize%d.emd' % (
                self.dataset, prop_short, self.num_walks, int(self.p), int(self.q), self.walk_length, self.dimensions,
                self.iter,
                self.window_size))

    def collab_similarities(self, user, item):

        # collaborative properties

        collaborative_properties = [prop for prop in self.properties if prop.typology == "collaborative"]

        sims = []

        for prop in collaborative_properties:

            sims.append(self.relatedness_score(prop.name, user, item))

        return list(sims)

    def content_similarities(self, user, item, items_liked_by_user):

        # content properties

        content_properties = [prop for prop in self.properties if prop.typology == "content"]

        sims = []

        if not items_liked_by_user:  # no past positive feedback

            sims = np.zeros(len(content_properties))

        else:

            for prop in content_properties:  # append a list of property-specific scores

                sims_prop = []

                for past_item in items_liked_by_user:

                    sims_prop.append(self.relatedness_score(prop.name, past_item, item))

                s = np.mean(sims_prop)

                sims.append(s)

        return list(sims)

    def social_similarities(self, user, item, users_liking_the_item):

        # social properties

        social_properties = [prop for prop in self.properties if prop.typology == "social"]

        sims = []

        if not users_liking_the_item:

            sims = np.zeros(len(social_properties))

        else:

            for prop in social_properties:  # append a list of property-specific scores

                sims_prop = []

                for past_user in users_liking_the_item:

                    sims_prop.append(self.relatedness_score(prop.name, past_user, user))

                s = np.mean(sims_prop)

                sims.append(s)

        return list(sims)

    def _compute_scores(self, user, item, items_liked_by_user, users_liking_the_item):

        collab_score = self.collab_similarities(user, item)

        content_scores = self.content_similarities(user, item, items_liked_by_user)

        social_scores = self.social_similarities(user, item, users_liking_the_item)

        return collab_score, content_scores, social_scores

    def compute_user_item_features(self, user, item, items_liked_by_user, users_liking_the_item):

        collab_scores, content_scores, social_scores = self._compute_scores(user, item, items_liked_by_user,
                                                                            users_liking_the_item)

        if self.collab_only:

            features = collab_scores

        elif self.content_only:

            features = content_scores

        elif self.social_only:

            features = social_scores

        else:

            features = collab_scores + content_scores + social_scores

        return features

    def fit(self, x_train, y_train, qids_train, x_val=None, y_val=None, qids_val=None,
            optimize='NDCG', N=None, lr=0.1, n_estimators=100, max_depth=5,
            max_features=None, user_to_cluster=None):

        # choose the metric to optimize during the fit process

        if not N:
            N = 10**8

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

        if user_to_cluster:

            self.models = {}

            self.user_to_cluster = user_to_cluster

            # map cluster id to users ids of the cluster
            cluster_to_users = defaultdict(list)

            for user, cluster in user_to_cluster.items():

                cluster_to_users[cluster].append(user)

            # iterate through cluster ids
            for user_cluster in list(set(self.user_to_cluster.values())):

                x_train_c, y_train_c, qids_train_c, x_val_c, y_val_c, qids_val_c = [],[],[],[],[],[]
                for i, qid in enumerate(qids_train):

                    if str(qid) in cluster_to_users[user_cluster]:

                        x_train_c.append(x_train[i])
                        y_train_c.append(y_train[i])
                        qids_train_c.append(qid)

                        x_val_c.append(x_val[i])
                        y_val_c.append(y_val[i])
                        qids_val_c.append(qids_val[i])

                """
                print(x_train_c)

                x_train_c = np.asarray(x_train_c).reshape(())
                y_train_c = np.asarray(y_train_c)
                qids_train_c = np.asarray(qids_train_c)
                items_train_c = np.asarray(items_train_c)

                x_val_c = np.asarray(x_val_c)
                y_val_c = np.asarray(y_val_c)
                qids_val_c = np.asarray(qids_val_c)
                items_val_c = np.asarray(items_val_c)
                """

                model = pyltr.models.LambdaMART(
                    metric=fit_metric,
                    n_estimators=n_estimators,
                    learning_rate=lr,
                    max_depth=max_depth,
                    max_features=max_features,
                    verbose=1,
                    random_state=1)
                    
                # Only needed if you want to perform validation (early stopping & trimming)

                if x_val is not None and y_val is not None and qids_val is not None:

                    monitor = pyltr.models.monitors.ValidationMonitor(
                        x_val_c, y_val_c, qids_val_c, metric=fit_metric)

                    model.fit(x_train_c, y_train_c, qids_train_c, monitor=monitor)

                else:

                    model.fit(x_train_c, y_train_c, qids_train_c)

                self.models[user_cluster] = model

        else:

            self.model = pyltr.models.LambdaMART(
                metric=fit_metric,
                n_estimators=n_estimators,
                learning_rate=lr,
                max_depth=max_depth,
                max_features=max_features,
                verbose=1,
                random_state=1
            )

            # Only needed if you want to perform validation (early stopping & trimming)

            if x_val is not None and y_val is not None and qids_val is not None:

                monitor = pyltr.models.monitors.ValidationMonitor(
                    x_val, y_val, qids_val, metric=fit_metric)

                self.model.fit(x_train, y_train, qids_train, monitor=monitor)

            else:

                self.model.fit(x_train, y_train, qids_train)


    def predict(self, x_test, qids_test):

        if self.user_to_cluster:
            
            preds = []
            
            for i, line in enumerate(x_test):

                qid = str(qids_test[i])

                cluster = self.user_to_cluster[qid]

                # retrieve the corresponding model of that cluster
                model = self.models[cluster]

                preds.append(model.predict(line.reshape(1, -1)))

            return preds

        else:

            return self.model.predict(x_test)

    def save_model(self, model_file):

        if not self.model:

            joblib.dump(self.model, model_file)

        else:

            raise AttributeError('Fit the model before saving it')

    def load_model(self, model_file):

        self.model = joblib.load(model_file)

    def recommend(self, user, qids_test, x_test, candidates, N=10):

        features_indeces = np.where(qids_test == user)

        features = x_test[features_indeces]

        preds = self.model.predict(features)

        items = np.lexsort(candidates, preds)[:N]

        return items
