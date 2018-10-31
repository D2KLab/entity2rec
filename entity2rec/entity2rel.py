from __future__ import print_function
from gensim.models.keyedvectors import KeyedVectors
from sparql import Sparql
import codecs
import time


class Entity2Rel(object):

    """
    Computes a set of relatedness scores between a pair of entities from a set of property-specific Knowledge Graph embeddings
    """

    def __init__(self, binary=True):

        self.binary = binary
        self.embedding_files = {}

    # add embedding file
    def add_embedding(self, property, embedding_file):

        self.embedding_files[property] = KeyedVectors.load_word2vec_format(embedding_file, binary=self.binary)

    def relatedness_score(self, property, uri1, uri2):

        emb_file = self.embedding_files[property]

        try:

            score = emb_file.similarity(uri1, uri2)

        except KeyError:

            score = 0.

        return score

    # parse ceccarelli benchmark line
    @staticmethod
    def parse_ceccarelli_line(line):

        line = line.split(' ')

        relevance = int(line[0])

        query_id = int((line[1].split(':'))[1])

        doc_id = line[-1]

        ids = line[-2].split('-')

        wiki_id_query = int(ids[0])

        wiki_id_candidate = int(ids[1])

        return wiki_id_query, query_id, wiki_id_candidate, relevance, doc_id

    # write line in the svm format
    def write_line(self, query_uri, qid, candidate_uri, relevance, file, doc_id):

        scores = self.relatedness_scores(query_uri, candidate_uri)

        file.write('%d qid:%d' %(relevance,qid))

        count = 1

        l = len(scores)

        for score in scores:

            if count == l:  # last score, end of line

                file.write(' %d:%f # %s-%s %d\n' %(count,score,query_uri,candidate_uri, int(doc_id)))

            else:

                file.write(' %d:%f' %(count,score))

                count += 1

    def feature_generator(self, data):

        data_name = (data.split('/')[-1]).split('.')[0]

        with codecs.open('features/ceccarelli/%s.svm' %data_name,'w', encoding='utf-8') as data_write:

            with codecs.open(data,'r', encoding='utf-8') as data_read:

                for i, line in enumerate(data_read):

                    wiki_id_query, qid, wiki_id_candidate, relevance, doc_id = self.parse_ceccarelli_line(line)

                    print(wiki_id_query)

                    uri_query = Sparql.get_uri_from_wiki_id(wiki_id_query)

                    uri_candidate = Sparql.get_uri_from_wiki_id(wiki_id_candidate)

                    self.write_line(uri_query, qid, uri_candidate, relevance, data_write, doc_id)

        print('finished writing features')

        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':

    # test

    start_time = time.time()

    uri1 = "http://dbpedia.org/resource/Pulp_Fiction"

    uri2 = "http://dbpedia.org/resource/Jackie_Brown_(film)"

    uri3 = "http://dbpedia.org/resource/Romeo_and_Juliet_(1996_movie)"

    embedding1 = "emb/movielens_1m/feedback/num500_p1_q4_l10_d500_iter5_winsize10.emd"

    embedding2 = "emb/movielens_1m/dbo:director/num500_p1_q4_l10_d500_iter5_winsize10.emd"

    rel = Entity2Rel()

    rel.add_embedding(embedding1)
    rel.add_embedding(embedding2)

    print('\n')
    print("Relatedness between Pulp Fiction and Jackie Brown is:\n")
    scores = rel.relatedness_scores(uri1, uri2)
    for s in scores:
        print(s)
        print('\n')

    print("Relatedness between Pulp Fiction and Romeo and Juliet is:\n")
    scores = rel.relatedness_scores(uri1, uri3)

    for s in scores:
        print(s)
        print('\n')

    print("--- %s seconds ---" % (time.time() - start_time))