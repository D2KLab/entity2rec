from __future__ import print_function
from SPARQLWrapper import SPARQLWrapper, JSON
import optparse
import codecs
from os import mkdir
import json
import subprocess


class Sparql(object):

    """SPARQL queries to define property list and get property-specific subgraphs"""

    def __init__(self, entities, config_file, dataset, endpoint, default_graph):

        self.entities = entities  # file containing a list of entities

        self.dataset = dataset

        self.wrapper = SPARQLWrapper(endpoint)

        self.wrapper.setReturnFormat(JSON)

        if default_graph:

            self.default_graph = default_graph

            self.wrapper.addDefaultGraph(self.default_graph)

        self.query_prop = "SELECT ?s ?o  WHERE {?s %s ?o. }"

        self.query_prop_uri = "SELECT ?s ?o  WHERE {?s %s ?o. FILTER (?s = %s)}"

        self._define_properties(config_file)

    def _define_properties(self, config_file):

        self.properties = []

        with codecs.open(config_file, 'r', encoding='utf-8') as config_read:

            property_file = json.loads(config_read.read())

            for property_name in property_file[self.dataset]['content']:

                if 'feedback_' in property_name:

                    property_name = property_name.replace('feedback_', '')

                self.properties.append(property_name)

    def get_property_graphs(self):

        properties = self.properties

        if 'feedback' in properties:
            properties.remove('feedback')  # don't query for the feedback property

        for prop in properties:  # iterate on the properties

            prop_short = prop

            prop_namespace = prop

            if '/' in prop:

                # avoid creating file with a '/' in the name
                prop_short = prop.split('/')[-1]

                # if it is actually a URI, surround by "<>"
                if prop.startswith("http"):
                    prop_namespace = '<' + prop + '>'

            try:
                mkdir('datasets/%s/' % self.dataset)
                mkdir('datasets/%s/graphs' % self.dataset)

            except:
                pass

            with codecs.open('datasets/%s/graphs/%s.edgelist' % (self.dataset, prop_short), 'w',
                             encoding='utf-8') as prop_graph:  # open a property file graph

                for uri in self.entities:

                    uri = '<' + uri + '>'

                    self.wrapper.setQuery(self.query_prop_uri % (prop_namespace, uri))

                    for result in self.wrapper.query().convert()['results']['bindings']:

                        subj = result['s']['value']

                        obj = result['o']['value']

                        print((subj, obj))

                        prop_graph.write('%s %s\n' % (subj, obj))

    @staticmethod
    def get_uri_from_wiki_id(wiki_id):

        sparql = SPARQLWrapper("http://dbpedia.org/sparql")

        sparql.setQuery("""select ?s where {?s <http://dbpedia.org/ontology/wikiPageID> %d
           }""" % int(wiki_id))

        sparql.setReturnFormat(JSON)

        try:
            uri = sparql.query().convert()['results']['bindings'][0]['s']['value']

        except:

            uri = None

        return uri

    @staticmethod
    def get_item_metadata(uri, item_type, thumbnail_exists):

        sparql = SPARQLWrapper("http://dbpedia.org/sparql")

        sparql.setQuery("""select ?labelo ?labelp ?labels ?description ?abstract ?homepage ?authorlabo ?authorlabp ?authorlabs
                        where {

                        OPTIONAL {
                          <%s> <http://dbpedia.org/ontology/label> ?labelo .
                          FILTER(lang(?labelo) = 'en' )
                         }

                        OPTIONAL {
                          <%s> <http://dbpedia.org/property/label> ?labelp .
                          FILTER(lang(?labelp) = 'en' )
                        }

                        OPTIONAL {
                          <%s> <http://www.w3.org/2000/01/rdf-schema#label> ?labels.
                          FILTER(lang(?labels) = 'en' )
                        }

                        OPTIONAL {
                        <%s> <http://purl.org/dc/terms/description> ?description .
                        FILTER (lang(?description) = 'en')
                        }
                        OPTIONAL{
                        <%s> <http://xmlns.com/foaf/0.1/homepage> ?homepage .
                        }
                        OPTIONAL {
                          <%s> <http://dbpedia.org/ontology/abstract> ?abstract .
                          FILTER (lang(?abstract) = 'en')
                        }
                        OPTIONAL {
                        <%s> dbo:author ?o.
                        ?o rdfs:label ?authorlabs.
                        FILTER (lang(?authorlabs) = 'en')
                        }
                        OPTIONAL {
                        <%s> dbo:author ?o.
                        ?o dbo:label ?authorlabo.
                        FILTER (lang(?authorlabo) = 'en')
                        }

                        OPTIONAL {
                        <%s> dbo:author ?o.
                        ?o dbp:label ?authorlabp.
                        FILTER (lang(?authorlabp) = 'en')
                        }

                        }""" % (uri, uri, uri, uri, uri, uri, uri, uri, uri))
        

        sparql.setReturnFormat(JSON)

        try:  # check whether it does not return an empty list

            result_raw = sparql.query().convert()['results']['bindings'][0]
            result = {}

            for key, value in result_raw.items():

                result[key] = value['value']

            c = 0

            try:

                result['label'] = result['labels']

            except KeyError:
                c+=1
                pass

            try:

                result['label'] = result['labelp']

            except KeyError:
                c+=1
                pass

            try:

                result['label'] = result['labelo']

            except KeyError:
                c+=1
                pass

            # at least one label must be there
            if c == 3: 
                result = None

            # same with author
            c = 0

            try:

                result['author'] = result['authorlabs']

            except KeyError:
                c+=1
                pass

            try:

                result['author'] = result['authorlabp']

            except KeyError:
                c+=1
                pass

            try:

                result['author'] = result['authorlabo']

            except KeyError:
                c+=1
                pass

            # at least one label must be there
            if c == 3: 
                result = None

            # either abstract or description must be there
            if 'abstract' not in result.keys() and 'description' not in result.keys():
                result = None

            if not thumbnail_exists:

                # scrape google for thumbnail

                out = subprocess.check_output(["googleimagesdownload", "--keywords", "\"%s %s %s\"" % (result['label'].replace(',',''), result['author'], item_type), "--print_urls", "-l", "1"])

                url = out.decode('utf-8').split('\n')[4].replace('Image URL: ','')

                result['thumbnail'] = url

                if not result['thumbnail']:  # skip item if there is not thumbnail
                    result = None

        except:

            result = None

        return result

if __name__ == '__main__':

    parser = optparse.OptionParser()
    parser.add_option('-e', '--entities', dest='entity_file', help='entity file name', default=False)
    parser.add_option('-c', '--config_file', default='config/properties.json', help='Path to configuration file')
    parser.add_option('-k', '--dataset', dest='dataset', help='dataset')
    parser.add_option('-m', '--endpoint', dest='endpoint', help='sparql endpoint')
    parser.add_option('-d', '--default_graph', dest='default_graph', help='default graph', default=False)

    (options, args) = parser.parse_args()

    entities = list()

    with open('datasets/%s/all.dat' % options.dataset, 'r') as read_all:

        for line in read_all:

            line_split = line.strip('\n').split(' ')

            entities.append(line_split[1])

    entities = list(set(entities))

    sparql_query = Sparql(entities, options.config_file, options.dataset, options.endpoint,
                          options.default_graph)

    sparql_query.get_property_graphs()
