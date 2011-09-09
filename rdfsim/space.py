import numpy as np
from operator import itemgetter
from tempfile import mkdtemp
import RDF
import pickle
import os

class Space(object):

    def __init__(self, path_to_rdf, format='ntriples', property='http://www.w3.org/2004/02/skos/core#broader'):
        self._path_to_rdf = 'file:' + path_to_rdf
        self._format = format
        self._property = property
        self._direct_parents = None
        self.generate_index(self._get_statement_stream())

    def _get_statement_stream(self):
        parser = RDF.Parser(name=self._format)
        return parser.parse_as_stream(self._path_to_rdf)

    def generate_index(self, stream):
        if self._direct_parents != None:
            return

        parents = {}
        z = 0

        for statement in stream:
            p = str(statement.predicate.uri)
            if statement.object.is_resource() and p == self._property:
                s = str(statement.subject.uri)
                o = str(statement.object.uri)
                if not parents.has_key(s):
                    parents[s] = [o]
                else:
                    parents[s].append(o)

            z += 1
            if z % 100000 == 0:
                print "Processed " + str(z) + " triples..."

        self._direct_parents = parents

    def parents(self, uri, done=None):
        if done is None:
            done = []
        # We stop after 5 recursions, otherwise we accumulate too much generic junk at the top of the hierarchy
        if len(done) > 5 or uri in done or not self._direct_parents.has_key(uri):
            return []
        done.append(uri)
        parents = self._direct_parents[uri]
        indirect_parents = []
        for parent in parents:
            indirect_parents.extend(self.parents(parent, done))
        parents.extend(indirect_parents)
        return list(set(parents))

    def to_vector(self, uri):
        v = {}
        uri_p = self.parents(uri)
        for p in uri_p:
            v[p] = 1.0
        return v

    def distance_uri(self, uri1, uri2):
        v1 = self.to_vector(uri1)
        v2 = self.to_vector(uri2)
        return self.distance(v1, v2)

    def distance(self, v1, v2):
        common_keys = list(set(v1.keys()) & set(v2.keys()))
        product = 0.0
        for key in common_keys:
            product += v1[key] * v2[key]
        return product / (self.norm(v1) * self.norm(v2))
        
    def norm(self, v):
        return np.sqrt(sum(np.power(v.values(), 2)))

    def centroid(self, vs):
        parents = {}
        p = float(sum(vs.values()))
        for uri in vs.keys():
            for parent in self.parents(uri):
                if parents.has_key(parent):
                    parents[parent] += vs[uri] / p
                else:
                    parents[parent] = vs[uri] / p
        return parents

    def save(self, file):
        f = open(file, 'w')
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def load(file):
        f = open(file)
        space = pickle.load(f)
        f.close()
        return space
