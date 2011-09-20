import numpy as np
from operator import itemgetter
from tempfile import mkdtemp
from scipy.sparse import lil_matrix, csr_matrix, issparse
import RDF
import pickle
import os

class Space(object):

    decay = 0.9
    max_depth = 5

    def __init__(self, path_to_rdf, format='ntriples', property='http://www.w3.org/2004/02/skos/core#broader'):
        self._path_to_rdf = 'file:' + path_to_rdf
        self._format = format
        self._property = property
        self._direct_parents = None
        self._index = {}
        self.generate_index(self._get_statement_stream())

    def _get_statement_stream(self):
        parser = RDF.Parser(name=self._format)
        return parser.parse_as_stream(self._path_to_rdf)

    def generate_index(self, stream):
        if self._direct_parents != None:
            return

        parents = {}
        z = 0
        k = 0

        for statement in stream:
            p = str(statement.predicate.uri)
            if statement.object.is_resource() and p == self._property:
                s = str(statement.subject.uri)
                o = str(statement.object.uri)
                if not parents.has_key(s):
                    parents[s] = [o]
                else:
                    parents[s].append(o)
                if not self._index.has_key(o):
                    self._index[o] = k
                    k += 1

            z += 1
            if z % 100000 == 0:
                print "Processed " + str(z) + " triples..."
        self._size = k

        self._direct_parents = parents

    def index(self, uri):
        return self._index[uri]

    def parents(self, uri, done=None, weight=1):
        if done is None:
            done = []
        # We stop after 5 recursions, otherwise we accumulate too much generic junk at the top of the hierarchy
        if len(done) > Space.max_depth or uri in done or not self._direct_parents.has_key(uri):
            return []
        done.append(uri)
        parents = [(direct_parent, weight) for direct_parent in self._direct_parents[uri]]
        indirect_parents = []
        for (parent, weight) in parents:
            indirect_parents.extend(self.parents(parent, done, weight * Space.decay))
        parents.extend(indirect_parents)
        # FIXME We will count twice parents that appear at different levels
        return list(set(parents))

    def to_vector(self, uri):
        v = lil_matrix((1, self._size))
        for (parent, weight) in self.parents(uri):
            v[0, self.index(parent)] += weight
        return v.tocsr()

    def distance_uri(self, uri1, uri2):
        v1 = self.to_vector(uri1)
        v2 = self.to_vector(uri2)
        return self.distance(v1, v2)

    def distance(self, v1, v2):
        return v1.dot(v2.T)[0, 0] / (self.sparse_norm(v1) * self.sparse_norm(v2))

    def sparse_norm(self, v):
        if issparse(v):
            return np.sqrt(v.dot(v.T)[0, 0])
        else:
            return np.linalg.norm(v)

    def centroid_weighted_uris(self, vs):
        vectors = []
        p = sum([w for (uri, w) in vs])
        for (uri, weight) in vs:
            vectors.append(self.to_vector(uri) * weight / p)
        return self.centroid(vectors)

    def centroid(self, vectors):
        return np.mean(vectors, axis=0)

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
