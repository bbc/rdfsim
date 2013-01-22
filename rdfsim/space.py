# Copyright (c) 2011 British Broadcasting Corporation
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from operator import itemgetter
from tempfile import mkdtemp
from scipy.sparse import lil_matrix, csr_matrix, issparse
import RDF
import pickle
import os

class Space(object):
    """ Base class for a vector space derived from a RDF hierarchy """

    decay = 0.9
    max_depth = 5

    def __init__(self, path_to_rdf, format='ntriples', property='http://www.w3.org/2004/02/skos/core#broader'):
        self._path_to_rdf = 'file:' + path_to_rdf
        self._format = format
        self._property = property
        self._direct_parents = None
        self._index = {}
        self._uri_to_vector = {}
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
        """ Gets the index of a particular URI in the vector space """
        return self._index[uri]

    def parents(self, uri, done=None, weight=1):
        """ Retrieves the parents of a particular URI """
        if done is None:
            done = []
        if len(done) > Space.max_depth or uri in done or not self._direct_parents.has_key(uri):
            return []
        done.append(uri)
        parents = [(direct_parent, weight) for direct_parent in self._direct_parents[uri]]
        indirect_parents = []
        for (parent, weight) in parents:
            indirect_parents.extend(self.parents(parent, list(done), weight * Space.decay))
        parents.extend(indirect_parents)
        return list(set(parents))

    def to_vector(self, uri):
        """ Converts a URI to a vector """
        if self._uri_to_vector.has_key(uri):
            return self._uri_to_vector[uri]
        v = lil_matrix((1, self._size))
        indices = []
        for (parent, weight) in self.parents(uri):
            index = self.index(parent)
            v[0, index] += weight
            indices.append(index)
        norm = 0.0
        for index in indices:
            norm += v[0, index] ** 2
        norm = np.sqrt(norm)
        v /= norm
        v = v.tocsr()
        self._uri_to_vector[uri] = v
        return v

    def cache_vectors(self):
        """ Pre-caches all category vectors in memory """
        # TODO: Changes of max_depth and decay parameter won't be
        # taken into account anymore, once a vector is cached
        z = 0
        for uri in self._direct_parents.keys():
            self.to_vector(uri)
            z += 1
            if z % 100 == 0:
                print "Generated " + str(z) + " category vectors..."

    def similarity_uri(self, uri1, uri2):
        """ Derives a cosine similarity between two URIs """
        v1 = self.to_vector(uri1)
        v2 = self.to_vector(uri2)
        return self.similarity(v1, v2)

    def similarity(self, v1, v2):
        """ Derives a cosine similarity between two normalized vectors """
        return v1.dot(v2.T)[0, 0]

    def similarity_all(self, vs, v2):
        """ Derives a set of cosine similarity between a set of vectors and a vector """
        products = vs.dot(v2.T)[:,0]
        similarities = []
        for i in range(0, products.shape[0]):
            similarities.append(products[i,0])
        return similarities

    def centroid_weighted_uris(self, vs):
        """ Returns the centroid of a set of weighted vectors """
        vectors = []
        for (uri, weight) in vs:
            vectors.append(self.to_vector(uri) * weight)
        return self.centroid(vectors)

    def centroid(self, vectors):
        """ Returns the centroid of a set of vectors """
        return np.mean(vectors, axis=0)

    def sum_weighted_uris(self, vs):
        """ Returns the sum of weighted vectors """
        vectors = []
        for (uri, weight) in vs:
            vectors.append(self.to_vector(uri) * weight)
        return self.sum(vectors)

    def sum(self, vectors):
        """ Returns the sum of vectors """
        return np.sum(vectors, axis=0)

    def sparse_norm(self, v):
        if issparse(v):
            return np.sqrt(v.dot(v.T)[0, 0])
        else:
            return np.linalg.norm(v)

    def save(self, file):
        """ Save the vector space in a file """
        f = open(file, 'w')
        pickle.dump(self, f)
        f.close()

    @staticmethod
    def load(file):
        """ Load a space instance from a file """
        f = open(file)
        space = pickle.load(f)
        f.close()
        return space
