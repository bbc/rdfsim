import numpy as np
import scipy.linalg as linalg
from scipy import *
from scipy.sparse import *
from sparsesvd import sparsesvd
from numpy.linalg import *
from operator import itemgetter
from tempfile import mkdtemp
import RDF
import pickle
import os
from sets import Set

class Space(object):

    def __init__(self, path_to_rdf, format='ntriples', property='http://www.w3.org/2004/02/skos/core#broader'):
        self._path_to_rdf = 'file:' + path_to_rdf
        self._format = format
        self._property = property
        self._index = None
        self._matrix = None
        self.generate_index(self._get_statement_stream())
        # self.generate_matrix()

    def _get_statement_stream(self):
        parser = RDF.Parser(name=self._format)
        return parser.parse_as_stream(self._path_to_rdf)

    def generate_index(self, stream):
        """
            Generates a dictionary of the form:
                URI a => list of URIs b such that { URI <self._property>* b }
            from a set of RDF triples.
        """
        if self._index != None:
            return

        index = {}
        r_index = {}
        z = 0

        for statement in stream:
            p = str(statement.predicate.uri)
            if statement.object.is_resource() and p == self._property:
                s = str(statement.subject.uri)
                o = str(statement.object.uri)
                if not index.has_key(s):
                    index[s] = [o]
                else:
                    index[s].append(o)
                if not index.has_key(o):
                    index[o] = []
                if not r_index.has_key(o):
                    r_index[o] = [s]
                else:
                    r_index[o].append(s)
                if index.has_key(o):
                    index[s].extend(index[o])
                index[s] = list(Set(index[s]))
                self._propagate(index, r_index, s)

            z += 1
            if z % 100000 == 0:
                print "Processed " + str(z) + " triples..."

        self._index = index

    def _propagate(self, index, r_index, s, done=[]):
        if s not in done and r_index.has_key(s):
            for r in r_index[s]:
                if index.has_key(r):
                    index[r].extend(index[s])
                    index[r] = list(Set(index[r]))
                    done.append(r)
                    self._propagate(index, r_index, r, done)


    def generate_matrix(self):
        if self._matrix != None:
            return
        if not self._index:
            raise Exception("Index not initialised")
        keys = self._index.keys()

        k = 0
        uri_index = {}
        for key in keys:
            uri_index[key] = k
            k += 1

        n = len(keys)
        path = os.path.join(mkdtemp(), 'matrix.dat')
        matrix = np.memmap(path, dtype='float32', mode='w+', shape=(n,n))

        i = 0
        for key in keys:
            matrix[i, i] = 1.0
            for uri in self._index[key]:
                j = uri_index[uri]
                matrix[i, j] = 1.0
            if i % 1000 == 0:
                print i
                # Flushing and reloading
                del matrix
                matrix = np.memmap(path, dtype='float32', mode='w+', shape=(n,n)) 
            i += 1

        for j in range(0, n):
            matrix[:, j] /= norm(matrix[:, j])

        self._matrix = matrix
