import numpy as np
import scipy.linalg as linalg
from scipy import *
from scipy.sparse import *
from sparsesvd import sparsesvd
from numpy.linalg import *
from operator import itemgetter
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
        self.generate_index(self._get_statement_stream())

    def _get_statement_stream(self):
        parser = RDF.Parser(name=self._format)
        return parser.parse_as_stream(self._path_to_rdf)

    def generate_index(self, stream):
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
                if not r_index.has_key(o):
                    r_index[o] = [s]
                else:
                    r_index[o].append(s)
                if index.has_key(o):
                    index[s].extend(index[o])
                if r_index.has_key(s):
                    for r in r_index[s]:
                        if index.has_key(r):
                            index[r].extend(index[s])
                            index[r] = list(Set(index[r]))
                index[s] = list(Set(index[s]))

            z += 1
            if z % 100000 == 0:
                print "Processed " + str(z) + " triples..."

        self._index = index
