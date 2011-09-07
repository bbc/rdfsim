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

    def parents(self, uri, done=[]):
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

    def distance(self, uri1, uri2):
        uri1_p = self.parents(uri1, [])
        uri2_p = self.parents(uri2, [])
        return float(len(list(set(uri1_p) & set(uri2_p)))) / (np.sqrt(len(uri1_p)) * np.sqrt(len(uri2_p))) 
