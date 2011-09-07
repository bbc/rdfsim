from nose.tools import *
import numpy as np
from numpy.testing import *
import rdfspace
from rdfspace.space import Space

def test_init():
    rdf_space = Space('tests/example.n3')
    assert_equal(rdf_space._path_to_rdf, 'file:tests/example.n3')
    assert_equal(rdf_space._format, 'ntriples')
    assert_equal(rdf_space._property, 'http://www.w3.org/2004/02/skos/core#broader')
    assert_equal(rdf_space._index, {
        'http://dbpedia.org/resource/Category:Categories_named_after_television_series': ['http://dbpedia.org/resource/Category:Foo'],
        'http://dbpedia.org/resource/Category:Star_Trek': [
            'http://dbpedia.org/resource/Category:Foo', 
            'http://dbpedia.org/resource/Category:Categories_named_after_television_series',
        ],
        'http://dbpedia.org/resource/Category:Futurama': [
            'http://dbpedia.org/resource/Category:New_York_City_in_fiction', 
            'http://dbpedia.org/resource/Category:Foo', 
            'http://dbpedia.org/resource/Category:Categories_named_after_television_series'
        ],
        'http://dbpedia.org/resource/Category:Foo': [],
        'http://dbpedia.org/resource/Category:New_York_City_in_fiction': [],
    })
