from nose.tools import *
import numpy as np
from rdfsim.space import Space

Space.decay = 0.9
Space.depth = 5

def test_init():
    space = Space('tests/example.n3')
    assert_equal(space._path_to_rdf, 'file:tests/example.n3')
    assert_equal(space._format, 'ntriples')
    assert_equal(space._property, 'http://www.w3.org/2004/02/skos/core#broader')
    assert_equal(space._direct_parents, {
        'http://dbpedia.org/resource/Category:Categories_named_after_television_series': ['http://dbpedia.org/resource/Category:Foo'],
        'http://dbpedia.org/resource/Category:Star_Trek': [
            'http://dbpedia.org/resource/Category:Categories_named_after_television_series',
        ],
        'http://dbpedia.org/resource/Category:Futurama': [
            'http://dbpedia.org/resource/Category:Categories_named_after_television_series',
            'http://dbpedia.org/resource/Category:New_York_City_in_fiction', 
        ],
    })
    assert_equal(space._index, {
        'http://dbpedia.org/resource/Category:Categories_named_after_television_series': 0,
        'http://dbpedia.org/resource/Category:New_York_City_in_fiction': 1,
        'http://dbpedia.org/resource/Category:Foo': 2,
    })
    assert_equal(space._size, 3)

def test_parents():
    space = Space('tests/example.n3')
    assert_equal(space.parents('http://dbpedia.org/resource/Category:Futurama'), [
        ('http://dbpedia.org/resource/Category:Categories_named_after_television_series', 1),
        ('http://dbpedia.org/resource/Category:New_York_City_in_fiction', 1),
        ('http://dbpedia.org/resource/Category:Foo', 0.9),
    ])
    assert_equal(space.parents('http://dbpedia.org/resource/Category:Star_Trek'), [
        ('http://dbpedia.org/resource/Category:Categories_named_after_television_series', 1),
        ('http://dbpedia.org/resource/Category:Foo', 0.9),
    ])
    assert_equal(space.parents('http://dbpedia.org/resource/Category:Foo'), [])

def test_to_vector():
    space = Space('tests/example.n3')
    np.testing.assert_array_equal(space.to_vector('http://dbpedia.org/resource/Category:Futurama').todense(), [[1, 1, 0.9]])
    np.testing.assert_array_equal(space.to_vector('http://dbpedia.org/resource/Category:Star_Trek').todense(), [[1, 0, 0.9]])

def test_distance_uri():
    space = Space('tests/example.n3')
    assert_equal(space.distance_uri('http://dbpedia.org/resource/Category:Futurama', 'http://dbpedia.org/resource/Category:Star_Trek'), (1 + 0.9 * 0.9) / (np.sqrt(2 + 0.9**2) * np.sqrt(1 + 0.9**2)))

def test_centroid_weighted_uris():
    space = Space('tests/example.n3')
    centroid = space.centroid_weighted_uris([('http://dbpedia.org/resource/Category:Futurama', 2), ('http://dbpedia.org/resource/Category:Star_Trek', 1)])
    np.testing.assert_allclose(np.asarray(centroid.todense()), [[0.5, 1.0 / 3, 3 * 0.9 / 6]])
