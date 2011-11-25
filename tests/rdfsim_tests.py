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

from nose.tools import *
import numpy as np
from scipy.sparse import lil_matrix
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
    np.testing.assert_array_equal(space.to_vector('http://dbpedia.org/resource/Category:Futurama').todense(), [[1/np.sqrt(2 + 0.9**2), 1/np.sqrt(2 + 0.9**2), 0.9/np.sqrt(2 + 0.9**2)]])
    np.testing.assert_array_equal(space.to_vector('http://dbpedia.org/resource/Category:Star_Trek').todense(), [[1/np.sqrt(1 + 0.9**2), 0, 0.9/np.sqrt(1 + 0.9**2)]])

def test_similarity_uri():
    space = Space('tests/example.n3')
    np.testing.assert_allclose(space.similarity_uri('http://dbpedia.org/resource/Category:Futurama', 'http://dbpedia.org/resource/Category:Star_Trek'), (1 + 0.9 * 0.9) / (np.sqrt(2 + 0.9**2) * np.sqrt(1 + 0.9**2)))

def test_similarity_all():
    space = Space('tests/example.n3')
    m = lil_matrix((2, 3))
    m[0,0] = 1 / np.sqrt(1 + 2*2 + 3*3)
    m[0,1] = 2 / np.sqrt(1 + 2*2 + 3*3)
    m[0,2] = 3 / np.sqrt(1 + 2*2 + 3*3)
    m[1,0] = 4 / np.sqrt(4*4 + 5*5 + 6*6)
    m[1,1] = 5 / np.sqrt(4*4 + 5*5 + 6*6)
    m[1,2] = 6 / np.sqrt(4*4 + 5*5 + 6*6)
    v = m[0,:]
    m = m.tocsr()
    similarities = space.similarity_all(m, v)
    assert_equal(similarities[0], 1)
    assert_equal(similarities[1], ((1*4 + 2*5 + 3*6)/(np.sqrt(1 + 2*2 + 3*3)*np.sqrt(4*4 + 5*5 + 6*6))))

def test_centroid_weighted_uris():
    space = Space('tests/example.n3')
    centroid = space.centroid_weighted_uris([('http://dbpedia.org/resource/Category:Futurama', 2), ('http://dbpedia.org/resource/Category:Star_Trek', 1)])
    np.testing.assert_allclose(np.asarray(centroid.todense()), [[(2/np.sqrt(2 + 0.9**2) + 1/np.sqrt(1 + 0.9**2))/2, (1/np.sqrt(2 + 0.9**2)), (2*0.9/np.sqrt(2 + 0.9**2) + 0.9/np.sqrt(1 + 0.9**2))/2]])

def test_sum_weighted_uris():
    space = Space('tests/example.n3')
    s = space.sum_weighted_uris([('http://dbpedia.org/resource/Category:Futurama', 2), ('http://dbpedia.org/resource/Category:Star_Trek', 1)])
    np.testing.assert_allclose(np.asarray(s.todense()), [[2/np.sqrt(2 + 0.9**2) + 1/np.sqrt(1 + 0.9**2), 2/np.sqrt(2 + 0.9**2), 2*0.9/np.sqrt(2 + 0.9**2) + 0.9/np.sqrt(1 + 0.9**2)]])
