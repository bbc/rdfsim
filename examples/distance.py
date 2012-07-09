import rdfsim
from rdfsim.space import Space

space = Space('skos_categories_1000.nt', property='http://www.w3.org/2004/02/skos/core#broader')

print "Distance betwen Dwarves and Elves:"
print space.similarity_uri('http://dbpedia.org/resource/Category:Middle-earth_Dwarves', 'http://dbpedia.org/resource/Category:Middle-earth_Elves')

print "Distance between Dwarves and Futurama:"
print space.similarity_uri('http://dbpedia.org/resource/Category:Middle-earth_Dwarves', 'http://dbpedia.org/resource/Category:Futurama')
