import nltk
from nltk.chunk import RegexpParser

grammar = '''
NP: {<DT>?<JJ>*<NN>*}
V: {<V.*>}'''
chunker = RegexpParser(grammar)
sentence = """We are both native New Yorkers, and have been on a quest for the last 28 years here in Arizona to find the perfect New York pizza."""
tokens = nltk.word_tokenize(sentence)
chunked = chunker.parse(tokens)

for n in chunked:
    if isinstance(n, nltk.tree.Tree):               
        if n.node == 'NP':
            print n.leaves()
        else:
            pass