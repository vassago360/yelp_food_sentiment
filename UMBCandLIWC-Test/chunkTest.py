import nltk.chunk
import itertools
 
class TagChunker(nltk.chunk.ChunkParserI):
    def __init__(self, chunk_tagger):
        self._chunk_tagger = chunk_tagger
 
    def parse(self, tokens):
        # split words and part of speech tags
        (words, tags) = zip(*tokens)
        # get IOB chunk tags
        chunks = self._chunk_tagger.tag(tags)
        # join words with chunk tags
        wtc = itertools.izip(words, chunks)
        # w = word, t = part-of-speech tag, c = chunk tag
        lines = [' '.join([w, t, c]) for (w, (t, c)) in wtc if c]
        # create tree from conll formatted chunk lines
        return nltk.chunk.conllstr2tree('\n'.join(lines))
    
sentence = """We are both native New Yorkers, and have been on a quest for the last 28 years here in Arizona to find the perfect New York pizza."""
# sentence should be a list of words
chunk_tagger = nltk.tag.ClassifierBasedPOSTagger(train=nltk.corpus.treebank.tagged_sents())

parser = nltk.ChartParser(nltk.grammar)
for tree in parser.parse(nltk.word_tokenize(sentence)):
    for subtree in tree.subtrees(filter=lambda t: t.node == 'NP'):
        # print the noun phrase as a list of part-of-speech tagged words
        print subtree.leaves()



chunk_tagger = nltk.tag.ClassifierBasedPOSTagger(train=nltk.corpus.treebank.tagged_sents())
chunker = TagChunker( chunk_tagger)
tagged = chunk_tagger.tag(sentence)
tree = chunker.parse(tagged)
# for each noun phrase sub tree in the parse tree
for subtree in tree.subtrees(filter=lambda t: t.node == 'NP'):
    # print the noun phrase as a list of part-of-speech tagged words
    print subtree.leaves()