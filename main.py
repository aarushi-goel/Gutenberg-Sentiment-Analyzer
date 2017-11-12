from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
#
# blob = TextBlob("I hate this library", analyzer=NaiveBayesAnalyzer())
# print (blob.sentiment)

from nltk.corpus import gutenberg
from nltk.sentiment import SentimentAnalyzer

sentim_analyzer = SentimentAnalyzer()
i = gutenberg.fileids()[0]

for i in gutenberg.fileids():
    blob = TextBlob(' '.join(gutenberg.words(i)), analyzer=NaiveBayesAnalyzer())
    print (i, blob.sentiment)
