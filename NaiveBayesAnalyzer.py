import nltk
from FeatureExtractor import FeatureExtractor


class NaiveBayesAnalyzer:

    def __init__(self, dict):
        self._dict = dict
        self._fe = FeatureExtractor()

    def train(self):
        train_data = []
        for k, v in self._dict.items():
            train_data = train_data + [(self._fe.default_feature_extractor(f), k) for f in v]

        self._classifier = nltk.classify.NaiveBayesClassifier.train(train_data)

    def analyze(self, text):
        feats = self._fe.default_feature_extractor(text)
        prob_dist = self._classifier.prob_classify(feats)

        classification=prob_dist.max()
        # print(classification)
        # for k in self._dict.keys():
        #     print (k, prob_dist.prob(k))
        return classification
