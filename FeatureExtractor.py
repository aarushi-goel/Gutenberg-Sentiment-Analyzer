class FeatureExtractor:

    def __init__(self):
        pass

    def default_feature_extractor(self, words):
        return dict((word, 1) for word in words)
