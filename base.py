class Metrics:
    def __init__(self, k, recommendations, holdout):
        self.k = k
        self.recommendations = recommendations
        self.holdout = holdout

    def hit_rate(self):
        pass

    def mean_reciprocal_rank(self):
        pass

    def recall(self):
        pass


class RecModel:
    def __init__(self):
        pass

    def build(self):
        pass

    def recommend(self):
        pass


class DataPreparator:
    def __init__(self):
        pass
