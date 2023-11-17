import numpy as np

class Metrics:
    def __init__(self, k, recommendations, holdout):
        self.k = k
        self.recommendations = recommendations[:, :self.k]
        self.holdout = holdout

    def hit_rate(self):
        return np.array(
            [
                np.intersect1d(self.holdout[i], self.recommendations[i]).any()
                for i in range(self.holdout.shape[0])
            ]
        ).mean()

    def mean_reciprocal_rank(self):
        n_test_users = self.holdout.shape[0]
        hits_rank = [
            np.where(np.in1d(self.holdout[i], self.recommendations[i]))[0]
            for i in range(self.holdout.shape[0])
        ]
        hits_rank = np.array([el[0] for el in hits_rank if el.shape[0]])
        return (1. / (hits_rank + 1)).sum() / n_test_users

    def recall(self):
        numerator = np.sum(
            [
                np.intersect1d(self.holdout[i], self.recommendations[i]).shape[0]
                for i in range(self.holdout.shape[0])
            ]
        )
        denominator = np.sum([min(self.k, self.holdout[i].shape[0]) for i in range(self.holdout.shape[0])])
        return numerator / denominator


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
