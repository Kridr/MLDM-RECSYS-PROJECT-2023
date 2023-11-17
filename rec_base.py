import numpy as np


class Metrics:
    def __init__(
            self,
            k: int,
            recommendations: np.ndarray,
            holdout: np.ndarray
            ):
        self.k = k
        self.recommendations = recommendations[:, :k]
        self.holdout = holdout

    def report(self):
        print(f"HR@{self.k} = {round(self.hit_rate(), 4)}")
        print(f"MRR@{self.k} = {round(self.mean_reciprocal_rank(), 4)}")
        print(f"Recall@{self.k} = {round(self.recall(), 4)}")
        print(f"Recall_Otto@{self.k} = {round(self.recall_otto(), 4)}")

    def hit_rate(self):
        return (
            self._get_hit_mask()
            .any(axis=1)
            .mean()
        )

    def mean_reciprocal_rank(self):
        hits_mask = self._get_hit_mask()

        idx = np.argwhere(hits_mask.argmax(axis=1)).squeeze(axis=1)

        return np.sum(
            1 / (hits_mask[idx].argmax(axis=1) + 1)
        ) / hits_mask.shape[0]

    def recall(self):
        hits_mask = self._get_hit_mask()
        return (
            hits_mask.sum(axis=1) / self.holdout.shape[1]
        ).sum() / self.recommendations.shape[0]

    def recall_otto(self):
        hits_mask = self._get_hit_mask()
        return (
            hits_mask.sum(axis=1) / min(self.k, self.holdout.shape[1])
        ).sum() / self.recommendations.shape[0]

    def _get_hit_mask(self) -> np.ndarray:
        return (
            self.holdout[..., None] ==
            np.expand_dims(self.recommendations[:, :self.k], axis=1)
        ).any(axis=1)


class RecModel:
    def __init__(self):
        pass

    def build(self):
        pass

    def score(self):
        pass

    def recommend(self):
        pass
