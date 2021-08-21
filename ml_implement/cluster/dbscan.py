"""
implement of DBSCAN

ref: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

TODO: add test
TODO: add validate input on fit method 
"""
import numpy as np
from ..utils.union_find import UnionFind

from typing import List


class DBSCAN:
    """クラスタリング手法

    ref: https://ja.wikipedia.org/wiki/DBSCAN
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5) -> None:
        """コンストラクタ

        Args:
            eps (float, optional): 到達可能半径. Defaults to 0.5.
            min_samples (int, optional): コア点とみなすサンプルの数. 自身を含む. Defaults to 5.
        """
        assert eps >= 0, f"eps is expected >= 0, but actual {eps}."
        assert (
            min_samples >= 1
        ), f"min_samples is expected >= 1, but actual {min_samples}."

        self.eps = eps
        self.min_samples = min_samples
        self._is_fitted = False

    @property
    def core_sample_indices_(self) -> np.array:
        """コアサンプルのインデックス属性.

        Returns:
            np.array: shape = (n_core_samples,)
        """
        assert self._is_fitted, f"This attributes must be called after fit."
        return np.array(self._core_sample_indices_)

    @property
    def components_(self) -> np.array:
        """コアサンプルのデータのコピー

        Returns:
            np.array: shape = (n_core_samples, n_features)
        """
        assert self._is_fitted, f"This attributes must be called after fit."
        return self._components

    @property
    def labels_(self) -> np.array:
        """全てのサンプルのラベル

        -1 は外れ値を表す

        Returns:
            np.array: shape = (n_samples,)
        """
        assert self._is_fitted, f"This attributes must be called after fit."
        return self._labels

    def fit(self, X: np.array):
        """サンプルのクラスタリングを行う

        Args:
            X (np.array): クラスタリングされるサンプル. shape = (n_samples, n_features)

        Returns:
            DBSCAN: インスタンス自身を返す
        """
        # validate input

        # クラスタリング
        # クラスタリング用のパラメータの初期化
        self._cluster = UnionFind(len(X))
        self._is_seen = np.full(shape=len(X), fill_value=False, dtype="bool")
        self._core_sample_indices_: List[int] = []

        # 全てのサンプルに対して, コア点か調べる
        for target_index in range(len(X)):
            if self._is_seen[target_index]:
                continue
            self._make_cluster(X, target_index)

        # クラスタリング結果をまとめ
        self._is_fitted = True
        self._components = X[self.core_sample_indices_].copy()
        self._labels = self._calc_labels(X)

        return self

    def fit_predict(self, X: np.array) -> np.array:
        """サンプルのクラスタリングを行い, ラベルを返す.

        Args:
            X (np.array): クラスタリングされるサンプル. shape = (n_samples, n_features)

        Returns:
            np.array: クラスタリングされた結果. ラベル. shape = (n_samples,)
        """
        self.fit(X)
        return self.labels_

    def _make_cluster(self, X: np.array, target_index: int) -> None:
        """ターゲットサンプルを中心にクラスタを生成

        Args:
            X (np.array): サンプル. shape = (n_samples, n_features).
            target_index (int): ターゲットサンプルのインデックス.
        """
        self._is_seen[target_index] = True

        is_nbhd = self._search_nbhd(X, target_index)
        is_core = is_nbhd.sum() >= self.min_samples
        if is_core:
            self._core_sample_indices_.append(target_index)
            reachable_indices = np.argwhere(~self._is_seen & is_nbhd).flatten()
            for reachable_index in reachable_indices:
                self._cluster.merge(target_index, reachable_index)
                self._make_cluster(X, reachable_index)

    def _search_nbhd(self, X: np.array, target_index: int) -> np.array:
        """ターゲットサンプルを中心とした近傍のインデックスを返す

        Args:
            X (np.array): サンプル. shape = (n_samples, n_features).
            target_index (int): ターゲットサンプルのインデックス.

        Returns:
            np.array: 近傍のインデックス
        """
        dist = np.linalg.norm(X - X[target_index], axis=1)
        return dist <= self.eps

    def _calc_labels(self, X: np.array) -> np.array:
        """クラスタリング後にラベルを割り振る

        Args:
            X (np.array): サンプル. shape = (n_samples, n_features).

        Returns:
            np.array: 各サンプルのラベル. shape = (n_samples,)
        """
        labels = np.full(shape=len(X), fill_value=-1, dtype="int")
        next_cluster_label = 0
        for group in self._cluster.groups():
            if len(group) >= self.min_samples:
                labels[group] = next_cluster_label
                next_cluster_label += 1
        return labels
