import argparse
import io
import os
import random
import warnings
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from typing import Iterable, Optional, Tuple

import numpy as np
import requests
# import tensorflow.compat.v1 as tf
from scipy import linalg
from tqdm.auto import tqdm

import torch



class Evaluator:
    def __init__(self):
        self.manifold_estimator = ManifoldEstimator()

    def compute_prec_recall(
        self, activations_ref: np.ndarray, activations_sample: np.ndarray
    ) -> Tuple[float, float]:
        
        radii_1 = self.manifold_estimator.manifold_radii(activations_ref)
        radii_2 = self.manifold_estimator.manifold_radii(activations_sample)
       
        pr = self.manifold_estimator.evaluate_pr(
            activations_ref, radii_1, activations_sample, radii_2
        )
        return (float(pr[0][0]), float(pr[1][0]))


class ManifoldEstimator:
    """
    A helper for comparing manifolds of feature vectors.

    Adapted from https://github.com/kynkaat/improved-precision-and-recall-metric/blob/f60f25e5ad933a79135c783fcda53de30f42c9b9/precision_recall.py#L57
    """

    def __init__(
        self,
        row_batch_size=10000,
        col_batch_size=10000,
        nhood_sizes=(3,),
        clamp_to_percentile=None,
        eps=1e-5,
    ):
        """
        Estimate the manifold of given feature vectors.

        :param session: the TensorFlow session.
        :param row_batch_size: row batch size to compute pairwise distances
                               (parameter to trade-off between memory usage and performance).
        :param col_batch_size: column batch size to compute pairwise distances.
        :param nhood_sizes: number of neighbors used to estimate the manifold.
        :param clamp_to_percentile: prune hyperspheres that have radius larger than
                                    the given percentile.
        :param eps: small number for numerical stability.
        """
        # self.distance_block = DistanceBlock(session)
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size
        self.nhood_sizes = nhood_sizes
        self.num_nhoods = len(nhood_sizes)
        self.clamp_to_percentile = clamp_to_percentile
        self.eps = eps

    def manifold_radii(self, features: np.ndarray) -> np.ndarray:
        num_images = features.shape[0]

        # Estimate manifold of features by calculating distances to k-NN of each sample.
        radii = np.zeros([num_images, self.num_nhoods], dtype=np.float32)
        distance_batch = np.zeros([self.row_batch_size, num_images], dtype=np.float32)
        seq = np.arange(max(self.nhood_sizes) + 1, dtype=np.int32)

        for begin1 in range(0, num_images, self.row_batch_size):
            end1 = min(begin1 + self.row_batch_size, num_images)
            row_batch = features[begin1:end1]

            for begin2 in range(0, num_images, self.col_batch_size):
                end2 = min(begin2 + self.col_batch_size, num_images)
                col_batch = features[begin2:end2]
                distance_batch[0 : end1 - begin1, begin2:end2] = self.pytorch_pairwise_distance(row_batch, col_batch)

            # Find the k-nearest neighbor from the current batch.
            radii[begin1:end1, :] = np.concatenate(
                [
                    x[:, self.nhood_sizes]
                    for x in _numpy_partition(distance_batch[0 : end1 - begin1, :], seq, axis=1)
                ],
                axis=0,
            )

        if self.clamp_to_percentile is not None:
            max_distances = np.percentile(radii, self.clamp_to_percentile, axis=0)
            radii[radii > max_distances] = 0
        return radii

    def pytorch_pairwise_distance(self, U,V, return_numpy=True):
        # U is an numpy array of shape M,D
        # V is an numpy array of shape N,D
        with torch.no_grad():
            U_t = torch.from_numpy(U).unsqueeze(0).cuda() # 1,M,D
            V_t = torch.from_numpy(V).unsqueeze(0).cuda() # 1,N,D
            DD = torch.cdist(U_t, V_t, p=2.0) # 1,M,N
            DD = DD.squeeze(0) # M,N
            if return_numpy:
                DD = DD.cpu().numpy()
        return DD

    def evaluate_pr(
        self,
        features_1: np.ndarray,
        radii_1: np.ndarray,
        features_2: np.ndarray,
        radii_2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate precision and recall efficiently.

        :param features_1: [N1 x D] feature vectors for reference batch.
        :param radii_1: [N1 x K1] radii for reference vectors.
        :param features_2: [N2 x D] feature vectors for the other batch.
        :param radii_2: [N x K2] radii for other vectors.
        :return: a tuple of arrays for (precision, recall):
                 - precision: an np.ndarray of length K1
                 - recall: an np.ndarray of length K2
        """
        
        features_1_status = np.zeros([features_1.shape[0], radii_2.shape[1]], dtype=np.bool)
        features_2_status = np.zeros([features_2.shape[0], radii_1.shape[1]], dtype=np.bool)
        for begin_1 in range(0, features_1.shape[0], self.row_batch_size):
            end_1 = begin_1 + self.row_batch_size
            batch_1 = features_1[begin_1:end_1]
            for begin_2 in range(0, features_2.shape[0], self.col_batch_size):
                end_2 = begin_2 + self.col_batch_size
                batch_2 = features_2[begin_2:end_2]
                batch_1_in, batch_2_in = self.pytorch_less_than(batch_1, radii_1[begin_1:end_1], batch_2, radii_2[begin_2:end_2])
                features_1_status[begin_1:end_1] |= batch_1_in
                features_2_status[begin_2:end_2] |= batch_2_in
        return (
            np.mean(features_2_status.astype(np.float64), axis=0),
            np.mean(features_1_status.astype(np.float64), axis=0),
        )

    def pytorch_less_than(self, batch_1, radii_1, batch_2, radii_2):
        # batch_1 M,D; radii_1, M,1
        # batch_2 N,D; radii_2, N,1
        radii_1t = torch.from_numpy(radii_1).cuda() # M,1
        radii_2t = torch.from_numpy(radii_2).cuda().transpose(0,1) # 1,N
        DD = self.pytorch_pairwise_distance(batch_1, batch_2, return_numpy=False) # M,N

        batch1_in = (DD <= radii_2t).long() # M,N
        batch1_in = batch1_in.max(dim=1)[0] # M
        batch1_in = batch1_in.cpu().numpy().astype(np.bool)
        batch1_in = batch1_in[:,np.newaxis] # M,1

        batch2_in = (DD <= radii_1t).long() # M,N
        batch2_in = batch2_in.max(dim=0)[0] # N
        batch2_in = batch2_in.cpu().numpy().astype(np.bool)
        batch2_in = batch2_in[:,np.newaxis] # N,1
        return batch1_in, batch2_in


def _numpy_partition(arr, kth, **kwargs):
    num_workers = min(cpu_count(), len(arr))
    chunk_size = len(arr) // num_workers
    extra = len(arr) % num_workers

    start_idx = 0
    batches = []
    for i in range(num_workers):
        size = chunk_size + (1 if i < extra else 0)
        batches.append(arr[start_idx : start_idx + size])
        start_idx += size

    with ThreadPool(num_workers) as pool:
        return list(pool.map(partial(np.partition, kth=kth, **kwargs), batches))

