import torch

from .base_sampler import BaseSampler
from .convex_sampling_result import ConvexSamplingResult

class ConvexPseudoSampler(BaseSampler):

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        raise NotImplementedError

    def sample(self, assign_result, points, gt_rbboxes, **kwargs):
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0).squeeze(-1).unique()
        gt_flags = points.new_zeros(points.shape[0], dtype=torch.uint8)
        sampling_result = ConvexSamplingResult(pos_inds, neg_inds, points, gt_rbboxes,
                                         assign_result, gt_flags)
        return sampling_result