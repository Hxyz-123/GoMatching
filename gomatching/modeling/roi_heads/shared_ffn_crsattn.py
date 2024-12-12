import copy
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels
from detectron2.utils.events import get_event_storage
import numpy as np

from detectron2.config import configurable
from detectron2.structures import Boxes, pairwise_iou, Instances

from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.roi_heads import ROIHeads
from .association_head import ATTWeightHead, FCHead4Query
from .transformer import Transformer

import sys

sys.path.insert(0, 'third_party/')
from adet.modeling.model.matcher import build_point_matcher
from adet.utils.misc import accuracy, is_dist_avail_and_initialized
from detectron2.utils.comm import get_world_size


def sigmoid_focal_loss(inputs, targets, num_inst, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss.ndim == 4:
        return loss.mean((1, 2)).sum() / num_inst
    elif loss.ndim == 3:
        return loss.mean(1).sum() / num_inst
    else:
        raise NotImplementedError(f"Unsupported dim {loss.ndim}")


@ROI_HEADS_REGISTRY.register()
class SHA_FFN_CRSATTN(torch.nn.Module):
    @configurable
    def __init__(self, **kwargs):
        cfg = kwargs.pop('cfg', None)
        self.cfg = cfg
        super().__init__()
        if cfg is None:
            return

        self.focal_alpha = cfg.MODEL.TRANSFORMER.LOSS.FOCAL_ALPHA
        self.focal_gamma = cfg.MODEL.TRANSFORMER.LOSS.FOCAL_GAMMA
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.point_matcher = build_point_matcher(cfg)

        self.proposal_append_gt = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        assert not cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.asso_on = cfg.MODEL.ASSO_ON
        assert self.asso_on
        self._init_asso_head(cfg)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        ret['cfg'] = cfg
        ret['input_shape'] = input_shape
        return ret

    def _sample_proposals(
            self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            gt_classes[matched_labels == 0] = self.num_classes
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def _init_asso_head(self, cfg):
        self.feature_dim = cfg.MODEL.ASSO_HEAD.FC_DIM
        self.num_fc = cfg.MODEL.ASSO_HEAD.NUM_FC
        self.asso_thresh_train = cfg.MODEL.ASSO_HEAD.ASSO_THRESH
        self.asso_thresh_test = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST
        self.asso_weight = cfg.MODEL.ASSO_HEAD.ASSO_WEIGHT
        self.neg_unmatched = cfg.MODEL.ASSO_HEAD.NEG_UNMATCHED
        self.with_temp_emb = cfg.MODEL.ASSO_HEAD.WITH_TEMP_EMB
        self.no_pos_emb = cfg.MODEL.ASSO_HEAD.NO_POS_EMB
        self.ctrs_weight = cfg.MODEL.ASSO_HEAD.CTRS_WEIGHT
        self.with_rescore = cfg.MODEL.ROI_HEADS.WITH_RESR

        self.asso_thresh_test = self.asso_thresh_test \
            if self.asso_thresh_test > 0 else self.asso_thresh_train

        num_encoder_layers = cfg.MODEL.ASSO_HEAD.NUM_ENCODER_LAYERS
        num_decoder_layers = cfg.MODEL.ASSO_HEAD.NUM_DECODER_LAYERS
        num_heads = cfg.MODEL.ASSO_HEAD.NUM_HEADS
        dropout = cfg.MODEL.ASSO_HEAD.DROPOUT
        norm = cfg.MODEL.ASSO_HEAD.NORM
        num_weight_layers = cfg.MODEL.ASSO_HEAD.NUM_WEIGHT_LAYERS
        no_decoder_self_att = cfg.MODEL.ASSO_HEAD.NO_DECODER_SELF_ATT
        no_encoder_self_att = cfg.MODEL.ASSO_HEAD.NO_ENCODER_SELF_ATT

        self.asso_head = FCHead4Query(
            input_channel=cfg.MODEL.TRANSFORMER.HIDDEN_DIM,
            point_nums=cfg.MODEL.TRANSFORMER.NUM_POINTS,
            fc_dim=self.feature_dim,
            num_fc=self.num_fc
        )

        if self.with_rescore:
            self.rescoring_head = nn.Linear(cfg.MODEL.TRANSFORMER.HIDDEN_DIM, 1)  # 1 for text

        self.asso_predictor = ATTWeightHead(
            self.feature_dim, num_layers=num_weight_layers, dropout=dropout)
        self.shared_matcher = Transformer(
            d_model=self.feature_dim,
            nhead=num_heads,
            num_encoder_layers=0,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=self.feature_dim,
            dropout=dropout,
            return_intermediate_dec=True,
            norm=norm,
            no_decoder_self_att=no_decoder_self_att,
            no_encoder_self_att=no_encoder_self_att,
            only_dec_crs_attn=True
        )

        self.asso_weight_local = cfg.MODEL.ASSO_HEAD.ASSO_WEIGHT_LOCAL
        self.local_asso_predictor = ATTWeightHead(
            self.feature_dim, num_layers=num_weight_layers, dropout=dropout)

        if not self.no_pos_emb:
            self.learn_pos_emb_num = 16
            self.pos_emb = nn.Embedding(
                self.learn_pos_emb_num * 4, self.feature_dim // 4)
            if self.with_temp_emb:
                self.learn_temp_emb_num = 16
                self.temp_emb = nn.Embedding(
                    self.learn_temp_emb_num, self.feature_dim)

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                               for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_res(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items()}

        indices = self.point_matcher(outputs_without_aux, targets)
        num_inst = sum(len(t['labels']) for t in targets)
        num_inst = torch.as_tensor(
            [num_inst], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_inst)
        num_inst = torch.clamp(num_inst / get_world_size(), min=1).item()

        assert 're_pred_logits' in outputs
        src_logits = outputs['re_pred_logits']
        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(src_logits.shape[:-1], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes_o = torch.cat([t["labels"][J]
                                      for t, (_, J) in zip(targets, indices)])
        if len(target_classes_o.shape) < len(target_classes[idx].shape):
            target_classes_o = target_classes_o[..., None]
        target_classes[idx] = target_classes_o

        shape = list(src_logits.shape)
        shape[-1] += 1
        target_classes_onehot = torch.zeros(shape,
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[..., :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_inst,
                                     alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        return {'loss_res': loss_ce}

    def _forward_asso(self, instances, targets=None):
        """
        """
        asso_thresh = self.asso_thresh_train if self.training \
            else self.asso_thresh_test
        fg_inds = [
            x.objectness_logits > asso_thresh for x in instances]
        proposals = [x[inds] for (x, inds) in zip(instances, fg_inds)]

        reid_features = torch.cat([x.query_features for x in proposals], dim=0)
        reid_features = self.asso_head(reid_features)
        reid_features = reid_features.view(
            1, -1, self.feature_dim)  # 1 x N x F
        n_t = [len(x) for x in proposals]
        if not self.training:
            instances = [inst[inds] for inst, inds in zip(instances, fg_inds)]
            features = reid_features.view(-1, self.feature_dim).split(n_t, dim=0)
            for inst, feat in zip(instances, features):
                inst.reid_features = feat
            return instances
        else:
            asso_outputs, pred_box, pred_time, query_inds = \
                self._forward_transformer(proposals, reid_features)
            assert len(proposals) == len(targets)
            target_box, target_time = self._get_boxes_time(targets)  # G x 4
            if sum(len(x) for x in targets) == 0 or \
                    max(x.gt_instance_ids.max().item() for x in targets if len(x) > 0) == 0:
                asso_loss_long = reid_features.new_zeros((1,), dtype=torch.float32)[0]
                asso_loss_short = reid_features.new_zeros((1,), dtype=torch.float32)[0]
                return {'loss_long_asso': asso_loss_long, 'loss_short_asso': asso_loss_short}
            target_inst_id = torch.cat(
                [x.gt_instance_ids for x in targets if len(x) > 0])

            asso_gt, match_cues = self._get_asso_gt(pred_box, pred_time, target_box, target_time, target_inst_id,
                                                    n_t)  # K x N,

            ### LT matcher
            asso_loss_long = 0
            for x in asso_outputs:
                asso_loss_long += self.detr_asso_loss(x, asso_gt, match_cues, n_t)

            ### ST matcher
            asso_loss_short = 0
            eff_num = 0
            for cur_id in range(1, len(proposals)):
                asso_outputs_local, pred_box_local, pred_time_local, query_inds_local = \
                    self._forward_transformer(proposals[cur_id - 1: cur_id + 1],
                                              reid_features[:, sum(n_t[:cur_id - 1]): sum(n_t[:cur_id + 1])],
                                              short_term=True)
                target_box_local, target_time_local = self._get_boxes_time(targets[cur_id - 1: cur_id + 1])
                if sum(len(x) for x in targets[cur_id - 1: cur_id + 1]) == 0 or \
                        max(x.gt_instance_ids.max().item() for x in targets[cur_id - 1: cur_id + 1] if len(x) > 0) == 0:
                    continue
                eff_num += 1
                target_inst_id_local = torch.cat(
                    [x.gt_instance_ids for x in targets[cur_id - 1: cur_id + 1] if len(x) > 0])
                asso_gt_local, match_cues_local = \
                    self._get_asso_gt(pred_box_local, pred_time_local, target_box_local, target_time_local,
                                      target_inst_id_local, n_t[cur_id - 1: cur_id + 1])
                for x in asso_outputs_local:
                    asso_loss_short += self.detr_asso_loss(x, asso_gt_local, match_cues_local,
                                                           n_t[cur_id - 1: cur_id + 1])
            asso_loss_short /= (eff_num + 1e-4)
            return {'loss_long_asso': self.asso_weight * asso_loss_long,
                    'loss_short_asso': self.asso_weight_local * asso_loss_short}

    def _forward_transformer(self, proposals, reid_features, query_frame=None, short_term=False):
        T = len(proposals)
        n_t = [len(x) for x in proposals]
        pred_box, pred_time = self._get_boxes_time(proposals)  # N x 4
        N = sum(n_t)
        D = self.feature_dim
        pos_emb = None

        query = None
        query_inds = None
        M = N
        if query_frame is not None:
            c = query_frame
            query_inds = [x for x in range(sum(n_t[:c]), sum(n_t[:c + 1]))]
            M = len(query_inds)

        if short_term:
            feats, memory = self.shared_matcher(
                reid_features, pos_embed=pos_emb, query_embed=query,
                query_inds=query_inds)
            # feats: L x [1 x M x F], memory: 1 x N x F
            asso_outputs = [self.local_asso_predictor(x, memory).view(M, N) \
                            for x in feats]  # L x [M x N]
        else:
            feats, memory = self.shared_matcher(
                reid_features, pos_embed=pos_emb, query_embed=query,
                query_inds=query_inds)
            # feats: L x [1 x M x F], memory: 1 x N x F
            asso_outputs = [self.asso_predictor(x, memory).view(M, N) \
                            for x in feats]  # L x [M x N]
        return asso_outputs, pred_box, pred_time, query_inds

    def _activate_asso(self, asso_output):
        asso_active = []
        for asso in asso_output:
            # asso: M x n_t
            asso = torch.cat(
                [asso, asso.new_zeros((asso.shape[0], 1))], dim=1).softmax(
                dim=1)[:, :-1]
            asso_active.append(asso)
        return asso_active

    def _get_asso_gt(self, pred_box, pred_time, \
                     target_box, target_time, target_inst_id, n_t):
        '''
        Inputs:
            pred_box: N x 4
            pred_time: N
            targer_box: G x 4
            targer_time: G
            target_inst_id: G
            K: len(unique(target_inst_id))
        Return:
            ret: K x N or K x T
            match_cues: K x 3 or N
        '''
        ious = pairwise_iou(Boxes(pred_box), Boxes(target_box))  # N x G
        ious[pred_time[:, None] != target_time[None, :]] = -1.
        inst_ids = torch.unique(target_inst_id[target_inst_id > 0])
        K, N = len(inst_ids), len(pred_box)
        match_cues = pred_box.new_full((N,), -1, dtype=torch.long)

        T = len(n_t)

        ret = pred_box.new_zeros((K, T), dtype=torch.long)
        ious_per_frame = ious.split(n_t, dim=0)  # T x [n_t x G]
        for k, inst_id in enumerate(inst_ids):
            target_inds = target_inst_id == inst_id  # G
            base_ind = 0
            for t in range(T):
                iou_t = ious_per_frame[t][:, target_inds]  # n_t x gk
                if iou_t.numel() == 0:
                    ret[k, t] = n_t[t]
                else:
                    val, inds = iou_t.max(dim=0)  # n_t x gk --> gk
                    ind = inds[val > 0.0]
                    assert (len(ind) <= 1), '{} {}'.format(
                        target_inst_id, n_t)
                    if len(ind) == 1:
                        obj_ind = ind[0].item()
                        ret[k, t] = obj_ind
                        match_cues[base_ind + obj_ind] = k
                    else:
                        ret[k, t] = n_t[t]
                base_ind += n_t[t]

        return ret, match_cues

    def detr_asso_loss(self, asso_pred, asso_gt, match_cues, n_t):
        '''
        Inputs:
            asso_pred: M x N
            asso_gt: K x N or K x T
            n_t: T (list of int)
        Return:
            float
        '''
        src_inds, target_inds = self._match(
            asso_pred, asso_gt, match_cues, n_t)

        loss = 0
        num_objs = 0
        zero = asso_pred.new_zeros((asso_pred.shape[0], 1))  # M x 1
        asso_pred_image = asso_pred.split(n_t, dim=1)  # T x [M x n_t]
        for t in range(len(n_t)):
            asso_pred_with_bg = torch.cat(
                [asso_pred_image[t], zero], dim=1)  # M x (n_t + 1)
            if self.neg_unmatched:
                asso_gt_t = asso_gt.new_full(
                    (asso_pred.shape[0],), n_t[t])  # M
                asso_gt_t[src_inds] = asso_gt[target_inds, t]  # M
            else:
                asso_pred_with_bg = asso_pred_with_bg[src_inds]  # K x (n_t + 1)
                asso_gt_t = asso_gt[target_inds, t]  # K
            num_objs += (asso_gt_t != n_t[t]).float().sum()
            loss += F.cross_entropy(
                asso_pred_with_bg, asso_gt_t, reduction='none')
        return loss.sum() / (num_objs + 1e-4)

    @torch.no_grad()
    def _match(self, asso_pred, asso_gt, match_cues, n_t):
        '''
        Inputs:
            asso_pred: M x N
            asso_gt: K x N or K x T
            match_cues: K x 3 or N
        Return:
            indices: 
        '''
        src_inds = torch.where(match_cues >= 0)[0]
        target_inds = match_cues[src_inds]
        return (src_inds, target_inds)

    def _get_boxes_time(self, instances):
        boxes, times = [], []
        for t, p in enumerate(instances):
            h, w = p._image_size
            if p.has('proposal_boxes'):
                p_boxes = p.proposal_boxes.tensor.clone()
            elif p.has('pred_boxes'):
                p_boxes = p.pred_boxes.tensor.clone()
            else:
                p_boxes = p.gt_boxes.tensor.clone()
            p_boxes[:, [0, 2]] /= w
            p_boxes[:, [1, 3]] /= h
            boxes.append(p_boxes)  # ni x 4
            times.append(p_boxes.new_full(
                (p_boxes.shape[0],), t, dtype=torch.long))
        boxes = torch.cat(boxes, dim=0)  # N x 4
        times = torch.cat(times, dim=0)  # N
        return boxes.detach(), times.detach()

    def _box_pe(self, boxes):
        '''
        '''
        N = boxes.shape[0]
        boxes = boxes.view(N, 4)
        xywh = torch.cat([
            (boxes[:, 2:] + boxes[:, :2]) / 2,
            (boxes[:, 2:] - boxes[:, :2])], dim=1)
        xywh = xywh * self.learn_pos_emb_num
        l = xywh.clamp(min=0, max=self.learn_pos_emb_num - 1).long()  # N x 4
        r = (l + 1).clamp(min=0, max=self.learn_pos_emb_num - 1).long()  # N x 4
        lw = (xywh - l.float())  # N x 4
        rw = 1. - lw
        f = self.pos_emb.weight.shape[1]
        pos_emb_table = self.pos_emb.weight.view(
            self.learn_pos_emb_num, 4, f)  # T x 4 x (F // 4)
        pos_le = pos_emb_table.gather(0, l[:, :, None].expand(N, 4, f))  # N x 4 x f
        pos_re = pos_emb_table.gather(0, r[:, :, None].expand(N, 4, f))  # N x 4 x f
        pos_emb = lw[:, :, None] * pos_re + rw[:, :, None] * pos_le
        return pos_emb.view(N, 4 * f)

    def _temp_pe(self, temps):
        '''
        '''
        N = temps.shape[0]
        temps = temps * self.learn_temp_emb_num
        l = temps.clamp(min=0, max=self.learn_temp_emb_num - 1).long()  # N x 4
        r = (l + 1).clamp(min=0, max=self.learn_temp_emb_num - 1).long()  # N x 4
        lw = (temps - l.float())  # N
        rw = 1. - lw
        le = self.temp_emb.weight[l]  # T x F --> N x F
        re = self.temp_emb.weight[r]  # N x F
        temp_emb = lw[:, None] * re + rw[:, None] * le
        return temp_emb.view(N, self.feature_dim)

    def forward(self, images, proposals, targets=None):
        """
        enable reid head
        enable association
        """
        # if self.training:
        #     proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            losses = {}
            losses.update(self._forward_asso(proposals, targets))
            return proposals, losses
        else:
            proposals = self._forward_asso(proposals)
            pred_instances = proposals
            for p in pred_instances:
                p.pred_boxes = p.proposal_boxes
                p.scores = p.objectness_logits
                p.pred_classes = torch.zeros(
                    (len(p),), dtype=torch.long, device=p.pred_boxes.device)
                p.remove('proposal_boxes')
                p.remove('objectness_logits')

            return pred_instances, {}
