import time

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from detectron2.config import configurable
from detectron2.structures import Boxes, pairwise_iou
from detectron2.modeling import build_backbone, build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.data import MetadataCatalog
from detectron2.layers import nms
import numpy as np
import sys
sys.path.insert(0, 'third_party/')

from adet.layers.pos_encoding import PositionalEncoding2D

from adet.modeling.model.detection_transformer_wobackbone import DETECTION_TRANSFORMER_WOBACKBONE
from adet.utils.misc import (
    NestedTensor,
)
from typing import List
from detectron2.structures import ImageList, Instances

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for _, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.output_shape = self.backbone.output_shape()
        self.feature_strides = [self.output_shape[f].stride for f in self.output_shape.keys()]
        self.num_channels = self.output_shape[list(self.output_shape.keys())[-1]].channels

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks

def detector_postprocess(results, output_height, output_width, min_size=None, max_size=None):
    """
    scale align
    """
    if min_size and max_size:
        # to eliminate the padding influence for ViTAE backbone results
        size = min_size * 1.0
        scale_img_size = min_size / min(output_width, output_height)
        if output_height < output_width:
            newh, neww = size, scale_img_size * output_width
        else:
            newh, neww = scale_img_size * output_height, size
        if max(newh, neww) > max_size:
            scale = max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        scale_x, scale_y = (output_width / neww, output_height / newh)
    else:
        scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])

    # scale points
    if results.has("ctrl_points"):
        ctrl_points = results.ctrl_points
        ctrl_points[:, 0::2] *= scale_x
        ctrl_points[:, 1::2] *= scale_y

    if results.has("pred_boxes") and not isinstance(results.bd, list):
        bd = results.bd
        bd[..., 0::2] *= scale_x
        bd[..., 1::2] *= scale_y

    return results

@META_ARCH_REGISTRY.register()
class GoMatching(nn.Module):
    @configurable
    def __init__(self, **kwargs):
        """
        """
        super().__init__()

        self.test_len = kwargs.pop('test_len')
        self.overlap_thresh = kwargs.pop('overlap_thresh')
        self.min_track_len = kwargs.pop('min_track_len')
        self.max_center_dist = kwargs.pop('max_center_dist')
        self.decay_time = kwargs.pop('decay_time')
        self.asso_thresh = kwargs.pop('asso_thresh')
        self.with_iou = kwargs.pop('with_iou')
        self.local_no_iou = kwargs.pop('local_no_iou')
        self.local_iou_only = kwargs.pop('local_iou_only')
        self.not_mult_thresh = kwargs.pop('not_mult_thresh')
        self.nms_thresh = kwargs.pop('nms_thresh')
        self.with_rescore = kwargs.pop('with_rescore')

        ### deeepsolo
        self.cfg = kwargs['cfg']
        self.metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused"
        )
        self.device = torch.device(kwargs['cfg'].MODEL.DEVICE)
        N_steps = kwargs['cfg'].MODEL.TRANSFORMER.HIDDEN_DIM // 2
        self.test_score_threshold = kwargs['cfg'].MODEL.TRANSFORMER.INFERENCE_TH_TEST
        self.min_size_test = None
        self.max_size_test = None
        if kwargs['cfg'].MODEL.BACKBONE.NAME == "build_vitaev2_backbone":
            self.min_size_test = kwargs['cfg'].INPUT.MIN_SIZE_TEST
            self.max_size_test = kwargs['cfg'].INPUT.MAX_SIZE_TEST

        d2_backbone = MaskedBackbone(kwargs['cfg'])
        self.backbone = Joiner(
            d2_backbone,
            PositionalEncoding2D(N_steps, kwargs['cfg'].MODEL.TRANSFORMER.TEMPERATURE, normalize=True)
        )
        self.backbone.num_channels = d2_backbone.num_channels
        self.backbone.output_shape = d2_backbone.output_shape
        self.detection_transformer = DETECTION_TRANSFORMER_WOBACKBONE(kwargs['cfg'])

        self.roi_heads = build_roi_heads(kwargs['cfg'], self.backbone.output_shape)

        pixel_mean = torch.Tensor(kwargs['cfg'].MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(kwargs['cfg'].MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    @classmethod
    def from_config(cls, cfg):
        ret = {}
        ret['test_len'] = cfg.INPUT.VIDEO.TEST_LEN
        ret['overlap_thresh'] = cfg.VIDEO_TEST.OVERLAP_THRESH     
        ret['asso_thresh'] = cfg.MODEL.ASSO_HEAD.ASSO_THRESH
        ret['min_track_len'] = cfg.VIDEO_TEST.MIN_TRACK_LEN
        ret['max_center_dist'] = cfg.VIDEO_TEST.MAX_CENTER_DIST
        ret['decay_time'] = cfg.VIDEO_TEST.DECAY_TIME
        ret['with_iou'] = cfg.VIDEO_TEST.WITH_IOU
        ret['local_no_iou'] = cfg.VIDEO_TEST.LOCAL_NO_IOU
        ret['local_iou_only'] = cfg.VIDEO_TEST.LOCAL_IOU_ONLY
        ret['not_mult_thresh'] = cfg.VIDEO_TEST.NOT_MULT_THRESH
        ret['nms_thresh'] = cfg.VIDEO_TEST.NMS_THRESH
        ret['with_rescore'] = cfg.MODEL.ROI_HEADS.WITH_RESR

        ### deepsolo cfg
        ret['cfg'] = cfg
        return ret

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            gt_classes = targets_per_image.gt_classes

            raw_ctrl_points = targets_per_image.polyline
            gt_texts = targets_per_image.texts
            gt_ctrl_points = raw_ctrl_points.reshape(-1, self.detection_transformer.num_points, 2) / \
                             torch.as_tensor([w, h], dtype=torch.float, device=self.device)[None, None, :]
            gt_ids = targets_per_image.gt_instance_ids
            new_targets.append(
                {
                    "labels": gt_classes,
                    "ctrl_points": gt_ctrl_points,
                    "texts": gt_texts,
                    "ids": gt_ids,
                }
            )
        return new_targets

    def forward(self, batched_inputs): # forward

        images = self.preprocess_image(batched_inputs)
        features, pos = self.backbone(images)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

        ### detection
        output = self.detection_transformer(features, pos, self.backbone)
        targets = self.prepare_targets(gt_instances)

        ## rescoring
        if self.with_rescore:
            output["re_pred_logits"] = self.roi_heads.rescoring_head(output["query_features"])
            res_loss = self.roi_heads.loss_res(output, targets)
        else:
            output["re_pred_logits"] = None

        ctrl_point_cls = output["pred_logits"]
        ctrl_point_cls_re = output["re_pred_logits"]
        ctrl_point_coord = output["pred_ctrl_points"]
        ctrl_point_text = output["pred_text_logits"]
        bd_points = output["pred_bd_points"]
        query_features = output["query_features"]
        det_results = self.detection(
            ctrl_point_cls,
            ctrl_point_cls_re,
            ctrl_point_coord,
            ctrl_point_text,
            bd_points,
            query_features,
            images.image_sizes
        )
        det_proposals = []
        for results_per_image in det_results:
            proposal = Instances(results_per_image.image_size)
            bd_pts = results_per_image.bd
            num_text = bd_pts.shape[0]
            if num_text > 0:
                bd_pts = bd_pts.reshape(bd_pts.shape[0], -1, 2)
                x_min, x_max = bd_pts[:, :, 0].min(dim=-1)[0][:, None], bd_pts[:, :, 0].max(dim=-1)[0][:, None]
                y_min, y_max = bd_pts[:, :, 1].min(dim=-1)[0][:, None], bd_pts[:, :, 1].max(dim=-1)[0][:, None]
                proposal.proposal_boxes = Boxes(torch.cat([x_min, y_min, x_max, y_max], dim=-1))
            else:
                proposal.proposal_boxes = Boxes([]).to(self.device)
            proposal.objectness_logits = results_per_image.scores
            proposal.query_features = results_per_image.query_features
            det_proposals.append(proposal)

        _, tracker_losses = self.roi_heads(images, det_proposals, gt_instances)
        losses = {}
        losses.update(tracker_losses)
        if self.with_rescore:
            losses.update(res_loss)
        return losses

    def inference(
        self,
        batched_inputs, time_cost
    ):
        assert not self.training
        start = time.time()
        images = self.preprocess_image(batched_inputs)
        time_cost['pre_process'] += time.time() - start

        start = time.time()
        features, pos = self.backbone(images)
        time_cost['backbone'] += time.time() - start

        ### detection
        start = time.time()
        output = self.detection_transformer(features, pos, self.backbone)
        time_cost['detector'] += time.time() - start

        if self.with_rescore:
            start = time.time()
            output["re_pred_logits"] = self.roi_heads.rescoring_head(output["query_features"])
            time_cost['rescore'] += time.time() - start
        else:
            output["re_pred_logits"] = None

        ctrl_point_cls = output["pred_logits"]
        ctrl_point_cls_re = output["re_pred_logits"]
        ctrl_point_coord = output["pred_ctrl_points"]
        ctrl_point_text = output["pred_text_logits"]
        bd_points = output["pred_bd_points"]
        query_features = output["query_features"]
        det_results = self.detection(
            ctrl_point_cls,
            ctrl_point_cls_re,
            ctrl_point_coord,
            ctrl_point_text,
            bd_points,
            query_features,
            images.image_sizes
        )
        det_proposals = []

        for results_per_image in det_results:
            proposal = Instances(results_per_image.image_size)
            bd_pts = results_per_image.bd
            num_text = bd_pts.shape[0]
            fields = results_per_image._fields
            if num_text > 0:
                bd_pts = bd_pts.reshape(bd_pts.shape[0], -1, 2)
                x_min, x_max = bd_pts[:, :, 0].min(dim=-1)[0][:, None], bd_pts[:, :, 0].max(dim=-1)[0][:, None]
                y_min, y_max = bd_pts[:, :, 1].min(dim=-1)[0][:, None], bd_pts[:, :, 1].max(dim=-1)[0][:, None]
                boxes = torch.cat([x_min, y_min, x_max, y_max], dim=-1)
                scores = results_per_image.scores
                keep = nms(boxes, scores, iou_threshold=self.nms_thresh)
                boxes = boxes[keep]
                for k, v in fields.items():
                    proposal.set(k, v[keep]) # v[keep]
                proposal.proposal_boxes = Boxes(boxes)
                proposal.objectness_logits = results_per_image.scores[keep] # [keep]
            else:
                proposal.proposal_boxes = Boxes([]).to(self.device)
                proposal.objectness_logits = results_per_image.scores
                for k, v in fields.items():
                    proposal.set(k, v)
            det_proposals.append(proposal)

        ### matching
        start = time.time()
        trk_results, _ = self.roi_heads(images, det_proposals, None)
        time_cost['tracker'] += time.time() - start
        results = []
        for trk_per_img, det_per_img in zip(trk_results, det_proposals):
            assert len(trk_per_img) == len(det_per_img)
            result = Instances(trk_per_img.image_size)
            result.reid_features = trk_per_img.reid_features
            result.pred_boxes = trk_per_img.pred_boxes
            result.scores = trk_per_img.scores
            result.pred_classes = trk_per_img.pred_classes

            result.ctrl_points = det_per_img.ctrl_points
            result.recs = det_per_img.recs
            result.bd = det_per_img.bd
            results.append(result)
        return results

    def batch_postprocess(self, instances, image_sizes):
        """
        Allow not clip box for MOT datasets
        """
        processed_results = []
        for results_per_image, image_size in zip(instances, image_sizes):
            height = image_size[0]
            width = image_size[1]
            r = detector_postprocess(
                results_per_image, height, width, self.min_size_test, self.max_size_test)
            processed_results.append({"instances": r})
        return processed_results

    def batch_inference(self, batched_inputs, batch_id, id_count, instances, time_cost): # batch_inference
        video_len = len(batched_inputs)
        start_frame_id = batch_id * 100 # 100
        for frame_id in range(video_len):
            instances_wo_id = self.inference(
                batched_inputs[frame_id: frame_id + 1], time_cost)
            instances.extend([x for x in instances_wo_id])

            real_frame_id = start_frame_id + frame_id
            if real_frame_id == 0: # first frame
                instances[0].track_ids = torch.arange(
                    1, len(instances[0]) + 1,
                    device=instances[0].reid_features.device)
                id_count = len(instances[0]) + 1
            elif real_frame_id == 1:
                start = time.time()
                instances[real_frame_id-1: real_frame_id+1], id_count = self.run_short_term_match(
                    instances[real_frame_id-1: real_frame_id+1], id_count=id_count)
                time_cost['short_match'] += time.time() - start
            else:
                start = time.time()
                instances[real_frame_id-1: real_frame_id+1], cur_id = self.run_short_term_match(
                    instances[real_frame_id-1: real_frame_id+1])
                time_cost['short_match'] += time.time() - start
                if -1 in cur_id:
                    win_st = max(0, real_frame_id + 1 - self.test_len)
                    win_ed = real_frame_id + 1
                    start = time.time()
                    instances[win_st: win_ed], id_count = self.run_long_term_match(
                        instances[win_st: win_ed],
                        k=min(self.test_len - 1, real_frame_id),
                        id_count=id_count,
                        cur_id=cur_id) # n_k x N
                    time_cost['long_match'] += time.time() - start
            assert len(instances[-1].track_ids) == len(torch.unique(instances[-1].track_ids))
            if real_frame_id - self.test_len >= 0:
                instances[real_frame_id - self.test_len].remove('reid_features')
        return instances, id_count

    def run_short_term_match(self, instances, id_count=None):
        n_t = [len(x) for x in instances]
        N, T = sum(n_t), len(n_t)

        reid_features = torch.cat(
            [x.reid_features for x in instances], dim=0)[None]
        asso_output, pred_boxes, _, _ = self.roi_heads._forward_transformer(
            instances, reid_features, 1, short_term=True)  # [n_k x N], N x 4

        asso_output = asso_output[-1].split(n_t, dim=1)  # T x [n_k x n_t]
        asso_output = self.roi_heads._activate_asso(asso_output)  # T x [n_k x n_t]
        asso_output = torch.cat(asso_output, dim=1)  # n_k x N

        n_k = len(instances[1])
        Np = N - n_k
        ids = torch.cat(
            [x.track_ids for t, x in enumerate(instances) if t != 1],
            dim=0).view(Np)  # Np
        k_inds = [x for x in range(n_t[0], sum(n_t))]
        nonk_inds = [i for i in range(N) if not i in k_inds]
        asso_nonk = asso_output[:, nonk_inds]  # n_k x Np
        k_boxes = pred_boxes[k_inds]  # n_k x 4
        nonk_boxes = pred_boxes[nonk_inds]  # Np x 4

        unique_ids = torch.unique(ids)  # M
        M = len(unique_ids)  # number of existing tracks
        id_inds = (unique_ids[None, :] == ids[:, None]).float()  # Np x M

        # (n_k x Np) x (Np x M) --> n_k x M
        traj_score = torch.mm(asso_nonk, id_inds)  # n_k x M
        if id_inds.numel() > 0:
            last_inds = (id_inds * torch.arange(
                Np, device=id_inds.device)[:, None]).max(dim=0)[1]  # M
            last_boxes = nonk_boxes[last_inds]  # M x 4
            last_ious = pairwise_iou(
                Boxes(k_boxes), Boxes(last_boxes))  # n_k x M
        else:
            last_ious = traj_score.new_zeros(traj_score.shape)

        if self.with_iou:
            traj_score = torch.max(traj_score, last_ious)

        match_i, match_j = linear_sum_assignment((- traj_score).cpu())  #
        track_ids = ids.new_full((n_k,), -1)
        for i, j in zip(match_i, match_j):
            thresh = self.overlap_thresh * id_inds[:, j].sum() \
                if not (self.not_mult_thresh) else self.overlap_thresh
            if traj_score[i, j] > thresh:
                track_ids[i] = unique_ids[j]
                
        if id_count:
            for i in range(n_k):
                if track_ids[i] < 0:
                    id_count = id_count + 1
                    track_ids[i] = id_count

        instances[1].track_ids = track_ids

        if id_count:
            return instances, id_count
        return instances, torch.unique(track_ids)

    def run_long_term_match(self, full_instances, k, id_count, cur_id):
        instances = []
        reid_idx = []
        for idx, p in enumerate(full_instances):
            instance = Instances(full_instances[0].image_size)
            if idx != len(full_instances) - 1:
                keep = [True if trk_id not in cur_id else False for trk_id in p.track_ids]
                instance.track_ids = p.track_ids[keep]
            else:
                keep = [True if trk_id == -1 else False for trk_id in p.track_ids]
                reid_idx = keep
            instance.reid_features = p.reid_features[keep]
            instance.pred_boxes = p.pred_boxes[keep]
            instance.scores = p.scores[keep]
            instance.pred_classes = p.pred_classes[keep]
            instance.ctrl_points = p.ctrl_points[keep]
            instance.recs = p.recs[keep]
            instance.bd = p.bd[keep]
            instances.append(instance)

        n_t = [len(x) for x in instances]
        N, T = sum(n_t), len(n_t)

        reid_features = torch.cat(
                [x.reid_features for x in instances], dim=0)[None]
        asso_output, pred_boxes, _, _ = self.roi_heads._forward_transformer(
            instances, reid_features, k) # [n_k x N], N x 4

        asso_output = asso_output[-1].split(n_t, dim=1) # T x [n_k x n_t]
        asso_output = self.roi_heads._activate_asso(asso_output) # T x [n_k x n_t]
        asso_output = torch.cat(asso_output, dim=1) # n_k x N

        n_k = len(instances[k])
        Np = N - n_k
        ids = torch.cat(
            [x.track_ids for t, x in enumerate(instances) if t != k],
            dim=0).view(Np) # Np
        k_inds = [x for x in range(sum(n_t[:k]), sum(n_t[:k + 1]))]
        nonk_inds = [i for i in range(N) if not i in k_inds]
        asso_nonk = asso_output[:, nonk_inds] # n_k x Np
        k_boxes = pred_boxes[k_inds] # n_k x 4
        nonk_boxes = pred_boxes[nonk_inds] # Np x 4

        unique_ids = torch.unique(ids) # M
        M = len(unique_ids) # number of existing tracks
        id_inds = (unique_ids[None, :] == ids[:, None]).float() # Np x M

        # (n_k x Np) x (Np x M) --> n_k x M
        if self.decay_time > 0:
            # (n_k x Np) x (Np x M) --> n_k x M
            dts = torch.cat([x.reid_features.new_full((len(x),), T - t - 2) \
                for t, x in enumerate(instances) if t != k], dim=0) # Np
            asso_nonk = asso_nonk * (self.decay_time ** dts[None, :])

        traj_score = torch.mm(asso_nonk, id_inds) # n_k x M
        if id_inds.numel() > 0:
            last_inds = (id_inds * torch.arange(
                Np, device=id_inds.device)[:, None]).max(dim=0)[1] # M
            last_boxes = nonk_boxes[last_inds] # M x 4
            last_ious = pairwise_iou(
                Boxes(k_boxes), Boxes(last_boxes)) # n_k x M
        else:
            last_ious = traj_score.new_zeros(traj_score.shape)
        
        if self.with_iou:
            traj_score = torch.max(traj_score, last_ious)
        
        if self.max_center_dist > 0.: # filter out too far-away trjactories
            # traj_score n_k x M
            k_boxes = pred_boxes[k_inds] # n_k x 4
            nonk_boxes = pred_boxes[nonk_inds] # Np x 4
            k_ct = (k_boxes[:, :2] + k_boxes[:, 2:]) / 2
            k_s = ((k_boxes[:, 2:] - k_boxes[:, :2]) ** 2).sum(dim=1) # n_k
            nonk_ct = (nonk_boxes[:, :2] + nonk_boxes[:, 2:]) / 2
            dist = ((k_ct[:, None] - nonk_ct[None, :]) ** 2).sum(dim=2) # n_k x Np
            norm_dist = dist / (k_s[:, None] + 1e-8) # n_k x Np
            # id_inds # Np x M
            valid = norm_dist < self.max_center_dist # n_k x Np
            valid_assn = torch.mm(
                valid.float(), id_inds).clamp_(max=1.).long().bool() # n_k x M
            traj_score[~valid_assn] = 0 # n_k x M

        match_i, match_j = linear_sum_assignment((- traj_score).cpu()) #
        track_ids = ids.new_full((n_k,), -1)
        for i, j in zip(match_i, match_j):
            thresh = self.overlap_thresh * id_inds[:, j].sum() \
                if not (self.not_mult_thresh) else self.overlap_thresh
            if traj_score[i, j] > thresh:
                track_ids[i] = unique_ids[j]

        for i in range(n_k):
            if track_ids[i] < 0:
                id_count = id_count + 1
                track_ids[i] = id_count
        full_instances[k].track_ids[reid_idx] = track_ids
        full_track_ids = full_instances[k].track_ids

        return full_instances, id_count

    def _remove_short_track(self, instances):
        ids = torch.cat([x.track_ids for x in instances], dim=0) # N
        unique_ids = ids.unique() # M
        id_inds = (unique_ids[:, None] == ids[None, :]).float() # M x N
        num_insts_track = id_inds.sum(dim=1) # M
        remove_track_id = num_insts_track < self.min_track_len # M
        unique_ids[remove_track_id] = -1
        ids = unique_ids[torch.where(id_inds.permute(1, 0))[1]]
        ids = ids.split([len(x) for x in instances])
        for k in range(len(instances)):
            instances[k] = instances[k][ids[k] >= 0]
        return instances

    def detection(
            self,
            ctrl_point_cls,
            ctrl_point_cls_re,
            ctrl_point_coord,
            ctrl_point_text,
            bd_points,
            query_features,
            image_sizes
    ):
        assert len(ctrl_point_cls) == len(image_sizes)
        results = []
        # cls shape: (b, nq, n_pts, voc_size)
        ctrl_point_text = torch.softmax(ctrl_point_text, dim=-1)
        prob = ctrl_point_cls.mean(-2).sigmoid()
        scores, labels = prob.max(-1)
        if ctrl_point_cls_re is not None:
            re_prob = ctrl_point_cls_re.mean(-2).sigmoid()
            re_scores, re_labels = re_prob.max(-1)
            final_scores = torch.where(scores > re_scores, scores, re_scores)
            final_labels = torch.where(scores > re_scores, labels, re_labels)
        else:
            final_scores = scores
            final_labels = labels

        if bd_points is not None:
            for scores_per_image, labels_per_image, ctrl_point_per_image, ctrl_point_text_per_image, bd, q, image_size in zip(
                    final_scores, final_labels, ctrl_point_coord, ctrl_point_text, bd_points, query_features, image_sizes
            ):
                selector = scores_per_image > self.test_score_threshold
                scores_per_image = scores_per_image[selector]
                labels_per_image = labels_per_image[selector]
                ctrl_point_per_image = ctrl_point_per_image[selector]
                ctrl_point_text_per_image = ctrl_point_text_per_image[selector]
                bd = bd[selector]
                q = q[selector]  # n x 25 x 256

                result = Instances(image_size)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                ctrl_point_per_image[..., 0] *= image_size[1]
                ctrl_point_per_image[..., 1] *= image_size[0]
                result.ctrl_points = ctrl_point_per_image.flatten(1)
                _, text_pred = ctrl_point_text_per_image.topk(1)
                result.recs = text_pred.squeeze(-1)
                bd[..., 0::2] *= image_size[1]
                bd[..., 1::2] *= image_size[0]
                result.bd = bd
                result.query_features = q
                results.append(result)
            return results
        else:
            for scores_per_image, labels_per_image, ctrl_point_per_image, ctrl_point_text_per_image, image_size in zip(
                    final_scores, final_labels, ctrl_point_coord, ctrl_point_text, image_sizes
            ):
                selector = scores_per_image > self.test_score_threshold
                scores_per_image = scores_per_image[selector]
                labels_per_image = labels_per_image[selector]
                ctrl_point_per_image = ctrl_point_per_image[selector]
                ctrl_point_text_per_image = ctrl_point_text_per_image[selector]

                result = Instances(image_size)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                result.rec_scores = ctrl_point_text_per_image
                ctrl_point_per_image[..., 0] *= image_size[1]
                ctrl_point_per_image[..., 1] *= image_size[0]
                result.ctrl_points = ctrl_point_per_image.flatten(1)
                _, text_pred = ctrl_point_text_per_image.topk(1)
                result.recs = text_pred.squeeze(-1)
                result.bd = [None] * len(scores_per_image)
                results.append(result)
            return results