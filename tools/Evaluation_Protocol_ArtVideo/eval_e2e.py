from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import argparse
import os
import numpy as np
import copy
import motmetrics as mm
import logging
from tqdm import  tqdm
from tracking_utils.io import read_results
import Levenshtein
from shapely.geometry import Polygon
from pycocotools import mask as mask_utils
import cv2

def parse_args():
    """Defines and parses command-line arguments."""
    parser = argparse.ArgumentParser(description="""
Compute metrics for trackers using MOTChallenge ground-truth data with data preprocess.

Files
-----
All file content, ground truth and test files, have to comply with the
format described in

Milan, Anton, et al.
"Mot16: A benchmark for multi-object tracking."
arXiv preprint arXiv:1603.00831 (2016).
https://motchallenge.net/

Structure
---------

Layout for ground truth data
    <GT_ROOT>/<SEQUENCE_1>/gt_video_1.json
    <GT_ROOT>/<SEQUENCE_2>/gt_video_2.json
    ...

Layout for test data
    <TEST_ROOT>/<SEQUENCE_1>/video_1.json
    <TEST_ROOT>/<SEQUENCE_2>/video_2.json
    ...

Sequences of ground truth and test will be matched according to the `<SEQUENCE_X>`
string in the seqmap.""", formatter_class=argparse.RawTextHelpFormatter)
    
    parser = argparse.ArgumentParser(description="evaluation on MMVText", formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--groundtruths', type=str,default='./Test/Annotation'
                        , help='Directory containing ground truth files.')
    parser.add_argument('--tests', type=str,default='../../output/artvideo/jsons',
                        help='Directory containing tracker result files')
    parser.add_argument('--log', type=str, help='a place to record result and outputfile of mistakes', default='')
    parser.add_argument('--loglevel', type=str, help='Log level', default='info')
    parser.add_argument('--fmt', type=str, help='Data format', default='mot15-2D')
    parser.add_argument('--solver', type=str, default="lap", help='LAP solver to use')
    parser.add_argument('--skip', type=int, default=0, help='skip frames n means choosing one frame for every (n+1) frames')
    parser.add_argument('--iou', type=float, default=0.5, help='special IoU threshold requirement for small targets')
    parser.add_argument("--curve", action='store_true', help="only eval curved text",)
    return parser.parse_args()


def cal_similarity(string1, string2):
    if string1 == "" and string2 == "":
        return 1.0
    if Levenshtein.distance(string1, string2) == 1 :
        return 0.95
    return 1 - Levenshtein.distance(string1, string2) / max(len(string1), len(string2))
    

def calculate_iou_polygen(box1, box2):
    box1 = np.array(box1).reshape(-1, 2)
    poly1 = Polygon(box1).convex_hull

    box2 = np.array(box2).reshape(-1, 2)
    poly2 = Polygon(box2).convex_hull
    if poly1.area < 0.01 or poly2.area < 0.01:
        return 0.0
    if not poly1.intersects(poly2):
        iou = 0
    else:
        inter_area = poly1.intersection(poly2).area
        union_area = poly1.area + poly2.area - inter_area
        iou = float(inter_area) / union_area
    return iou

def calculate_iou_mask(mask1,mask2):
    inter_area = cv2.bitwise_and(mask1, mask2).sum()
    if inter_area < 1:
        return 0.0
    union_area = cv2.bitwise_or(mask1, mask2).sum()
    mask_iou = inter_area / union_area
    return mask_iou


def iou_matrix_polygen(objs, gt_transcription, hyps, transcription, max_iou=1., max_similarity=0.9):
    if np.size(objs) == 0 or np.size(hyps) == 0:
        return np.empty((0, 0))
    m = len(objs)
    n = len(hyps)

    dist_mat = np.zeros((m, n))

    for x, row in enumerate(range(m)):
        for y, col in enumerate(range(n)):
            iou = calculate_iou_mask(objs[row], hyps[col])
            dist = iou

            gt_trans = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "",
                              gt_transcription[row]).lower()
            hyps_trans = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "",
                                transcription[col]).lower()

            if dist < max_iou or cal_similarity(gt_trans, hyps_trans) < 0.9:
                dist = np.nan

            dist_mat[row][col] = dist
    return dist_mat

class Evaluator(object):
    def __init__(self, data_root, seq_name, data_type, iou_thr, only_curve):
        self.data_root = data_root
        self.seq_name = seq_name
        self.data_type = data_type
        self.load_annotations()
        self.reset_accumulator()
        self.iou_thr = iou_thr
        self.only_curve = only_curve

    def load_annotations(self):
        assert self.data_type in ('mot', 'text')
        if self.data_type == 'mot':
            gt_filename = os.path.join(self.data_root, self.seq_name, 'gt', 'gt.txt')
            
        else:
            name = self.seq_name
            gt_filename = os.path.join(self.data_root,name) 

        self.org_gt_frame_dict = read_results(gt_filename, self.data_type, is_gt=True)
        self.gt_frame_dict = {}
        for frame_id in range(1, len(self.org_gt_frame_dict['frame']) + 1):
            self.gt_frame_dict[str(frame_id)] = []
            self.img_h = self.org_gt_frame_dict['frame'][0]["height"]
            self.img_w = self.org_gt_frame_dict['frame'][0]["width"]
        for ann in self.org_gt_frame_dict['annotations']:
            frame_id = ann['frame_id']
            annotation = {}
            annotation["points"] = np.array(ann['point'], dtype=np.float32).astype(np.int32).reshape(-1)
            annotation["transcription"] = ann['Transcription']
            annotation["ID"] = ann['obj_id']
            mask = mask_utils.decode(ann['segmentation'])
            annotation["mask"] = mask
            annotation["text_type"] = ann['text_type']
            self.gt_frame_dict[str(frame_id)].append(annotation)


    def reset_accumulator(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def eval_frame(self, frame_id, trk_tlwhs, trk_ids, trk_transcription, rtn_events=False):
        # results
        trk_tlwhs = np.copy(trk_tlwhs)
        trk_ids = np.copy(trk_ids)
        trk_transcription = np.copy(trk_transcription)

        gt_objs = self.gt_frame_dict[frame_id]
            
        gts = []
        ids = []
        transcription = []
        ignored = []
        for gt in gt_objs:
            if self.only_curve:
                if gt["transcription"] == "###" or gt["transcription"] == "#1" or gt["text_type"] == 'Straight': #curve spotting
                    ignored.append(gt["mask"])
                else:
                    gts.append(gt["mask"])
                    ids.append(gt["ID"])
                    transcription.append(gt["transcription"])
            else:
                if gt["transcription"] == "###" or gt["transcription"] == "#1": #spotting
                    ignored.append(gt["mask"])
                else:
                    gts.append(gt["mask"])
                    ids.append(gt["ID"])
                    transcription.append(gt["transcription"])

        gt_objs = gts

        gt_tlwhs = gt_objs
        gt_ids = ids
        gt_transcription = transcription

        # filter 
        trk_tlwhs_ = []
        trk_ids_ = []
        trk_transcription_ = []
        
        for idx,box1 in enumerate(trk_tlwhs):
            flag = 0
            for box2 in ignored:
                iou = calculate_iou_mask(box1, box2)
                if iou > self.iou_thr:
                    flag=1
            if flag == 0:
                trk_tlwhs_.append(trk_tlwhs[idx])
                trk_ids_.append(trk_ids[idx])
                trk_transcription_.append(trk_transcription[idx])
                
        trk_tlwhs = trk_tlwhs_
        trk_ids = trk_ids_
        trk_transcription = trk_transcription_
        
        iou_distance = iou_matrix_polygen(gt_tlwhs,gt_transcription, trk_tlwhs, trk_transcription, max_iou=self.iou_thr, max_similarity=0.9)
        
        # acc
        self.acc.update(gt_ids, trk_ids, iou_distance)

        if rtn_events and iou_distance.size > 0 and hasattr(self.acc, 'last_mot_events'):
            events = self.acc.last_mot_events
        else:
            events = None
        return events

    def eval_file(self, filename):
        self.reset_accumulator()
        result_frame_dict = read_results(filename, self.data_type, is_gt=False)

        for frame_id in range(len(self.gt_frame_dict)):
            frame_id += 1
            if str(frame_id) in result_frame_dict.keys():
                trk_objs = result_frame_dict[str(frame_id)]
                
                trk_tlwhs = []
                trk_ids = []
                trk_transcription = []
                for trk in trk_objs:
                    if 'segmentation' in trk:
                        points = np.array(trk["points"], dtype=np.float32).astype(np.int32)
                        if isinstance(trk["segmentation"], list):
                            blank = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
                            mask = cv2.fillPoly(blank, np.array(trk["segmentation"]), 1)
                        else:
                            mask = mask_utils.decode(trk["segmentation"])
                        trk_tlwhs.append(mask) # points mask
                    else:
                        points = np.array(trk["points"], dtype=np.float32).astype(np.int32)
                        mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
                        mask = cv2.fillPoly(mask, [np.array(points, dtype=np.int32).reshape(-1, 2)], 1)
                        trk_tlwhs.append(mask) # points mask
                    trk_ids.append(np.array(trk["ID"], dtype=np.int32))
                    try:
                        trk_transcription.append(trk["transcription"])
                    except:
                        trk_transcription.append("error")
                        print(trk)
            else:
                trk_tlwhs = np.array([])
                trk_ids = np.array([])
                trk_transcription = []
                
            self.eval_frame(str(frame_id), trk_tlwhs, trk_ids,trk_transcription, rtn_events=False)

        return self.acc

    @staticmethod
    def get_summary(accs, names, metrics=('mota','motp', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall')):
        names = copy.deepcopy(names)
        if metrics is None:
            metrics = mm.metrics.motchallenge_metrics
        metrics = copy.deepcopy(metrics)

        mh = mm.metrics.create()
        summary = mh.compute_many(
            accs,
            metrics=metrics,
            names=names,
            generate_overall=True
        )
        return summary

    @staticmethod
    def save_summary(summary, filename):
        import pandas as pd
        writer = pd.ExcelWriter(filename)
        summary.to_excel(writer)
        writer.save()
        
def main():
    args = parse_args()
    
    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {} '.format(args.loglevel))
    logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')

    if args.solver:
        mm.lap.default_solver = args.solver
        mm.lap.default_solver = 'lap'

    only_curve = False
    if args.curve:
        only_curve = True

    data_type = 'text'
    
    filter_seqs = []
    for video_ in os.listdir(args.groundtruths):
        if video_ == ".ipynb_checkpoints":
            continue
        filter_seqs.append(video_)
    
    
    accs = []
    for seq in tqdm(filter_seqs):
        # eval
        result_path = os.path.join(args.tests, seq)
        evaluator = Evaluator(args.groundtruths, seq, data_type, args.iou, only_curve)
        accs.append(evaluator.eval_file(result_path))
        
    # metric names
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, filter_seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    
    print(strsummary)
    return summary['mota']['OVERALL']

if __name__ == '__main__':
    main()


