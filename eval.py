import argparse
from glob import glob
import multiprocessing as mp
import numpy as np
import os
import cv2
import tqdm
import sys

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode

from gomatching.config import add_gom_config
sys.path.insert(0, 'third_party/')
from adet.config import add_deepsolo_cfg

from gomatching.text_track_visualizer import TextTrackingVisualizer, GoMBatchPredictor
from tqdm.contrib import tqdm
from xml.dom.minidom import Document
import xml.etree.cElementTree as ET
from collections import OrderedDict, defaultdict

# constants
WINDOW_NAME = "GoMatching"


class StorageDictionary(object):
    @staticmethod
    def dict2file(file_name, data_dict):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # import pickle
        output = open(file_name, 'wb')
        pickle.dump(data_dict, output)
        output.close()

    @staticmethod
    def file2dict(file_name):
        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # import pickle
        pkl_file = open(file_name, 'rb')
        data_dict = pickle.load(pkl_file)
        pkl_file.close()
        return data_dict

    @staticmethod
    def dict2file_json(file_name, data_dict):
        import json, io
        with io.open(file_name, 'w', encoding='utf-8') as fp:
            fp.write((json.dumps(data_dict, ensure_ascii=False, indent=4)))

    @staticmethod
    def file2dict_json(file_name):
        import json, io
        with io.open(file_name, 'r', encoding='utf-8') as fp:
            data_dict = json.load(fp)
        return data_dict


def Generate_Json_annotation(TL_Cluster_Video_dict, Outpu_dir, xml_dir_):
    '''   '''
    ICDAR21_DetectionTracks = {}
    text_id = 1
    doc = Document()
    video_xml = doc.createElement("Frames")

    for frame in TL_Cluster_Video_dict.keys():

        doc.appendChild(video_xml)
        aperson = doc.createElement("frame")
        aperson.setAttribute("ID", str(frame))
        video_xml.appendChild(aperson)

        ICDAR21_DetectionTracks[frame] = []
        for text_list in TL_Cluster_Video_dict[frame]:
            if len(text_list) == 11:
                ICDAR21_DetectionTracks[frame].append(
                    {"points": text_list[:8], "ID": text_list[8], "transcription": text_list[9],
                     "segmentation": text_list[10]})
            else:
                ICDAR21_DetectionTracks[frame].append(
                    {"points": text_list[:8], "ID": text_list[8], "transcription": text_list[9]})

            # xml
            object1 = doc.createElement("object")
            object1.setAttribute("ID", str(text_list[8]))
            object1.setAttribute("Transcription", str(text_list[9]))
            aperson.appendChild(object1)

            for i in range(4):
                name = doc.createElement("Point")
                object1.appendChild(name)
                name.setAttribute("x", str(int(text_list[i * 2])))
                name.setAttribute("y", str(int(text_list[i * 2 + 1])))

    StorageDictionary.dict2file_json(Outpu_dir, ICDAR21_DetectionTracks)

    # xml
    f = open(xml_dir_, "w")
    f.write(doc.toprettyxml(indent="  "))
    f.close()


def getBboxesAndLabels_icd131(annotations):
    bboxes = []
    Transcriptions = []
    IDs = []
    confidences = []
    for annotation in annotations:
        object_boxes = []
        for point in annotation:
            object_boxes.append([int(point.attrib["x"]), int(point.attrib["y"])])

        points = np.array(object_boxes).reshape((-1))
        points = cv2.minAreaRect(points.reshape((4, 2)))

        points = cv2.boxPoints(points).reshape((-1))
        IDs.append(annotation.attrib["ID"])
        Transcriptions.append(annotation.attrib["Transcription"])
        confidences.append(1)
        bboxes.append(points)

    if bboxes:
        IDs = np.array(IDs, dtype=np.int64)
        bboxes = np.array(bboxes, dtype=np.float32)
    else:
        bboxes = np.zeros((0, 8), dtype=np.float32)
        IDs = np.array([], dtype=np.int64)
        Transcriptions = []
        confidences = []

    return bboxes, IDs, Transcriptions, confidences

def parse_xml_rec(annotation_path):
    utf8_parser = ET.XMLParser(encoding='utf-8')  # utf-8 gbk
    with open(annotation_path, 'r', encoding='utf-8') as load_f: # utf-8 gbk
        tree = ET.parse(load_f, parser=utf8_parser)
    root = tree.getroot()

    ann_dict = {}
    for idx, child in enumerate(root):
        bboxes, IDs, Transcriptions, confidences = \
            getBboxesAndLabels_icd131(child)
        ann_dict[child.attrib["ID"]] = [bboxes, IDs, Transcriptions, confidences]
    return ann_dict

def sort_key(old_dict, reverse=False):
    keys = [int(i) for i in old_dict.keys()]
    keys = sorted(keys, reverse=reverse)
    new_dict = OrderedDict()

    for key in keys:
        new_dict[str(key)] = old_dict[str(key)]
    return new_dict

def get_dir(path):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return path
    return os.path.split(path)[0]

def make_parent_dir(path):
    parent_dir = get_dir(path)
    if not os.path.exists(parent_dir):
        os.makedirs(path)

def write_lines(p, lines):
    p = os.path.abspath(p)
    make_parent_dir(p)
    with open(p, 'w') as f:
        for line in lines:
            f.write(line)

def getid_text(new_xml_dir_):
    for xml in tqdm(os.listdir(new_xml_dir_)):
        id_trans = {}
        id_cond = {}
        if ".txt" in xml or "ipynb" in xml:
            continue

        lines = []
        xml_one = os.path.join(new_xml_dir_, xml)
        ann = parse_xml_rec(xml_one)
        for frame_id_ann in ann:
            points, IDs, Transcriptions, confidences = ann[frame_id_ann]
            for ids, trans, confidence in zip(IDs, Transcriptions, confidences):
                if str(ids) in id_trans:
                    id_trans[str(ids)].append(trans)
                    id_cond[str(ids)].append(float(confidence))
                else:
                    id_trans[str(ids)] = [trans]
                    id_cond[str(ids)] = [float(confidence)]

        id_trans = sort_key(id_trans)
        id_cond = sort_key(id_cond)
        #         print(xml)
        for i in id_trans:
            txts = id_trans[i]
            confidences = id_cond[i]
            txt = max(txts, key=txts.count)
            lines.append('"' + i + '"' + "," + '"' + txt + '"' + "\n")
        write_lines(os.path.join(new_xml_dir_, xml.replace("xml", "txt")), lines)

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_deepsolo_cfg(cfg)
    add_gom_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.ASSO_HEAD.ASSO_THRESH_TEST = cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--show",
        action='store_true',
        help="Visulize results",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    xml_dir = os.path.join(args.output, 'preds')
    os.makedirs(xml_dir, exist_ok=True)
    save_dir = os.path.join(args.output, 'results')
    os.makedirs(save_dir, exist_ok=True)
    json_dir = os.path.join(args.output, 'jsons')
    os.makedirs(json_dir, exist_ok=True)

    os.system('cp ' + args.config_file + ' ' + args.output)
    preded_videos = []
    for preded_video in glob(xml_dir + '/*.xml'):
        preded_videos.append(preded_video.split('//')[-1].split('res_')[-1].split('.xml')[0])

    assert os.path.isdir(args.input[0])
    videos_dir = args.input[0]
    video_files = []
    if 'DSText' in videos_dir:
        data_type = 'DSText'
    elif 'ICDAR15' in videos_dir:
        data_type = 'ICDAR15'
    elif 'BOVText' in videos_dir:
        data_type = 'BOVText'
    else:
        data_type = 'OTHER'
    for video in os.listdir(videos_dir):
        if data_type == 'DSText' or data_type == 'BOVText':
            for video_file in os.listdir(os.path.join(videos_dir, video)):
                video_files.append(os.path.join(videos_dir, video, video_file))
        else:
            video_files.append(os.path.join(videos_dir, video))

    video_text_spotter = GoMBatchPredictor(cfg)

    metadata = MetadataCatalog.get("__unused")
    instance_mode = ColorMode.IMAGE
    tracker_visualizer = TextTrackingVisualizer(metadata, cfg, instance_mode)
    total_frame = 0
    time_cost = {'total_time': 0, 'pre_process': 0, 'backbone': 0, 'detector': 0, 'rescore': 0, 'tracker': 0, 'long_match': 0,
                 'short_match': 0, 'post_process': 0}
    for video in tqdm(video_files):
        img_paths = []
        for img_file in os.listdir(video):
            img_paths.append(os.path.join(video, img_file))
        img_paths.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

        frames_batch = defaultdict(list)
        frames = []
        total_frames = []

        video_name = video.split('/')[-1].split('.')[0]
        if video_name == 'Cls1_Livestreaming_video40' or video_name in preded_videos: # filter bovtext damaged video
            continue

        print('processing {}...'.format(video_name))
        if args.show:
            save_img_dir = os.path.join(save_dir, video_name)
            os.makedirs(save_img_dir, exist_ok=True)

        h, w = read_image(img_paths[0], format="BGR").shape[:2]

        for idx, path in enumerate(img_paths):
            img = read_image(path, format="BGR")
            total_frames.append(img)
            frames_batch[idx // 100].append(img) # 100

        per_video_time = 0
        video_frames = len(total_frames)
        annotation = {}
        instances = []
        last_batch = False
        id_count = 0
        for batch_id in tqdm(range(len(frames_batch))):
            frames = frames_batch[batch_id]
            if batch_id == len(frames_batch) - 1:
                last_batch = True
            instances, id_count, per_batch_time = video_text_spotter(frames, instances, batch_id, id_count, last_batch, time_cost, return_time=True)
            per_video_time += per_batch_time
            time_cost['total_time'] += per_batch_time
            total_frame += len(frames)

        for frame_id, (frame, prediction, save_path) in enumerate(zip(total_frames, instances, img_paths)):
            lines = []
            prediction = tracker_visualizer.pre_vis_process(prediction["instances"].to('cpu'))
            ins_texts = prediction.texts
            ins_polys = prediction.polys
            ins_scores = prediction.scores
            ins_ids = prediction.track_ids
            for poly, ID, score, text in zip(ins_polys, ins_ids, ins_scores, ins_texts):
                rect = cv2.minAreaRect(poly)
                box = np.array(cv2.boxPoints(rect)).reshape([8])
                x1, y1, x2, y2, x3, y3, x4, y4 = [int(i) for i in box[:8]]
                max_x, min_x = max(x1, x2, x3, x4), min(x1, x2, x3, x4)
                max_y, min_y = max(y1, y2, y3, y4), min(y1, y2, y3, y4)
                if max_y - min_y < 5 or max_x - min_x < 5:
                    continue
                blank = np.zeros((h, w), dtype=np.uint8)
                seg = [poly.astype(int).tolist()]
                lines.append([x1, y1, x2, y2, x3, y3, x4, y4, int(ID), text, seg])
            if args.show:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                vis_frame = tracker_visualizer.draw_instance_predictions(frame, prediction)
                vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
                out_filename = os.path.join(save_img_dir, os.path.basename(save_path))
                cv2.imwrite(out_filename, vis_frame)
            annotation.update({str(frame_id + 1): lines})
        print("Video: ", video_name, "per_img_time: ", per_video_time / video_frames, ", FPS: ", video_frames / per_video_time)
        if data_type == 'ICDAR15':
            xml_name = video_name.split("_")
            xml_name = (xml_name[0] + "_" + xml_name[1]).replace("V","v")
            xml_path = os.path.join(xml_dir, "res_{}.xml".format(xml_name))
        else:
            xml_path = os.path.join(xml_dir, "res_{}.xml".format(video_name))
        json_path = os.path.join(json_dir, "{}.json".format(video_name))
        Generate_Json_annotation(annotation, json_path, xml_path)

    getid_text(xml_dir)
    print("total_time: ", time_cost['total_time'], ", per_video_time: ", time_cost['total_time'] / len(video_files), ", per_img_time: ", time_cost['total_time'] / total_frame, ", FPS: ", total_frame / time_cost['total_time'])
    print(time_cost)

# python eval.py --config-file configs/GoMatching_DSText.yaml --input datasets/DSText/frame_test/ --output output/GoMatching/DSText --opts MODEL.WEIGHTS trained_models/GoMatching_dstext.pth
# python eval.py --config-file configs/GoMatching_PP_DSText.yaml --input datasets/DSText/frame_test/ --output output/GoMatching++/DSText --opts MODEL.WEIGHTS trained_models/GoMatching_pp_dstext.pth