import os
import contextlib
from detectron2.utils.file_io import PathManager
import io
import pycocotools.mask as mask_util
import logging

from fvcore.common.timer import Timer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import numpy as np

logger = logging.getLogger(__name__)

CTLABELS = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
            'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19,
            'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, '0': 26, '1': 27, '2': 28, '3': 29,
            '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, '9': 35}

import numpy as np

def distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def bezier_points(p1, p2, num_points):
    pts = []
    pts.append(p1)
    for i in range(1, num_points+1):
        t = i / (num_points+1)
        x = int((1-t)*p1[0] + t*p2[0])
        y = int((1-t)*p1[1] + t*p2[1])
        pts.append([x, y])
    pts.append(p2)
    return pts

def longest_edges(rect):
    pts_arr = np.array(rect)
    ctr = np.mean(pts_arr, axis=0, dtype=int)
    ids = np.argsort(np.arctan2(pts_arr[:, 1] - ctr[1], pts_arr[:, 0] - ctr[0])) # clockwise
    poly = pts_arr[ids]
    edges = [(poly[i], poly[(i+1) % 4]) for i in range(4)]
    edges_sorted = sorted(edges, key=lambda e: -distance(*e))
    return edges_sorted[:2]

def cpt_bezier_pts(rect):
    edges = longest_edges(rect)
    bezier_pts = []
    for edge in edges:
        bezier_pts.extend(bezier_points(*edge, 2))
    bzr_arr = np.array(bezier_pts)
    bzr_ctr = np.mean(bzr_arr, axis=0, dtype=int)
    ids = np.argsort(np.arctan2(bzr_arr[:, 1] - bzr_ctr[1], bzr_arr[:, 0] - bzr_ctr[0]))  # clockwise
    bzr_clk = bzr_arr[ids]
    return bzr_clk

def load_video_json(
    json_file, image_root, dataset_name=None, extra_annotation_keys=None,
    map_inst_id=False):
    """
    add video id to image record
    enable inst_id
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    if map_inst_id:
        assert 'instance_id' in extra_annotation_keys
        instance_ids = set(
            x['instance_id'] for x in coco_api.dataset['annotations'] \
                if x['instance_id'] > 0)
        inst_id_map = {x: i + 1 for i, x in enumerate(sorted(instance_ids))}
        if len(instance_ids) > 0: 
            print('Maping instances len/ min/ max', \
              len(inst_id_map), min(inst_id_map.values()), max(inst_id_map.values()))
        inst_id_map[0] = 0
        inst_id_map[-1] = 0
        
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    imgs_anns = list(zip(imgs, anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])

    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]
        # modified 
        video_id = img_dict.get('video_id', -1)
        record['video_id'] = video_id
        # finish modified
        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

            obj = {key: anno[key] for key in ann_keys if key in anno}

            segm = anno.get("segmentation", None)
            if segm:  # either list[list[float]] or dict(RLE)
                if isinstance(segm, dict):
                    if isinstance(segm["counts"], list):
                        # convert to compressed RLE
                        segm = mask_util.frPyObjects(segm, *segm["size"])
                else:
                    # filter out invalid polygons (< 3 points)
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                obj["segmentation"] = segm

            keypts = anno.get("keypoints", None)
            if keypts:  # list[int]
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        keypts[idx] = v + 0.5
                obj["keypoints"] = keypts

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            if map_inst_id:
                obj['instance_id'] = inst_id_map[obj['instance_id']]

            texts = anno.get("transcription", None)
            if texts:
                text_str = texts.lower()
                text = np.full([25], 37, dtype=np.int32)
                if text_str == '###':
                    text[0] = 36
                else:
                    for idx, ch in enumerate(text_str):
                        if ch in CTLABELS:
                            text[idx] = CTLABELS[ch]
                        else:
                            text[idx] = 36
                obj['texts'] = text

            bezierpts = anno.get("poly", None)
            if bezierpts:
                bezierpts = cpt_bezier_pts(bezierpts)
                bezierpts = np.array(bezierpts).reshape(-1, 2)
                center_bezierpts = (bezierpts[:4] + bezierpts[4:][::-1, :]) / 2
                obj["beziers"] = center_bezierpts
                bezierpts = bezierpts.reshape(2, 4, 2).transpose(0, 2, 1).reshape(4, 4)
                u = np.linspace(0, 1, 25) # num_pts_cfg: 25
                boundary = np.outer((1 - u) ** 3, bezierpts[:, 0]) \
                           + np.outer(3 * u * ((1 - u) ** 2), bezierpts[:, 1]) \
                           + np.outer(3 * (u ** 2) * (1 - u), bezierpts[:, 2]) \
                           + np.outer(u ** 3, bezierpts[:, 3])
                obj["boundary"] = np.hstack([boundary[:, :2], boundary[:, 2:][::-1, :]]).reshape(-1, 2)
                obj["polyline"] = (boundary[:, :2] + boundary[:, 2:][::-1, :]) / 2

            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_vts_instances(name, metadata, json_file, image_root):
    """
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    DatasetCatalog.register(name, lambda: load_video_json(
        json_file, image_root, name, extra_annotation_keys=['instance_id'],
        map_inst_id=True))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, 
        evaluator_type="vts", **metadata
    )

categories = [
    {'id': 1, 'name': 'text'},
]

def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

_PREDEFINED_SPLITS = {
    "icdar15_train": ("ICDAR15/frame/",
        "ICDAR15/vts_train.json"),
    "icdar15_test": ("ICDAR15/frame_test/",
        "ICDAR15/vts_test_wo_anno.json"),
    "dstext_train": ("DSText/frame/",
        "DSText/vts_train.json"),
    "dstext_test": ("DSText/frame_test/",
        "DSText/vts_test_wo_anno.json"),

}

for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
    register_vts_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("./datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("./datasets", image_root),  # datasets
    )