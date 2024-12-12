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
from shapely.geometry import LinearRing
from .bezier_tools import polygon2rbox, cpt_bezier_pts, polygon_to_bezier_pts

logger = logging.getLogger(__name__)

CTLABELS = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
            'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19,
            'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25, '0': 26, '1': 27, '2': 28, '3': 29,
            '4': 30, '5': 31, '6': 32, '7': 33, '8': 34, '9': 35}

import numpy as np

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

        # if len(anno_dict_list) > 200: # query 100
        #     continue
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
            text_category = anno.get("text_category", None)
            text = np.full([25], 37, dtype=np.int32)
            if texts:
                text_str = texts.lower()
                if text_str == '###' or text_category == 'nonalphanumeric':
                    text[0] = 36
                else:
                    for idx, ch in enumerate(text_str):
                        if idx > 24:
                            break
                        if ch in CTLABELS:
                            text[idx] = CTLABELS[ch]
                        else:
                            text[idx] = 36
            else:
                text[0] = 36
            obj['texts'] = text
            
            bezierpts = None
            if "bezier_pts" in anno:
                bezierpts = anno.get("bezier_pts", None)

            elif "poly" in anno:
                polys = anno.get("poly", None)
                polys = np.array(polys).reshape((-1, 2)).astype(np.float32)
                if len(polys) == 4:
                    bezierpts = polygon2rbox(polys, record["height"], record["width"])
                    pRing = LinearRing(bezierpts)
                    if not pRing.is_ccw:
                        bezierpts.reverse()
                    bezierpts = cpt_bezier_pts(bezierpts)
                elif len(polys) == 14:
                    bezierpts = polygon_to_bezier_pts(polys)
                else:
                    raise ValueError('Error Num of points')

            if bezierpts is not None:
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
            else:
                pass

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
        "ICDAR15/train.json"),
    "dstext_train": ("DSText/frame/",
        "DSText/train.json"),
    "artvideo_train": ("ArTVideo/Train/frame/",
        "ArTVideo/Train/train.json"),
    "bov_train": ("BOVText/frame/",
        "BOVText/train.json"),

}

for key, (image_root, json_file) in _PREDEFINED_SPLITS.items():
    register_vts_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets/", json_file) if "://" not in json_file else json_file, 
        os.path.join("datasets/", image_root),  
    )