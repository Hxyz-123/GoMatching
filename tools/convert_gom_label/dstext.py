import glob
import cv2
import os
import json
import tqdm
import codecs
import xml.etree.ElementTree as ET
import random

xml_dir_path = r'datasets/DSText/Train_annotation/'
json_save_path = r'datasets/DSText/train.json'
frames_save_dir = r'datasets/DSText/frame'


seq_dirs = []
for seq_dir in os.listdir(xml_dir_path):
    seq_dirs.append(seq_dir) 

out = {'images': [], 'annotations': [],
        'categories': [{"supercategory": "beverage", "id": 1,
                        "keypoints": ["mean", "xmin", "x2", "x3", "xmax", "ymin", "y2", "y3", "ymax", "cross"],
                        "name": "text"}],
        'videos': []}

video_id = 0
img_id = 0
ann_cnt = 0
for seq_dir in seq_dirs:
    xml_dir = os.path.join(xml_dir_path, seq_dir)
    xml_files = []
    for xml_file in os.listdir(xml_dir):
        xml_files.append(xml_file)
    print(seq_dir)
    for xml_file in tqdm.tqdm(xml_files):
        video_id += 1
        file_name = xml_file.split('_GT')[0]
        print(file_name)
        xml_path = os.path.join(xml_dir, xml_file)
        out['videos'].append({
            'id': video_id,
            'file_name': file_name,
            'data_source': 'DSText'})
        image_path = os.path.join(frames_save_dir, seq_dir, file_name)
        num_images = len(glob.glob(image_path + '/*.jpg'))
        image = cv2.imread(image_path + '/1.jpg')
        h, w = image.shape[:2]

        # read_xml
        tree = ET.parse(xml_path)
        root = tree.getroot()
        Frames = root.findall('frame')
        assert num_images == len(Frames)
        for frame in Frames:

            frame_id = int(frame.attrib['ID'])
            objects = frame.findall('object')

            img_id += 1
            img_info = {
                    "file_name": '{}/{}/{}'.format(seq_dir, file_name, str(frame_id) + '.jpg'),
                    "id": img_id,
                    "height": h,
                    "width": w,
                    "frame_id": frame_id,
                    "prev_image_id": img_id - 1 if frame_id > 1 else -1,
                    "next_image_id": img_id + 1 if frame_id < num_images else -1,
                    "video_id": video_id
                    }
            out['images'].append(img_info)

            obj_ids = []
            for object in objects:
                object_detail = object.attrib
                obj_id = object_detail['ID']
                if obj_id in obj_ids:
                    continue
                else:
                    obj_ids.append(obj_id)
                    ann_cnt += 1
                if object_detail['Transcription'] == '##DONT#CARE##':
                    transcription = '###'
                    text_category = 'other'
                else:
                    transcription = object_detail['Transcription']
                    if object_detail['language'] == 'Chinese':
                        text_category = 'nonalphanumeric'
                    else:
                        text_category = 'alphanumeric'

                points = object.findall('Point')
                xs = []
                ys = []
                poly = []
                for point in points:
                    box_detail = point.attrib
                    x, y = int(box_detail['x']), int(box_detail['y'])
                    xs.append(x)
                    ys.append(y)
                    poly.append([x, y])
                x_min = min(xs)
                x_max = max(xs)
                y_min = min(ys)
                y_max = max(ys)
                box_w = x_max - x_min
                box_h = y_max - y_min
                ann_info = {'id': ann_cnt,
                    'category_id': 1,
                    'text_category': text_category,
                    'transcription': transcription,
                    'image_id': img_id,
                    'instance_id': int(obj_id),
                    'bbox': [x_min, y_min, box_w, box_h],
                    'poly': poly,
                    'anno_type': 'word',
                    'box_type': 'quadrilateral',
                    'iscrowd': 0}
                out['annotations'].append(ann_info)

json_fp = codecs.open(json_save_path, 'w', encoding='utf-8')  # use codecs to speed up dump
json_str = json.dumps(out, indent=2, ensure_ascii=False)
json_fp.write(json_str)
json_fp.close()
print('video_num: ', video_id, ' img_num: ', img_id, ' instance_num: ', ann_cnt)

