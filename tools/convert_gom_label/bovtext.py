import glob
import cv2
import os
import json
import tqdm
import codecs
import numpy as np


json_dir_path = r'datasets/BOVText/Train/train_annotation/'
json_save_path = r'datasets/BOVText/train.json'
frames_save_dir = r'datasets/BOVText/frame'


seq_dirs = ['Cls1_Livestreaming', 'Cls2_Cartoon', 'Cls3_Sports', 'Cls4_Celebrity', 'Cls5_Advertising', 'Cls6_NewsReport', 'Cls7_Game',
            'Cls8_Comedy', 'Cls9_Activity', 'Cls10_Program','Cls11_Movie', 'Cls12_Interview', 'Cls13_Introduction',
            'Cls14_Talent', 'Cls15_Photograph', 'Cls16_Government', 'Cls17_Speech', 'Cls18_Travel', 'Cls19_Fashion',
            'Cls20_Campus', 'Cls21_Vlog', 'Cls22_Driving', 'Cls23_International', 'Cls24_Fishery', 'Cls25_ShortVideo',
            'Cls26_Technology', 'Cls27_Education', 'Cls28_BeautyIndustry', 'Cls29_Makeup', 'Cls30_Dance', 'Cls31_Eating', 'Cls32_Unknown']

# 'Cls1_Livestreaming', 'Cls2_Cartoon', 'Cls3_Sports', 'Cls4_Celebrity', 'Cls5_Advertising', 'Cls6_NewsReport', 'Cls7_Game'   313
# 'Cls8_Comedy', 'Cls9_Activity', 'Cls10_Program','Cls11_Movie', 'Cls12_Interview', 'Cls13_Introduction'   288
# 'Cls14_Talent', 'Cls15_Photograph', 'Cls16_Government', 'Cls17_Speech', 'Cls18_Travel', 'Cls19_Fashion'   322
# 'Cls20_Campus', 'Cls21_Vlog', 'Cls22_Driving', 'Cls23_International', 'Cls24_Fishery', 'Cls25_ShortVideo'   317
# 'Cls26_Technology', 'Cls27_Education', 'Cls28_BeautyIndustry', 'Cls29_Makeup', 'Cls30_Dance', 'Cls31_Eating', 'Cls32_Unknown'  300

out = {'images': [], 'annotations': [],
        'categories': [{"supercategory": "beverage", "id": 1,
                        "keypoints": ["mean", "xmin", "x2", "x3", "xmax", "ymin", "y2", "y3", "ymax", "cross"],
                        "name": "text"}],
        'videos': []}

video_id = 0
img_id = 0
ann_cnt = 0
for seq_dir in seq_dirs:
    json_dir = os.path.join(json_dir_path, seq_dir)
    json_files = []
    for json_file in os.listdir(json_dir):
        json_files.append(json_file)
    print(seq_dir)
    color_map = {}
    for json_file in tqdm.tqdm(json_files):
        video_id += 1
        file_name = json_file.split('.')[0]
        print(file_name)
        json_path = os.path.join(json_dir, json_file)
        out['videos'].append({
        'id': video_id,
        'file_name': file_name,
        'data_source': 'BOVText'})
        image_path = os.path.join(frames_save_dir, seq_dir, file_name)
        num_images = len(glob.glob(image_path + '/*.jpg'))
        image = cv2.imread(image_path + '/1.jpg')
        h, w = image.shape[:2]

        # read_json
        with open(json_path, "r", encoding='utf-8') as f:
            Frames = json.load(f)
        f.close()
        # assert num_images == len(Frames)
        for frame_id, objects in Frames.items():
            frame_id = int(frame_id)
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

            for object in objects:
                ann_cnt += 1
                obj_id = object['ID']
                if object['transcription'] == '##DONT#CARE##':
                    transcription = '###'
                    text_category = 'other'
                else:
                    transcription = object['transcription']
                    if object['language'] == 'Chinese':
                        text_category = 'nonalphanumeric'
                    else:
                        text_category = 'alphanumeric'
                points = object['points']
                points = np.array(points, dtype=np.float32).astype(np.int32)
                xs = points[::2].tolist()
                ys = points[1::2].tolist()
                x_min = min(xs)
                x_max = max(xs)
                y_min = min(ys)
                y_max = max(ys)
                box_w = x_max - x_min
                box_h = y_max - y_min
                poly = points.reshape(-1,2).tolist()
                ann_info = {'id': ann_cnt,
                            'category_id': 1,
                            'text_category': text_category,
                            'transcription': transcription,
                            'image_id': img_id,
                            'instance_id': int(obj_id),
                            'bbox': [x_min, y_min, box_w, box_h],
                            'poly': poly,
                            'anno_type': 'line',
                            'box_type': 'quadrilateral',
                            'iscrowd': 0}
                out['annotations'].append(ann_info)

json_fp = codecs.open(json_save_path, 'w', encoding='utf-8')  # use codecs to speed up dump
json_str = json.dumps(out, indent=2, ensure_ascii=False)
json_fp.write(json_str)
json_fp.close()
print('video_num: ', video_id, ' img_num: ', img_id, ' instance_num: ', ann_cnt)

