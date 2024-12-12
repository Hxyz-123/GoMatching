import cv2
import os
from tqdm import tqdm
video_dir = r'datasets/ICDAR15/ICDAR15_test/'

video_len = 24
save_dir = r'datasets/ICDAR15/frame_test'


if not os.path.exists(save_dir):
        os.mkdir(save_dir)

video_path_list = []
for dir in os.listdir(video_dir):
    if os.path.isdir(os.path.join(video_dir, dir)):
         sub_dir = os.path.join(video_dir, dir)
         sub_save_path = os.path.join(save_dir, dir)
         if not os.path.exists(sub_save_path):
              os.mkdir(sub_save_path)
         for file in os.listdir(sub_dir):
              if '.mp4' not in file and '.avi' not in file:
                   continue
              else:
                   video_path_list.append(os.path.join(dir, file))
    else:
        if '.mp4' not in dir and '.avi' not in dir:
            continue
        else:
            video_path_list.append(dir)
try:         
    assert len(video_path_list) == video_len
except:
     raise RuntimeError('video num: ', video_len, ' truth num: ', len(video_path_list))

total_frame_num = 0
for video_path in tqdm(video_path_list):
    video_name = video_path.split('\\')[-1].split('.')[0]
    save_path = os.path.join(save_dir, video_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    video_object = cv2.VideoCapture(os.path.join(video_dir, video_path))
    # video_object.set(cv2.CAP_PROP_POS_FRAMES, visual_frame_index - 1)
    frames_num = int(video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frame_num += frames_num
    print('video_name: ', video_name, ', frames_num: ', frames_num)

    for i in tqdm(range(frames_num)):
        ret, frame = video_object.read()
        frame_name = "{}.jpg".format(i + 1)
        if 'BOVText' in video_dir:
            cv2.imwrite(os.path.join(save_path, frame_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        else:
            cv2.imwrite(os.path.join(save_path, frame_name), frame)

print(total_frame_num)