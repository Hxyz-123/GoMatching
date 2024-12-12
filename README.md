<h1 align="center"> GoMatching: A Simple Baseline for Video Text Spotting via Long and Short Term Matching<a href="https://arxiv.org/abs/2401.07080"><img src="https://img.shields.io/badge/arXiv-Paper-<color>"></a> </h1>
<p align="center">
<h4 align="center">This is the official repository of the paper <a href="https://arxiv.org/abs/2401.07080">GoMatching: A Simple Baseline for Video Text Spotting via Long and Short Term Matching</a>.</h4>
<h5 align="center"><em>Haibin He, Maoyuan Ye, Jing Zhang, Juhua Liu, Bo Du, Dacheng Tao</em></h5>
<p align="center">
  <a href="#introduction">Introduction</a> |
  <a href="#news">News</a> |
  <a href="#usage">Usage</a> |
  <a href="#main results">Main Results</a> |
  <a href="#statement">Statement</a>
</p>




# Introduction
<figure>
<img src="figs/framework.png">
</figure>

1. We identify a main bottleneck in the state-of-the-art video text spotter: the limited recognition capability. In response to this issue, we propose to efficiently turn an off-the-shelf query-based image text spotter into a specialist on video and present a simple baseline termed GoMatching, along with its extension version, GoMatching++.
2. We introduce a rescoring mechanism and long-short term matching module to adapt image text spotter to video datasets and enhance the tracker's capabilities.
3. We establish the ArTVideo test set for addressing the absence of curved texts in current video text spotting datasets and evaluating the performance of video text spotters on videos with arbitrary-shaped text. ArTVideo contains 60 video clips, featuring over 30% curved text.
4. GoMatching only requires 3 hours training on one Nvidia RTX 3090 GPU for ICDAR15-video. For video text spotting task, GoMatching achieves 72.20 MOTA on ICDAR15-video, setting a new record on the leaderboard. We reveal the probability of freezing off-the-shelf ITS part and focusing on tracking, thereby saving training budgets while reaching SOTA performance. 



# News

***13/01/2024***

- The paper is uploaded to arxiv! 

***20/01/2024***

- Update ArTVideo and refresh a new record on ICDAR15-video! 

***25/09/2024***

- The paper is accepted by NeurIPS 2024 Conference!

***12/12/2024***

- Update GoMatching NeurIPS version and a more concise and effective implementation version, GoMacthing++. The paper of GoMatching++ will be released soon.




# Usage
### Dataset

Videos in [ICDAR15-video](https://rrc.cvc.uab.es/?ch=3&com=downloads),  [DSText](https://rrc.cvc.uab.es/?ch=22&com=downloads)  and [BOVText](https://github.com/weijiawu/BOVText-Benchmark) should be extracted into frames. And using json format annotation files [[ICDAR15-video](https://drive.google.com/file/d/18_d5oN4yvcXCV1nUb8OlQJSDzLAZ1Mlz/view?usp=drive_link) & [DSText](https://drive.google.com/drive/folders/1D49hsIsYQtDNzYsgoYNEEmUgXh8QW68y?usp=drive_link)] we provide for training or processing the raw data by the provided codes.  For [ArTVideo](https://drive.google.com/drive/folders/1k2Z-DZHn4kVmiyG6pRbJFjteP0VIwt1a), you can download it to `./datasets`. The prepared Data organization is as follows:

```
|- ./datasets
		|--- ICDAR15
		|      |--- frame
		|            |--- Video_10_1_1
		|                       |--- 1.jpg
		|                       └---  ...
		|            └--- ...
		|      |--- frame_test
		|            |--- Video_11_4_1
		|                       |--- 1.jpg
		|                       └---  ...
		|            └--- ...
		|      └--- train.json
		|
		|--- DSText
		|      |--- frame
		|            |--- Activity
		|                     |--- Video_163_6_3
		|                               |--- 1.jpg
		|                               └---  ...
		|                     └--- ...
		|            └--- ...
		|      |--- frame_test
		|            |--- Activity
		|                     |--- Video_162_6_2
		|                               |--- 1.jpg
		|                               └---  ...
		|                     └--- ...
		|            └--- ...
		|      |--- dstext_test_e2e_gt.zip
		|      |--- dstext_test_trk_gt.zip
		|      └--- train.json
		|
		|--- BOVText
		|      |--- frame
		|            |--- Cls1_Livestreaming
		|            		|--- Cls1_Livestreaming_video1
		|                       		|--- 1.jpg
		|                       		└---  ...
		|            		└--- ...
		|            		└--- ...
		|
		|      |--- frame_test
		|            |--- Cls1_Livestreaming
		|            		|--- Cls1_Livestreaming_video5
		|                       		|--- 1.jpg
		|                       		└---  ...
		|            		└--- ...
		|            └--- ...
		|      |--- Test
		|            |--- test_annotation
		|            └--- Video
		|      |--- Train
		|            |--- train_annotation
		|            └--- Video
		|      └--- train.json
		|
		|--- ArTVideo
		|      |--- Train
		|            |--- frame
		|                   |--- video1
		|                           |--- 1.jpg
		|                           └---  ...
		|                   └--- ...
		|            └--- train.json
		|      |--- Test
		|            |--- frame
		|                   |--- video1
		|                           |--- 1.jpg
		|                           └---  ...
		|                   └--- ...
		|            |--- json
		|                   |--- video_1.json
		|                   └--- ...
```

processing the raw data

```python
1. ###   turn video to frame
python tools/video2frame.py

2. ###  generate json files to train GoMatching
python tools/convert_gom_label/icdar15.py # bovtext.py  dstext.py
```



### Installation

Python_3.8 + PyTorch_1.9.0 + CUDA_11.1 + Detectron2_v0.6

```python
git clone https://github.com/Hxyz-123/GoMatching.git
cd GoMatching
conda create -n gomatching python=3.8 -y
conda activate gomatching
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
cd third_party
python setup.py build develop
```

### Pre-trained model 

We share the trained [deepsolo weights](https://drive.google.com/drive/folders/1xqjTsrUvMEiGGReLI3wLenkSiPXN4-5d?usp=drive_link) we use in GoMatching. You can download it to `./pretrained_models`. If you want to use other model weights in [official deepsolo](https://github.com/ViTAE-Transformer/DeepSolo), run following code to decouple the backbone and transformer in deepsolo before training GoMatching.

```python
python tools/decouple_deepsolo.py --input path_to_original_weights  --output output_path
```

### Train

ICDAR15

```python
python train_net.py --num-gpus 1 --config-file configs/GoMatching_ICDAR15.yaml # for GoMatching
python train_net.py --num-gpus 1 --config-file configs/GoMatching_PP_ICDAR15.yaml # for GoMatching++
```

DSText

```python
python train_net.py --num-gpus 1 --config-file configs/GoMatching_DSText.yaml # for GoMatching
python train_net.py --num-gpus 1 --config-file configs/GoMatching_PP_DSText.yaml # for GoMatching++
```

BOVText

```python
python train_net.py --num-gpus 1 --config-file configs/GoMatching_BOVText.yaml # for GoMatching
python train_net.py --num-gpus 1 --config-file configs/GoMatching_PP_BOVText.yaml # for GoMatching++
```

ArTVideo

```python
python train_net.py --num-gpus 1 --config-file configs/GoMatching_ArTVideo.yaml # for GoMatching
python train_net.py --num-gpus 1 --config-file configs/GoMatching_PP_ArTVideo.yaml # for GoMatching++
```



### Evaluation

**ICDAR15**

```python
python eval.py --config-file configs/GoMatching_ICDAR15.yaml --input ./datasets/ICDAR15/frame_test/ --output output/icdar15 --opts MODEL.WEIGHTS trained_models/ICDAR15/xxx.pth ### for GoMatching

python eval.py --config-file configs/GoMatching_PP_ICDAR15.yaml --input ./datasets/ICDAR15/frame_test/ --output output/icdar15 --opts MODEL.WEIGHTS trained_models/ICDAR15/xxx.pth ### for GoMatching++

cd output/icdar15/preds
zip -r ../preds.zip ./*
zip -r ../track.zip ./*.xml
```

Then you can submit the `zip` file to the [official websit](https://rrc.cvc.uab.es/?ch=3&com=evaluation&task=4) for evaluation.



**DSText**

online evaluation

```python
python eval.py --config-file configs/GoMatching_DSText.yaml --input ./datasets/DSText/frame_test/ --output output/dstext --opts MODEL.WEIGHTS trained_models/DSText/xxx.pth  ### for GoMatching

python eval.py --config-file configs/GoMatching_PP_DSText.yaml --input ./datasets/DSText/frame_test/ --output output/dstext --opts MODEL.WEIGHTS trained_models/DSText/xxx.pth  ### for GoMatching

cd output/dstext/preds
zip -r ../track.zip ./*.xml
zip -r ../preds.zip ./*.*
```

Then you can submit the `zip` file to the [official websit](https://rrc.cvc.uab.es/?ch=22&com=evaluation&task=2) for evaluation.



offline evaluation

1. download the [grouth-truths](https://drive.google.com/drive/folders/1D49hsIsYQtDNzYsgoYNEEmUgXh8QW68y?usp=drive_link) label to `./datasets/DSText`

2. used the codes we provide:

```python
python eval.py --config-file configs/GoMatching_DSText.yaml --input ./datasets/DSText/frame_test/ --output output/dstext --opts MODEL.WEIGHTS trained_models/DSText/xxx.pth

cd output/dstext/preds
zip -r ../track.zip ./*.xml
zip -r ../preds.zip ./*.*

### return to the main directory 
python tools/Evaluation_Protocol_DSText/Evaluation_DSText_tracking/Track_video_2_0.py  # for tracking
python tools/Evaluation_Protocol_DSText/Evaluation_DSText_E2E/E2E_video_2_0.py  # for spotting
```



**BOVText**  

```python
python eval.py --config-file configs/GoMatching_BOVText.yaml --input ./datasets/BOVText/frame_test/ --output output/bovtext --opts MODEL.WEIGHTS trained_models/GoM_BOVText/xxx.pth ### for GoMatching

python eval.py --config-file configs/GoMatching_PP_BOVText.yaml --input ./datasets/BOVText/frame_test/ --output output/bovtext --opts MODEL.WEIGHTS trained_models/GoMPP_BOVText/xxx.pth ### for GoMatching++

### evaluation 
python tools/Evaluation_Protocol_BOV_Text/Task1_VideoTextTracking/evaluation.py --groundtruths ./datasets/BOVText/Test/test_annotaion/ --tests output/bovtext/jsons/ # for tracking


# 2. eval spotting
python tools/Evaluation_Protocol_BOV_Text/Task2_VideoTextSpotting/evaluation.py --groundtruths ./datasets/BOVText/Test/test_annotaion/ --tests output/bovtext/jsons/ # for spotting

```



**ArTVideo**  The standard of evaluation is consistent with [BOVText](https://github.com/weijiawu/BOVText-Benchmark).

```python
python eval.py --config-file configs/GoMatching_ArTVideo.yaml --input ./datasets/ArTVideo/Test/frame/ --output output/artvideo --opts MODEL.WEIGHTS trained_models/GoM_ArTVideo/xxx.pth ### for GoMatching

python eval.py --config-file configs/GoMatching_PP_ArTVideo.yaml --input ./datasets/ArTVideo/Test/frame/ --output output/artvideo --opts MODEL.WEIGHTS trained_models/GoMPP_ArTVideo/xxx.pth ### for GoMatching++

### evaluation
# 1. eval tracking on straight and curve text
python tools/Evaluation_Protocol_ArtVideo/eval_trk.py --groundtruths ./datasets/ArTVideo/json/ --tests output/artvideo/jsons/

# 2. eval tracking on curve text only
python tools/Evaluation_Protocol_ArtVideo/eval_trk.py --groundtruths ./datasets/ArTVideo/json/ --tests output/artvideo/jsons/ --curve

# 3. eval spotting on straight and curve text
python tools/Evaluation_Protocol_ArtVideo/eval_e2e.py --groundtruths ./datasets/ArTVideo/json/ --tests output/artvideo/jsons/

# 4. eval spotting on curve text only
python tools/Evaluation_Protocol_ArtVideo/eval_e2e.py --groundtruths ./datasets/ArTVideo/json/ --tests output/artvideo/jsons/ --curve
```

**Note:** If you want to visualize the results, you can add `--show` argument as follow:

```python
python eval.py --config-file configs/GoMatching_ICDAR15.yaml --input ./datasets/ICDAR15/frame_test/ --output output/icdar15 --show --opts MODEL.WEIGHTS trained_models/ICDAR15/xxx.pth
```



# Main Results
**[ICDAR15-video Video Text Spotting challenge](https://rrc.cvc.uab.es/?ch=3&com=evaluation&task=4)**

|    Method    | MOTA  | MOTP  | IDF1  | Trainable Parameters(M) |                            Weight                            |
| :----------: | :---: | :---: | :---: | :---------------------: | :----------------------------------------------------------: |
|  GoMatching  | 72.04 | 78.53 | 80.11 |          32.79          | [GoogleDrive](https://drive.google.com/file/d/1B5fM9rd1I7nW8J4Ef_S1_ZpH58ohiP9c/view?usp=drive_link) |
| GoMatching++ | 72.20 | 78.52 | 80.11 |          11.80          | [GoogleDrive](https://drive.google.com/file/d/1lZ3JXbZxK-jYAg8RPPTTq0FzoNewQMwx/view?usp=drive_link) |



**[DSText Video Text Spotting challenge](https://rrc.cvc.uab.es/?ch=22&com=evaluation&task=2)**

|    Method    | MOTA  | MOTP  | IDF1  | Trainable Parameters(M) |                            Weight                            |
| :----------: | :---: | :---: | :---: | :---------------------: | :----------------------------------------------------------: |
|  GoMatching  | 22.83 | 80.43 | 46.06 |          32.79          | [GoogleDrive](https://drive.google.com/file/d/1U2BtwKipT4wY50W1-8tqKWJo8dAmRS1b/view?usp=drive_link) |
| GoMatching++ | 23.23 | 80.24 | 46.24 |          11.80          | [GoogleDrive](https://drive.google.com/file/d/1OMeBOjxnrfoEGBL2TRe0brv42G540aG0/view?usp=drive_link) |



**BOVText Video Text Spotting**

|    Method    | MOTA | MOTP | IDF1 | Trainable Parameters(M) |                            Weight                            |
| :----------: | :--: | :--: | :--: | :---------------------: | :----------------------------------------------------------: |
| GoMatching++ | 52.9 | 87.2 | 62.8 |          11.80          | [GoogleDrive](https://drive.google.com/file/d/1bFXs9rb2AJhghbH4v22BrhwpZwdrs3OO/view?usp=drive_link) |



**ArTVideo Video Text Spotting**

|    Method    | MOTA | MOTP | IDF1 | Trainable Parameters(M) |                            Weight                            |
| :----------: | :--: | :--: | :--: | :---------------------: | :----------------------------------------------------------: |
| GoMatching++ | 75.7 | 83.5 | 82.3 |          11.80          | [GoogleDrive](https://drive.google.com/file/d/16BHRf5UWIzBjx-EKAxmD6to6TTyc1_gl/view?usp=drive_link) |



# Statement

This project is for research purpose only. For any other questions please contact [haibinhe@whu.edu.cn](mailto:haibinhe@whu.edu.cn).



## Citation

If you find GoMatching helpful, please consider giving this repo a star and citing:

```bibtex
@article{he2024gomatching,
  title={GoMatching: A Simple Baseline for Video Text Spotting via Long and Short Term Matching},
  author={He, Haibin and Ye, Maoyuan and Zhang, Jing and Liu, Juhua and Tao, Dacheng},
  journal={arXiv preprint arXiv:2401.07080},
  year={2024}
}

@inproceedings{ye2023deepsolo,
  title={DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting},
  author={Ye, Maoyuan and Zhang, Jing and Zhao, Shanshan and Liu, Juhua and Liu, Tongliang and Du, Bo and Tao, Dacheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19348--19357},
  year={2023}
}
```
# Acknowledgements

------

This project is based on [DeepSolo](https://github.com/ViTAE-Transformer/DeepSolo), [GTR](https://github.com/xingyizhou/GTR), [TransDETR](https://github.com/weijiawu/TransDETR) and [BOVText](https://github.com/weijiawu/BOVText-Benchmark).

