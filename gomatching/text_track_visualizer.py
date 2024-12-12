import cv2
import numpy as np
import torch
import colorsys
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer, random_color
from detectron2.utils.video_visualizer import _create_text_labels
from detectron2.utils.visualizer import ColorMode, Visualizer, VisImage
import pickle
import matplotlib.colors as mplc
import matplotlib.font_manager as mfm
from shapely.geometry import LineString
import matplotlib as mpl
from tqdm.contrib import tzip
import time


class TextTrackingVisualizer(VideoVisualizer):
    def __init__(self, metadata, cfg, instance_mode=ColorMode.IMAGE):
        """
        """
        self.metadata = metadata
        self._old_instances = []
        assert instance_mode in [
            ColorMode.IMAGE,
            ColorMode.IMAGE_BW,
        ], "Other mode not supported yet."
        self._instance_mode = instance_mode
        self._assigned_colors = {}
        self._max_num_instances = 10000
        self._num_colors = 500
        self._color_pool = [random_color(rgb=True, maximum=1) \
            for _ in range(self._num_colors)]
        self.color_idx = 0

        self.voc_size = cfg.MODEL.TRANSFORMER.VOC_SIZE
        self.use_customer_dictionary = cfg.MODEL.TRANSFORMER.CUSTOM_DICT
        if self.voc_size == 96:
            self.CTLABELS = [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1',
                             '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C',
                             'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                             'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
                             'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
                             'z', '{', '|', '}', '~']
        elif self.voc_size == 37:
            self.CTLABELS = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                             's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        else:
            with open(self.use_customer_dictionary, 'rb') as fp:
                self.CTLABELS = pickle.load(fp)
        # voc_size includes the unknown class, which is not in self.CTABLES
        assert (int(self.voc_size - 1) == len(
            self.CTLABELS)), "voc_size is not matched dictionary size, got {} and {}.".format(int(self.voc_size - 1),
                                                                                              len(self.CTLABELS))

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        polygon = mpl.patches.Polygon(
            segment,
            fill=True,
            facecolor=mplc.to_rgb(color) + (alpha,),
            edgecolor=edge_color,
            linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
        )
        self.output.ax.add_patch(polygon)
        return self.output

    def pre_vis_process(self, predictions):
        polys = []
        texts = []
        recs = predictions.recs
        bd_pts = predictions.bd.cpu().numpy()
        for bd, rec in zip(bd_pts, recs):
            bd = np.hsplit(bd, 2)
            bd = np.vstack([bd[0], bd[1][::-1]])
            text = self._ctc_decode_recognition(rec)
            polys.append(bd)
            texts.append(text)
        predictions.texts = texts
        predictions.polys = polys
        predictions.track_ids = predictions.track_ids.numpy()
        predictions.ctrl_points = predictions.ctrl_points.numpy()
        return predictions

    def draw_instance_predictions(self, frame, predictions):
        """
        """
        self.img = np.asarray(frame).clip(0, 255).astype(np.uint8)
        self.output = VisImage(self.img)
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 70, 10 // 1.0  # // 70 for ic15, dstext  50 for bovtext
        )
        num_instances = len(predictions)
        if num_instances == 0:
            return self.output

        self._assign_colors_by_track_id(predictions.track_ids)

        self.overlay_instances(
            bd_pts=predictions.polys,
            ctrl_pnts=predictions.ctrl_points,
            alpha=0.5,
            texts=predictions.texts,
            track_ids=predictions.track_ids,
            scores=predictions.scores
        )

        return self.output

    def overlay_instances(
        self,
        bd_pts=None,
        ctrl_pnts=None,
        alpha=0.5,
        texts=None,
        track_ids=None,
        scores=None,
    ):
        num_instances = 0
        if bd_pts is not None:
            num_instances = len(bd_pts)
        if num_instances == 0:
            return self.output

        for i in range(num_instances):
            color = self._assigned_colors[track_ids[i]]
            if bd_pts is not None:
                self.draw_polygon(bd_pts[i], color, alpha=alpha)
            horiz_align = "left"

            text = texts[i]
            if self.voc_size == 37:
                text = text.upper()
            score = scores[i]

            text = "({}){}".format(track_ids[i],text)

            line = self._process_ctrl_pnt(ctrl_pnts[i])
            line_ = LineString(line)
            center_point = np.array(line_.interpolate(0.5, normalized=True).coords[0], dtype=np.int32)
            lighter_color = self._change_color_brightness(color, brightness_factor=0)
            font_size = self._default_font_size
            text_pos = center_point
            ## draw text
            self.draw_text(
                text,
                text_pos,
                color=lighter_color,
                horizontal_alignment=horiz_align,
                font_size=font_size,
                draw_chinese=False if self.voc_size == 37 or self.voc_size == 96 else True
            )
        return self.output

    def _process_ctrl_pnt(self, pnt):
        points = pnt.reshape(-1, 2)
        return points

    def _ctc_decode_recognition(self, rec):
        last_char = '###'
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    if self.voc_size == 37 or self.voc_size == 96:
                        s += self.CTLABELS[c]
                        last_char = c
                    else:
                        s += str(chr(self.CTLABELS[c]))
                        last_char = c
            else:
                last_char = '###'
        return s

    def _change_color_brightness(self, color, brightness_factor):
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color

    def _assign_colors_by_track_id(self, track_ids):
        for text_id in track_ids:
            if text_id in self._assigned_colors:
                continue
            else:
                self.color_idx = (self.color_idx + 1) % self._num_colors
                color = self._color_pool[self.color_idx]
                self._assigned_colors[text_id] = color

    def draw_text(
            self,
            text,
            position,
            *,
            font_size=None,
            color="g",
            horizontal_alignment="center",
            rotation=0,
            draw_chinese=False
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW
        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        if draw_chinese:
            font_path = "./simsun.ttc"
            prop = mfm.FontProperties(fname=font_path)
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
                fontproperties=prop
            )
        else:
            self.output.ax.text(
                x,
                y,
                text,
                size=font_size * self.output.scale,
                family="sans-serif",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
                verticalalignment="top",
                horizontalalignment=horizontal_alignment,
                color=color,
                zorder=10,
                rotation=rotation,
            )
        return self.output

class GoMPredictor(DefaultPredictor):
    @torch.no_grad()
    def __call__(self, original_frames, return_time=False):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
        """
        if self.input_format == "RGB":
            original_frames = \
                [x[:, :, ::-1] for x in original_frames]
        height, width = original_frames[0].shape[:2]
        frames = [self.aug.get_transform(x).apply_image(x) \
            for x in original_frames]
        frames = [torch.as_tensor(x.astype("float32").transpose(2, 0, 1))\
            for x in frames]
        inputs = [{"image": x, "height": height, "width": width, "video_id": 0} \
            for x in frames]
        start_time = time.time()
        predictions = self.model(inputs)
        if return_time:
            return predictions, time.time() - start_time
        return predictions

class GoMBatchPredictor(DefaultPredictor):
    @torch.no_grad()
    def __call__(self, original_frames, instances, batch_id, id_count, last_batch, time_cost, return_time=False):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
        """

        # test_in = torch.rand([3, 1280, 1280], device='cuda')
        # flops, n_paras = profile(self.model, inputs=([test_in], 0, 0, []))
        # print(f"FLOPS: {flops / 1e9} G")
        # print(f"N_PARAS: {n_paras / 1e6} M")
        # trainable_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # total_parameters = sum(p.numel() for p in self.model.parameters())
        # print('trainble params:{} M, total params:{} M'.format(trainable_parameters / 1e6, total_parameters / 1e6))

        if self.input_format == "RGB":
            original_frames = \
                [x[:, :, ::-1] for x in original_frames]
        height, width = original_frames[0].shape[:2]
        frames = [self.aug.get_transform(x).apply_image(x) \
            for x in original_frames]
        frames = [torch.as_tensor(x.astype("float32").transpose(2, 0, 1))\
            for x in frames]
        inputs = [{"image": x, "height": height, "width": width, "video_id": 0} \
            for x in frames]
        start_time = time.time()
        instances, id_count = self.model.batch_inference(inputs, batch_id, id_count, instances, time_cost)
        if last_batch:
            start = time.time()
            if self.model.min_track_len > 0:
                instances = self.model._remove_short_track(instances)
            instances = self.model.batch_postprocess(instances, [(height, width) for _ in range((batch_id + 1) * 100 + len(inputs))]) # 100
            time_cost['post_process'] += time.time() - start
        if return_time:
            return instances, id_count, time.time() - start_time
        return instances, id_count


class TextVisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.video_predictor = GoMBatchPredictor(cfg)
        self.cfg = cfg


    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break


    def _process_predictions(self, tracker_visualizer, frame, predictions):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = predictions["instances"].to(self.cpu_device)
        predictions = tracker_visualizer.pre_vis_process(predictions)
        vis_frame = tracker_visualizer.draw_instance_predictions(
            frame, predictions)
        vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
        return vis_frame


    def run_on_video(self, video):
        """
        """
        tracker_visualizer = TextTrackingVisualizer(self.metadata, self.cfg, self.instance_mode)
        frames = [x for x in self._frame_from_video(video)]
        outputs = self.video_predictor(frames)
        print("generating video results...")
        for frame, instances in tzip(frames, outputs):
            yield self._process_predictions(tracker_visualizer, frame, instances)


    def run_on_images(self, frames):
        """
        """
        tracker_visualizer = TextTrackingVisualizer(self.metadata, self.cfg, self.instance_mode)
        outputs = self.video_predictor(frames)
        for frame, instances in zip(frames, outputs):
            yield self._process_predictions(tracker_visualizer, frame, instances)