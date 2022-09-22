#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import glob
import json
import logging
import math
import os
import pickle
import sys
from timeit import default_timer
from typing import Any, ClassVar, Dict, List

import cv2
import numpy as np
import torch

from torch.profiler import profile, record_function, ProfilerActivity

from detectron2.config import CfgNode, get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor, BatchPredictor
from detectron2.structures import BoxMode
from detectron2.structures.instances import Instances
from detectron2.utils.logger import setup_logger

from densepose import add_densepose_config
from densepose.structures import DensePoseChartPredictorOutput, DensePoseEmbeddingPredictorOutput
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer, BoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases, DensePoseOutputsVertexAnnotatedVisualizer,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,
)

from densepose.vis.detection_rects_wrangler import DetectionRectsWranglerCSE, DetectionRectsWranglerIUV
from densepose.vis.detection_rects_config_parser import DetectionRectsConfigParser

DOC = """Apply Net - a tool to print / visualize DensePose results
"""

LOGGER_NAME = "apply_net"
logger = logging.getLogger(LOGGER_NAME)

_ACTION_REGISTRY: Dict[str, "Action"] = {}


class Action(object):
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        parser.add_argument(
            "-v",
            "--verbosity",
            action="count",
            help="Verbose mode. Multiple -v options increase the verbosity.",
        )


def register_action(cls: type):
    """
    Decorator for action classes to automate action registration
    """
    global _ACTION_REGISTRY
    _ACTION_REGISTRY[cls.COMMAND] = cls
    return cls


class InferenceAction(Action):
    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(InferenceAction, cls).add_arguments(parser)
        parser.add_argument("cfg", metavar="<config>", help="Config file")
        parser.add_argument("model", metavar="<model>", help="Model file")
        parser.add_argument("input", metavar="<input>", help="Input data")
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=[],
            nargs=argparse.REMAINDER,
        )
        parser.add_argument(
            "--is_video",
            help="Input is a video",
            default=False,
            action="store_true"
        )


    @classmethod
    def execute(cls: type, args: argparse.Namespace):
        logger.info(f"Loading config from {args.cfg}")
        opts = []
        cfg = cls.setup_config(args.cfg, args.model, args, opts)
        logger.info(f"Loading model from {args.model}")
        predictor = BatchPredictor(cfg) if args.is_video else DefaultPredictor(cfg)
        logger.info(f"Loading data from {args.input}")
        file_list = cls._get_input_file_list(args.input)
        if len(file_list) == 0:
            logger.warning(f"No input images for {args.input}")
            return
        context = cls.create_context(args, cfg)
        if args.is_video:

            def trace_handler(p):
                trace_output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=30)
                print(trace_output)
                p.export_chrome_trace("trace_" + str(p.step_num) + ".json")

            #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #             schedule=torch.profiler.schedule(
            #                 wait=1,
            #                 warmup=1,
            #                 active=100),
            #             on_trace_ready=trace_handler) as prof:
            if True:
                video_path = file_list[0]
                video = cv2.VideoCapture(video_path)

                video.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                total_frames = video.get(cv2.CAP_PROP_POS_FRAMES)
                video.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)

                video_frame_rate = video.get(cv2.CAP_PROP_FPS)
                context["json_output_header"] = {
                    "fps": video_frame_rate,
                    "source": video_path,
                    "size": [
                        video.get(cv2.CAP_PROP_FRAME_WIDTH),
                        video.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    ]
                }
                duration = float(total_frames) / float(video_frame_rate)

                start_time = default_timer()
                first_frame_to_process = context.get("first_frame_to_process", 0)

                frame_index = 0
                target_process_fps = 8
                frame_interval = math.ceil(video_frame_rate / target_process_fps)

                print('duration %f s, fps %f, %i frames' % (duration, video_frame_rate, total_frames))
                print('frame increment %i -> processing at %f fps' % (frame_interval, video_frame_rate / float(frame_interval)))
                if first_frame_to_process > 0:
                    print("seeking to frame", first_frame_to_process)

                processed_frame_count = 0
                batched_imgs = []
                batch_size = 4
                while video.isOpened():
                    success, img = video.read()
                    frame_index += 1
                    if not success:
                        break
                    if (frame_index-1) < first_frame_to_process:
                        if (frame_index % 1000) == 0:
                            print("...", frame_index, end='\r')
                        continue
                    batched_imgs.append((frame_index, img))
                    #print("collected", len(batched_imgs), "frames")
                    if len(batched_imgs) >= batch_size:
                        with torch.no_grad():
                            print(" processing frame %i (%f%%)           " % (frame_index-1, float(100 * frame_index) / float(total_frames)), end='\r')
                            with record_function("inference"):
                                outputs_all = predictor([img for (frame_index, img) in batched_imgs])
                            with record_function("postprocess"):
                                for (input_idx, outputs) in enumerate(outputs_all):
                                    (this_frame_index, img) = batched_imgs[input_idx]
                                    cls.execute_on_outputs(context, {"file_name": video_path,
                                                                "image": img,
                                                                "frame_number": this_frame_index-1}, outputs["instances"])
                            processed_frame_count += len(batched_imgs)
                            #prof.step()
                            batched_imgs = []
                            # partial save, check context["is_finished"]
                            if (processed_frame_count % (32*batch_size)) == 0:
                                cls.postexecute(context)
                                current_time = default_timer()
                                total_time = (current_time - start_time)
                                process_fps = float(processed_frame_count) / total_time
                                frames_remaining = (total_frames - frame_index) / frame_interval
                                seconds_remaining = frames_remaining / process_fps
                                print(
                                    "processed %i frames in %f seconds (process %f fps, %fx playback rate), %im %is remaining" % (
                                    processed_frame_count + 1, total_time, process_fps,
                                    process_fps * float(frame_interval) / float(video_frame_rate),
                                    int(seconds_remaining / 60), int(seconds_remaining % 60)))

                    # step forward by frame_interval
                    for i in range(frame_interval-1):
                        _, frame = video.read()
                        frame_index += 1

        else:
            for file_name in file_list:
                try:
                    img = read_image(file_name, format="BGR")  # predictor expects BGR image.
                except Exception as e:
                    print(e)
                    continue
                with torch.no_grad():
                    outputs = predictor(img)["instances"]
                    cls.execute_on_outputs(context, {"file_name": file_name, "image": img}, outputs)

        context["is_finished"] = True
        cls.postexecute(context)

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        cfg = get_cfg()
        add_densepose_config(cfg)
        cfg.merge_from_file(config_fpath)
        cfg.merge_from_list(args.opts)
        if opts:
            cfg.merge_from_list(opts)
        cfg.MODEL.WEIGHTS = model_fpath
        cfg.freeze()
        return cfg

    @classmethod
    def _get_input_file_list(cls: type, input_spec: str):
        if os.path.isdir(input_spec):
            file_list = [
                os.path.join(input_spec, fname)
                for fname in os.listdir(input_spec)
                if os.path.isfile(os.path.join(input_spec, fname))
            ]
        elif os.path.isfile(input_spec):
            file_list = [input_spec]
        else:
            file_list = glob.glob(input_spec)
        return file_list


@register_action
class DumpAction(InferenceAction):
    """
    Dump action that outputs results to a pickle file
    """

    COMMAND: ClassVar[str] = "dump"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Dump model outputs to a file.")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(DumpAction, cls).add_arguments(parser)
        parser.add_argument(
            "--output",
            metavar="<dump_file>",
            default="results.pkl",
            help="File name to save dump to",
        )

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")
        result = {"file_name": image_fpath}
        if outputs.has("scores"):
            result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                    extractor = DensePoseResultExtractor()
                elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                    extractor = DensePoseOutputsExtractor()
                result["pred_densepose"] = extractor(outputs)[0]
        context["results"].append(result)

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode):
        context = {"results": [], "out_fname": args.output}
        return context

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        out_fname = context["out_fname"]
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_fname, "wb") as hFile:
            pickle.dump(context["results"], hFile)
            logger.info(f"Output saved to {out_fname}")

@register_action
class WriteRectsAction(InferenceAction):
    """
    Make detection rects and write to json
    """
    COMMAND: ClassVar[str] = "write_rects"

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Make detection rects from model outputs and write to json file.")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(WriteRectsAction, cls).add_arguments(parser)
        parser.add_argument(
            "detector_type",
            metavar="<detector type>",
            help="Detector type. Possible values are \"iuv\" (chart-based) or \"cse\"."
        )
        parser.add_argument(
            "--output",
            metavar="<result_file>",
            default="make.json",
            help="File name to save rects to",
        )
        parser.add_argument(
            "--min_score",
            metavar="<score>",
            default=0.8,
            type=float,
            help="Minimum detection score to visualize",
        )
        parser.add_argument(
            "--nms_thresh", metavar="<threshold>", default=None, type=float, help="NMS threshold"
        )
        parser.add_argument(
            "--rects_config",
            metavar="<detection_rects_config_file>",
            default="detectionRectsConfig.yml",
            help="YAML file that describes how to make detection rects from labels and U/V data"
        )

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
        opts.append(str(args.min_score))
        if args.nms_thresh is not None:
            opts.append("MODEL.ROI_HEADS.NMS_THRESH_TEST")
            opts.append(str(args.nms_thresh))
        cfg = super(WriteRectsAction, cls).setup_config(config_fpath, model_fpath, args, opts)
        return cfg

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        image_fpath = entry["file_name"]
        frame_id = ("frame_%05d" % entry["frame_number"])
        #logger.info(f"Processing {image_fpath} frame {frame_id}")
        result = {"frame": frame_id}
        #if outputs.has("scores"):
        #    result["scores"] = outputs.get("scores").cpu()
        if outputs.has("pred_boxes"):
            #result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
            if outputs.has("pred_densepose"):
                wrangler = context["wrangler"]
                # this level is 'detections' and top leve is too
                result["detections"] = wrangler.wrangle(outputs)
        context["results"].append(result)

        #if "detections" in result.keys():
        #    wrangler_results = result["detection_rects"]
        #    bbvis = BoundingBoxVisualizer()
        #    image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        #    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        #    bboxes_xywh = list(map(lambda x: BoxMode.convert(x["bbox_xyxy"], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS), wrangler_results))
        #    image_vis = bbvis.visualize(image, bboxes_xywh)
        #    entry_idx = context["entry_idx"] + 1
        #    out_fname = cls._get_out_fname(entry_idx, "/tmp/wrangler-test.png")
        #    out_dir = os.path.dirname(out_fname)
        #    if len(out_dir) > 0 and not os.path.exists(out_dir):
        #        os.makedirs(out_dir)
        #    cv2.imwrite(out_fname, image_vis)
        #    print("wrote to", out_fname)
        context["entry_idx"] += 1

    @classmethod
    def _get_out_fname(cls: type, entry_idx: int, fname_base: str):
        base, ext = os.path.splitext(fname_base)
        return base + ".{0:04d}".format(entry_idx) + ext


    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode):
        detection_rects_definitions = DetectionRectsConfigParser.parse(args.rects_config)
        wrangler = DetectionRectsWranglerIUV(detection_rects_definitions) if args.detector_type == "iuv" else DetectionRectsWranglerCSE(cfg, detection_rects_definitions)
        out_fname = args.output
        out_fname_partial = args.output+".part"
        context = {"results": [],
                   "out_fname": out_fname,
                   "out_fname_partial": out_fname_partial,
                   "wrangler": wrangler,
                   "entry_idx": 0,
                   "is_finished": False
                   }
        if args.is_video:
            if os.path.exists(out_fname):
                print("output file", out_fname, "already exists")
                exit(2)
            if os.path.exists(out_fname_partial):
                with open(out_fname_partial, "r") as hFile:
                    partial_json_output = json.load(hFile)
                    context["results"] = partial_json_output["detections"]
                    frame_ids = [int(x['frame'][6:]) for x in context["results"]]
                    last_frame = sorted(frame_ids)[-1]
                    context["first_frame_to_process"] = int(last_frame)
        return context

    @classmethod
    def prepare_for_json(cls: type, object: Dict) -> Dict:
        result = {}
        for k, v in object.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.tolist()
            elif isinstance(v, Dict):
                result[k] = cls.prepare_for_json(v)
            else:
                result[k] = v
        return result



    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        out_fname = context["out_fname"] if context["is_finished"] else context["out_fname_partial"]
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_fname, "w") as hFile:
            # copy header
            json_output = context["json_output_header"]
            # set results
            json_output["detections"] = context["results"]
            results_for_json = cls.prepare_for_json(json_output)
            json.dump(results_for_json, hFile)
            logger.info(f"Output saved to {out_fname}")

        if context["is_finished"]:
            os.unlink(context["out_fname_partial"])

@register_action
class ShowAction(InferenceAction):
    """
    Show action that visualizes selected entries on an image
    """

    COMMAND: ClassVar[str] = "show"
    VISUALIZERS: ClassVar[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
        "dp_cse_texture": DensePoseOutputsTextureVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer,
        "dp_vertex_SMPL6980_annotations": DensePoseOutputsVertexAnnotatedVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }

    @classmethod
    def add_parser(cls: type, subparsers: argparse._SubParsersAction):
        parser = subparsers.add_parser(cls.COMMAND, help="Visualize selected entries")
        cls.add_arguments(parser)
        parser.set_defaults(func=cls.execute)

    @classmethod
    def add_arguments(cls: type, parser: argparse.ArgumentParser):
        super(ShowAction, cls).add_arguments(parser)
        parser.add_argument(
            "visualizations",
            metavar="<visualizations>",
            help="Comma separated list of visualizations, possible values: "
            "[{}]".format(",".join(sorted(cls.VISUALIZERS.keys()))),
        )
        parser.add_argument(
            "--min_score",
            metavar="<score>",
            default=0.8,
            type=float,
            help="Minimum detection score to visualize",
        )
        parser.add_argument(
            "--nms_thresh", metavar="<threshold>", default=None, type=float, help="NMS threshold"
        )
        parser.add_argument(
            "--texture_atlas",
            metavar="<texture_atlas>",
            default=None,
            help="Texture atlas file (for IUV texture transfer)",
        )
        parser.add_argument(
            "--texture_atlases_map",
            metavar="<texture_atlases_map>",
            default=None,
            help="JSON string of a dict containing texture atlas files for each mesh",
        )
        parser.add_argument(
            "--output",
            metavar="<image_file>",
            default="outputres.png",
            help="File name to save output to",
        )

    @classmethod
    def setup_config(
        cls: type, config_fpath: str, model_fpath: str, args: argparse.Namespace, opts: List[str]
    ):
        opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
        opts.append(str(args.min_score))
        if args.nms_thresh is not None:
            opts.append("MODEL.ROI_HEADS.NMS_THRESH_TEST")
            opts.append(str(args.nms_thresh))
        cfg = super(ShowAction, cls).setup_config(config_fpath, model_fpath, args, opts)
        return cfg

    @classmethod
    def execute_on_outputs(
        cls: type, context: Dict[str, Any], entry: Dict[str, Any], outputs: Instances
    ):
        import cv2
        import numpy as np

        visualizer = context["visualizer"]
        extractor = context["extractor"]
        image_fpath = entry["file_name"]
        logger.info(f"Processing {image_fpath}")
        image = cv2.cvtColor(entry["image"], cv2.COLOR_BGR2GRAY)
        image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
        data = extractor(outputs)
        image_vis = visualizer.visualize(image, data)
        entry_idx = context["entry_idx"] + 1
        out_fname = cls._get_out_fname(entry_idx, context["out_fname"])
        out_dir = os.path.dirname(out_fname)
        if len(out_dir) > 0 and not os.path.exists(out_dir):
            os.makedirs(out_dir)
        cv2.imwrite(out_fname, image_vis)
        logger.info(f"Output saved to {out_fname}")
        context["entry_idx"] += 1

    @classmethod
    def postexecute(cls: type, context: Dict[str, Any]):
        pass

    @classmethod
    def _get_out_fname(cls: type, entry_idx: int, fname_base: str):
        base, ext = os.path.splitext(fname_base)
        return base + ".{0:04d}".format(entry_idx) + ext

    @classmethod
    def create_context(cls: type, args: argparse.Namespace, cfg: CfgNode) -> Dict[str, Any]:
        vis_specs = args.visualizations.split(",")
        visualizers = []
        extractors = []
        for vis_spec in vis_specs:
            texture_atlas = get_texture_atlas(args.texture_atlas)
            texture_atlases_dict = get_texture_atlases(args.texture_atlases_map)
            vis = cls.VISUALIZERS[vis_spec](
                cfg=cfg,
                texture_atlas=texture_atlas,
                texture_atlases_dict=texture_atlases_dict
            )
            visualizers.append(vis)
            extractor = create_extractor(vis)
            extractors.append(extractor)
        visualizer = CompoundVisualizer(visualizers)
        extractor = CompoundExtractor(extractors)
        context = {
            "extractor": extractor,
            "visualizer": visualizer,
            "out_fname": args.output,
            "entry_idx": 0,
        }
        return context


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DOC,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=120),
    )
    parser.set_defaults(func=lambda _: parser.print_help(sys.stdout))
    subparsers = parser.add_subparsers(title="Actions")
    for _, action in _ACTION_REGISTRY.items():
        action.add_parser(subparsers)
    return parser


def main():
    parser = create_argument_parser()
    args = parser.parse_args()
    verbosity = args.verbosity if hasattr(args, "verbosity") else None
    global logger
    logger = setup_logger(name=LOGGER_NAME)
    logger.setLevel(verbosity_to_level(verbosity))
    print("cuda available: ", torch.cuda.is_available())
    args.func(args)


if __name__ == "__main__":
    main()
