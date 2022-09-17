from typing import Optional, List

import numpy as np
import torch
from traitlets import Int

from detectron2.structures import Boxes, BoxMode
from .densepose_outputs_vertex import get_xyz_vertex_embedding
from .extractor import DensePoseOutputsExtractor
from ..data.utils import get_class_to_mesh_name_mapping
from ..modeling import build_densepose_embedder
from ..modeling.cse.utils import get_closest_vertices_mask_from_ES
from ..structures import DensePoseChartResult, DensePoseChartResultWithConfidences, DensePoseChartPredictorOutput, \
    DensePoseEmbeddingPredictorOutput
from .detection_rects_config_parser import DetectionRectsConfig

from scipy.ndimage import measurements

from .dilate_erode import Dilation2d, Erosion2d

class DetectionRectsWranglerIUV:

    def __init__(self, rects_config: DetectionRectsConfig):
        self.rects_config = rects_config

    def wrangle(self, outputs):

        prediction_indices = torch.nonzero(outputs.scores >= self.rects_config.score_threshold)
        skipped = torch.nonzero(outputs.scores < self.rects_config.score_threshold)
        if len(skipped) > 0:
            print("skipping", len(skipped), "predictions for too low scores")

        # outputs has coarse_segm = (1, 2, 112, 112)
        # course_segm(0, 0, :, :) are all samples outside the silhouette
        # course_segm(0, 1, :, :) are all samples inside the silhouette
        # -> course_segm.argmax(dim=1) gives in/out labels

        # outputs also has fine_segm = (1, 25, 112, 112)
        # each of dim=1 represents the different classes, eg 1 is torso back, 2 is torso front
        # -> course_segm.argmax(dim=1) gives class labels
        # but also:
        # -> course_segm[:, 2, :, :] gives probabilities that this pixel is in class 2

        image_size = outputs.image_size
        rects = []

        for prediction_index in prediction_indices:
            this_predictor_output: DensePoseChartPredictorOutput = outputs.pred_densepose[prediction_index]
            pred_bbox_yxyx = outputs.pred_boxes[prediction_index].tensor[0].tolist()
            pred_bbox_yxhw = BoxMode.convert(pred_bbox_yxyx, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            coarse_segm = this_predictor_output.coarse_segm
            fine_segm = this_predictor_output.fine_segm
            labels = (
                    fine_segm.argmax(dim=1)
                    * (coarse_segm.argmax(dim=1) > 0).long()
            )
            u = this_predictor_output.u
            v = this_predictor_output.v

            for rect_def in self.rects_config.rect_definitions:
                rect_name = rect_def["name"]
                label_index = rect_def["label"]
                min_u = rect_def["uMin"]/255
                max_u = rect_def["uMax"]/255
                min_v = rect_def["vMin"]/255
                max_v = rect_def["vMax"]/255
                mask_label = np.zeros(labels.shape, dtype=np.uint8)
                mask_label[labels == label_index] = 1

                this_u = u[0, label_index, :, :]
                mask_ulow = np.zeros(this_u.shape, dtype=np.uint8)
                mask_uhigh = np.zeros(this_u.shape, dtype=np.uint8)
                mask_ulow[this_u >= min_u] = 1
                mask_uhigh[this_u <= max_u] = 1
                mask_u = mask_ulow * mask_uhigh

                this_v = v[0, label_index, :, :]
                mask_vlow = np.zeros(this_v.shape, dtype=np.uint8)
                mask_vhigh = np.zeros(this_v.shape, dtype=np.uint8)
                mask_vlow[this_v >= min_v] = 1
                mask_vhigh[this_v <= max_v] = 1
                mask_v = mask_vlow * mask_vhigh
                mask = mask_label * mask_u * mask_v

                this_rects = find_rects_in_mask(pred_bbox_yxhw, mask)
                for r in this_rects:
                    rects.append({
                        'label': rect_name,
                        'bbox_xyxy': r
                    })



        #print("all done",rects)
        return rects

        #pred_boxes = torch.index_select(outputs.pred_d, 0, prediction_indices)
        #pred_denseposes = outputs.index_select(0, prediction_indices)

        #print("image size", image_size, "with predictions at", pred_boxes)


class DetectionRectsWranglerCSE:

    def __init__(self, cfg, rects_config: DetectionRectsConfig, ):
        self.rects_config = rects_config
        self.embedder = build_densepose_embedder(cfg)
        default_class = 0
        self.default_class = default_class

        self.class_to_mesh_name = get_class_to_mesh_name_mapping(cfg)

        self.mesh_vertex_embeddings = {
            mesh_name: self.embedder(mesh_name)
            for mesh_name in self.class_to_mesh_name.values()
            if self.embedder.has_embeddings(mesh_name)
        }

    def wrangle(self, outputs):

        extractor = DensePoseOutputsExtractor()
        outputs, extracted_boxes_xywh, extracted_pred_classes = extractor(outputs)

        S, E, N, bboxes_xywh, pred_classes = self.extract_and_check_outputs_and_boxes(
            outputs, extracted_boxes_xywh, extracted_pred_classes
        )
        device = outputs.coarse_segm.device
        truncate_vertex_index = 6980

        rects = []
        for n in range(N):
            pred_bbox_yxhw = bboxes_xywh[n]
            x, y, w, h = pred_bbox_yxhw.int().tolist()
            if w == 0 or h == 0:
                continue
            mesh_name = self.class_to_mesh_name[pred_classes[n]]
            mesh_vertex_embeddings = self.mesh_vertex_embeddings[mesh_name]
            if truncate_vertex_index is not None:
                mesh_vertex_embeddings = mesh_vertex_embeddings[:truncate_vertex_index]
            closest_vertices, mask = get_closest_vertices_mask_from_ES(
                E[[n]],
                S[[n]],
                h,
                w,
                mesh_vertex_embeddings,
                device,
            )
            #mask_numpy = mask.cpu().numpy().astype(dtype=np.uint8)

            for group in self.rects_config.smpl_6980_vertex_groups:
                vertices = group['vertices']
                name = group['name']
                selected_vertices_mask = torch.isin(closest_vertices, torch.Tensor(vertices))
                this_mask = mask * selected_vertices_mask

                this_rects_xyxy_list = find_rects_in_mask(pred_bbox_yxhw, this_mask.unsqueeze(0).to(torch.float16))
                if len(this_rects_xyxy_list) > 0:
                    this_rects_xyxy = torch.Tensor(this_rects_xyxy_list)
                    w = this_rects_xyxy[:, 2] - this_rects_xyxy[:, 0]
                    h = this_rects_xyxy[:, 3] - this_rects_xyxy[:, 1]
                    area = w * h
                    largest = area.argmax()
                    rects.append({
                        'label': name,
                        'bbox_xyxy': this_rects_xyxy_list[largest]
                    })


        return rects


    def extract_and_check_outputs_and_boxes(
            self,
            densepose_output: Optional[DensePoseEmbeddingPredictorOutput],
            bboxes_xywh: Optional[Boxes],
            pred_classes: Optional[List[Int]]
    ):

        if pred_classes is None:
            pred_classes = [self.default_class] * len(bboxes_xywh)

        assert isinstance(
            densepose_output, DensePoseEmbeddingPredictorOutput
        ), "DensePoseEmbeddingPredictorOutput expected, {} encountered".format(
            type(densepose_output)
        )

        S = densepose_output.coarse_segm
        E = densepose_output.embedding
        N = S.size(0)
        assert N == E.size(
            0
        ), "CSE coarse_segm {} and embeddings {}" " should have equal first dim size".format(
            S.size(), E.size()
        )
        assert N == len(
            bboxes_xywh
        ), "number of bounding boxes {}" " should be equal to first dim size of outputs {}".format(
            len(bboxes_xywh), N
        )
        assert N == len(pred_classes), (
            "number of predicted classes {}"
            " should be equal to first dim size of outputs {}".format(len(bboxes_xywh), N)
        )

        return S, E, N, bboxes_xywh, pred_classes


def map_smpl_6980_to_smpl_27554(smpl_6980_vertices: torch.Tensor) -> torch.Tensor:
    # https://github.com/facebookresearch/detectron2/issues/3233#issuecomment-879145446
    # vkhalidov commented on 13 Jul 2021
    # Indeed, for DensePose we use an upsampled mesh that contains 27554 vertices. The first 6890 vertices on the
    # upsampled mesh correspond to the 6890 canonical SMPL vertices. The other ones can be mapped to the closest one
    # of those 6890 vertices by using the geodesic distances tensor (available [here]
    # (https://dl.fbaipublicfiles.com/densepose/meshes/geodists/geodists_smpl_27554.pkl) (6GB), see data/meshes/builtin.py).
    return torch.Tensor()


def find_rects_in_mask(pred_bbox_yxhw, mask) -> [list]:

    mask_tensor_padded = torch.unsqueeze(torch.Tensor(mask), dim=0)
    if mask_tensor_padded.size()[2] != mask_tensor_padded.size()[3]:
        size = max(mask_tensor_padded.size())
        mask_tensor_padded = torch.nn.functional.pad(mask_tensor_padded, (
                                                0, size-mask_tensor_padded.size()[3],
                                                0, size-mask_tensor_padded.size()[2]
        ))
    mask_dilated = Dilation2d(1, 1, 5, soft_max=False)(mask_tensor_padded)
    mask_dilated_eroded = Erosion2d(1, 1, 3, soft_max=False)(mask_dilated)

    def find_islands(src):
        # https://stackoverflow.com/questions/25664682/how-to-find-cluster-sizes-in-2d-numpy-array
        islands, num_islands = measurements.label(src[0])
        result = []
        for island_index in range(1, num_islands + 1):
            coords = np.where(islands == island_index)
            min_x = np.min(coords[1])  # x and y are swapped in coords
            min_y = np.min(coords[0])
            max_x = np.max(coords[1])
            max_y = np.max(coords[0])

            print("island", island_index, "pixel count", len(coords[0]), "bbox xyxy", str((min_x, min_y, max_x, max_y)))
            result.append([len(coords[0]), min_x, min_y, max_x, max_y])
        return result

    rects = []
    islands = find_islands(mask_dilated_eroded[0])
    if len(islands) > 0:
        # for island_index in range(len(islands)):
        biggest_island_index = np.argmax(islands, axis=0)[0]
        island_bbox_xyxy_relative_to_pred_bbox = islands[biggest_island_index][1:]
        pred_bbox_ox = pred_bbox_yxhw[0]
        pred_bbox_oy = pred_bbox_yxhw[1]
        pred_bbox_width = pred_bbox_yxhw[2]
        pred_bbox_height = pred_bbox_yxhw[3]
        # ignore padding by using mask rather than mask_tensor_padded
        left = pred_bbox_ox + pred_bbox_width * island_bbox_xyxy_relative_to_pred_bbox[0] / mask.shape[2]
        top = pred_bbox_oy + pred_bbox_height * island_bbox_xyxy_relative_to_pred_bbox[1] / mask.shape[1]
        right = pred_bbox_ox + pred_bbox_width * island_bbox_xyxy_relative_to_pred_bbox[2] / mask.shape[2]
        bottom = pred_bbox_oy + pred_bbox_height * island_bbox_xyxy_relative_to_pred_bbox[3] / mask.shape[1]

        island_bbox_xyxy = [round(left.item()), round(top.item()), round(right.item()), round(bottom.item())]
        rects.append(island_bbox_xyxy)

    return rects
