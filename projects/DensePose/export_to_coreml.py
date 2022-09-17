import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.export.flatten import TracingAdapter
from detectron2.modeling import build_model
from densepose import add_densepose_config

import coremltools as ct

def inference(model, image):
    inputs = [{"image": image}]
    output = model.inference(inputs, do_postprocess=False)[0]
    return output

model_fpath = "workspace/models/densepose_rcnn_R_50_FPN_s1x-CSE-c4ea5f.pkl"
config_fpath = "configs/cse/densepose_rcnn_R_50_FPN_s1x.yaml"

min_score = 0.8
opts = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", str(min_score)]

cfg = get_cfg()
add_densepose_config(cfg)
cfg.merge_from_file(config_fpath)
cfg.MODEL.WEIGHTS = model_fpath
cfg.merge_from_list(opts)
cfg.freeze()

print(f"Loading model from", model_fpath)
model = build_model(cfg).eval()
if len(cfg.DATASETS.TEST):
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)

example_input = torch.rand([3, 512, 512])
model.eval()

adapter = TracingAdapter(model, example_input, inference)
adapter.eval()
traced_model = torch.jit.trace(adapter, adapter.flattened_inputs)
traced_model.eval()

"""
::
outputs = model(inputs)  # inputs/outputs may be rich structure
adapter = TracingAdapter(model, inputs)

# can now trace the model, with adapter.flattened_inputs, or another
# tuple of tensors with the same length and meaning
traced = torch.jit.trace(adapter, adapter.flattened_inputs)

# traced model can only produce flattened outputs (tuple of tensors)
flattened_outputs = traced(*adapter.flattened_inputs)
# adapter knows the schema to convert it back (new_outputs == outputs)
new_outputs = adapter.outputs_schema(flattened_outputs)
"""


flattened_output = traced_model(*adapter.flattened_inputs)

print("model code:")
print(traced_model.code_with_constants)
print("-")
print("graph:")
print(traced_model.inlined_graph)


converted_model = ct.convert(
    traced_model,
    source="pytorch",
    convert_to="mlprogram",
    inputs=[ct.TensorType(name="image", shape=example_input.shape)]
 )
