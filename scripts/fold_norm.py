#!/usr/bin/env python3
"""Fold the /255 normalisation into the model graph so the input can be UINT8.

Every flow in this study ships [1,3,640,640] FP32 = 4.69 MB per frame, because
the client does the letterbox AND the divide-by-255 before handing the tensor
over. But the pixels started life as 8-bit and the divide is one multiply - so
we are paying 4x the wire cost to deliver data that only needs 8 bits.

Flipping --inputIOFormats=uint8:chw alone does NOT work:

    Error 3: /model.0/conv/Conv: only activation types allowed as input

The cast has to happen inside the graph. So splice a two-node preprocessing
head onto the front:

    images_u8 (uint8) -> Cast(to=FLOAT) -> Div(255.0) -> <original input name>

The original input name is reused as the Div output, so every downstream node
is untouched. TensorRT is then free to fold the scale into the first
convolution, which is the outcome we want: 4x less wire, no added GPU work.

Usage: python3 fold_norm.py <in.onnx> <out.onnx>
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SRC, DST = sys.argv[1], sys.argv[2]
m = onnx.load(SRC)
g = m.graph

old = g.input[0]
name = old.name
shape = [d.dim_value or 1 for d in old.type.tensor_type.shape.dim]
assert old.type.tensor_type.elem_type == TensorProto.FLOAT, "expected an fp32 input"

# New uint8 input. The original input NAME is reused for the Div output, so no
# downstream node needs rewiring.
u8 = helper.make_tensor_value_info(name + "_u8", TensorProto.UINT8, shape)
cast = helper.make_node("Cast", [u8.name], [name + "_f32"], to=TensorProto.FLOAT,
                        name="prep_cast")
scale = numpy_helper.from_array(np.array(255.0, dtype=np.float32), "prep_scale")
div = helper.make_node("Div", [name + "_f32", "prep_scale"], [name], name="prep_div")

g.input.remove(old)
g.input.insert(0, u8)
g.initializer.append(scale)
g.node.insert(0, div)
g.node.insert(0, cast)

onnx.checker.check_model(m)
onnx.save(m, DST)
print("%s -> %s" % (SRC.split("/")[-1], DST.split("/")[-1]))
print("  input  %s uint8 %s  (%.2f MB)"
      % (u8.name, "x".join(map(str, shape)), np.prod(shape) / 1e6))
print("  was    %s fp32  %s  (%.2f MB)"
      % (name, "x".join(map(str, shape)), np.prod(shape) * 4 / 1e6))
