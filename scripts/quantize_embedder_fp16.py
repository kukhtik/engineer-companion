"""
Convert model.onnx (FP32) to FP16 using onnxconverter-common.
FP16 halves weight storage with ~1e-4 absolute error — effectively lossless for cosine similarity.
ORT on Android ARM uses FP16 natively.
"""
import pathlib, sys

ROOT = pathlib.Path(r"E:\engineer-companion")
ONNX_DIR = ROOT / "android" / "app" / "src" / "main" / "assets" / "models" / "embedder_onnx"
SRC = ONNX_DIR / "model.onnx"
DST = ONNX_DIR / "model_fp16.onnx"

import onnx

# Try onnxconverter_common first
try:
    from onnxconverter_common import float16
    model = onnx.load(str(SRC))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)
    onnx.save(model_fp16, str(DST))
    method = "onnxconverter_common"
except ImportError:
    # Fallback: pure onnx approach with numpy float16 conversion
    import numpy as np
    from onnx import numpy_helper, TensorProto

    model = onnx.load(str(SRC))
    for tensor in model.graph.initializer:
        if tensor.data_type == TensorProto.FLOAT:
            # Convert float32 initializer to float16
            np_array = numpy_helper.to_array(tensor).astype(np.float16)
            new_tensor = numpy_helper.from_array(np_array, name=tensor.name)
            tensor.CopyFrom(new_tensor)
    # Update graph inputs/outputs to fp16
    for node_input in model.graph.input:
        if node_input.type.tensor_type.elem_type == TensorProto.FLOAT:
            node_input.type.tensor_type.elem_type = TensorProto.FLOAT16
    for node_output in model.graph.output:
        if node_output.type.tensor_type.elem_type == TensorProto.FLOAT:
            node_output.type.tensor_type.elem_type = TensorProto.FLOAT16
    onnx.save(model, str(DST))
    method = "manual_numpy"

src_mb = SRC.stat().st_size / 1e6
dst_mb = DST.stat().st_size / 1e6
print(f"Method: {method}")
print(f"FP32: {src_mb:.1f} MB")
print(f"FP16: {dst_mb:.1f} MB")
print(f"Reduction: {(1 - dst_mb/src_mb)*100:.1f}%")
