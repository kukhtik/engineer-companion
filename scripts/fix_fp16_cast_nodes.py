"""
Fix Cast nodes in the FP16-converted model:
onnxconverter_common converts initializers to fp16 but leaves Cast-to-FLOAT (to=1)
nodes pointing at FLOAT32. Since surrounding ops are now fp16, ORT sees a type mismatch.
Solution: rewrite Cast-to-FLOAT to Cast-to-FLOAT16 (to=10).
"""
import pathlib
import onnx
from onnx import TensorProto

ONNX_DIR = pathlib.Path(r"E:\engineer-companion\android\app\src\main\assets\models\embedder_onnx")
SRC = ONNX_DIR / "model_fp16.onnx"
DST = ONNX_DIR / "model_fp16_fixed.onnx"

model = onnx.load(str(SRC))

patched = 0
for node in model.graph.node:
    if node.op_type == "Cast":
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.FLOAT:
                attr.i = TensorProto.FLOAT16
                patched += 1

print(f"Patched {patched} Cast nodes (FLOAT -> FLOAT16)")

# Also fix value_info type annotations for Cast outputs
fixed_vi = 0
for vi in list(model.graph.value_info):
    if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
        vi.type.tensor_type.elem_type = TensorProto.FLOAT16
        fixed_vi += 1
print(f"Fixed {fixed_vi} value_info type annotations")

# Fix graph outputs
fixed_out = 0
for out in model.graph.output:
    if out.type.tensor_type.elem_type == TensorProto.FLOAT:
        out.type.tensor_type.elem_type = TensorProto.FLOAT16
        fixed_out += 1
print(f"Fixed {fixed_out} graph output type annotations")

onnx.save(model, str(DST))
size_mb = DST.stat().st_size / 1e6
print(f"Saved to {DST} ({size_mb:.1f} MB)")
