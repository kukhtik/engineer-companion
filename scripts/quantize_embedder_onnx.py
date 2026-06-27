"""
Convert the FP32 ONNX embedder (model.onnx) to a corrected FP16 model.

This replaces the old INT8 dynamic-quantization script, which produced models
with ORT type-mismatch errors on Android. FP16 halves weight storage with
~1e-4 absolute error — effectively lossless for cosine-similarity retrieval.
ORT on Android ARM can run FP16 natively.

Two-step process baked into one script:
  Step 1: FP32 → FP16 via onnxconverter-common (or manual numpy fallback).
  Step 2: Patch Cast nodes — onnxconverter_common leaves Cast-to-FLOAT (to=1)
          intact even though surrounding ops are now FP16, causing ORT type
          errors on Android. We rewrite them to Cast-to-FLOAT16 (to=10) and
          fix value_info / graph-output type annotations.

Output: model_fp16_fixed.onnx (~224 MB)

Usage:
    python scripts/quantize_embedder_onnx.py

To adopt the result as the active model:
    Copy-Item .../embedder_onnx/model_fp16_fixed.onnx .../embedder_onnx/model.onnx -Force
"""

from __future__ import annotations

import pathlib
import sys

ONNX_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "android"
    / "app"
    / "src"
    / "main"
    / "assets"
    / "models"
    / "embedder_onnx"
)

SRC_FP32 = ONNX_DIR / "model.onnx"
INTERMEDIATE = ONNX_DIR / "model_fp16.onnx"
DST = ONNX_DIR / "model_fp16_fixed.onnx"


# ---------------------------------------------------------------------------
# Step 1 — FP32 → FP16
# ---------------------------------------------------------------------------

def convert_to_fp16() -> None:
    import onnx

    if not SRC_FP32.exists():
        print(f"[FAIL] Source model not found: {SRC_FP32}", file=sys.stderr)
        sys.exit(1)

    src_mb = SRC_FP32.stat().st_size / 1e6
    print(f"Source (FP32): {SRC_FP32}  ({src_mb:.1f} MB)")
    print("Step 1: Converting FP32 → FP16 ...")

    try:
        from onnxconverter_common import float16

        model = onnx.load(str(SRC_FP32))
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)
        onnx.save(model_fp16, str(INTERMEDIATE))
        method = "onnxconverter_common"
    except ImportError:
        print("  onnxconverter_common not found — falling back to manual numpy conversion.")
        import numpy as np
        from onnx import TensorProto, numpy_helper

        model = onnx.load(str(SRC_FP32))

        # Convert float32 initializers to float16
        for tensor in model.graph.initializer:
            if tensor.data_type == TensorProto.FLOAT:
                arr = numpy_helper.to_array(tensor).astype(np.float16)
                new_tensor = numpy_helper.from_array(arr, name=tensor.name)
                tensor.CopyFrom(new_tensor)

        # Update graph input types
        for node_input in model.graph.input:
            if node_input.type.tensor_type.elem_type == TensorProto.FLOAT:
                node_input.type.tensor_type.elem_type = TensorProto.FLOAT16

        # Update graph output types
        for node_output in model.graph.output:
            if node_output.type.tensor_type.elem_type == TensorProto.FLOAT:
                node_output.type.tensor_type.elem_type = TensorProto.FLOAT16

        onnx.save(model, str(INTERMEDIATE))
        method = "manual_numpy"

    inter_mb = INTERMEDIATE.stat().st_size / 1e6
    print(f"  Method: {method}")
    print(f"  Intermediate (FP16, unfixed): {inter_mb:.1f} MB  ({(1 - inter_mb/src_mb)*100:.1f}% reduction)")


# ---------------------------------------------------------------------------
# Step 2 — Fix Cast nodes and type annotations
# ---------------------------------------------------------------------------

def fix_cast_nodes() -> None:
    import onnx
    from onnx import TensorProto

    print("Step 2: Fixing Cast nodes and value_info type annotations ...")

    model = onnx.load(str(INTERMEDIATE))

    patched_cast = 0
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    attr.i = TensorProto.FLOAT16
                    patched_cast += 1

    fixed_vi = 0
    for vi in list(model.graph.value_info):
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16
            fixed_vi += 1

    fixed_out = 0
    for out in model.graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT:
            out.type.tensor_type.elem_type = TensorProto.FLOAT16
            fixed_out += 1

    onnx.save(model, str(DST))

    dst_mb = DST.stat().st_size / 1e6
    print(f"  Cast nodes patched (FLOAT→FLOAT16): {patched_cast}")
    print(f"  value_info type annotations fixed:  {fixed_vi}")
    print(f"  graph output type annotations fixed: {fixed_out}")
    print(f"  Output (FP16, fixed): {DST}  ({dst_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    src_mb = SRC_FP32.stat().st_size / 1e6 if SRC_FP32.exists() else 0

    convert_to_fp16()
    fix_cast_nodes()

    dst_mb = DST.stat().st_size / 1e6
    print()
    print(f"Done. FP32 {src_mb:.1f} MB  →  FP16 {dst_mb:.1f} MB  ({(1 - dst_mb/src_mb)*100:.1f}% reduction)")
    print()
    print("To adopt as the active model, run:")
    print(f'  Copy-Item "{DST}" "{SRC_FP32}" -Force')
    print()
    print("Or from the repo root in PowerShell:")
    dir_var = str(ONNX_DIR)
    print(f'  $d = "{dir_var}"')
    print(r'  Copy-Item "$d\model_fp16_fixed.onnx" "$d\model.onnx" -Force')

    # Clean up the unfixed intermediate to avoid confusion
    if INTERMEDIATE.exists():
        INTERMEDIATE.unlink()
        print(f"\n  (Removed intermediate {INTERMEDIATE.name})")


if __name__ == "__main__":
    main()
