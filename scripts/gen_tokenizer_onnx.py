"""
Generate tokenizer.onnx for XLM-RoBERTa SentencePiece tokenizer.

Uses SentencepieceTokenizer custom op from onnxruntime-extensions with fairseq=True
to match HF XLMRobertaTokenizer token IDs exactly.

The graph takes:
  input_text: STRING [1]   (single query string)
Outputs:
  input_ids:      INT64 [seq]   (token IDs matching HF exactly, with <s>=0 and </s>=2)
  attention_mask: INT64 [seq]   (all ones — no padding for single-string input)
  token_type_ids: INT64 [seq]   (all zeros, required by the ONNX embedding model)

Run from repo root:
    python scripts/gen_tokenizer_onnx.py
"""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
ONNX_OUT_DIR = ROOT / "android" / "app" / "src" / "main" / "assets" / "models" / "embedder_onnx"
SPM_MODEL = ONNX_OUT_DIR / "sentencepiece.bpe.model"
OUT_PATH = ONNX_OUT_DIR / "tokenizer.onnx"

print(f"SentencePiece model: {SPM_MODEL} ({SPM_MODEL.stat().st_size:,} bytes)")
print(f"Output path: {OUT_PATH}")

import onnx
from onnx import helper, TensorProto, numpy_helper
import numpy as np

spm_bytes = SPM_MODEL.read_bytes()

def sc(name, dtype, val):
    return numpy_helper.from_array(np.array(val, dtype=dtype), name=name)

# ── Node 1: SentencepieceTokenizer (fairseq=True for XLM-RoBERTa offset) ─────
# Inputs:  input_text (STRING), nbest_size (INT64), alpha (FLOAT),
#          add_bos (BOOL), add_eos (BOOL), reverse (BOOL), fairseq (BOOL)
# Outputs: tokens (INT32 flat), instance_indices (INT64), token_indices (INT32)
# For a single string input, 'tokens' == the full sequence of token IDs.
node_spm = helper.make_node(
    op_type="SentencepieceTokenizer",
    inputs=["input_text", "nbest_size", "alpha", "add_bos", "add_eos", "reverse", "fairseq"],
    outputs=["spm_tokens_i32", "instance_indices", "token_char_indices"],
    domain="ai.onnx.contrib",
    name="spm_tok",
    model=spm_bytes,
)

# ── Node 2: Cast tokens int32 → int64 (input_ids) ────────────────────────────
node_cast_ids = helper.make_node(
    "Cast", inputs=["spm_tokens_i32"], outputs=["input_ids"],
    to=TensorProto.INT64, name="cast_ids",
)

# ── Nodes 3-4: attention_mask = ones(shape_of(input_ids), int64) ─────────────
node_shape = helper.make_node(
    "Shape", inputs=["input_ids"], outputs=["ids_shape"], name="ids_shape",
)
node_ones = helper.make_node(
    "ConstantOfShape", inputs=["ids_shape"], outputs=["attention_mask"],
    name="ones_fill",
    value=numpy_helper.from_array(np.array([1], dtype=np.int64)),
)

# ── Nodes 5-6: token_type_ids = zeros(shape_of(input_ids), int64) ────────────
node_zeros = helper.make_node(
    "ConstantOfShape", inputs=["ids_shape"], outputs=["token_type_ids"],
    name="zeros_fill",
    value=numpy_helper.from_array(np.array([0], dtype=np.int64)),
)

# ── Initializers ──────────────────────────────────────────────────────────────
inits = [
    sc("nbest_size", np.int64,   0),      # greedy decode
    sc("alpha",      np.float32, 0.0),
    sc("add_bos",    np.bool_,   True),   # prepend <s> (id=0 for XLM-R with fairseq)
    sc("add_eos",    np.bool_,   True),   # append </s> (id=2)
    sc("reverse",    np.bool_,   False),
    sc("fairseq",    np.bool_,   True),   # CRITICAL: applies +1 offset for XLM-RoBERTa vocab
]

# ── Graph ─────────────────────────────────────────────────────────────────────
inp_text = helper.make_tensor_value_info("input_text",      TensorProto.STRING, [None])
out_ids  = helper.make_tensor_value_info("input_ids",       TensorProto.INT64,  [None])
out_mask = helper.make_tensor_value_info("attention_mask",  TensorProto.INT64,  [None])
out_tti  = helper.make_tensor_value_info("token_type_ids",  TensorProto.INT64,  [None])
# intermediate (instance_indices, token_char_indices not exposed as graph outputs)

graph = helper.make_graph(
    nodes=[node_spm, node_cast_ids, node_shape, node_ones, node_zeros],
    name="xlmr_spm_tokenizer",
    inputs=[inp_text],
    outputs=[out_ids, out_mask, out_tti],
    initializer=inits,
)

model_proto = helper.make_model(
    graph,
    opset_imports=[
        helper.make_opsetid("", 17),
        helper.make_opsetid("ai.onnx.contrib", 1),
    ],
)
model_proto.doc_string = (
    "XLM-RoBERTa SentencePiece tokenizer (fairseq=True). "
    "INPUT: 'input_text' STRING[1] — single query string, no prefix. "
    "OUTPUTS: 'input_ids' INT64[seq], 'attention_mask' INT64[seq] (ones), "
    "'token_type_ids' INT64[seq] (zeros). "
    "BOS=0 (<s>), EOS=2 (</s>). No padding (single query at a time). "
    "Kotlin: reshape input_ids to [1, seq] before passing to model.onnx."
)

onnx.save(model_proto, str(OUT_PATH))
print(f"Saved: {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")

# ── Smoke test ────────────────────────────────────────────────────────────────
print("\nSmoke-testing tokenizer.onnx with ORT + extensions ...")
import onnxruntime as ort
import onnxruntime_extensions as ortx
from transformers import AutoTokenizer

so = ort.SessionOptions()
so.register_custom_ops_library(ortx.get_library_path())
sess = ort.InferenceSession(str(OUT_PATH), so, providers=["CPUExecutionProvider"])

print(f"Session inputs:  {[i.name for i in sess.get_inputs()]}")
print(f"Session outputs: {[o.name for o in sess.get_outputs()]}")

hf_tok = AutoTokenizer.from_pretrained(str(ROOT / "assets" / "models" / "embedder"))
test_texts = ["interlock reset", "beam on", "что такое MLC?", "absolute dose calibration"]

print("\nParity vs HF tokenizer:")
all_ok = True
for text in test_texts:
    res = sess.run(None, {"input_text": np.array([text], dtype=object)})
    ids_onnx, mask, tti = res
    hf_ids = hf_tok(text)["input_ids"]
    match = ids_onnx.tolist() == hf_ids
    if not match:
        all_ok = False
    status = "MATCH" if match else "DIFF"
    print(f"  {repr(text[:35]):40s} {status}  onnx={ids_onnx[:5].tolist()}  hf={hf_ids[:5]}")
    assert all(t == 0 for t in tti.tolist()), "token_type_ids must be zeros"
    assert all(m == 1 for m in mask.tolist()), "attention_mask must be ones"

print()
if all_ok:
    print("ALL PARITY CHECKS PASSED")
else:
    print("PARITY FAILED — check fairseq flag")
    sys.exit(1)

print(f"""
Tokenizer.onnx I/O contract for Kotlin OnnxRetriever:
  Input:  'input_text'      STRING  shape [1]    (single query, no prefix)
  Output: 'input_ids'       INT64   shape [seq]  (includes BOS=0, EOS=2)
  Output: 'attention_mask'  INT64   shape [seq]  (all ones, no padding)
  Output: 'token_type_ids'  INT64   shape [seq]  (all zeros)
  Note:   reshape outputs to [1, seq] before feeding model.onnx
""")
