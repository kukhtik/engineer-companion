import io, sys, pathlib, struct, json
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = pathlib.Path(r"E:\engineer-companion")
ONNX_DIR = ROOT / "android" / "app" / "src" / "main" / "assets" / "models" / "embedder_onnx"
TOK_DIR = ROOT / "assets" / "models" / "embedder"
INDEX_PATH = ROOT / "android" / "app" / "src" / "main" / "assets" / "db" / "chunks_index.bin"

TEST_STRINGS = [
    "что такое MLC?",
    "interlock reset",
    "absolute dose calibration",
    "The TrueBeam system uses a sophisticated interlocking mechanism to ensure patient safety during radiation therapy treatment delivery.",
    "beam on",
    "дозиметрическая калибровка",
]

import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(str(TOK_DIR))

def mean_pool(token_embs, attn_mask):
    mask = attn_mask[..., np.newaxis].astype(np.float32)
    summed = (token_embs * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    return summed / counts

def embed(sess, text):
    input_names = {inp.name for inp in sess.get_inputs()}
    enc = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
    inputs = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in input_names:
        inputs["token_type_ids"] = enc.get("token_type_ids", np.zeros_like(enc["input_ids"])).astype(np.int64)
    out = sess.run(None, inputs)
    # out[0] may be fp16 — cast to float32
    token_embs = out[0].astype(np.float32)
    pooled = mean_pool(token_embs, enc["attention_mask"].astype(np.float32))[0]
    normed = pooled / np.linalg.norm(pooled).clip(min=1e-12)
    return normed.astype(np.float32)

def load_index():
    with open(str(INDEX_PATH), "rb") as f:
        num_rows, vec_dim = struct.unpack("<ii", f.read(8))
        print(f"Index: {num_rows} rows, {vec_dim}-dim")
        embs = np.zeros((num_rows, vec_dim), dtype=np.float32)
        metas = []
        for i in range(num_rows):
            emb = np.frombuffer(f.read(vec_dim * 4), dtype=np.float32).copy()
            meta_len = struct.unpack("<i", f.read(4))[0]
            meta = json.loads(f.read(meta_len).decode("utf-8"))
            embs[i] = emb
            metas.append(meta)
    return embs, metas

def top_k(q, embs, k=8):
    scores = embs @ q
    idxs = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in idxs]

print("Loading FP32 model...")
fp32_sess = ort.InferenceSession(str(ONNX_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
fp32_embs_list = [embed(fp32_sess, t) for t in TEST_STRINGS]

fp16_path = ONNX_DIR / "model_fp16_fixed.onnx"
if not fp16_path.exists():
    print("model_fp16.onnx not found!")
    sys.exit(1)

print("\nLoading FP16 model...")
fp16_sess = ort.InferenceSession(str(fp16_path), providers=["CPUExecutionProvider"])

cosines = []
for i, t in enumerate(TEST_STRINGS):
    fp16_emb = embed(fp16_sess, t)
    cos = float(np.dot(fp32_embs_list[i], fp16_emb))
    cosines.append(cos)
    label = t[:62] + ".." if len(t) > 64 else t
    print(f"  {label:<64}  cosine={cos:.6f}")

mean_cos = float(np.mean(cosines))
print(f"\nMEAN COSINE (FP32 vs FP16): {mean_cos:.6f}")

print("\nLoading index for ranking check...")
index_embs, metas = load_index()

mlc_query = "что такое MLC?"
fp32_mlc = embed(fp32_sess, mlc_query)
fp16_mlc = embed(fp16_sess, mlc_query)

fp32_top8 = top_k(fp32_mlc, index_embs, 8)
fp16_top8 = top_k(fp16_mlc, index_embs, 8)

print("\nMLC RANKING COMPARISON (FP32 vs FP16):")
print(f"{'Rank':<5} {'FP32 source/page/text':<55} {'FP16 source/page/text':<55} Same?")
for r in range(8):
    fi, fs = fp32_top8[r]
    qi, qs = fp16_top8[r]
    fm = metas[fi]
    qm = metas[qi]
    fl = f"{fm.get('source','?')[:25]} p{fm.get('page','?')}"
    ql = f"{qm.get('source','?')[:25]} p{qm.get('page','?')}"
    same = "OK" if fi == qi else "  "
    print(f"{r+1:<5} {fl:<55} {ql:<55} {same}")

fp32_top5_set = set(i for i,_ in fp32_top8[:5])
fp16_top5_set = set(i for i,_ in fp16_top8[:5])
overlap = len(fp32_top5_set & fp16_top5_set)
rank1_ok = fp32_top8[0][0] == fp16_top8[0][0]

print(f"\nTop-5 set overlap: {overlap}/5")
print(f"Rank-1 exact match: {rank1_ok}")
print(f"Mean cosine: {mean_cos:.6f}")

passes = mean_cos >= 0.985 and overlap >= 4 and rank1_ok
print(f"\nVERDICT: {'PASS -- ADOPT FP16' if passes else 'FAIL -- reject FP16 too'}")
