"""
Build the flat binary vector index for the Android OnnxRetriever.

Reads all chunks from assets/db/engineer.db (LanceDB 'chunks' table) and writes
android/app/src/main/assets/db/chunks_index.bin.

Binary format (little-endian throughout):
  Header:
    uint32  num_rows   — total number of chunk records
    uint32  vec_dim    — embedding dimension (384)
  Per row (repeated num_rows times):
    float32[vec_dim]   — the embedding vector (384 x 4 = 1536 bytes)
    uint32  meta_len   — byte length of the UTF-8 JSON metadata blob
    uint8[meta_len]    — UTF-8 JSON: {"source":…,"page":…,"section":…,"text":…}

The Kotlin OnnxRetriever reads this file sequentially; each record is at a fixed
stride of (vec_dim*4 + 4 + meta_len) bytes, so seek is via a pre-scan index built
at load time (or stream from start since the file fits in ~21 MB RAM).

Usage:
    python scripts/build_android_index.py

Outputs:
    android/app/src/main/assets/db/chunks_index.bin
"""
import io
import json
import pathlib
import struct
import sys

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parent.parent
DB_PATH   = ROOT / "assets" / "db" / "engineer.db"
OUT_PATH  = ROOT / "android" / "app" / "src" / "main" / "assets" / "db" / "chunks_index.bin"
VEC_DIM   = 384

print(f"Opening LanceDB: {DB_PATH}")
import lancedb
db    = lancedb.connect(str(DB_PATH))
table = db.open_table("chunks")
total = table.count_rows()
print(f"  rows: {total}")

print(f"Loading all rows …")
df = table.to_pandas()
print(f"  columns: {list(df.columns)}")
print(f"  loaded {len(df)} rows")

# Ensure vector column is list-of-floats or numpy array
import numpy as np
vec_col = "vector"
if vec_col not in df.columns:
    raise RuntimeError(f"No 'vector' column; got: {list(df.columns)}")

print(f"Writing {OUT_PATH} …")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

rows_written = 0
with open(OUT_PATH, "wb") as f:
    # Write header (placeholder num_rows — we'll overwrite at end)
    f.write(struct.pack("<II", 0, VEC_DIM))

    for _, row in df.iterrows():
        vec = np.asarray(row[vec_col], dtype=np.float32)
        if vec.shape != (VEC_DIM,):
            raise ValueError(f"Expected vector dim {VEC_DIM}, got {vec.shape} at row {rows_written}")

        # Write vector
        f.write(vec.tobytes())  # 384 * 4 = 1536 bytes, little-endian float32

        # Write metadata JSON
        meta = {
            "source":  str(row.get("source", "")),
            "page":    int(row.get("page", 0)) if row.get("page") is not None else 0,
            "section": str(row.get("section", "")),
            "text":    str(row.get("text", "")),
        }
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")
        f.write(struct.pack("<I", len(meta_bytes)))
        f.write(meta_bytes)

        rows_written += 1
        if rows_written % 1000 == 0:
            print(f"  … {rows_written}/{total}")

    # Patch header with actual row count
    f.seek(0)
    f.write(struct.pack("<II", rows_written, VEC_DIM))

size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
print(f"\nDone: {rows_written} rows written, {size_mb:.1f} MB -> {OUT_PATH}")

# ── Verify: re-read header ────────────────────────────────────────────────────
print("\nVerifying header …")
with open(OUT_PATH, "rb") as f:
    num_rows_hdr, dim_hdr = struct.unpack("<II", f.read(8))
print(f"  header: num_rows={num_rows_hdr}, vec_dim={dim_hdr}")
assert num_rows_hdr == rows_written, "Header mismatch!"
assert dim_hdr == VEC_DIM, "Dim mismatch!"
print("  header OK")

# ── Sanity: top-5 for test query via ONNX embedder ───────────────────────────
print("\nRunning sanity search: 'что такое MLC?' via ONNX …")

import onnxruntime as ort
from transformers import AutoTokenizer

ONNX_DIR = ROOT / "android" / "app" / "src" / "main" / "assets" / "models" / "embedder_onnx"
TOKENIZER_DIR = ROOT / "assets" / "models" / "embedder"  # use original tokenizer

sess = ort.InferenceSession(str(ONNX_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
input_names = {inp.name for inp in sess.get_inputs()}

QUERY = "что такое MLC?"
enc = tokenizer(QUERY, return_tensors="np", padding=True, truncation=True, max_length=512)
inputs = {
    "input_ids":      enc["input_ids"].astype(np.int64),
    "attention_mask": enc["attention_mask"].astype(np.int64),
}
if "token_type_ids" in input_names:
    # XLM-RoBERTa doesn't return token_type_ids; supply zeros (all same segment)
    inputs["token_type_ids"] = np.zeros_like(enc["input_ids"], dtype=np.int64)

outputs = sess.run(None, inputs)
token_embs   = outputs[0]  # (1, seq, 384)
mask         = enc["attention_mask"][..., np.newaxis].astype(np.float32)
pooled       = (token_embs * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)
query_vec    = pooled[0]
query_vec   /= np.linalg.norm(query_vec)

# Load all vectors from the bin file into a matrix
print("Loading vectors from bin for cosine search …")
all_vecs = np.stack([np.asarray(row[vec_col], dtype=np.float32) for _, row in df.iterrows()])
# already normalized? check norm of first row
norms = np.linalg.norm(all_vecs, axis=1)
print(f"  Vector norm range: {norms.min():.4f} – {norms.max():.4f} (should be ~1.0 if normalized)")

scores = all_vecs @ query_vec  # cosine because both are unit-normed
top5_idx = np.argsort(scores)[::-1][:5]

print(f"\nTop-5 results for '{QUERY}':")
print(f"  {'score':>6}  {'source':<50}  page  section[:40]")
print("  " + "-" * 100)
for rank, idx in enumerate(top5_idx, 1):
    row = df.iloc[idx]
    score = scores[idx]
    src   = str(row.get("source", ""))[-48:]
    page  = row.get("page", "?")
    sect  = str(row.get("section", ""))[:40]
    print(f"  {score:6.4f}  {src:<50}  {page:<5} {sect}")

print("\nBuild complete.")
