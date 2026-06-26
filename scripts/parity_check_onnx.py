"""
Parity check: sentence-transformers (torch) vs ONNX export.

Runs as TWO subprocesses to avoid the torch+onnxruntime DLL collision on Windows.
Step 1 (this script with --torch): compute torch embeddings -> _parity_torch_embs.npy
Step 2 (this script with --onnx):  compute ONNX embeddings -> compare & report

Typical usage (from repo root):
    python scripts/parity_check_onnx.py

This wrapper launches both steps automatically.

PREPROCESSING SPEC for Kotlin OnnxRetriever (verified by this test):
  prefix:          NONE  (raw text — no 'query: ' / 'passage: ' prefix)
  tokenizer:       XLM-RoBERTa SentencePiece (sentencepiece.bpe.model)
  max_length:      512, padding=True, truncation=True
  token_type_ids:  required by ONNX graph; supply all-zeros (XLM-R tokenizer
                   does not produce them automatically)
  pooling:         mean over token hidden states, weighted by attention_mask
  normalize:       L2 to unit norm
  similarity:      cosine = dot product (both vectors unit-normed)
"""
import io
import subprocess
import sys
import pathlib

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT   = pathlib.Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".venv-win" / "Scripts" / "python.exe"
THIS   = pathlib.Path(__file__).resolve()

TEST_STRINGS = [
    "что такое MLC?",
    "interlock reset",
    "absolute dose calibration",
    "The TrueBeam system uses a sophisticated interlocking mechanism to ensure patient safety during radiation therapy treatment delivery.",
    "beam on",
]

if "--torch" in sys.argv:
    # ── Step 1: sentence-transformers embeddings ──────────────────────────────
    import numpy as np
    EMBEDDER_DIR = ROOT / "assets" / "models" / "embedder"
    OUT_FILE     = ROOT / "scripts" / "_parity_torch_embs.npy"

    print("Loading SentenceTransformer …")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(EMBEDDER_DIR), device="cpu")
    embeddings = []
    for text in TEST_STRINGS:
        emb = model.encode(text, normalize_embeddings=True, device="cpu").astype(np.float32)
        embeddings.append(emb)
        print(f"  encoded: {text[:60]}")
    np.save(str(OUT_FILE), np.stack(embeddings))
    print(f"Saved ({len(embeddings)}, 384) to {OUT_FILE}")

elif "--onnx" in sys.argv:
    # ── Step 2: ONNX embeddings + parity comparison ───────────────────────────
    import numpy as np
    ONNX_DIR      = ROOT / "android" / "app" / "src" / "main" / "assets" / "models" / "embedder_onnx"
    TOKENIZER_DIR = ROOT / "assets" / "models" / "embedder"  # use original tokenizer vocab
    TORCH_FILE    = ROOT / "scripts" / "_parity_torch_embs.npy"

    torch_embs = np.load(str(TORCH_FILE))

    import onnxruntime as ort
    from transformers import AutoTokenizer

    print("Loading ONNX session …")
    sess        = ort.InferenceSession(str(ONNX_DIR / "model.onnx"), providers=["CPUExecutionProvider"])
    tokenizer   = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    input_names = {inp.name for inp in sess.get_inputs()}
    print(f"  ORT inputs: {input_names}")

    def mean_pool(token_embs, attn_mask):
        mask   = attn_mask[..., np.newaxis].astype(np.float32)
        summed = (token_embs * mask).sum(axis=1)
        counts = mask.sum(axis=1).clip(min=1e-9)
        return summed / counts

    onnx_embs = []
    for text in TEST_STRINGS:
        enc    = tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=512)
        inputs = {
            "input_ids":      enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in input_names:
            # XLM-R tokenizer doesn't return token_type_ids; supply zeros
            tti = enc.get("token_type_ids", np.zeros_like(enc["input_ids"]))
            inputs["token_type_ids"] = tti.astype(np.int64)

        out    = sess.run(None, inputs)
        pooled = mean_pool(out[0], enc["attention_mask"])[0]
        normed = pooled / np.linalg.norm(pooled).clip(min=1e-12)
        onnx_embs.append(normed.astype(np.float32))
        print(f"  encoded: {text[:60]}")

    print()
    print("── Parity check ──────────────────────────────────────────────────────────")
    print(f"  {'Text':<65}  cosine_sim  result")
    print("  " + "-" * 78)
    all_ok = True
    for i, text in enumerate(TEST_STRINGS):
        a   = torch_embs[i]
        b   = onnx_embs[i]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        ok  = "PASS" if cos >= 0.999 else "FAIL"
        if cos < 0.999:
            all_ok = False
        lbl = text[:63] + ".." if len(text) > 65 else text
        print(f"  {lbl:<65}  {cos:.6f}  {ok}")

    print()
    if all_ok:
        print("OVERALL: PASS — all cosine similarities >= 0.999")
    else:
        print("OVERALL: FAIL — check pooling/prefix/normalization")
        sys.exit(1)

else:
    # ── Orchestrator: run both steps in sequence ──────────────────────────────
    env = {"PYTHONPATH": str(ROOT)}
    import os
    env.update(os.environ)

    print("=== Step 1: torch embeddings ===")
    r1 = subprocess.run([str(PYTHON), str(THIS), "--torch"], env=env,
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r1.stdout)
    if r1.stderr.strip():
        print(r1.stderr, file=sys.stderr)
    if r1.returncode != 0:
        print(f"Step 1 failed (exit {r1.returncode})", file=sys.stderr)
        sys.exit(1)

    print("=== Step 2: ONNX embeddings + parity ===")
    r2 = subprocess.run([str(PYTHON), str(THIS), "--onnx"], env=env,
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r2.stdout)
    if r2.stderr.strip():
        print(r2.stderr, file=sys.stderr)
    if r2.returncode != 0:
        print(f"Step 2 failed (exit {r2.returncode})", file=sys.stderr)
        sys.exit(1)

    print()
    print("parity_check_onnx.py: DONE")
