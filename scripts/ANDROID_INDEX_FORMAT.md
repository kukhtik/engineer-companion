# chunks_index.bin — Android Flat Vector Index Format

**File:** `android/app/src/main/assets/db/chunks_index.bin`  
**Built by:** `scripts/build_android_index.py`  
**Current stats:** 11,629 rows, 23.1 MB

---

## Binary Layout (all values little-endian)

```
[ HEADER ]
  uint32   num_rows      — total number of chunk records
  uint32   vec_dim       — embedding dimension (always 384)

[ RECORD 0 ]
  float32[384]           — embedding vector (1,536 bytes)
  uint32   meta_len      — byte length of the JSON metadata blob
  uint8[meta_len]        — UTF-8 JSON: {"source":…,"page":…,"section":…,"text":…}

[ RECORD 1 ]
  … (same structure)

… repeated num_rows times
```

Total header size: **8 bytes**.  
Per-record fixed portion: **1,536 + 4 = 1,540 bytes** + variable JSON.

Because metadata length varies, records are **not** at fixed offsets. The Kotlin
loader must scan sequentially or build an offset table at startup (recommended:
scan once, store `LongArray` of offsets for O(1) seek).

---

## Metadata JSON Schema

Each metadata blob is a compact JSON object (no extra whitespace):

```json
{
  "source":  "TrueBeam Administrators Guide.pdf",
  "page":    55,
  "section": "MLC",
  "text":    "The Multi-Leaf Collimator (MLC) …"
}
```

| Field     | Type    | Description                                |
|-----------|---------|--------------------------------------------|
| `source`  | string  | PDF filename (basename)                    |
| `page`    | integer | 1-indexed page number in the source PDF    |
| `section` | string  | Heading/section extracted by the chunker   |
| `text`    | string  | Full chunk text shown in the answer panel  |

---

## Embedder Preprocessing (MUST match on Android side)

The index was built with **intfloat/multilingual-e5-small** (XLM-RoBERTa backbone,
mean pooling, L2-normalized) using `sentence-transformers`.

**Kotlin OnnxRetriever must replicate exactly:**

| Step          | Value / Rule                                                   |
|---------------|----------------------------------------------------------------|
| Query prefix  | **NONE** — raw text, no `"query: "` or `"passage: "` prepended |
| Tokenizer     | XLM-RoBERTa SentencePiece (`sentencepiece.bpe.model`)          |
| Max length    | 512 tokens, truncation=true, padding=true                      |
| `input_ids`   | int64                                                          |
| `attention_mask` | int64                                                       |
| `token_type_ids` | int64, all zeros (required by ONNX graph; not produced by XLM-R tokenizer automatically) |
| Pooling       | Mean of token hidden states weighted by `attention_mask` (ignore padding) |
| Normalize     | L2 to unit norm after pooling                                  |
| Similarity    | Cosine (= dot product, since both vectors are unit-normed)      |

**ONNX model inputs** (verified from exported graph):
- `input_ids` (int64)
- `attention_mask` (int64)
- `token_type_ids` (int64, zeros)

**ONNX model output:** `last_hidden_state` shape `[batch, seq_len, 384]`

Parity verified: cosine similarity between torch and ONNX paths = **1.000000** on
5 test strings (RU + EN, short + long).

---

## Kotlin Pseudocode

```kotlin
// Build offset table at startup (one-time scan, ~23 MB sequential read)
val offsets = LongArray(numRows)
var pos = 8L  // skip header
for (i in 0 until numRows) {
    offsets[i] = pos
    pos += 384 * 4          // vector
    val metaLen = buf.getInt(pos.toInt()); pos += 4
    pos += metaLen
}

// Cosine search (brute force — 11.6k × 384 fits in ~18 MB float array)
val allVecs = FloatArray(numRows * 384)  // pre-loaded
fun cosineSimilarity(a: FloatArray, b: FloatArray): Float { … }

fun topK(queryVec: FloatArray, k: Int): List<ChunkResult> {
    return (0 until numRows)
        .map { i -> i to cosineSimilarity(queryVec, allVecs.slice(i*384, 384)) }
        .sortedByDescending { it.second }
        .take(k)
        .map { (i, score) -> ChunkResult(score, metadata[i]) }
}
```

---

## Files

| File | Size | Notes |
|------|------|-------|
| `android/app/src/main/assets/db/chunks_index.bin` | 23.1 MB | gitignored (large binary) |
| `android/app/src/main/assets/models/embedder_onnx/model.onnx` | 448 MB | gitignored |
| `android/app/src/main/assets/models/embedder_onnx/tokenizer.json` | 16.3 MB | include in repo |
| `android/app/src/main/assets/models/embedder_onnx/sentencepiece.bpe.model` | 4.8 MB | include in repo |
| `android/app/src/main/assets/models/embedder_onnx/tokenizer_config.json` | <1 KB | include in repo |
| `android/app/src/main/assets/models/embedder_onnx/special_tokens_map.json` | <1 KB | include in repo |
| `scripts/build_android_index.py` | — | rebuilds the .bin from engineer.db |
| `scripts/ANDROID_INDEX_FORMAT.md` | — | this file |

Add to `.gitignore`:
```
android/app/src/main/assets/db/chunks_index.bin
android/app/src/main/assets/models/embedder_onnx/model.onnx
```
