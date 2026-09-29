# GPU steps (embeddings and neighbour search)

These three scripts produced the test embeddings and all nearest-neighbour lists used by the pipeline
(`BER_EMB=1`, see `../embeddings.py`). They were run on a rented CUDA GPU (one NVIDIA L4, 23 GB) because
they take hours on the Mac; they also run on any CUDA machine. The instance is gone; recreate it if the
embeddings ever have to be rebuilt.

| Script | Where | Does |
|---|---|---|
| `prepare_upload.py` | Mac | builds the encoder input text for every test record (v1 recipe) and a 5,000-record training check set with the stored vectors |
| `encode_test.py` | GPU | encodes the test text with `Embeddings/sc_encoder` (fp16, L2-normalised) into `test_emb.npy`, row i = test rid i. **It first re-encodes the 5,000 check records and stops unless they match the stored training vectors (cosine > 0.995)**: this guards against a checkpoint / recipe mismatch |
| `knn.py` | GPU | exact cosine nearest neighbours per country: every S1 -> top-20 S2/S3 records, every S2/S3 -> top-10 S1 (`knn_<world>_{fwd,rev}.parquet`). Worlds: `train`, `train --exclude-s1 testlike_removed_s1.parquet` (test-like), `test` |

Encoder input text (v1): `query: name: {name_norm} | address: {addr_norm} | country: {country}`, with
`name_norm` / `addr_norm` lower-cased, NFKC-normalised, punctuation stripped, `&` -> `and`.
Measured speeds on the L4: encoding ~4,500 records/s (11.7M test records in 45 min); neighbour search
23 min (train), 13 min (test-like), 13 min (test).
Copy `test_emb.npy` and the six `knn_*.parquet` files back to `work/emb/`.
