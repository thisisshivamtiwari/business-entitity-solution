# Model A: embedding blocking, cross-encoder judge, LightGBM combiner

```bash
cd src
../.venv/bin/python -u model_a/run_model_a.py      # CUDA GPU needed; finished stages are skipped on a re-run
```

| Stage | File | What it does |
|---|---|---|
| 1 | `blocking/prepare.py`, `blocking/common.py` | cleans the raw files (case, script and accents kept), writes the ground-truth links and fold 0 |
| 2 | `blocking/train_sc.py` | fine-tunes `intfloat/multilingual-e5-small` with a supervised contrastive loss on folds 1-4 (4,000 steps, 128 clusters x 2 records per batch, 40% of clusters from same-name groups) |
| 3 | `blocking/encode.py` | embeds every record in two views: `name | address` and name only (384-d, float16) |
| 4 | `blocking/retrieve.py`, `blocking/evaluate_blocking.py` | exact top-k cosine search per country on the GPU: top 25 on the full view + top 10 on the name view + same-country records with the identical name key (groups of at most 30) |
| 5 | `matching/build_pairs.py` | labels each (S1, candidate) pair from the ground truth; 5% of the training S1 are kept away from the judge |
| 6 | `matching/cross_encoder.py train` | the judge: `FacebookAI/xlm-roberta-base` reads both records in one sequence; 2.4M pairs (1 positive : 4 negatives), 2 epochs, total batch 192, learning rate 2e-5, 96 tokens, bf16 |
| 7 | `matching/cross_encoder.py score` | judge probability for every candidate pair |
| 8 | `matching/features.py` | 25 pair features: name and address similarities, house number, postal code, missing-address and script flags, blocking cosines |
| 9 | `matching/train_gbm.py`, `matching/decide.py` | LightGBM combiner trained on the 5% group; Model A's own fold-0 score |
| 10-12 | same scripts, `--split test` | candidates, judge scores, features and combiner scores for the test set |

Training augmentation of the judge (labels unchanged): in 30% of the pairs one random edit (drop a token, drop an
address, swap the two sides, change the case, shuffle the address parts); in addition, addresses are removed from
true pairs until 30% of them have a name-only side.

**Output:** `work/ensemble/model_a/fold0_pair_probs.parquet` and `test_pair_probs.parquet` with
`s1_id, cand_id, rank, judge_prob, combiner_prob, blocking_cosine, cand_addr_missing` (+ `label` on fold 0).

**Measured on 2 x RTX 4090:** encoder 12 min (on one NVIDIA L4), encoding 12 min per split, candidate search 4 min (fold 0), 13 min
(training S1) and 10 min (test), judge training 2 h 17 min, judge scoring 7,000-10,000 pairs per second per GPU
(test set 68 min), features 20 min for the test set. About 5.5 hours in total. Disk: about 60 GB.

**Fold 0:** 14,427,665 candidate pairs (32.6 per S1), 99.65% of the true pairs among them; macro F0.5 of Model A
alone 0.9824. Test set: 56,342,681 candidate pairs (32.5 per S1).
