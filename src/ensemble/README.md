# Stacker and decision rule

| File | What it does |
|---|---|
| `ensemble_test.py` | joins both models' pair tables (union of candidates), fits the LightGBM stacker on fold 0 with 22 pool-safe features (5-fold cross-fitted by S1 for the reported score, then fitted on all of fold 0), scores the test union and writes `work/ensemble/test_stacked.parquet` |
| `ensemble_apply.py` | assignment (each S2/S3 record goes to the S1 that scores it highest) and the bars `t_abs = t_noaddr = 0.70`, `delta = 0.2`; writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` |
| `ensemble_fold0.py` | analysis on fold 0 (each model alone, error overlap, weighted averages); holds the shared functions `load_union`, `add_features`, `scorer_for` |

Inputs: `work/ensemble/model_a/*.parquet` (`model_a/run_model_a.py`) and `work/ensemble/model_b/*.parquet`
(`export_pair_probs.py`). Run by stages E1 and E2 of `run_all.py`; about 40 minutes on CPU. We ran them on the GPU machine (125 GB RAM).
