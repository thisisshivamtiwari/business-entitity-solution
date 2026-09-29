# Methodology: linking business records across three sources

**Team:** Abhimanyu Vashishtha, Puneet Mishra, Kashif Khan, Shivam Tiwari  
**Date:** 2026-09-29

The task: for every business record in Source 1, find the records in Source 2 and Source 3 that describe the same
real-world business. Names and addresses are noisy, some are written in Indian scripts, and one country appears
only in the test set. The score is macro F0.5: one F0.5 per Source 1 record, averaged, so a wrong merge costs
more than a missed match and a record with no match must get an empty list.

---

## 1. Summary
Our solution is a stacked ensemble of two independent matchers: **Model A**, a cross-encoder pipeline that reads each pair of records jointly (`xlm-roberta-base` judge plus a LightGBM combiner), and **Model B**, a feature-based pipeline (compound-key blocking, embedding neighbours, a learned ranker and a two-pass LightGBM matcher over 113 features). A LightGBM stacker, trained only on a held-out fold where both models' scores are out of sample, combines their pair probabilities. It scored **0.9866 macro F0.5 on the hidden-label test set** and 0.9904 on our validation fold, against 0.9801 and 0.9883 for the better single model.

---

## 2. Methodology

### 2.1 Problem Analysis
Key findings from the exploratory analysis of the training data:

- **No shared identifiers.** Train and test share no IDs, so the models must learn matching patterns.
- **Cluster structure.** Each Source 1 (S1) record has 0 to 11 matches (mean 3.46); 5.6% have none. No Source 2/3 record ever belongs to two S1, so at most one S1 may claim each record.
- **Decoys.** About 26% of S2/S3 records in train match no S1. Three independent signals put the test share near 40%: about 2.3 look-alike decoys per S1, against 1.2 in train. Validation on the raw training data therefore overstates test performance.
- **Name noise.** Only 21% of true matches have an identical name. 21% differ only in legal form, 30% are reordered or contain a typo, 11% have heavier edits, 5% are website names, 7% (18% in India) are written in one of nine Indian scripts, and 3% are unrelated names at a matching address.
- **Names are not unique.** 56–70% of S1 names are unique; generic US names repeat up to 250 times.
- **Address noise.** About 3% of S2/S3 records have no address and 3% have placeholders; there are no postcodes. States appear as codes, full names or native script; France uses regions or departments. House numbers agree in 69% of true pairs, are truncated in 8% and differ in 10%.
- **One field is usually intact.** Name noise and address noise rarely hit the same record.
- **Unseen country.** The test set adds France (15% of test S1) with no French training data, so country is treated as an open set.

### 2.2 Solution Strategy
**Approach Type:** Hybrid. Two blocking-plus-classifier pipelines with different designs, combined by a stacked meta-classifier.  
**Core Innovation:** (1) Two deliberately different matchers: one reads the text jointly with a cross-encoder, the other uses engineered similarity, uniqueness and competition features. Where they disagree, each fixes errors the other makes. (2) A candidate channel from a fine-tuned multilingual embedding model, merged with lexical blocking by a learned ranker, which raised Model B's candidate recall from 97.0% to 99.4% at 12 candidates per S1. (3) Validation in a "test-like" world with the test set's decoy density, used to choose thresholds and experiments.

---

## 3. Candidate Generation (Blocking)

### 3.1 Model B (feature-based pipeline)
- **Blocking keys used:** compound keys, counted per country and kept only if rare on both sides (at most 50 S1 and 200 S2/S3 records per key): pairs of name words, name word plus address word, name plus region, house number plus address word, adjacent address words, and the name without spaces. The top 100 raw candidates are kept per S1.
- **Embedding channel:** a fine-tuned `intfloat/multilingual-e5-small` encoder (118M parameters, MIT licence; the first version of Model A's encoder, trained on lower-cased text) embeds `name | address | country`. Exact per-country cosine search gives each S1 its 20 nearest S2/S3 records, and each S2/S3 record its 10 nearest S1.
- **Learned ranker:** a HistGradientBoosting model scores the pool (keyword top 30 plus embedding top 20) with key-type scores, competition statistics, cosine and forward/reverse neighbour ranks, and keeps the **top 12 per S1**.
- **Name-only search:** character-trigram TF-IDF adds up to 5 S1 for each S2/S3 record that has no address.
- **Candidate pairs generated:** 21,492,924 on the test set (12.4 per S1). On the validation fold, 99.4% of true pairs are candidates.

### 3.2 Model A (cross-encoder pipeline)
- **Encoder:** `intfloat/multilingual-e5-small`, fine-tuned with a supervised contrastive loss. Each ground-truth cluster (an S1 record and its matches) is one class; a batch holds 128 clusters with 2 records each, and 40% of the clusters come from groups of different businesses that share a name, so the encoder has to separate them by address. 4,000 steps, folds 1–4 only.
- **Text:** case, script and accents are kept; only invisible characters, leading junk (`--`, `<<`, `#`) and null-like address parts are removed. Every record is embedded in two views: `name | address` and name only.
- **Blocking keys used:** none. Candidates are the nearest neighbours by cosine, searched **exactly** on the GPU within each country: the top 25 on the full view, the top 10 on the name-only view (for records without an address) and same-country records with the identical name key (groups of at most 30).
- **Why exact search:** an approximate index (FAISS HNSW) found 98.88% of the true pairs. A brute-force check showed that 68% of its misses were inside the exact top 25: many businesses share a name, and the index loses records in these dense groups. Exact search found 99.65% and was faster (14 minutes against 80).
- **Candidate pairs generated:** 56,342,681 on the test set (32.5 per S1); 14,427,665 on the validation fold (32.6 per S1), with 99.65% of the true pairs among them (99.63% for Indian-script names, 95.5% when the matching record has no address).

### 3.3 Final candidate set
- The stacker scores the **union** of both models' candidates: 62,884,758 test pairs (14,950,847 proposed by both models). On the validation fold, 99.81% of true pairs are in the union.
- `output/candidate_pairs.tsv` lists this union, best stacker score first.
- **How we ensured true matches were not lost:** each blocking stage was tuned on validation recall, per country and per record type (no address, Indian script, website names). The two models' blocking methods are different, so the union recovers most pairs either one misses.

---

## 4. Matching Model

### 4.1 Model B: two-pass LightGBM
**Features used (113):**
- Name features: RapidFuzz ratios (simple, token-set, partial), IDF-weighted word overlap, character trigrams, the name without the city, the rarest unshared word, legal-form relation, website and alias (dba) handling. Indian-script names are mapped to English with a word dictionary learned from the training pairs (1,346 words, 9 scripts).
- Address features: fuzzy and IDF-weighted overlap, house-number relations (equal, truncated, off by a few, different), region agreement after canonicalisation (including native-script states learned from the data and French department→region), placeholder and missing-address flags.
- Other: name uniqueness (how many S1 share the name), decoy signals (legal form × house-number mismatch, extra words, name-length gap), each feature's gap to the S1's best candidate, the ranker score, embedding cosine, forward and reverse neighbour rank (the "mutual neighbour" signal).

**Model type:** LightGBM, binary objective, learning rate 0.05, 127 leaves. A second pass adds cluster-consistency features: each candidate's similarity to the S1's other confident matches, from out-of-fold first-pass scores. The final model is trained on all 27.4M training pairs (first pass 2,880 trees, second pass 1,242 trees).

### 4.2 Model A: cross-encoder judge and combiner
**Model type:** an `xlm-roberta-base` cross-encoder (278M parameters, MIT licence), the "judge", reads both records in one sequence (`name | address [SEP] name | address`, 96 tokens) and outputs `judge_prob`. A LightGBM combiner turns the judge score and 26 simple features into `combiner_prob`.

**Judge training:** 2.4M labelled pairs from folds 1–4: every candidate of a training S1 is a pair, labelled from the ground truth, so the negatives are the look-alikes that blocking returns (1 positive to 4 negatives). 2 epochs, total batch 192, learning rate 2e-5, binary cross-entropy, bf16. Augmentation, with labels unchanged: in 30% of the pairs one random edit (drop a token, drop an address, swap the sides, change the case, shuffle the address parts), and addresses are removed from true pairs until 30% of them have a name-only side. The training S1 are those with at least one match.

**Features used by the combiner (27):**
- Name features: RapidFuzz ratio, token-set and partial ratio, token and character-trigram Jaccard, the same without legal suffixes, length ratio, legal-suffix agreement, script mismatch.
- Address features: fuzzy ratio and token overlap, house-number and postal-code agreement, overlap of all numbers, missing-address flags. Missing values stay empty.
- Other: `judge_prob`, cosine on both embedding views, gap to the S1's best cosine, blocking rank, number of candidates.

**Combiner:** LightGBM (600 rounds, 63 leaves), trained on the 5% of training S1 (82,725 S1, 2.7M pairs) that the judge never saw. The judge score carries about 8 times the gain of the next feature. On the validation fold: judge alone 0.9792, with the assignment rule 0.9806, with the combiner 0.9824.

### 4.3 Stacker
**Model type:** LightGBM (binary, learning rate 0.05, 63 leaves, min 200 samples per leaf, feature and bagging fraction 0.8, L2 1.0, 400 rounds).

**Features (22), all "pool-safe":**
- pair: `ber_prob` (Model B), `judge_prob`, `combiner_prob` (Model A), `emb_cosine`, `blocking_cosine`, `rank_a`, `rank_b`, `in_a`, `in_b` (which model proposed the pair), `cand_addr_missing`, `from_namesearch`;
- blends: `mean_prob`, `min_prob`, `max_prob`, `abs_diff`;
- S1 context: best score of each model in the S1, number of candidates, number of high-scoring candidates per model, gap to the S1's best score per model.

"Pool-safe" means no feature depends on how many S1 compete for a candidate: fold 0 has 442k S1 and the test set 1.73M, so such counts would shift between validation and test. The most important features by gain are `mean_prob`, `ber_prob` and `max_prob`.

**Training:** both base models were trained without fold 0 (S1 with `hash(entity_id) % 5 == 0`, 442,001 S1). Their fold-0 scores are therefore out of sample. The stacker is 5-fold cross-fitted by S1 inside fold 0 (all reported scores are out of fold), then fitted on all of fold 0 and applied to the test union.

**Threshold selection method (decision rule):**
1. **Assignment:** each S2/S3 record may only go to the S1 that gives it the highest score.
2. **Keep** a candidate if its score is at least the bar (`t_abs = t_noaddr = 0.70`) and within `delta = 0.2` of the S1's best score.

Macro F0.5 on fold 0 is flat for bars 0.40–0.70 (0.9904). The fold-0 optimum was 0.575; the bar was raised to 0.70 because the test set has about twice the decoys, which costs nothing on fold 0. A test-set probe supported it: removing the 7,200 accepted pairs that our reproduction of the stacker scored below 0.80 (a stricter cut) scored 0.986605, against 0.986650 for the final file.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9904** on the validation fold (442,001 S1, out of fold); **0.9866** on the hidden-label test set (1,732,544 S1).

| Method (validation fold, macro F0.5) | F0.5 | False merges | Missed links |
|---|---|---|---|
| Model A alone | 0.9824 | 9,781 | 59,289 |
| Model B alone | 0.9883 | 2,782 | 46,090 |
| Weighted average of the two | 0.9890 | 2,839 | 46,538 |
| **Stacker (final)** | **0.9904** | **1,807** | **41,941** |
| Oracle: better model per S1 | 0.9927 | | |

On pairs both models scored, their probabilities correlate at 0.988. They fall on opposite sides of 0.5 for 1.16% of pairs; there, Model B is right 68.5% of the time and Model A 31.5%.

**Test-set results:** Model B alone 0.9801 → stacked ensemble **0.9866**. Two probes replaced only the French rows (15% of test S1) of the ensemble file: with Model B's French rows 0.9857, with Model A's French rows 0.9835. So the ensemble is the best choice in every country, and 86% of its gain over Model B comes from the US and India.

- **Common false positives (wrong merges):** a similar name at the same address (a different business in the same building, frequent with French templated names such as "Lille Club SARL"); a house number off by a few digits; a same-name record without an address that belongs to another branch; orphan copies of a business that has no S1 record.
- **Common false negatives (missed matches):** records with **no address whose name is shared by several S1** make up 77% of the remaining misses. The ground truth assigns each to exactly one branch, and neither the normalised nor the raw name identifies which one; abstaining is F0.5-optimal. The rest are heavy name edits and unrelated names at a matching address.

---

## 6. Conclusion
Two matchers with different designs, a cross-encoder and a feature-based gradient-boosting pipeline, combined by a stacker trained on out-of-sample scores, beat either model alone by a wide margin on the test set (0.9866 against 0.9801). The largest single gains came from new ways of reading the text: embedding neighbours for candidate generation (+0.0064 for Model B) and the cross-encoder as a second opinion in the stack. The main lessons: validate under test conditions (decoy density), keep features "pool-safe" when validation and test pools differ in size, and accept that some errors (no-address records of shared-name businesses) cannot be resolved from the data.

---

## Appendix

### A. Code Artefacts
All code is in `src/`, with `README.md` and `requirements.txt` at the repository root.

- **Entry point:** `src/run_all.py` runs every stage in order: Model B experiment and fold-0 scoring (B1–B3), Model B final run on all training data and test scoring (B4–B5), Model A (A1), the stacker (E1), the decision rule writing `output/matching_results.tsv` and `output/candidate_pairs.tsv` (E2), and the output check (V). The embeddings Model B reads are built first (stage G1 and `src/gpu/`).
- **Model B:** `convert.py` → `translit.py` → `normalize.py` → `block.py` → `embeddings.py` / `rerank.py` → `namesearch.py` → `features.py` and related → `train.py` / `final.py` → `predict.py`; runner `run_pipeline.py`; memory watchdog `memguard.py`.
- **Embeddings for Model B (GPU):** `src/encoder_v1/` (text recipe, contrastive fine-tuning, training embeddings), `src/gpu/encode_test.py`, `src/gpu/knn.py`.
- **Model A:** `src/model_a/run_model_a.py` with `blocking/` (`prepare.py`, `train_sc.py`, `encode.py`, `retrieve.py`) and `matching/` (`build_pairs.py`, `cross_encoder.py`, `features.py`, `train_gbm.py`, `decide.py`).
- **Stacker:** `src/ensemble/` (`ensemble_test.py` fits and scores, `ensemble_apply.py` applies the decision rule, `ensemble_fold0.py` holds the analysis and shared functions).
- **Hardware:** Model B on an Apple M4 Pro (48 GB RAM). Encoders, Model A and the stacker on rented GPUs (one NVIDIA L4, then 2 × RTX 4090 with 125 GB RAM); Model A takes about 5.5 hours there.
- **Reproducibility:** all models are retrained from the provided data with fixed seeds. GPU training is not bit-exact, so a rerun reproduces the final files up to training randomness.
- **Validation tools:** `testlike.py` (test-like world), `validate.py`, `evaluate.py` (exact macro F0.5).

**Compliance:** no external data, APIs, geocoding or lookups; every model is trained on the provided training data only, and the test set is only scored. Pretrained models: `intfloat/multilingual-e5-small` (118M parameters, MIT) and `FacebookAI/xlm-roberta-base` (278M, MIT). Other models: LightGBM (MIT), scikit-learn (BSD-3). All are far below 8 billion parameters.

### B. Additional Results

**Model B development** (test-like validation F0.5):

| Iteration | Change | Test-like F0.5 |
|---|---|---|
| iter2 | learned ranker, house-number and legal-form features | 0.9719 |
| iter3 | Indian-script dictionary and spelling variants | 0.9752 |
| iter5–7 | LightGBM, uniqueness and decoy features, cluster second pass | 0.9789 |
| iter8 | wider key caps (50 / 200) | 0.9802 |
| iter9c | French normalisation, trigram features, name-only search | 0.9809 |
| iter10a | embedding channel in the ranker and the matcher | **0.9873** |

**Stacker sensitivity to the bar** (validation fold, delta 0.2):

| Bar | F0.5 | False merges | Missed links |
|---|---|---|---|
| 0.40 | 0.9904 | 1,996 | 41,750 |
| 0.60 | 0.9904 | 1,844 | 41,881 |
| **0.70 (final)** | **0.9904** | **1,807** | **41,941** |
| 0.85 | 0.9900 | 1,449 | 43,587 |
| 0.95 | 0.9884 | 698 | 51,744 |

**Final output by country:**

| Country | S1 records | Left empty | Mean matches per matched S1 |
|---|---|---|---|
| India | 809,986 | 5.8% | 3.58 |
| US | 663,106 | 5.8% | 3.60 |
| France | 259,452 | 5.5% | 3.58 |
| **All** | **1,732,544** | **5.8%** | **3.58** |
