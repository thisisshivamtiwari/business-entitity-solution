"""Matching stage 5: collective assignment + per-entity decision rule + macro-F0.5 (vectorised).

Assignment (holds on 100% of the labels): every S2/S3 record belongs to at most one S1
entity, so a candidate pair is eligible only if this S1 is the highest-scoring claimant
of that S2/S3 record (--assign, on by default).
Rule, per S1 entity with candidate scores s_i (a candidate without an address uses the
stricter bar t_noaddr instead of t_abs):
    keep candidate i  iff  eligible_i  and  s_i >= bar_i  and  s_i >= best - delta
where best = max_i s_i over eligible candidates. Nothing kept = singleton (empty list).

tune : grid-search (t_abs, t_noaddr, delta) for macro F0.5 on half A of the scored
       entities, report on half B, then swap (honest estimate); final params tuned on all.
apply: write matching_results.tsv with the given parameters.
"""
import argparse
import itertools
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


class Scorer:
    """Holds the arrays the rule needs; every decide() call is pure numpy."""

    def __init__(self, df, truth=None, score_col="score", assign=True):
        df = df.sort_values(["s1_id", "rank"]).reset_index(drop=True)
        self.ent, self.code = np.unique(df.s1_id.to_numpy(), return_inverse=True)
        self.n = len(self.ent)
        self.score = df[score_col].to_numpy(dtype=np.float32)
        self.noaddr = df.addr2_missing.to_numpy() == 1
        self.cand = df.cand_id.to_numpy()
        # collective assignment: eligible only if this S1 is the best claimant of the S2/S3 record
        if assign:
            cid, ccode = np.unique(self.cand, return_inverse=True)
            cbest = np.full(len(cid), -1.0, dtype=np.float32)
            np.maximum.at(cbest, ccode, self.score)
            self.eligible = self.score >= cbest[ccode] - 1e-6
        else:
            self.eligible = np.ones(len(self.score), dtype=bool)
        self.best = np.full(self.n, -1.0, dtype=np.float32)
        np.maximum.at(self.best, self.code[self.eligible], self.score[self.eligible])
        if truth is not None:
            self.n_true = np.array([len(truth.get(e, ())) for e in self.ent])
            self.is_true = np.array([c in truth.get(e, ()) for e, c in zip(df.s1_id.to_numpy(), self.cand)])

    def keep_mask(self, t_abs, t_noaddr, delta):
        bar = np.where(self.noaddr, t_noaddr, t_abs)
        return self.eligible & (self.score >= bar) & (self.score >= self.best[self.code] - delta)

    def f05_per_entity(self, keep):
        n_pred = np.bincount(self.code[keep], minlength=self.n)
        tp = np.bincount(self.code[keep & self.is_true], minlength=self.n)
        f = np.zeros(self.n)
        both_empty = (n_pred == 0) & (self.n_true == 0)
        f[both_empty] = 1.0
        ok = tp > 0
        p = tp[ok] / n_pred[ok]; r = tp[ok] / self.n_true[ok]
        f[ok] = 1.25 * p * r / (0.25 * p + r)
        return f, n_pred, tp

    def macro(self, keep, subset=None):
        f, _, _ = self.f05_per_entity(keep)
        return float(f.mean() if subset is None else f[subset].mean())

    def lists(self, keep):
        out = {e: [] for e in self.ent}
        for code, c in zip(self.code[keep], self.cand[keep]):
            out[self.ent[code]].append(c)
        return out


GRID = list(itertools.product(np.round(np.arange(0.20, 0.96, 0.05), 2), (0.0, 0.1, 0.2, 0.3),
                              (0.05, 0.1, 0.2, 0.35, 0.5, 1.0)))


def bayes(sc, subset, n_trials=150, seed=0):
    """Bayesian search (Optuna TPE) over the three bars; same objective as grid()."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(trial):
        t_abs = trial.suggest_float("t_abs", 0.05, 0.99)
        t_noaddr = trial.suggest_float("t_noaddr", t_abs, 0.995)
        delta = trial.suggest_float("delta", 0.02, 1.0, log=True)
        return sc.macro(sc.keep_mask(t_abs, t_noaddr, delta), subset)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.enqueue_trial(dict(t_abs=0.75, t_noaddr=0.75, delta=0.35))          # cycle-1 optimum as a seed point
    st.optimize(objective, n_trials=n_trials)
    p = st.best_params
    return st.best_value, dict(t_abs=round(p["t_abs"], 4), t_noaddr=round(p["t_noaddr"], 4), delta=round(p["delta"], 4))


def grid(sc, subset, coarse_frac=0.1, seed=0):
    """Coarse-to-fine: score the full grid on a random 10% of entities, then re-score
    the 12 best settings (plus a finer neighbourhood) on all entities of `subset`."""
    n = sc.n
    rng = np.random.RandomState(seed)
    sub = np.zeros(n, dtype=bool); sub[rng.choice(n, max(1, int(n * coarse_frac)), replace=False)] = True
    if subset is not None:
        sub &= subset
    scored = []
    for t_abs, extra, delta in GRID:
        t_noaddr = min(0.99, float(t_abs) + extra)
        scored.append((sc.macro(sc.keep_mask(float(t_abs), t_noaddr, delta), sub), float(t_abs), t_noaddr, delta))
    scored.sort(reverse=True)
    cands = set()
    for _, t_abs, t_noaddr, delta in scored[:12]:
        for da in (-0.025, 0.0, 0.025):
            for dn in (-0.025, 0.0, 0.025):
                cands.add((round(min(0.99, max(0.05, t_abs + da)), 3), round(min(0.99, max(0.05, t_noaddr + dn)), 3), delta))
    best = (-1.0, None)
    for t_abs, t_noaddr, delta in sorted(cands):
        val = sc.macro(sc.keep_mask(t_abs, t_noaddr, delta), subset)
        if val > best[0]:
            best = (val, dict(t_abs=t_abs, t_noaddr=t_noaddr, delta=delta))
    return best


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--scored", required=True); t.add_argument("--ground-truth", required=True)
    t.add_argument("--records", required=True); t.add_argument("--params-out", required=True)
    t.add_argument("--pred-out", default=None)
    t.add_argument("--score-col", default="score", help="column to threshold (score = combiner, ce_score = judge alone)")
    t.add_argument("--search", default="grid", choices=["grid", "bayes"],
                   help="grid = coarse-to-fine grid; bayes = Optuna TPE, 150 trials")
    t.add_argument("--claims", nargs="*", default=[], help="extra scored parquets whose S1 entities compete in the "
                   "assignment but are not evaluated (global assignment, as at test time)")
    t.add_argument("--no-assign", action="store_true", help="disable the one-S1-per-record constraint")
    a = sub.add_parser("apply")
    a.add_argument("--scored", required=True); a.add_argument("--params", required=True)
    a.add_argument("--s1-file", required=True, help="source1 TSV: every entity gets a row")
    a.add_argument("--out", required=True)
    a.add_argument("--score-col", default="score"); a.add_argument("--no-assign", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.scored)
    if args.cmd == "tune":
        eval_ents = set(df.s1_id.unique())
        for c in args.claims:
            extra = pd.read_parquet(c, columns=list(df.columns))
            extra = extra[~extra.s1_id.isin(eval_ents)]
            print(f"[decide] + {len(extra):,} claim pairs from {extra.s1_id.nunique():,} extra S1 entities ({c})")
            df = pd.concat([df, extra], ignore_index=True)
        gt = pd.read_csv(args.ground_truth, sep="\t", dtype=str, keep_default_na=False, quoting=3)
        ents = eval_ents
        truth = {e: set(x for x in l.split(",") if x) for e, l in zip(gt.source1_entity_id, gt.matched_entity_ids) if e in ents}
        sc = Scorer(df, truth, score_col=args.score_col, assign=not args.no_assign)
        print(f"[decide] score column '{args.score_col}', assignment {'off' if args.no_assign else 'on'}: "
              f"{(~sc.eligible).sum():,} of {len(sc.eligible):,} pairs blocked by a stronger claimant")
        evalmask = np.fromiter((e in ents for e in sc.ent), dtype=bool, count=len(sc.ent))   # np.isin on object arrays is pathologically slow
        half = np.array([zlib.crc32(f"half:{e}".encode()) % 2 for e in sc.ent])
        A, B = evalmask & (half == 0), evalmask & (half == 1)
        search = bayes if args.search == "bayes" else grid
        scA, pA = search(sc, A); scB, pB = search(sc, B)
        heldB = sc.macro(sc.keep_mask(**pA), B); heldA = sc.macro(sc.keep_mask(**pB), A)
        print(f"[decide] tuned on half A {scA:.4f} -> held-out half B {heldB:.4f}  params {pA}")
        print(f"[decide] tuned on half B {scB:.4f} -> held-out half A {heldA:.4f}  params {pB}")
        honest = (heldA + heldB) / 2
        print(f"[decide] honest macro F0.5 (mean of held-out halves): {honest:.4f}")
        scAll, pAll = search(sc, evalmask)
        print(f"[decide] final params tuned on all {evalmask.sum():,} evaluated entities: {pAll}  (in-sample {scAll:.4f})")
        keep = sc.keep_mask(**pAll)
        f, n_pred, tp = sc.f05_per_entity(keep)
        f, n_pred, tp, ent_ids, n_true = f[evalmask], n_pred[evalmask], tp[evalmask], sc.ent[evalmask], sc.n_true[evalmask]
        single = n_true == 0
        meta = pd.read_parquet(args.records, columns=["entity_id", "country"]).set_index("entity_id").country
        country = meta.loc[ent_ids].to_numpy()
        print("[decide] breakdown with final params (in-sample):")
        print(f"  singletons ({single.sum():,}): predicted empty {(n_pred[single] == 0).mean():.4f}")
        print(f"  matched entities ({(~single).sum():,}): F0.5 {f[~single].mean():.4f}, "
              f"wrongly emptied {(n_pred[~single] == 0).mean():.4f}")
        print(f"  false merges (pairs) {(n_pred - tp).sum():,}   missed links {(n_true - tp).sum():,}")
        print("  by country:", {c: round(float(f[country == c].mean()), 4) for c in np.unique(country)})
        Path(args.params_out).write_text(json.dumps(dict(pAll, honest_f05=honest, in_sample_f05=scAll), indent=2))
        if args.pred_out:
            pd.DataFrame(dict(entity=ent_ids, f05=f, n_pred=n_pred, tp=tp, n_true=n_true,
                              country=country)).to_parquet(args.pred_out, index=False)
    else:
        params = json.loads(Path(args.params).read_text())
        sc = Scorer(df, score_col=args.score_col, assign=not args.no_assign)
        lists = sc.lists(sc.keep_mask(params["t_abs"], params["t_noaddr"], params["delta"]))
        s1 = pd.read_csv(args.s1_file, sep="\t", dtype=str, keep_default_na=False, quoting=3, usecols=["entity_id"]).entity_id
        n_nonempty = 0
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("source1_entity_id\tmatched_entity_ids\n")
            for e in s1:
                l = lists.get(e, [])
                n_nonempty += bool(l)
                fh.write(f"{e}\t{','.join(l)}\n")
        print(f"[decide] wrote {len(s1):,} rows ({n_nonempty:,} with matches) -> {args.out}")


if __name__ == "__main__":
    main()
