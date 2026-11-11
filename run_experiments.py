"""Run all experiments: 2 datasets x N seeds x 4 models. Writes results/ (tables, plots, model json)."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from src.data import prepare, META
from src.ga import GeneticFuzzyLearner
from src.fuzzy import memberships, SETS
from src.explain import explain, format_text

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
LAM = 0.05
BASE, PROP = "GA-Fuzzy base (accuracy-only)", "GA-Fuzzy proposed (accuracy + compact rules)"
LR, RF = "Logistic Regression", "Random Forest (black box)"


def best_threshold(y, s):
    grid = np.quantile(s, np.linspace(0.05, 0.95, 91))
    bal = [((s[y == 1] >= t).mean() + (s[y == 0] < t).mean()) / 2 for t in grid]
    return float(grid[int(np.argmax(bal))])


def metrics(y, s, thr):
    p = s >= thr
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    rec, prec = tp / max(tp + fn, 1), tp / max(tp + fp, 1)
    return dict(AUC=roc_auc_score(y, s), BalAcc=(rec + tn / max(tn + fp, 1)) / 2, Recall=rec,
                Precision=prec, F1=2 * prec * rec / max(prec + rec, 1e-9))


def run(dataset, seed, ga_kw):
    d = prepare(dataset, seed)
    Xtr, ytr, Xva, yva, Xte, yte = (d[k] for k in ("Xtr", "ytr", "Xva", "yva", "Xte", "yte"))
    models, rows = {}, []
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
    rf = RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    fz_base = GeneticFuzzyLearner(lam=0.0, prune=False, seed=seed, **ga_kw)
    fz_prop = GeneticFuzzyLearner(lam=LAM, prune=True, seed=seed, **ga_kw)
    mb = fz_base.fit(Xtr, ytr, d["features"])
    mp = fz_prop.fit(Xtr, ytr, d["features"])
    scorers = {LR: lambda X: lr.predict_proba(X)[:, 1], RF: lambda X: rf.predict_proba(X)[:, 1],
               BASE: mb.score, PROP: mp.score}
    for name, fn in scorers.items():
        thr = best_threshold(yva, fn(Xva))
        m = metrics(yte, fn(Xte), thr)
        m.update(model=name, dataset=dataset, seed=seed,
                 rules=(mb.n_rules() if name == BASE else mp.n_rules() if name == PROP else np.nan),
                 avg_conds=(mb.avg_conditions() if name == BASE else mp.avg_conditions() if name == PROP else np.nan))
        rows.append(m)
    return rows, dict(d=d, scorers=scorers, mb=mb, mp=mp, fz_base=fz_base, fz_prop=fz_prop, lr=lr, rf=rf)


def plots(dataset, art):
    d, feats = art["d"], art["d"]["features"]
    # 1) ROC
    plt.figure(figsize=(5.5, 5))
    for name, fn in art["scorers"].items():
        fpr, tpr, _ = roc_curve(d["yte"], fn(d["Xte"]))
        plt.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(d['yte'], fn(d['Xte'])):.3f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1); plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title(f"ROC on held-out test set - {dataset}"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(OUT / f"roc_{dataset}.png", dpi=150); plt.close()
    # 2) GA convergence
    plt.figure(figsize=(5.5, 4))
    for lab, ga in [("base (lam=0)", art["fz_base"]), (f"proposed (lam={LAM})", art["fz_prop"])]:
        h = pd.DataFrame(ga.history)
        plt.plot(h.gen, h.auc, label=f"{lab}: AUC"); plt.plot(h.gen, h.fitness, "--", label=f"{lab}: fitness")
    plt.xlabel("Generation"); plt.ylabel("Best of generation (train sample)")
    plt.title(f"GA convergence - {dataset}"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(OUT / f"convergence_{dataset}.png", dpi=150); plt.close()
    # 3) evolved membership functions of the proposed model
    m, meta = art["mp"], META[dataset]
    xs = np.linspace(0, 1, 201)
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.5))
    for i, ax in enumerate(axs.ravel()):
        M = memberships(np.tile(xs[:, None], (1, len(feats))), m.knots)[:, i, 1:]
        for j, s in enumerate(SETS):
            ax.plot(xs, M[:, j], label=s)
        ax.set_title(meta["labels"][feats[i]], fontsize=8); ax.set_ylim(-0.05, 1.05)
    axs[0, 0].legend(fontsize=7); fig.suptitle(f"Evolved linguistic sets (proposed model) - {dataset}")
    fig.tight_layout(); fig.savefig(OUT / f"membership_functions_{dataset}.png", dpi=150); plt.close(fig)


def save_bundle(dataset, art):
    d, m, meta = art["d"], art["mp"], META[dataset]
    thr = best_threshold(d["yva"], m.score(d["Xva"]))
    Xtr = d["Xtr"]
    good = {}
    for i, f in enumerate(d["features"]):
        dr = meta["direction"].get(f)
        good[f] = float(np.percentile(Xtr[:, i], 10 if dr == "lower" else 90)) if dr else float(np.median(Xtr[:, i]))
    s = m.score(d["Xte"])
    picks = [int(np.argsort(s)[k]) for k in (5, len(s) // 2, len(s) - 6)]        # low / mid / high risk
    picks += [int(np.where((s > thr) & (d["yte"] == 1))[0][0]), int(np.where((s < thr) & (d["yte"] == 0))[0][0])]
    samples = [dict(name=n, x=d["Xte"][i].tolist(), actual_default=int(d["yte"][i]))
               for n, i in zip(["Applicant A (low risk)", "Applicant B (middle)", "Applicant C (high risk)",
                                "Applicant D (caught defaulter)", "Applicant E (correctly cleared)"], picks)]
    bundle = dict(dataset=dataset, model=m.to_dict(), threshold=thr, good_values=good,
                  meta=meta, samples=samples, feature_names=d["features"],
                  rules=[m.rule_text(r, meta["labels"]) for r in np.where(m.active_mask())[0]])
    (OUT / f"model_{dataset}.json").write_text(json.dumps(bundle, indent=2))
    lines = [f"=== {dataset} | proposed model: {m.n_rules()} rules ===", *bundle["rules"], ""]
    for sp in samples:
        e = explain(m, sp["x"], meta, thr, good)
        lines += [f"--- {sp['name']} (true outcome: {'defaulted' if sp['actual_default'] else 'repaid'}) ---", format_text(e), ""]
    (OUT / f"sample_explanations_{dataset}.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-taiwan", type=int, default=5)
    ap.add_argument("--seeds-german", type=int, default=10)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=60)
    a = ap.parse_args()
    ga_kw = dict(pop_size=a.pop, generations=a.gens)
    all_rows = []
    for dataset, n in [("taiwan", a.seeds_taiwan), ("german", a.seeds_german)]:
        for seed in range(n):
            rows, art = run(dataset, seed, ga_kw)
            all_rows += rows
            print(f"{dataset} seed {seed}: " + " | ".join(f"{r['model'][:14]} AUC={r['AUC']:.3f}" for r in rows), flush=True)
            if seed == 0:
                plots(dataset, art); save_bundle(dataset, art)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT / "metrics_raw.csv", index=False)
    cols = ["AUC", "BalAcc", "Recall", "Precision", "F1", "rules", "avg_conds"]
    g = df.groupby(["dataset", "model"], sort=False)[cols]
    summ = g.mean().round(3).astype(str) + " +/- " + g.std().round(3).astype(str)
    summ.to_csv(OUT / "summary.csv")
    (OUT / "summary.md").write_text(summ.to_markdown())
    print(summ.to_string())
