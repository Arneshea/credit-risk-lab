"""Genetic learner for a descriptive fuzzy rule base (Pittsburgh style: 1 individual = 1 rule base).

Chromosome
  knots   (F,3) : shared Low/Medium/High partition per feature (evolved, not hand-picked)
  genes   (K,F) : 0 = feature unused, 1/2/3 = Low/Medium/High
  cons    (K,)  : rule consequent (1 = default risk, 0 = repays)
  enabled (K,)  : rule on/off switch (lets the GA shrink the rule base)

Fitness = AUC  -  lam * complexity           (lam = 0  ->  accuracy-only baseline)
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from .fuzzy import FuzzyRuleBase, memberships, rule_strengths, risk_from_strengths

GAP = 0.03


def fix_knots(k):
    k = np.sort(np.clip(k, 0, 1), axis=1)
    k[:, 2] = np.clip(k[:, 2], 2 * GAP, 1.0)
    k[:, 1] = np.clip(k[:, 1], GAP, k[:, 2] - GAP)
    k[:, 0] = np.clip(k[:, 0], 0.0, k[:, 1] - GAP)
    return k


class GeneticFuzzyLearner:
    def __init__(self, n_rules=8, pop_size=60, generations=60, lam=0.05, prune=False,
                 seed=0, max_train=5000, p_cross=0.8, tournament=3, elites=2):
        self.K, self.pop_size, self.generations = n_rules, pop_size, generations
        self.lam, self.prune, self.seed, self.max_train = lam, prune, seed, max_train
        self.p_cross, self.tournament, self.elites = p_cross, tournament, elites
        self.history = []

    # ---------- chromosome helpers ----------
    def _new(self, rng, F, even):
        if even:
            knots = fix_knots(np.array([0.0, 0.5, 1.0])[None, :] + rng.normal(0, 0.08, (F, 3)))
        else:
            knots = fix_knots(rng.random((F, 3)))
        genes = np.zeros((self.K, F), int)
        for r in range(self.K):
            n = rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
            fs = rng.choice(F, size=min(n, F), replace=False)
            genes[r, fs] = rng.integers(1, 4, size=len(fs))
        return dict(knots=knots, genes=genes, cons=rng.integers(0, 2, self.K),
                    enabled=rng.random(self.K) < 0.85)

    def _crossover(self, a, b, rng):
        take = rng.random(self.K) < 0.5
        genes = np.where(take[:, None], a["genes"], b["genes"])
        cons = np.where(take, a["cons"], b["cons"])
        enabled = np.where(take, a["enabled"], b["enabled"])
        kt = rng.random(a["knots"].shape[0]) < 0.5
        knots = np.where(kt[:, None], a["knots"], b["knots"])
        blend = rng.random(len(kt)) < 0.3
        knots = np.where(blend[:, None], 0.5 * (a["knots"] + b["knots"]), knots)
        return dict(knots=fix_knots(knots.copy()), genes=genes.copy(), cons=cons.copy(), enabled=enabled.copy())

    def _mutate(self, ind, rng):
        F = ind["genes"].shape[1]
        ind["knots"] = fix_knots(ind["knots"] + rng.normal(0, 0.05, ind["knots"].shape) * (rng.random(ind["knots"].shape) < 0.3))
        m = rng.random(ind["genes"].shape) < 0.06
        ind["genes"] = np.where(m, rng.integers(0, 4, ind["genes"].shape), ind["genes"])
        ind["cons"] = np.where(rng.random(self.K) < 0.04, 1 - ind["cons"], ind["cons"])
        ind["enabled"] = np.where(rng.random(self.K) < 0.04, ~ind["enabled"], ind["enabled"])
        return ind

    # ---------- fitness ----------
    def _complexity(self, ind):
        F = ind["genes"].shape[1]
        act = ind["enabled"] & (ind["genes"] > 0).any(1)
        n_cond = (ind["genes"] > 0).sum(1)
        return float(((1 + n_cond) * act).sum() / (self.K * (1 + F)))

    def _evaluate(self, ind, X, y, prior):
        act = ind["enabled"] & (ind["genes"] > 0).any(1)
        S = rule_strengths(memberships(X, ind["knots"]), ind["genes"], act)
        s = risk_from_strengths(S, ind["cons"], prior)
        auc = roc_auc_score(y, s)
        comp = self._complexity(ind)
        return auc - self.lam * comp, auc, comp

    def _tournament(self, pop, fits, rng):
        idx = rng.choice(len(pop), self.tournament, replace=False)
        return pop[idx[np.argmax(fits[idx])]]

    # ---------- main loop ----------
    def fit(self, X, y, feature_names):
        rng = np.random.default_rng(self.seed)
        X, y = np.asarray(X, float), np.asarray(y, int)
        if len(y) > self.max_train:                       # stratified subsample keeps GA fast
            keep = np.concatenate([rng.choice(np.where(y == c)[0],
                                   int(self.max_train * (y == c).mean()), replace=False) for c in (0, 1)])
            X, y = X[keep], y[keep]
        F, prior = X.shape[1], y.mean()
        pop = [self._new(rng, F, even=(i % 2 == 0)) for i in range(self.pop_size)]
        res = [self._evaluate(p, X, y, prior) for p in pop]
        fits = np.array([r[0] for r in res])
        self.history = []
        for g in range(self.generations):
            order = np.argsort(-fits)
            new_pop = [pop[i] for i in order[:self.elites]]
            new_res = [res[i] for i in order[:self.elites]]
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(pop, fits, rng)
                child = (self._crossover(p1, self._tournament(pop, fits, rng), rng)
                         if rng.random() < self.p_cross else {k: v.copy() for k, v in p1.items()})
                child = self._mutate(child, rng)
                new_pop.append(child)
                new_res.append(self._evaluate(child, X, y, prior))
            pop, res = new_pop, new_res
            fits = np.array([r[0] for r in res])
            b = int(np.argmax(fits))
            self.history.append(dict(gen=g, fitness=res[b][0], auc=res[b][1], complexity=res[b][2]))
        best = pop[int(np.argmax(fits))]
        model = FuzzyRuleBase(feature_names, best["knots"], best["genes"], best["cons"], best["enabled"], prior)
        if self.prune:
            model = self._prune(model, X, y)
        self.model_ = model
        return model

    def _prune(self, model, X, y, tol=0.002):
        """Greedy simplification: drop any rule whose removal costs < tol AUC."""
        base = roc_auc_score(y, model.score(X))
        changed = True
        while changed:
            changed = False
            for r in np.where(model.active_mask())[0]:
                trial = FuzzyRuleBase(model.feature_names, model.knots, model.genes, model.cons,
                                      model.enabled & (np.arange(len(model.enabled)) != r), model.prior)
                if trial.n_rules() and roc_auc_score(y, trial.score(X)) >= base - tol:
                    model, changed = trial, True
                    base = max(base, roc_auc_score(y, model.score(X)))
        return model
