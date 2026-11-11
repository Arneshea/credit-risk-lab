"""Fuzzy rule base: shared Low/Medium/High partitions + IF-THEN rules.

Descriptive fuzzy system (in the sense of Hoffmann et al. 2007): every rule uses the
*same* linguistic sets per feature, so rules read like plain sentences.
"""
import numpy as np

SETS = ["Low", "Medium", "High"]          # gene value 0 = 'feature not used in this rule'


def memberships(X, knots):
    """X (n,F) in [0,1]; knots (F,3) = k1<k2<k3.  Returns M (n,F,4); M[...,0]=1 (don't care)."""
    k1, k2, k3 = knots[:, 0], knots[:, 1], knots[:, 2]
    eps = 1e-9
    low = np.clip((k2 - X) / (k2 - k1 + eps), 0, 1)
    high = np.clip((X - k2) / (k3 - k2 + eps), 0, 1)
    med = np.where(X <= k2, np.clip((X - k1) / (k2 - k1 + eps), 0, 1),
                   np.clip((k3 - X) / (k3 - k2 + eps), 0, 1))
    M = np.ones(X.shape + (4,))
    M[..., 1], M[..., 2], M[..., 3] = low, med, high
    return M


def rule_strengths(M, genes, active):
    """genes (K,F) ints 0..3; active (K,) bool -> firing strengths (n,K), AND = min."""
    F = genes.shape[1]
    G = M[:, np.arange(F)[None, :], genes]       # (n,K,F)
    return G.min(axis=2) * active[None, :]


def risk_from_strengths(S, cons, prior, eps=0.05):
    """Risk in [0,1]: share of firing strength that supports 'default'. Falls back to the prior."""
    d = (S * (cons == 1)[None, :]).sum(1)
    g = (S * (cons == 0)[None, :]).sum(1)
    return (d + eps * prior) / (d + g + eps)


class FuzzyRuleBase:
    def __init__(self, feature_names, knots, genes, cons, enabled, prior):
        self.feature_names = list(feature_names)
        self.knots = np.array(knots, float)
        self.genes = np.array(genes, int)
        self.cons = np.array(cons, int)
        self.enabled = np.array(enabled, bool)
        self.prior = float(prior)

    def active_mask(self):
        return self.enabled & (self.genes > 0).any(1)

    def score(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        S = rule_strengths(memberships(X, self.knots), self.genes, self.active_mask())
        return risk_from_strengths(S, self.cons, self.prior)

    def strengths(self, x):
        x = np.atleast_2d(np.asarray(x, float))
        return rule_strengths(memberships(x, self.knots), self.genes, self.active_mask())[0]

    def rule_conditions(self, r):
        return [(self.feature_names[f], SETS[g - 1]) for f, g in enumerate(self.genes[r]) if g > 0]

    def rule_text(self, r, labels=None):
        labels = labels or {}
        conds = " AND ".join(f"{labels.get(f, f)} is {s}" for f, s in self.rule_conditions(r))
        return f"IF {conds} THEN {'HIGH default risk' if self.cons[r] == 1 else 'LOW default risk'}"

    def n_rules(self):
        return int(self.active_mask().sum())

    def avg_conditions(self):
        a = self.active_mask()
        return float((self.genes[a] > 0).sum(1).mean()) if a.any() else 0.0

    def to_dict(self):
        return dict(feature_names=self.feature_names, knots=self.knots.tolist(),
                    genes=self.genes.tolist(), cons=self.cons.tolist(),
                    enabled=self.enabled.tolist(), prior=self.prior)

    @classmethod
    def from_dict(cls, d):
        return cls(d["feature_names"], d["knots"], d["genes"], d["cons"], d["enabled"], d["prior"])
