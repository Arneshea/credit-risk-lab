"""Reproducible held-out comparison for the Neuro-Fuzzy Soft Computing Lab UI."""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

from .data import META, prepare
from .neuro_fuzzy import GeneticNeuroFuzzyClassifier


_CACHE = {}


def _best_threshold(y, scores):
    grid = np.quantile(scores, np.linspace(0.05, 0.95, 91))
    balanced = [((scores[y == 1] >= threshold).mean()
                 + (scores[y == 0] < threshold).mean()) / 2 for threshold in grid]
    return float(grid[int(np.argmax(balanced))])


def _metrics(y, scores, threshold):
    predicted = scores >= threshold
    positive = y == 1
    negative = ~positive
    tp = int((predicted & positive).sum())
    fp = int((predicted & negative).sum())
    fn = int((~predicted & positive).sum())
    tn = int((~predicted & negative).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "auc": float(roc_auc_score(y, scores)),
        "balanced_accuracy": float((recall + specificity) / 2),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "threshold": float(threshold),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def run_comparison(dataset="taiwan"):
    """Train GA initialized neuro-fuzzy and ANN models on the same seed-0 split."""
    if dataset not in ("taiwan", "german"):
        raise ValueError("dataset must be 'taiwan' or 'german'")
    if dataset in _CACHE:
        return _CACHE[dataset]

    split = prepare(dataset, seed=0)
    labels = META[dataset]["labels"]
    display_features = [labels[name] for name in split["features"]]

    neuro_fuzzy = GeneticNeuroFuzzyClassifier(
        seed=0, n_rules=8, population_size=24, generations=18,
        backprop_epochs=70, penalty=0.045,
    ).fit(split["Xtr"], split["ytr"], display_features)
    fuzzy_validation = neuro_fuzzy.score(split["Xva"])
    fuzzy_threshold = _best_threshold(split["yva"], fuzzy_validation)
    fuzzy_scores = neuro_fuzzy.score(split["Xte"])

    # Inputs are engineered onto [0, 1], so the ANN can use them directly.
    ann = MLPClassifier(
        hidden_layer_sizes=(24, 12), activation="relu", solver="adam",
        alpha=0.001, learning_rate_init=0.003, max_iter=300,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=18, random_state=42,
    )
    ann.fit(split["Xtr"], split["ytr"])
    ann_validation = ann.predict_proba(split["Xva"])[:, 1]
    ann_threshold = _best_threshold(split["yva"], ann_validation)
    ann_scores = ann.predict_proba(split["Xte"])[:, 1]

    fuzzy_auc = float(roc_auc_score(split["yte"], fuzzy_scores))
    rules = neuro_fuzzy.rules(reference_default_rate=float(np.mean(split["ytr"])))
    result = {
        "dataset": dataset,
        "dataset_name": "Taiwan Credit Card Default" if dataset == "taiwan" else "German Credit",
        "n_train": int(len(split["ytr"])),
        "n_validation": int(len(split["yva"])),
        "n_test": int(len(split["yte"])),
        "default_rate": float(np.mean(split["ytr"])),
        "features": display_features,
        "models": [
            {
                "key": "ga_anfis",
                "name": "GA-Optimized Neuro-Fuzzy System",
                "family": "Fuzzy inference + genetic search + neural backpropagation",
                "details": "GA evolves rule antecedents and Gaussian sets; backprop tunes memberships and zero-order Sugeno rule outputs.",
                "metrics": _metrics(split["yte"], fuzzy_scores, fuzzy_threshold),
                "rule_count": neuro_fuzzy.n_rules_,
                "avg_conditions": neuro_fuzzy.avg_conditions_,
                "evolution_generations": neuro_fuzzy.generations_,
                "backprop_epochs": neuro_fuzzy.backprop_epochs_,
                "evolution_auc": neuro_fuzzy.evolution_history_[-1],
                "rules": rules,
            },
            {
                "key": "ann",
                "name": "Artificial Neural Network",
                "family": "Multilayer perceptron baseline",
                "details": "MLP with 24 and 12 hidden units, early stopping, and a threshold chosen on validation data.",
                "metrics": _metrics(split["yte"], ann_scores, ann_threshold),
                "epochs": int(ann.n_iter_),
                "rules": [],
            },
        ],
        "method": {
            "split": "Stratified 60 / 20 / 20 train, validation, and held-out test split; seed 0.",
            "threshold": "Each model uses a threshold selected on validation data for balanced accuracy.",
            "neuro_fuzzy": "A genetic algorithm searches fuzzy antecedents and Gaussian membership sets. Backpropagation then tunes those memberships and Sugeno rule-output logits using binary cross-entropy.",
            "warning": "These datasets are not gig-worker or UPI datasets. Results demonstrate the algorithms, not real gig-worker lending performance.",
        },
    }
    _CACHE[dataset] = result
    return result
