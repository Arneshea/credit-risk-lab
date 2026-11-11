"""Genetic-search initialized neuro-fuzzy credit-default classifier.

The evolutionary stage searches rule antecedents and Gaussian membership sets.
The neural stage then uses backpropagation to tune the membership centers,
widths, and zero-order Sugeno consequent logits with binary cross-entropy.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


LABELS = ("Low", "Medium", "High")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35, 35)))


def _firing(X, genes, centers, widths):
    """Return normalized Gaussian rule firing strengths for rows in X."""
    log_memberships = -0.5 * ((X[:, :, None] - centers[None, :, :])
                              / widths[None, :, :]) ** 2
    log_memberships = np.maximum(log_memberships, -40.0)
    # Antecedents encode 0 as "ignore" and 1..3 as Low/Medium/High,
    # while the membership tensor uses zero-based indices 0..2.
    membership_index = np.maximum(genes - 1, 0)
    selected = log_memberships[:, np.arange(X.shape[1])[None, :], membership_index]
    log_strength = np.where(genes[None, :, :] > 0, selected, 0.0).sum(axis=2)
    log_strength -= log_strength.max(axis=1, keepdims=True)
    strength = np.exp(log_strength)
    return strength / np.maximum(strength.sum(axis=1, keepdims=True), 1e-12)


def _force_active(genes, rng):
    """Avoid empty rules, which otherwise act as duplicate global priors."""
    for row in genes:
        if not np.any(row):
            row[rng.integers(0, len(row))] = rng.integers(1, 4)
    return genes


def _new_individual(rng, n_features, n_rules, centered):
    genes = np.zeros((n_rules, n_features), dtype=np.int8)
    for rule in range(n_rules):
        count = int(rng.integers(1, min(4, n_features + 1)))
        features = rng.choice(n_features, count, replace=False)
        genes[rule, features] = rng.integers(1, 4, size=count)
    if centered:
        centers = np.tile(np.array([0.16, 0.5, 0.84]), (n_features, 1))
        centers += rng.normal(0.0, 0.07, centers.shape)
    else:
        centers = np.sort(rng.uniform(0.05, 0.95, (n_features, 3)), axis=1)
    centers = np.clip(centers, 0.01, 0.99)
    widths = np.clip(rng.normal(0.25, 0.07, (n_features, 3)), 0.08, 0.55)
    return {"genes": genes, "centers": centers, "widths": widths}


def _repair(individual, rng):
    individual["genes"] = np.clip(individual["genes"], 0, 3).astype(np.int8)
    _force_active(individual["genes"], rng)
    individual["centers"] = np.clip(individual["centers"], 0.01, 0.99)
    order = np.argsort(individual["centers"], axis=1)
    individual["centers"] = np.take_along_axis(individual["centers"], order, axis=1)
    individual["widths"] = np.take_along_axis(individual["widths"], order, axis=1)
    individual["widths"] = np.clip(individual["widths"], 0.06, 0.65)
    return individual


def _initial_consequents(X, y, individual, ridge=0.08):
    firing = _firing(X, individual["genes"], individual["centers"], individual["widths"])
    lhs = firing.T @ firing + ridge * np.eye(firing.shape[1])
    rhs = firing.T @ y
    fitted = np.linalg.solve(lhs, rhs)
    fitted = np.clip(fitted, 0.015, 0.985)
    return np.log(fitted / (1.0 - fitted))


def _complexity(genes):
    conditions = (genes > 0).sum(axis=1)
    return float((len(conditions) + conditions.sum())
                 / (len(conditions) * (genes.shape[1] + 1)))


def _evaluate(X, y, individual, penalty):
    consequent = _initial_consequents(X, y, individual)
    probabilities = _sigmoid(_firing(X, individual["genes"], individual["centers"],
                                     individual["widths"]) @ consequent)
    auc = roc_auc_score(y, probabilities)
    score = auc - penalty * _complexity(individual["genes"])
    return float(score), float(auc), consequent


def _crossover(parent_a, parent_b, rng):
    n_rules = len(parent_a["genes"])
    choose_a = rng.random(n_rules) < 0.5
    genes = np.where(choose_a[:, None], parent_a["genes"], parent_b["genes"])
    choose_a_feature = rng.random(parent_a["centers"].shape[0]) < 0.5
    centers = np.where(choose_a_feature[:, None], parent_a["centers"], parent_b["centers"])
    widths = np.where(choose_a_feature[:, None], parent_a["widths"], parent_b["widths"])
    blend = rng.random(len(choose_a_feature)) < 0.25
    centers = np.where(blend[:, None], 0.5 * (parent_a["centers"] + parent_b["centers"]), centers)
    widths = np.where(blend[:, None], 0.5 * (parent_a["widths"] + parent_b["widths"]), widths)
    return {"genes": genes, "centers": centers.copy(), "widths": widths.copy()}


def _mutate(individual, rng):
    genes = individual["genes"]
    gene_mask = rng.random(genes.shape) < 0.075
    genes[gene_mask] = rng.integers(0, 4, size=int(gene_mask.sum()))
    center_mask = rng.random(individual["centers"].shape) < 0.24
    individual["centers"] += center_mask * rng.normal(0.0, 0.06, individual["centers"].shape)
    width_mask = rng.random(individual["widths"].shape) < 0.2
    individual["widths"] += width_mask * rng.normal(0.0, 0.04, individual["widths"].shape)
    return _repair(individual, rng)


def _evolve(X, y, seed, n_rules=8, population_size=24, generations=18, penalty=0.045):
    rng = np.random.default_rng(seed)
    population = [_new_individual(rng, X.shape[1], n_rules, i % 2 == 0)
                  for i in range(population_size)]
    evaluated = [_evaluate(X, y, item, penalty) for item in population]
    fitness = np.array([entry[0] for entry in evaluated])
    best_history = []

    def tournament():
        idx = rng.choice(population_size, size=3, replace=False)
        return population[int(idx[np.argmax(fitness[idx])])]

    for _ in range(generations):
        order = np.argsort(-fitness)
        next_population = [population[i] for i in order[:2]]
        next_evaluated = [evaluated[i] for i in order[:2]]
        while len(next_population) < population_size:
            first = tournament()
            second = tournament()
            child = _crossover(first, second, rng)
            child = _mutate(child, rng)
            next_population.append(child)
            next_evaluated.append(_evaluate(X, y, child, penalty))
        population, evaluated = next_population, next_evaluated
        fitness = np.array([entry[0] for entry in evaluated])
        best = int(np.argmax(fitness))
        best_history.append(evaluated[best][1])

    best = int(np.argmax(fitness))
    individual = population[best]
    consequent = evaluated[best][2]
    return individual, consequent, best_history


def _backprop(X, y, genes, centers, widths, consequent, seed,
              epochs=70, batch_size=512, learning_rate=0.018):
    """Adam backprop for Gaussian premise parameters and Sugeno rule logits."""
    rng = np.random.default_rng(seed)
    params = [centers.copy(), widths.copy(), consequent.copy()]
    first = [np.zeros_like(value) for value in params]
    second = [np.zeros_like(value) for value in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    n_steps = 0
    n_rows, n_features = X.shape
    selected_masks = [genes[:, f] > 0 for f in range(n_features)]
    selected_sets = [genes[:, f] - 1 for f in range(n_features)]

    for _ in range(epochs):
        order = rng.permutation(n_rows)
        for start in range(0, n_rows, batch_size):
            idx = order[start:start + batch_size]
            xb, yb = X[idx], y[idx]
            firing = _firing(xb, genes, params[0], params[1])
            logits = firing @ params[2]
            predicted = _sigmoid(logits)
            delta = (predicted - yb) / max(len(yb), 1)
            grad_consequent = firing.T @ delta
            grad_log_firing = (firing * (params[2][None, :] - logits[:, None])
                               * delta[:, None])
            grad_centers = np.zeros_like(params[0])
            grad_widths = np.zeros_like(params[1])
            for feature in range(n_features):
                used = selected_masks[feature]
                if not np.any(used):
                    continue
                sets = selected_sets[feature][used]
                diff = xb[:, feature, None] - params[0][feature, sets][None, :]
                sigma = params[1][feature, sets][None, :]
                np.add.at(grad_centers[feature], sets,
                          np.sum(grad_log_firing[:, used] * diff / (sigma ** 2), axis=0))
                np.add.at(grad_widths[feature], sets,
                          np.sum(grad_log_firing[:, used] * (diff ** 2) / (sigma ** 3), axis=0))
            gradients = [grad_centers, grad_widths, grad_consequent]
            n_steps += 1
            for index, grad in enumerate(gradients):
                first[index] = beta1 * first[index] + (1.0 - beta1) * grad
                second[index] = beta2 * second[index] + (1.0 - beta2) * (grad ** 2)
                m_hat = first[index] / (1.0 - beta1 ** n_steps)
                v_hat = second[index] / (1.0 - beta2 ** n_steps)
                params[index] -= learning_rate * m_hat / (np.sqrt(v_hat) + eps)
            params[0] = np.clip(params[0], 0.01, 0.99)
            order_sets = np.argsort(params[0], axis=1)
            params[0] = np.take_along_axis(params[0], order_sets, axis=1)
            params[1] = np.take_along_axis(params[1], order_sets, axis=1)
            first[0] = np.take_along_axis(first[0], order_sets, axis=1)
            second[0] = np.take_along_axis(second[0], order_sets, axis=1)
            first[1] = np.take_along_axis(first[1], order_sets, axis=1)
            second[1] = np.take_along_axis(second[1], order_sets, axis=1)
            params[1] = np.clip(params[1], 0.06, 0.65)
            params[2] = np.clip(params[2], -8.0, 8.0)
    return params


class GeneticNeuroFuzzyClassifier:
    """GA search followed by neural backprop in a zero-order Sugeno classifier."""

    def __init__(self, seed=0, n_rules=8, population_size=24, generations=18,
                 backprop_epochs=70, penalty=0.045, max_evolution_rows=3500):
        self.seed = seed
        self.n_rules = n_rules
        self.population_size = population_size
        self.generations = generations
        self.backprop_epochs = backprop_epochs
        self.penalty = penalty
        self.max_evolution_rows = max_evolution_rows

    def fit(self, X, y, feature_names):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        rng = np.random.default_rng(self.seed)
        if len(y) > self.max_evolution_rows:
            parts = []
            for label in (0, 1):
                ids = np.flatnonzero(y == label)
                take = min(len(ids), max(1, round(self.max_evolution_rows * len(ids) / len(y))))
                parts.append(rng.choice(ids, size=take, replace=False))
            evolution_idx = np.concatenate(parts)
            rng.shuffle(evolution_idx)
        else:
            evolution_idx = np.arange(len(y))

        individual, consequent, history = _evolve(
            X[evolution_idx], y[evolution_idx], self.seed, self.n_rules,
            self.population_size, self.generations, self.penalty,
        )
        centers, widths, consequent = _backprop(
            X, y, individual["genes"], individual["centers"], individual["widths"],
            consequent, self.seed + 1, epochs=self.backprop_epochs,
        )
        self.feature_names_ = list(feature_names)
        self.genes_ = individual["genes"].copy()
        self.centers_ = centers
        self.widths_ = widths
        self.consequent_ = consequent
        self.evolution_history_ = history
        self.n_rules_ = int(len(self.genes_))
        self.avg_conditions_ = float((self.genes_ > 0).sum(axis=1).mean())
        self.evolution_rows_ = int(len(evolution_idx))
        self.generations_ = int(self.generations)
        self.backprop_epochs_ = int(self.backprop_epochs)
        return self

    def decision_function(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return _firing(X, self.genes_, self.centers_, self.widths_) @ self.consequent_

    def predict_proba(self, X):
        risk = _sigmoid(self.decision_function(X))
        return np.column_stack([1.0 - risk, risk])

    def score(self, X):
        return self.predict_proba(X)[:, 1]

    def rules(self, reference_default_rate=0.22):
        output = []
        for rule_index, genes in enumerate(self.genes_):
            conditions = [f"{self.feature_names_[feature]} is {LABELS[gene - 1]}"
                          for feature, gene in enumerate(genes) if gene > 0]
            contribution = float(_sigmoid(np.array([self.consequent_[rule_index]]))[0])
            risk_label = "elevated" if contribution >= reference_default_rate else "lower"
            output.append({
                "text": f"IF {' AND '.join(conditions)} THEN {risk_label} default-risk contribution",
                "conditions": len(conditions),
                "risk_contribution": contribution,
            })
        return output
