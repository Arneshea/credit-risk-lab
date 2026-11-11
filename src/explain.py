"""Turn a fired fuzzy rule base into a plain-language explanation for one applicant."""
import numpy as np
from .fuzzy import SETS


def _fmt(meta, f, v):
    mult, tpl = meta["unit"][f]
    return tpl.format(v * mult)


def _dominant_set(model, f_idx, v):
    from .fuzzy import memberships
    m = memberships(np.array([[v]]).repeat(len(model.feature_names), 1), model.knots)[0, f_idx, 1:]
    return SETS[int(np.argmax(m))]


def explain(model, x, meta, thr, good_values, min_strength=0.02, top_rules=3, top_tips=2):
    x = np.asarray(x, float)
    risk = float(model.score(x)[0])
    S = model.strengths(x)
    order = [r for r in np.argsort(-S) if S[r] >= min_strength][:top_rules]

    reasons = []
    for r in order:
        conds = " and ".join(f"{meta['labels'][f]} is {s}" for f, s in model.rule_conditions(r))
        verdict = "raises" if model.cons[r] == 1 else "lowers"
        reasons.append(dict(text=f"Because {conds} - this {verdict} your risk.",
                            strength=float(S[r]), rule=model.rule_text(r, meta["labels"]),
                            supports_default=bool(model.cons[r] == 1)))

    tips = []
    for i, f in enumerate(model.feature_names):
        d = meta["direction"].get(f)
        if d is None:
            continue
        best = good_values[f]
        worse = x[i] > best if d == "lower" else x[i] < best
        if not worse:
            continue
        x2 = x.copy(); x2[i] = best
        new = float(model.score(x2)[0])
        if risk - new > 0.01:
            tips.append(dict(feature=f, gain=risk - new, new_risk=new,
                             text=f"If your {meta['labels'][f]} improved from {_fmt(meta, f, x[i])} "
                                  f"to about {_fmt(meta, f, best)}, your risk would drop from "
                                  f"{risk:.2f} to {new:.2f}."))
    tips = sorted(tips, key=lambda t: -t["gain"])[:top_tips]

    return dict(risk=risk, credit_score=int(round(100 * (1 - risk))),
                decision="Approve" if risk < thr else "Refer for review",
                reasons=reasons, tips=tips)


def format_text(e):
    lines = [f"Credit score: {e['credit_score']}/100  (risk {e['risk']:.2f})  ->  {e['decision']}",
             "Main reasons:"]
    lines += [f"  - {r['text']}  [rule strength {r['strength']:.2f}]" for r in e["reasons"]] or ["  - (no strong rule fired)"]
    if e["tips"]:
        lines.append("What would help most:")
        lines += [f"  - {t['text']}" for t in e["tips"]]
    return "\n".join(lines)
