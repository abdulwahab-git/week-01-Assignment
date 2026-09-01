# Capstone Report — Organic Search Decline Prediction

- **Author:** Abdul Wahab Baig
- **Lane:** Machine Learning Intern
- **Repo:** abdulwahab-git/week-01-Assignment
- **Date:** July 2026

## 1. Problem framing

The unit of analysis is one **(client, content page)** pair. The output is a continuous
**risk score** (0–1) that a Random Forest assigns to each page, converted into a ranked
review queue with three tiers: `CRITICAL_URGENT_SALVAGE`, `OPTIMIZE_CONTENT_FOOTPRINT`, and
`MONITOR_STABLE`. The action a human takes: an SEO/content specialist reviews pages at or
above the `OPTIMIZE_CONTENT_FOOTPRINT` threshold (score ≥ 0.50) and decides whether to
refresh, redirect, or leave them.

Cost of a wrong call: a false alarm wastes limited editorial review time on a page that
didn't need it; a missed decline (false negative) lets organic visibility keep eroding
undetected until it's much harder to recover. ML helps here because the signal isn't a
single fixed rule — page performance depends on interacting factors (prior volume, tracking
consistency, client) that are impractical to hand-encode as nested `if` statements, and a
learned ranking model consistently beats a flat rule at surfacing the pages worth reviewing
first.

## 2. Data safety

**Used:** `fact_content_daily_performance` (2026 monthly Parquet partitions) from the public
FlyRank ML Internship warehouse, aggregated to `impressions_last_90d`, `impressions_prior`,
and `active_days_count` per client/content pair.

**Deliberately excluded:**
- `client_hash_id` / `content_hash_id` — used only for grouping/identification, never as
  model features.
- `trend_direction` and `trend_pct` (present in the earlier starter dataset explored in
  Week 1–2) — these are label-derived from the same signal the model is meant to predict,
  so using them would be leakage.
- `impressions_last_90d` itself is used only to *construct* the label (`action_score`,
  `is_decline_target`), not as an independent input feature — the model only sees
  `impressions_prior` and `active_days_count`.

No client names, raw URLs, or private search queries appear anywhere in this report or the
committed notebooks — all client and content identifiers are pre-hashed by the warehouse.

## 3. Baseline

Two baselines were used at different stages:

- **Rule-based baseline (Week 4):** `score = impressions_90d × (1 + clicks_90d)`, ranking
  pages by existing visibility/engagement. This is fully transparent (no fitted weights) and
  established that raw visibility volume is a reasonable starting signal, but it can't
  distinguish declining pages from simply high-traffic ones.
- **Majority-class baseline (capstone, reported below):** predicts the training set's most
  common label for every evaluation row. On the client-grouped holdout it scores 28.17%
  accuracy / 28.17% precision / 100% recall / 43.96% F1 — the 100% recall is a trivial
  artifact of always predicting the majority class, not a meaningful signal. This is the fair
  comparison because it's evaluated on the exact same split as the model.

## 4. Model / analysis

**Method:** Random Forest classifier (100 trees, max depth 8, seed 42). Chosen because the
underlying tabular features are skewed and heavy-tailed (confirmed in the Week-4/5 signal
audit), and tree ensembles handle that kind of non-linearity and outlier behavior without
manual transforms.

**Features:** `impressions_prior` (historical impressions before the most recent 90-day
window) and `active_days_count` (distinct reporting days observed). Nothing else — a
deliberately small, leakage-checked feature set.

**Target:** `action_score = impressions_last_90d / (impressions_prior + 1)`; a page is a
positive "decline" case when `action_score < 0.8`, i.e. its most recent 90-day impressions
fell below 80% of its prior baseline.

## 5. Evaluation

**Split:** client-grouped 80/20 — all pages for a given client sit entirely in train or
entirely in eval, never split across both. This matters because a plain random split lets
the model see some of a client's pages in training and other pages from the *same* client in
test, letting it partly memorize client-specific quirks instead of learning a pattern that
generalizes. That difference is measurable, not theoretical: the same model scored **76.58%
accuracy under a random split** vs. **66.54% under the client-grouped split** — roughly a
10-point drop once client leakage is removed. The grouped number is the one reported as the
honest estimate.

**Metrics (model vs. baseline, same client-grouped holdout):**

| Metric | Majority-class baseline | Random Forest |
|---|---:|---:|
| Accuracy | 28.17% | 66.54% |
| Precision | 28.17% | 42.85% |
| Recall | 100.00% | 56.22% |
| F1-score | 43.96% | 48.63% |

Base rate for context: the positive ("decline") class is ~28% of the evaluation set, which
is exactly the majority-class baseline's accuracy — a useful sanity check that the baseline
isn't accidentally inflated.

**Error analysis:** The model's precision dips on pages with very high historical impressions
that see a short, strong seasonal spike — normal post-holiday or post-peak fall-off can look
like a programmatic decline to the model. Separately, 8,533 evaluation-set pages carry a
decline label but have zero prior-period impressions, meaning the model has no real baseline
to compare against for them; these are flagged for manual exclusion rather than trusted to
the automated queue.

## 6. Interpretation

Feature importances are lopsided: `active_days_count` accounts for **~87%** of the model's
decision weight, `impressions_prior` for **~13%**. In plain words, the model leans mainly on
how consistently a page has been tracked/active over time, and only secondarily on how much
historical traffic it had. This is a useful finding but also a caution: a page that looks
"at risk" may partly be reflecting sparse tracking history rather than a genuine visibility
problem, which is why edge cases with little tracking history are routed to manual review
rather than the automated queue.

No causal claims are made or supported by this design — the client-grouped holdout only
tells us the model's pattern generalizes to unseen clients, not that any recommended action
(a content refresh) will causally improve a page's performance.

## 7. Recommendation

Each page's model probability becomes a risk score used to rank a review queue in three
tiers:

- **`CRITICAL_URGENT_SALVAGE`** (score ≥ 0.80) — highest priority, review immediately.
- **`OPTIMIZE_CONTENT_FOOTPRINT`** (0.50 ≤ score < 0.80) — elevated-risk, review soon.
- **`MONITOR_STABLE`** (score < 0.50) — stays in the watch pool, no immediate action.

An SEO/content specialist would pull the exported queue (`final_action_playbook.csv`), start
at the top, and sign off before any change is made — anything scoring ≥ 0.50 requires human
sign-off by design. **Confidence:** directional and decision-support only, ~66.5% accuracy on
unseen clients. **Limits:** unreliable during search-algorithm updates, site migrations, or
strong seasonal swings, and unreliable for pages with little or no prior tracking history
(the 8,533 edge cases above) — these should be manually reviewed, not queued automatically.

## 8. Reproducibility

**Re-run from a fresh clone:**
```
git clone https://github.com/abdulwahab-git/week-01-Assignment
cd week-01-Assignment
# open work/notebooks/capstone.ipynb in Colab or Jupyter
# Runtime → Run all (prompts for a Hugging Face READ token for FlyRank/internship-warehouse)
```
**Seed:** `random_state=42` (Random Forest and any shuffling). **Model:** `RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)`
from `scikit-learn`. **Key dependency:** `duckdb` (10GB memory limit set explicitly) for
querying the Hugging Face-hosted Parquet warehouse directly, plus `pandas`. Outputs
(`final_action_playbook.csv`) are written to `work/outputs/` and are treated as regenerable
artifacts, not source-controlled data.

---

> **Claims checklist before submitting:** observed / measured / directional / decision-support
> language everywhere · no causal claims without an experiment or causal design · no
> "predicted Google's algorithm" · no client-identifying details · numbers in this report
> match a fresh re-run.
