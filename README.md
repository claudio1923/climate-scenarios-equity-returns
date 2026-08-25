# Stock market returns under different climate scenarios

Is the Green–Brown return differential *within the same sector* a structural feature of a
portfolio, or does it depend on which climate scenario the economy follows? This repository
contains the Python replication of the modelling pipeline built to answer that question, together
with the main results of the thesis.

## Replication note

The estimation in the thesis was done in MATLAB. The code here is a Python replication of that
pipeline: same panel, same feature-engineering rule, same 73-feature winning set, same Gradient
Boosting configuration, refit with scikit-learn. The replication results track the thesis closely
but do not match it exactly, because the two Gradient Boosting implementations differ in how they
search splits, break ties and fit leaves. Both sets of numbers are reported side by side and are
never mixed: figures and tables state which pipeline produced them, and the replication values are
always labelled as such. Where the replication does *not* reproduce a thesis finding, this is
stated in the text rather than smoothed over — see [What the model finds](#4-what-the-model-finds).

## Setup

22 US sector portfolios (11 GICS sectors, each split into an ESG top-25% "Green" and a bottom-25%
"Brown" leg), monthly data 2005–2024. Four models are compared out-of-sample on a sealed
2021–2024 test block; the winner is projected on the NGFS scenarios to 2050; the final metric is
the within-sector Green/Brown log-ratio.

## Repository layout

```
src/
  build_features.py   24 drivers -> 552 candidates -> 73-feature winning set,
                      validated column by column against the MATLAB design matrix
  train_gb.py         refit of the winning Gradient Boosting configuration
  evaluate.py         out-of-sample R2 and RMSE on the sealed test block
  interpret.py        PDP and ICE curves for the six defence pairs
  scenarios.py        2025-2050 projection and Green/Brown log-ratio
  matlab_policy_gb.py LSBoost with MATLAB's tree growth policy, in numpy:
                      breadth-first growth under a split budget, with the
                      per-level undo rule that scikit-learn does not offer
scripts/
  make_figures.py     regenerates everything in figures/
  make_tables.py      builds results/mean_differences_table.csv
  verify_237_months.py  reads the four .mat inputs directly and compares the
                      189-month and 237-month refits
  phase0_diagnostics.py  importance fingerprint, capacity probe, extrapolation
  compare_growth_policies.py  the three growth policies side by side
  task1_bestfirst_fitb.py     best-first upper bound on the projection
  task2_matlab_policy.py      the MATLAB policy on both samples
results/              thesis exports + replication outputs
figures/              all figures used below
```

Run order, from the repository root:

```bash
python src/build_features.py && python src/train_gb.py && python src/evaluate.py && python src/interpret.py && python src/scenarios.py && python scripts/make_tables.py && python scripts/make_figures.py
```

The `src/` steps need `data_private/`, which is not distributed (see
[Data availability](#data-availability)). `scripts/make_tables.py` and the thesis-based figures run
from the CSVs in `results/` alone.

### Regenerable outputs

Four monthly projection dumps are excluded from version control, at roughly 14 MB each. They are
deterministic, so re-running the script that wrote them reproduces them byte for byte:

| File | Recreated by |
| --- | --- |
| `results/verifica_237_mesi/scenario_monthly_fitA.csv` | `python scripts/verify_237_months.py` |
| `results/verifica_237_mesi/scenario_monthly_fitB.csv` | `python scripts/verify_237_months.py` |
| `results/task1/scenario_monthly_fitB_bestfirst.csv` | `python scripts/task1_bestfirst_fitb.py` |
| `results/task2/scenario_monthly_fitB_matlabpolicy.csv` | `python scripts/task2_matlab_policy.py` |

The aggregated log-ratio files and the text reports in the same folders are small and stay tracked,
so the conclusions are readable without regenerating anything. Note that these four scripts read the
`.mat` inputs directly and therefore need the private data.

The MATLAB exports in `results/` are a different matter and stay tracked:
`results/logratio_green_brown.csv` in particular is the reference the replication is compared
against, and it cannot be rebuilt without MATLAB.

---

## 1. Why machine learning is necessary, not decorative

The linear panel cannot answer the research question, by construction. In that specification Green
enters only as an intercept shift (δ ≈ −0.63) with no interaction terms. Projected on the five
scenarios, it returns the same differential in all five. A linear model of this form can only ever
say "structural"; it has no way to represent a scenario-dependent answer.

The differential lives in the interactions. Any effect common to both legs of a sector cancels in
the log-ratio, so only a driver tied to one specific portfolio can open a gap. That is what the
candidate space is built for: 552 features, made of 24 driver-lags interacted with 22 entities plus
the 24 main effects. The panel is split 70-10-20, with the last block sealed and untouched until
the final evaluation.

`src/build_features.py` rebuilds this space from the base panel and checks itself against the
matrix built in MATLAB: on the tr70 + va10 sample the two agree column by column, with a maximum
absolute difference of 0.

## 2. Model selection by four criteria, not by R²

![Out-of-sample R2 by model](figures/fig_model_comparison.png)

The four models are nearly tied out-of-sample: Elastic Net 0.389, Random Forest 0.401, Panel 0.404,
Gradient Boosting 0.406. A gap of 0.017 across four very different specifications is not a ranking.

The choice was made by progressive elimination on four criteria:

| Criterion | Consequence |
|---|---|
| No standardisation required | rules out Elastic Net |
| Non-linearity needed | rules out Elastic Net and the panel |
| Determinism | rules out Random Forest |
| Parsimony | 73 features against 155 |

Gradient Boosting is the only model not dominated on any of the four. Its configuration — 73
features, depth 4, learning rate 0.03, 300 trees, no subsampling — comes from the thesis and is
refit here without any further tuning. Fit is confirmation, not reason.

**Replication against thesis, sealed 2021–2024 test block:**

| Metric | Thesis (MATLAB) | Replication (Python) |
|---|---|---|
| R² out-of-sample | 0.406 | 0.4104 |
| RMSE out-of-sample | 5.108 | 5.0913 |
| R² in-sample | 0.7065 | 0.6503 |
| RMSE in-sample | 3.8182 | 4.1679 |

The out-of-sample figures agree closely. The in-sample gap is the expected signature of a different
boosting implementation: scikit-learn fits the training sample somewhat less tightly at the same
nominal settings, which leaves the out-of-sample number essentially unchanged.

![Feature importance](figures/fig_feature_importance.png)

The importance ranking survives the change of implementation: the contemporaneous market factor
dominates in both (68.7% of the budget in the thesis, 74.4% in the replication), followed by the
same oil, market and temperature interactions in almost the same order.

## 3. Interpretability with a built-in falsification test

![Partial dependence, six pairs](figures/fig_pdp_2x3.png)

The partial dependence curves are step functions even in the channels that are close to linear —
the market channel has a linear R² of 0.95 and still produces steps. The same variable takes
different shapes on different portfolios: this is exactly the heterogeneity the linear panel cannot
see, because there the same coefficient is imposed everywhere.

![ICE curves, six pairs](figures/fig_ice_2x3.png)

The ICE curves are parallel. That matters: it means the average curve is representative of the
individual observations and not an artefact of averaging over conflicting shapes. If the individual
curves crossed, the PDP would be a summary of nothing.

Both figures are recomputed in Python on the replicated model, following the sweep logic of the
thesis: an interaction feature is non-zero only on the rows of its own entity, so the grid is
applied to those rows and the predictions are averaged over them.

The falsification test is the pivot between two sectors. Communication carries the heaviest
interaction budget — 24% of it, of which 13.8 sits on temperature — and it barely separates the
scenarios (range 0.11). Energy carries less — 16%, of which 14.3 sits on fuel — and it separates
everything (range 0.39). The reason is not in the model but in the drivers: temperature paths do
not diverge across scenarios by 2050, fuel paths do. What matters is the channel, not the weight.
When the model does not separate, it is because the driver does not separate. In the thesis model,
interactions take 23% of the total importance budget (19.9% in the replication).

## 4. What the model finds

![Energy trajectories](figures/fig_energy_trajectories.png)

Energy is the only sector where the scenario flips the sign of the cumulative log-ratio. In the
transition component at 2050: Net Zero +0.29 and Delayed transition +0.16 put the green leg ahead,
while NDCs −0.70, Below 2°C −0.82 and Fragmented World −0.93 put the brown leg ahead.

The ordering is not a simple ranking of ambition. It takes ambition *and* speed together: Net Zero
and Delayed transition have both, Below 2°C has ambition without speed, Fragmented World has speed
without ambition, NDCs has neither.

The left panel is the other half of the finding, and it is kept in the figure on purpose: under the
physical component the five scenarios lie exactly on top of each other. The Energy spread across
scenarios is 0.00 for physical against 1.22 for transition. All of the separation is transition
risk.

![Materials trajectories](figures/fig_materials_trajectories.png)

Materials behaves differently. The brown leg is ahead under every pathway; only the size of the gap
changes, from −1.67 under Net Zero (tightest) to −2.62 under NDCs (widest). The shock enters as a
cost rather than as an advantage: green does not become better, it becomes less bad.

### Where the replication diverges

The sign reversal in Energy does **not** fully reproduce in the Python replication, and the
difference is worth stating precisely:

| Scenario | Thesis (MATLAB) | Replication (Python) |
|---|---|---|
| Net Zero 2050 | 0.2854 | 0.3049 |
| Delayed transition | 0.1644 | 0.2484 |
| Below 2°C | −0.8242 | 0.0926 |
| NDCs | −0.7012 | 0.0502 |
| Fragmented World | −0.9345 | 0.0606 |

What survives: Energy is still the most scenario-sensitive sector in the replication, and Net Zero
and Delayed transition still sit clearly at the top. What does not survive: the three
low-ambition-or-low-speed scenarios stay slightly positive instead of turning negative, so the
spread compresses from 1.22 to 0.25 and the zero line is never crossed. Materials compresses much
further, from a spread of 0.95 to 0.02.

The mechanism is visible in the design file. The projected fuel path over 2025–2050 stays inside a
narrow band, roughly 0.04 to 1.85, which sits close to the median of the training distribution of
the same driver. A boosted-tree model is a step function, so what it returns over such a short
interval depends on exactly where its split points fall. The MATLAB model happens to place steps
inside that band and the scikit-learn model does not, which is enough to move the level of the
brown Energy leg without changing the ordering across scenarios. This is a real limitation of
projecting a step-function model over a narrow driver range, and it applies to the thesis model as
much as to the replication.

The risk-free path is not the explanation. The scenario design file carries no risk-free series, so
the replication compounds excess returns while the thesis compounds total returns; recomputing the
thesis log-ratio on excess returns alone moves the Energy 2050 value from −0.8242 to −0.8255, which
is negligible.

### Mean 2025–2050 differences, thesis results

Average over 2025–2050 of the monthly difference between the predicted return of the Green and the
Brown portfolio of each sector, by scenario, with the range across scenarios in the last column.
These come from the thesis output, not from the replication. No significance tests are reported
here; the tests are in the thesis.

**Transition component**

| Sector | Net Zero | Delayed | Below 2C | NDCs | Fragmented | Range |
|---|---|---|---|---|---|---|
| CDSC | 0.224 | 0.215 | 0.224 | 0.224 | 0.215 | 0.009 |
| COMM | -2.051 | -2.150 | -2.036 | -2.085 | -2.136 | 0.114 |
| CSTP | -0.246 | -0.225 | -0.246 | -0.231 | -0.231 | 0.021 |
| ENRG | 0.092 | 0.053 | -0.267 | -0.227 | -0.303 | 0.395 |
| FIN | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.001 |
| HLTC | -0.165 | -0.165 | -0.165 | -0.165 | -0.165 | 0.000 |
| IND | -0.112 | -0.108 | -0.112 | -0.112 | -0.108 | 0.004 |
| INFT | -0.623 | -0.648 | -0.594 | -0.616 | -0.622 | 0.054 |
| MATS | -0.544 | -0.660 | -0.545 | -0.856 | -0.660 | 0.312 |
| REIT | -0.012 | -0.016 | -0.012 | -0.014 | -0.014 | 0.004 |
| UTIL | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

**Physical component**

| Sector | Net Zero | Delayed | Below 2C | NDCs | Fragmented | Range |
|---|---|---|---|---|---|---|
| CDSC | 0.224 | 0.224 | 0.224 | 0.224 | 0.224 | 0.001 |
| COMM | -1.970 | -2.091 | -1.974 | -2.081 | -2.073 | 0.121 |
| CSTP | -0.246 | -0.231 | -0.246 | -0.231 | -0.231 | 0.016 |
| ENRG | -0.323 | -0.323 | -0.323 | -0.323 | -0.323 | 0.000 |
| FIN | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| HLTC | -0.165 | -0.165 | -0.165 | -0.165 | -0.165 | 0.000 |
| IND | -0.112 | -0.112 | -0.112 | -0.112 | -0.112 | 0.000 |
| INFT | -0.605 | -0.598 | -0.605 | -0.600 | -0.600 | 0.007 |
| MATS | -1.102 | -1.099 | -1.102 | -1.099 | -1.099 | 0.002 |
| REIT | -0.020 | -0.019 | -0.020 | -0.020 | -0.020 | 0.001 |
| UTIL | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Full table: [`results/mean_differences_table.csv`](results/mean_differences_table.csv).

## 5. Conclusion: conditional, but local

Out of 11 sectors, 9 are structural: the differential is what it is, whatever the pathway. Two
move. Only one — Energy — inverts its sign. The answer to the research question is therefore not
"structural" and not "conditional", but conditional in a specific place, through a specific
channel, and structural everywhere else.

## Limitations and future work

**Why the gains over the linear benchmark are modest.** Most of the predictable variance comes from
the contemporaneous market factor, which is linear, so the ceiling for improvement was low by
construction of the problem. Monthly equity returns have a very low signal-to-noise ratio, and
roughly 190 effective training months limit what trees can learn — consistent with what Gu, Kelly
and Xiu report for machine learning in asset pricing. Machine learning was chosen here for
expressiveness, not for fit: the linear model cannot even represent a scenario-dependent
differential, by construction. R² measures total-return prediction, while the object of interest is
a second-order differential that R² barely sees.

**Style factors are excluded.** No scenario path exists for them, so they cannot be projected. The
within-sector design attenuates the omission but does not remove it.

**The time dimension is narrow from both sides.** The data are monthly while the NGFS scenarios are
annual, and ESG coverage thins out before 2005.

**A single ESG provider.** Ratings diverge across providers, and the Green/Brown sorting inherits
that choice.

**Scenario paths are smooth by construction.** The projections should be read as directions of
adjustment, not as magnitudes. The divergence documented above is a concrete illustration: over a
narrow driver range, the level of a step-function projection is not robust across implementations,
even when the ordering across scenarios is.

**The two scenario-sensitive sectors depend on the design.** A richer model might find others.
Communication hints at why: its interaction budget sits on temperature, which does not separate
scenarios by 2050.

**Future work.** Longer or higher-frequency samples; penalised linear models with explicit
interaction terms as a sharper benchmark; news-based transition-risk indicators alongside physical
anomalies; neural approaches if more data becomes available; uncertainty quantification on the
scenario projections.

## Data availability

The underlying data are proprietary. Market and ESG data come from Datastream, macroeconomic series
from FRED, and climate variables from ISIMIP. None of them are redistributed here, and the folder
that holds them, `data_private/`, is excluded from version control.

What is published is the complete modelling code and the aggregated results: the thesis exports and
the replication outputs in `results/`, and the figures in `figures/`. The code in `src/` is
therefore readable end to end but not runnable without `data_private/`; every module fails with an
explicit message pointing to this section when an input file is missing. The scripts in `scripts/`
run from `results/` and need no private data.

### Provenance of the MATLAB reference

`results/logratio_green_brown.csv` is the file every comparison in this repository is measured
against: 165 rows by 312 monthly columns, 51,480 values, no missing cells. It is a direct MATLAB
export, not a file rebuilt along the way. The chain is
`s3_K62_leaf10.m` → `Scenario_Yhat_K62L10_2005_2024.mat` → `s4_cumulate_logratio.m` →
`Scenario_CumLogRatio_2005_2024.mat` → `writetable` in `export_highlights_data.m`. The copy in the
MATLAB working folder and the two copies in this repository are byte-identical, and no Python module
here writes that filename; they only read it.

One limit is worth stating plainly. The `.m` scripts are not under version control. The timestamps
of the `.mat` files are consistent with the chain above and with the scripts as they read today, but
consistency is not proof: nothing rules out that a script was edited after the `.mat` it produced.
The provenance is documented, not verified.

## Citation

Claudio Andreassi, PhD in Economics, University of Perugia (2026). Supervisor: Prof. Marco
Nicolosi.

## License

MIT — see [LICENSE](LICENSE).
