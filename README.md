# Stock market returns under different climate scenarios

Is the Green–Brown return differential *within the same sector* a structural feature of a
portfolio, or does it depend on which climate scenario the economy follows? This repository
contains the Python replication of the modelling pipeline built to answer that question, together
with the main results of the thesis.

## Replication note

The estimation in the thesis was done in MATLAB. The code here is a Python replication of that
pipeline: same panel, same feature-engineering rule, same 73-feature winning set, same Gradient
Boosting configuration, same two estimation windows.

The replication reproduces the thesis. On the sealed test block the in-sample figures agree to nine
decimals; over the 51,480 monthly log-ratio values of the scenario projection the largest deviation
from the MATLAB export is 1.4e-05. Two features of the pipeline carry most of the weight in that
agreement, and both are documented below: the projection comes from a second fit on a longer window,
and MATLAB's tree growth policy has no scikit-learn equivalent, so it is implemented directly in
[`src/matlab_policy_gb.py`](src/matlab_policy_gb.py).

Thesis values and replication values are labelled wherever both appear, and never mixed.

A word on what "reproduces" means here. Agreement to 1.4e-05 says the Python code walks the same
path as the MATLAB code and arrives where it arrives. It says nothing about how firmly that
destination is fixed, which is a separate question and one the thesis addresses in its own terms:
the scenario projections are to be read as directions of adjustment, not as calibrated magnitudes.

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
  train_gb.py         the two fits (189 months for metrics, 237 for the
                      projection) and the three growth policies
  evaluate.py         out-of-sample R2 and RMSE on the sealed test block
  interpret.py        PDP and ICE curves for the six defence pairs
  scenarios.py        2025-2050 projection and Green/Brown log-ratio
  matlab_policy_gb.py LSBoost with MATLAB's tree growth policy, in numpy:
                      breadth-first growth under a split budget, with the
                      per-level undo rule that scikit-learn does not offer
scripts/
  make_figures.py     regenerates everything in figures/
  make_tables.py      builds results/mean_differences_table.csv and
                      results/growth_policy_table.csv
  reexport_data_private.py    rewrites the private CSVs at %.17g so they
                      round-trip exactly; run once, before anything else
results/              thesis exports + replication outputs
figures/              all figures used below
```

Run order, from the repository root:

```bash
python src/build_features.py && python src/train_gb.py && python src/evaluate.py && python src/interpret.py && python src/scenarios.py && python scripts/make_tables.py && python scripts/make_figures.py
```

The `src/` steps need `data_private/`, which is not distributed (see
[Data availability](#data-availability)). The thesis-based figures and the mean-differences table
run from the CSVs in `results/` alone; the growth-policy table inside `make_tables.py` fits models
and therefore needs the private inputs, and is skipped with a message when they are absent.

The MATLAB exports in `results/` cannot be rebuilt without MATLAB and stay tracked.
`results/logratio_green_brown.csv` in particular is the reference every comparison here is measured
against; its provenance is described under [Data availability](#data-availability).

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

**Replication against thesis, FIT A on the sealed 2021–2024 test block:**

| Metric | Thesis (MATLAB) | Replication (Python) |
|---|---|---|
| R² in-sample | 0.706548873843745 | 0.706548874290826 |
| RMSE in-sample | 3.8182342232388 | 3.818234220330205 |
| R² out-of-sample | 0.4064289 | 0.4066954 |
| RMSE out-of-sample | 5.1082591 | 5.1071122 |

The in-sample figures agree to nine decimals, which is what identifies the growth policy as correct.
The out-of-sample figures differ in the fourth, and the asymmetry has a mechanical cause: split
thresholds are midpoints between adjacent *training* values, so no training row ever sits near a
boundary, while test rows have no such protection and a few of them fall on the other side of a
threshold that differs in its last bits.

![Feature importance](figures/fig_feature_importance.png)

The importance ranking comes across too: the contemporaneous market factor takes 68.69% of the
budget against 68.71% in the thesis, aggregate terms take 76.8% against roughly 77%, and one
predictor has zero importance in both.

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

Energy is the sector where the scenario matters most, and the robust part of the finding is which
pathways put the green leg ahead. In the transition component at 2050, Net Zero +0.29 and Delayed
transition +0.16 are both positive; NDCs −0.70, Below 2°C −0.82 and Fragmented World −0.93 are
below them.

The ordering is not a simple ranking of ambition. It takes ambition *and* speed together: Net Zero
and Delayed transition have both, Below 2°C has ambition without speed, Fragmented World has speed
without ambition, NDCs has neither. Net Zero sits at the green extreme and Delayed transition beside
it, and that placement is the durable part of the finding.

The crossing of zero is a different kind of statement, and a weaker one. Whether the remaining three
pathways land below zero or merely below Net Zero is a property of the *level* of the log-ratio, and
the level shifts with the estimation window. Appendix A.3 of the thesis treats it that way, and so
should a reader of this repository: the result to take away is which pathways put the green leg
ahead, not that Energy inverts its sign — the latter is a description of this particular fit rather
than a property of the sector.

The left panel is the other half of the finding, and it is kept in the figure on purpose: under the
physical component the five scenarios lie exactly on top of each other. The Energy spread across
scenarios is 0.00 for physical against 1.22 for transition. All of the separation is transition
risk.

![Materials trajectories](figures/fig_materials_trajectories.png)

Materials behaves differently. The brown leg is ahead under every pathway; only the size of the gap
changes, from −1.67 under Net Zero (tightest) to −2.62 under NDCs (widest). The shock enters as a
cost rather than as an advantage: green does not become better, it becomes less bad.

### Two estimation windows, and a tree growth policy with no scikit-learn equivalent

Two features of the MATLAB pipeline are easy to miss when reading the code, and both change the
answer.

**The estimation window.** The same configuration is fitted twice, for two different purposes.
`refit_finale_K62.m` fits 189 months, up to December 2020, and certifies the out-of-sample metrics
on the sealed block. `s3_K62_leaf10.m` fits 237 months, everything up to December 2024, and it is
this second fit that produces the projection; appendix A.3 of the thesis states the design and the
reason, which is that the scenario anchors are recomputed on the same window as the estimation so
each configuration is internally consistent. The two are kept apart here as well: `prepare_fit_a`
and `prepare_fit_b` in [`src/train_gb.py`](src/train_gb.py), and the projection calls the second.
Using the 189-month fit to project means projecting from a model that stops before the test block.

**The growth policy.** `MaxNumSplits = 15` is a budget on the *number of splits*, spent
breadth-first: the tree grows level by level, and when a level would overrun the budget, the least
productive splits of that level are undone. It coincides with `max_depth = 4` only for a complete
tree, since 1 + 2 + 4 + 8 = 15 branch nodes. On this design the trees are not complete — many nodes
cannot be split because an interaction column is constant inside them — so the two policies part
company, and neither of the two scikit-learn options is the MATLAB procedure. That is what
[`src/matlab_policy_gb.py`](src/matlab_policy_gb.py) implements, and its equivalence test pins it
down: given a budget of `2**d - 1` on data where every node can split, it must reproduce
scikit-learn at `max_depth = d` exactly, which it does at depths 2, 3 and 4.

The three policies differ in *where* the splits go, not only in how many there are. A depth bound
stops at the fourth level whether or not the budget is spent; a leaf-count bound spends the whole
budget wherever the gain is largest; the MATLAB rule spends the whole budget too, but breadth
before depth, which produces unbalanced trees reaching depth 12.

Fitting all three on the same design puts numbers on that:

| Growth policy | R² in-sample | R² out-of-sample | splits/tree | Energy 2050 spread |
|---|---|---|---|---|
| scikit-learn `max_depth=4`, level-wise, truncated at depth 4 | 0.6503 | 0.4103 | 8.74 | 0.5213 |
| **numpy builder, level-wise under a 15-split budget** | **0.7065** | **0.4070** | **15.00** | **1.2199** |
| scikit-learn `max_leaf_nodes=16`, best-first | 0.7192 | 0.3965 | 15.00 | 1.4251 |
| thesis (MATLAB) | 0.7065 | 0.4064 | not reported | 1.2199 |

The budget policy is bracketed by the two approximations on in-sample fit, on out-of-sample fit and
on the projection at the same time, and it lands where breadth-first growth predicts on each. Three
independent axes agreeing is why the match to the thesis is not treated as a coincidence.

The two scikit-learn policies remain available by name in `train_gb.build_model`, and
[`scripts/make_tables.py`](scripts/make_tables.py) regenerates the table above from
`results/growth_policy_table.csv`.

**The risk-free path.** `s3_K62_leaf10.m` compounds `total = yhat + RF`, and the projection here
does the same. The scenario design file carries no risk-free column, so the path is joined from the
thesis predictions export, where RF is constant across entities within a scenario, component and
month.

### Hyperparameter optimization

The four values quoted above — depth 4, learning rate 0.03, 300 trees, minimum leaf size 10 — are
the output of a hyperparameter optimization carried out in the thesis. They are not tuned in this
repository, and they are not hand-picked: the procedure, its search space and its selection
criterion are all fixed in advance, and are recorded here so that a reader can see which they are.

**Feature selection is settled before the search begins.** A deliberately permissive reference model
— learning rate 0.01, four leaves, minimum leaf size 10 — is fitted on all 552 candidates, and 101
of them come out with positive importance. Reading that ranking at 99% of cumulative importance
gives the smallest top-K window that reaches the threshold, K = 62, to which the thesis adds 23
forced market terms (contemporaneous ExMkt and its 22 entity interactions, eleven of which fall
outside the window and are restored). The resulting 73-feature set is fixed from that point on. It
is chosen before the grid is entered and never revisited afterwards, so the search cannot quietly
select features and hyper-parameters against the same data.

**Search space.** An exhaustive grid, defined a priori, over

| Axis | Values |
|---|---|
| learning rate η | 0.02, 0.03, 0.05, 0.07, 0.10 |
| number of learners M | tied to η as M = round(900 × 0.01 / η): 450, 300, 180, 129, 90 |
| tree depth | 3, 4, 5 |
| minimum leaf size | 5, 8, 10, 12, 15, 20, 30 |
| subsample rate | 0.5, 0.8, 1.0 |

That is 5 × 3 × 7 × 3 = **315 configurations**. The number of learners is not a free axis: it is
pinned to the learning rate so that η × M is the same across the grid — 9 in every cell, up to the
rounding of M — which holds the total amount of learning constant and stops the comparison turning
into a contest between long slow runs and short fast ones.

**Validation window and selection criterion.** Selection is scored by RMSE on a held-out 24-month
validation window, in two stages. The first pass scores all 315 configurations — five seeds each for
the stochastic settings, a single fit for the deterministic ones, which are deterministic precisely
because subsampling is off. The finalists are then re-scored over 30 seeds, so that a configuration
cannot win on a lucky initialisation: the second stage separates the signal from initialisation
variance.

**Sealed test set.** The 2021–2024 block takes no part in any of this. It is not used to score
candidates, not used to pick finalists, and not consulted between stages. The selected configuration
is refitted on the 189 training months and evaluated against that block exactly once, which is what
makes the reported out-of-sample figures an out-of-sample result rather than a selection statistic.

**Refit for the projection.** For the 2025–2050 projection the same hyper-parameters are held fixed
and only the estimation sample changes, to the full 237 months. Appendix A.3 of the thesis describes
that refit and the reason for it. No re-optimization happens on the longer window.

One thing the table of growth policies above is *not*: it is not part of this search.
`max_depth=4` and `max_leaf_nodes=16` were never candidates, and no criterion in the thesis ever
compared them. They are two ways of approximating the single constraint `MaxNumSplits = 15` in a
library that does not offer it, and they appear only to show where the correct policy sits between
them.

### Reading a CSV is not free

The private inputs are CSV exports of MATLAB doubles, and getting them back intact takes two
separate precautions. Neither is sufficient alone:

| Written with | Read with | Deviation from the MATLAB doubles |
|---|---|---|
| default precision | default parser | 4.97e-14 |
| default precision | `float_precision="round_trip"` | 4.97e-14 |
| `%.17g` | default parser | 1.42e-14 |
| **`%.17g`** | **`float_precision="round_trip"`** | **0** |

The original files simply carried too few digits, so no parser could recover them. The re-exported
files carry enough, but pandas' default CSV parser is fast rather than correctly rounded and puts
about 1e-14 back in. Both together give an exact round-trip, and only then does the pipeline
reproduce 1.219885 rather than 1.4631.

This is worth knowing outside this project: a 5e-14 discrepancy in an input file is invisible in
every diagnostic anyone normally looks at, and here it moved the headline result by twenty per cent.

### How closely the replication matches

Over the full projection — 11 sectors x 5 scenarios x 3 components x 312 months, 51,480 values —
the largest deviation from the MATLAB export is **1.410e-05** and the mean deviation is 1.006e-06.
The worst cell is Communication under Delayed transition in December 2050, where the log-ratio is
−6.528841 against −6.528855, a relative error of 2.2e-06. That residual sits above the six-decimal
rounding of the export, so it is a real numerical difference and not a display artefact: the
floating-point tie-breaks inside the two splitters are not exposed by either library.

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

The response to the pathway is graded, not binary, and it is concentrated. Ranking the 11 sectors by
the range of their mean 2025–2050 differential across scenarios: Energy 0.395 and Materials 0.312
stand apart; Communication 0.114 and Information Technology 0.054 are small but not nothing; the
remaining seven are at 0.02 or below, four of them at 0.00.

Sectors are ranked by the size of that response rather than counted by whether their log-ratio
crosses zero, and the choice matters. How much a sector's differential moves when the pathway
changes is a property of the sector: it is the gap between what the model predicts under one set of
driver paths and another, and nothing outside the sector fixes its size. Crossing zero is a
different kind of quantity. It depends on the level the differential happens to sit around, and
that level shifts with the estimation window — appendix A.3 of the thesis makes the same point.
Ranking by response also needs no cut-off, so nothing hinges on where a boundary is drawn: Energy
first by a wide margin, Materials second, everything else far behind.

So the answer to the research question is not "structural" and not "conditional", but conditional in
a specific place and through a specific channel. In Energy the channel is fuel, and the pathways
that put the green leg ahead are the ones combining ambition with speed.

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
adjustment, not as magnitudes, which is the conclusion the thesis reaches from the low volatility of
the NGFS paths. The design of the projection gives a second reason for the same caution. The
projected fuel path over 2025–2050 spans roughly 0.04 to 1.85, about three per cent of the range the
model was trained on and sitting close to its median. A boosted tree is a step function, so over an
interval that narrow the level of the answer depends on whether a step edge happens to fall inside
it, while the ranking across scenarios does not.

**No MATLAB was re-run.** Every comparison in this repository is against exported files, not against
a live MATLAB session. The `.m` scripts are not under version control; their provenance is
documented below but not verified. The average number of splits and leaves in the MATLAB trees is
not reported in the thesis and not present in the exports, so the 15.00 and 16.00 produced by the
builder cannot be checked against the original directly — the in-sample agreement to nine decimals
is the indirect evidence that stands in for it.

**One rule in the growth policy is a choice, not a transcription.** When the split budget forces the
per-level undo, ties in impurity gain have to be broken somehow, and the MathWorks documentation
does not say how. The builder resolves them in favour of the lower node index and says so. The fit
reported here does not appear to turn on that choice, since it reproduces the MATLAB result, but
agreement is not proof that the rule matches.

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
