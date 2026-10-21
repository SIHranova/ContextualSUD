# Proposed restructuring of `3_degradation.ipynb`

This document records the proposed restructuring and implementation progress for `notebooks/3_degradation.ipynb`, using `notebooks/3b_devaluation.ipynb` and the shared helpers in `notebooks/utilities.py` as references.

## Progress so far

- The setup/run cell in `notebooks/3_degradation.ipynb` has been reorganized around a `pars` dictionary, trialwise reward schedules, simulation parameter combinations, dataframe construction, and CSV export. It still contains duplicate imports, and the earlier notebook-local helper definitions have not yet been removed.
- `notebooks/utilities.py` has been edited to accept both trialwise and context-indexed reward schedules in `run_simulation`, and its gradual-change dataframe helper has been adapted for reward-changing and utility-changing runs.
- A reduced simulation smoke check previously passed for both experiment modes and produced dataframes. The helper implementation has changed since that check, so rerun the smoke check against the current files before treating compatibility as verified.
- No full 400-trial, 200-repetition simulation has been run as part of this work.

The utility helpers are not yet confirmed compatible with both notebooks in their current state. The current degradation setup uses `decay_rate` and the `changing_rewards` / `changing_utility` flags, while the devaluation setup currently passes `utility_decay_rate`; the dataframe helper must consistently support these parameter shapes and names. Preserve the user's current edits in both `utilities.py` and the notebook when completing this integration.

## Current file structure

| File | Structure and role |
|---|---|
| `notebooks/3_degradation.ipynb` | 15 cells. Extensive introductory notes; imports and three local functions; experiment description; simulation settings; posterior/action plots; fit exclusions and diagnostics; switching-time plots; interpretation. |
| `notebooks/3b_devaluation.ipynb` | 7 cells. Brief title; one main setup/run/export cell using shared utilities; posterior/action plots; fit exclusions and diagnostics; switching-time plots. |
| `notebooks/utilities.py` | Shared reward-regime plotting, simulation initialization, posterior fitting, and dataframe construction for multiple-lever and gradual-change experiments. |
| `notebooks/classes.py` | `Agent` handles inference, learning, and action selection; `MultiArmedBandit` generates states, observations, and rewards; `World` coordinates trials and belief updates. |

## Experimental distinction

| | Degradation | Devaluation |
|---|---|---|
| Changes over trials | Reward contingencies, `Rho` | Outcome utility |
| Remains fixed | Utility `[0.025, 0.95, 0.025]` | Reward contingencies |
| Slow/fast conditions | `k = 0.02, 3` | `k = 0.02, 3` |
| Common design | 400 trials, 200 repetitions per condition, three habit settings, context persistence parameter `0.99` | Same |

In degradation, reward probability for the initially rewarded action falls from `0.9` toward `0.1`, while the alternative improves. Both notebooks then examine context inference, action persistence, and fitted switching times.

## Proposed notebook organization

Follow the devaluation notebook's workflow and formatting, with clearly separated blocks:

1. **Title and experiment description**
   - Replace the opening devaluation title and lengthy exploratory notes with a concise description of gradual contingency change.
   - Retain the relevant hypothesis and interpretation.

2. **Imports and simulation parameters**
   - Import shared functions explicitly.
   - Consolidate settings into `pars`, including seed, dimensions, habit settings, change rates, trial counts, and category mappings.

3. **Reward schedules and fixed utility**
   - Construct one trialwise `Rho` per change rate using the existing degradation logistic schedule.
   - Keep utility fixed.
   - Plot the schedules using the shared plotting helper, with degradation-specific axis labels.

4. **Simulation combinations, execution, and export**
   - Build `sim_params` with `product` and assign unique simulation IDs.
   - Call the shared runner and construct long dataframes.
   - Concatenate results, set category ordering, and export using a parameter-derived filename.

5. **Posterior and executed-action plots**
   - Preserve the current context posterior, policy posterior, and action-frequency plots.
   - Format these like the devaluation notebook.

6. **Curve-fitting exclusions and diagnostics**
   - Retain degradation's existing exclusion rules initially.
   - Remove obsolete commented alternatives and document the active criteria.

7. **Switching-time plots and interpretation**
   - Preserve the box/strip and point plots.
   - Correct stale descriptions: for example, the final text currently mentions violin plots although the code produces boxplots.

This removes the notebook-local definitions of `run_simulation`, `fit_posterior_traces`, `run_degradation_simulations`, and `plot_reward_regime`.

## Remaining shared-code work

- **Verify reward-schedule selection:** retain support for both trialwise reward schedules and context-indexed schedules, and confirm the two notebook parameter shapes select the intended path.
- **Align dataframe parameters:** support the `utility_decay_rate` parameter used by devaluation and the `decay_rate` parameter currently used by degradation. Keep the corresponding decay-rate dataframe column and constant versus trialwise utility columns correct for each experiment.
- **Preserve initialization behavior:** check whether degradation's prior counts and initial context probability are intended to differ from devaluation's. If so, expose explicit options and pass them from the degradation setup; do not silently change either experiment.
- **Remove obsolete local definitions and imports:** the degradation notebook still contains its earlier local simulation helpers. Remove or retire them only after the shared implementations have passed the compatibility checks.
- **Check the compatibility alias/import:** the degradation notebook may still refer to the plural `create_long_df_gradual_changes` name in legacy code; either update the import/call to the singular shared function or retain a documented alias if that legacy surface is still required.
- **Rerun focused validation** against the current code: a small run for each notebook's parameter format, expected dataframe columns and utility values, reward schedule shape, and plotting helpers. Then check imports and syntax before any full simulation.

## Recommended remaining scope and validation

Use a matching notebook layout plus small, backward-compatible utility changes. No changes to `classes.py` are expected to be necessary.

Preserve degradation's parameters and analysis rules. Before running the full experiment, verify a small simulation produces:

- The intended slow and fast reward schedules.
- Constant utility.
- The expected dataframe columns.
- Working analysis plots.

The notebook setup and utility helpers have been partially edited. The remaining compatibility work and validation are still pending; do not treat the full restructuring as complete until both notebook workflows pass focused checks.
