# # """
# # analyze_study1.py

# # Reads results_study1/*.csv (written by run_study1.py) and produces:

# #   results_study1/table_summary.csv   mean and 95% CI per model x error
# #   fig_learning_curves.png            return per episode, mean +- 95% CI,
# #                                      one panel per radius error
# #   fig_vs_error.png                   training return, frozen-eval success
# #                                      and 8-step model error vs radius error
# #   + a printed table and a paired hybrid-vs-physics_only comparison

# # Uses only numpy + matplotlib (no pandas / scipy needed).
# # 95% CIs use the t-distribution over seeds.

# # Usage:
# #     python analyze_study1.py
# # """

# # import csv
# # import math
# # import os
# # from collections import defaultdict

# # import numpy as np
# # import matplotlib.pyplot as plt

# # RESULTS_DIR = 'results_study1'
# # MODELS = ['physics_only', 'hybrid', 'black_box']
# # COLORS = {'physics_only': '#2a78d6', 'hybrid': '#eb6834', 'black_box': '#1baf7a'}
# # LABELS = {'physics_only': 'Physics only', 'hybrid': 'Hybrid (physics + residual)',
# #           'black_box': 'Black box'}

# # # two-sided 95% t critical values, df = 1..30 (df > 30 -> 1.96)
# # _T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
# #         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
# #         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


# # def mean_ci(values):
# #     v = np.array([x for x in values if x is not None and not np.isnan(x)], float)
# #     n = len(v)
# #     if n == 0:
# #         return np.nan, np.nan, 0
# #     if n == 1:
# #         return v[0], np.nan, 1
# #     t = _T95[n - 2] if n - 1 <= 30 else 1.96
# #     return v.mean(), t * v.std(ddof=1) / math.sqrt(n), n


# # def to_num(x):
# #     if x in ('', 'None', 'nan', None):
# #         return np.nan
# #     if x in ('True', 'False'):
# #         return 1.0 if x == 'True' else 0.0
# #     return float(x)


# # def read_csv(name):
# #     with open(os.path.join(RESULTS_DIR, name)) as f:
# #         rows = list(csv.DictReader(f))
# #     # keep the LAST row for any duplicated key (safety for interrupted runs)
# #     return rows


# # def err_key(r):
# #     e = to_num(r['radius_error_pct'])
# #     return None if np.isnan(e) else e


# # def main():
# #     summary = read_csv('summary.csv')
# #     episodes = read_csv('episodes.csv')

# #     dedup = {}
# #     for r in summary:
# #         dedup[(r['model'], r['radius_error_pct'], r['seed'])] = r
# #     summary = list(dedup.values())

# #     errors = sorted({err_key(r) for r in summary if err_key(r) is not None})
# #     by = defaultdict(list)                     # (model, err) -> rows
# #     for r in summary:
# #         if r['model'] == 'black_box':
# #             for e in errors:                   # black_box is the same at every error
# #                 by[('black_box', e)].append(r)
# #         else:
# #             by[(r['model'], err_key(r))].append(r)

# #     # ------------------------------------------------------------ table
# #     metrics = [
# #         ('steps_to_reliable_success', 'Steps to 3-in-a-row (incl. warmup)', 1),
# #         ('mean_return', 'Mean training return (all 15 ep.)', 1),
# #         ('success_rate_all', 'Training success rate', 100),
# #         ('eval_success_rate', 'Frozen-eval success, new tasks (%)', 100),
# #         ('eval_mean_return', 'Frozen-eval mean return', 1),
# #         ('final_onestep_pos_err', 'Final 1-step position error (mm)', 1000),
# #         ('final_kstep_pos_err', 'Final 8-step position error (mm)', 1000),
# #         ('eval_onpolicy_onestep_pos_err', 'On-policy 1-step position error (mm), eval paths', 1000),
# #         ('eval_onpolicy_kstep_pos_err', 'On-policy 8-step position error (mm), eval paths', 1000),
# #         ('mean_epochs_per_retrain', 'Mean epochs per retrain', 1),
# #     ]
# #     out_rows = []
# #     print("\nmean +- 95% CI over seeds   (n = number of seeds)\n")
# #     for key, title, scale in metrics:
# #         print(title)
# #         print(f"  {'error':>6s}  " + "  ".join(f"{LABELS[m][:22]:>24s}" for m in MODELS))
# #         for e in errors:
# #             cells = []
# #             for m in MODELS:
# #                 vals = [to_num(r[key]) * scale for r in by[(m, e)]]
# #                 mu, ci, n = mean_ci(vals)
# #                 n_never = sum(np.isnan(v) for v in vals)
# #                 extra = f" ({n_never} never)" if key == 'steps_to_reliable_success' and n_never else ""
# #                 cells.append(f"{mu:9.2f} +- {ci:6.2f} n={n}{extra}")
# #                 out_rows.append({'metric': key, 'model': m, 'radius_error_pct': e,
# #                                  'mean': mu, 'ci95': ci, 'n': n})
# #             print(f"  {e:+5.0f}%  " + "  ".join(f"{c:>24s}" for c in cells))
# #         print()

# #     with open(os.path.join(RESULTS_DIR, 'table_summary.csv'), 'w', newline='') as f:
# #         w = csv.DictWriter(f, fieldnames=['metric', 'model', 'radius_error_pct',
# #                                           'mean', 'ci95', 'n'])
# #         w.writeheader()
# #         w.writerows(out_rows)

# #     # ----------------------------------- paired hybrid vs physics_only
# #     print("Paired difference, hybrid minus physics_only (same seed), mean training return")
# #     for e in errors:
# #         p = {r['seed']: to_num(r['mean_return']) for r in by[('physics_only', e)]}
# #         h = {r['seed']: to_num(r['mean_return']) for r in by[('hybrid', e)]}
# #         diffs = [h[s] - p[s] for s in h if s in p]
# #         mu, ci, n = mean_ci(diffs)
# #         print(f"  {e:+5.0f}%   {mu:+7.2f} +- {ci:5.2f}  (n={n})")
# #     print("Paired difference in steps to 3-in-a-row (incl. warmup), hybrid minus physics_only")
# #     print("  negative = hybrid needed fewer real steps.")
# #     for e in errors:
# #         p = {r['seed']: to_num(r['steps_to_reliable_success']) for r in by[('physics_only', e)]}
# #         h = {r['seed']: to_num(r['steps_to_reliable_success']) for r in by[('hybrid', e)]}
# #         diffs = [h[s] - p[s] for s in h if s in p]
# #         mu, ci, n = mean_ci(diffs)
# #         print(f"  {e:+5.0f}%   {mu:+7.1f} +- {ci:5.1f}  (n={n})")

# #     # ------------------------------------------- learning-curve figure
# #     curves = defaultdict(lambda: defaultdict(list))   # (model, err) -> ep -> [returns]
# #     for r in episodes:
# #         m = r['model']
# #         targets = errors if m == 'black_box' else [err_key(r)]
# #         for e in targets:
# #             curves[(m, e)][int(r['episode'])].append(to_num(r['return']))

# #     n_e = len(errors)
# #     ncol = min(4, n_e)
# #     nrow = math.ceil(n_e / ncol)
# #     fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow),
# #                              sharey=True, squeeze=False)
# #     for idx, e in enumerate(errors):
# #         ax = axes[idx // ncol][idx % ncol]
# #         for m in MODELS:
# #             eps = sorted(curves[(m, e)])
# #             if not eps:
# #                 continue
# #             stats = [mean_ci(curves[(m, e)][k]) for k in eps]
# #             mu = np.array([s[0] for s in stats])
# #             ci = np.nan_to_num(np.array([s[1] for s in stats]))
# #             ax.plot(eps, mu, color=COLORS[m], lw=2, label=LABELS[m])
# #             ax.fill_between(eps, mu - ci, mu + ci, color=COLORS[m], alpha=0.15, lw=0)
# #         ax.set_title(f"radius error {e:+.0f}%", fontsize=10)
# #         ax.grid(alpha=0.25, lw=0.5)
# #         for s in ('top', 'right'):
# #             ax.spines[s].set_visible(False)
# #         if idx % ncol == 0:
# #             ax.set_ylabel('episode return')
# #         if idx // ncol == nrow - 1:
# #             ax.set_xlabel('training episode')
# #     for j in range(n_e, nrow * ncol):
# #         axes[j // ncol][j % ncol].axis('off')
# #     handles, labels = axes[0][0].get_legend_handles_labels()
# #     fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
# #     fig.tight_layout(rect=(0, 0, 1, 0.94))
# #     fig.savefig('fig_learning_curves.png', dpi=150)
# #     plt.close(fig)

# #     # ---------------------------------------------- metrics vs error
# #     panels = [('mean_return', 'mean training return', 1),
# #               ('eval_success_rate', 'frozen-eval success (%)', 100),
# #               ('final_kstep_pos_err', '8-step error, random actions (mm)', 1000),
# #               ('eval_onpolicy_kstep_pos_err', '8-step error, on-policy (mm)', 1000)]
# #     fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
# #     for ax, (key, ylabel, scale) in zip(axes, panels):
# #         for m in MODELS:
# #             stats = [mean_ci([to_num(r[key]) * scale for r in by[(m, e)]]) for e in errors]
# #             mu = np.array([s[0] for s in stats])
# #             ci = np.nan_to_num(np.array([s[1] for s in stats]))
# #             if m == 'black_box':      # same at every error: a flat band
# #                 ax.axhspan(mu[0] - ci[0], mu[0] + ci[0], color=COLORS[m], alpha=0.12, lw=0)
# #                 ax.axhline(mu[0], color=COLORS[m], lw=2, ls='--', label=LABELS[m])
# #                 continue
# #             ax.errorbar(errors, mu, yerr=ci, color=COLORS[m], lw=2,
# #                         marker='o', ms=5, capsize=3, label=LABELS[m])
# #         ax.set_xlabel('radius error (%)   [+ = wheel assumed smaller]')
# #         ax.set_ylabel(ylabel)
# #         ax.grid(alpha=0.25, lw=0.5)
# #         for s in ('top', 'right'):
# #             ax.spines[s].set_visible(False)
# #     axes[0].legend(frameon=False, fontsize=8)
# #     fig.tight_layout()
# #     fig.savefig('fig_vs_error.png', dpi=150)
# #     plt.close(fig)

# #     print("\nSaved: fig_learning_curves.png, fig_vs_error.png, "
# #           f"{RESULTS_DIR}/table_summary.csv")


# # if __name__ == '__main__':
# #     main()

# """
# analyze_study1.py

# Reads results_study1/*.csv (written by run_study1.py) and produces:

#   results_study1/table_summary.csv   mean and 95% CI per model x error
#   fig_learning_curves.png            return per episode, mean +- 95% CI,
#                                      one panel per radius error
#   fig_vs_error.png                   training return, frozen-eval success
#                                      and 8-step model error vs radius error
#   + a printed table and a paired hybrid-vs-physics_only comparison

# Uses only numpy + matplotlib (no pandas / scipy needed).
# 95% CIs use the t-distribution over seeds.

# Usage:
#     python analyze_study1.py
# """

# import csv
# import math
# import os
# from collections import defaultdict

# import numpy as np
# import matplotlib.pyplot as plt

# RESULTS_DIR = 'results_study1'
# MODELS = ['physics_only', 'hybrid', 'black_box']
# COLORS = {'physics_only': '#2a78d6', 'hybrid': '#eb6834', 'black_box': '#1baf7a'}
# LABELS = {'physics_only': 'Physics only', 'hybrid': 'Hybrid (physics + residual)',
#           'black_box': 'Black box'}

# # two-sided 95% t critical values, df = 1..30 (df > 30 -> 1.96)
# _T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
#         2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
#         2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


# def mean_ci(values):
#     v = np.array([x for x in values if x is not None and not np.isnan(x)], float)
#     n = len(v)
#     if n == 0:
#         return np.nan, np.nan, 0
#     if n == 1:
#         return v[0], np.nan, 1
#     t = _T95[n - 2] if n - 1 <= 30 else 1.96
#     return v.mean(), t * v.std(ddof=1) / math.sqrt(n), n


# def to_num(x):
#     if x in ('', 'None', 'nan', None):
#         return np.nan
#     if x in ('True', 'False'):
#         return 1.0 if x == 'True' else 0.0
#     return float(x)


# def read_csv(name):
#     with open(os.path.join(RESULTS_DIR, name)) as f:
#         rows = list(csv.DictReader(f))
#     # keep the LAST row for any duplicated key (safety for interrupted runs)
#     return rows


# def err_key(r):
#     e = to_num(r['radius_error_pct'])
#     return None if np.isnan(e) else e


# def main():
#     summary = read_csv('summary.csv')
#     episodes = read_csv('episodes.csv')

#     dedup = {}
#     for r in summary:
#         dedup[(r['model'], r['radius_error_pct'], r['seed'])] = r
#     summary = list(dedup.values())

#     errors = sorted({err_key(r) for r in summary if err_key(r) is not None})
#     by = defaultdict(list)                     # (model, err) -> rows
#     for r in summary:
#         if r['model'] == 'black_box':
#             for e in errors:                   # black_box is the same at every error
#                 by[('black_box', e)].append(r)
#         else:
#             by[(r['model'], err_key(r))].append(r)

#     # ------------------------------------------------------------ table
#     metrics = [
#         ('steps_to_reliable_success', 'Steps to 3-in-a-row (incl. warmup)', 1),
#         ('mean_return', 'Mean training return (all 15 ep.)', 1),
#         ('success_rate_all', 'Training success rate', 100),
#         ('eval_success_rate', 'Frozen-eval success, new tasks (%)', 100),
#         ('eval_mean_return', 'Frozen-eval mean return', 1),
#         ('final_onestep_pos_err', 'Final 1-step position error (mm)', 1000),
#         ('final_kstep_pos_err', 'Final 8-step position error (mm)', 1000),
#         ('eval_onpolicy_onestep_pos_err', 'On-policy 1-step position error (mm), eval paths', 1000),
#         ('eval_onpolicy_kstep_pos_err', 'On-policy 8-step position error (mm), eval paths', 1000),
#         ('mean_epochs_per_retrain', 'Mean epochs per retrain', 1),
#     ]
#     out_rows = []
#     print("\nmean +- 95% CI over seeds   (n = number of seeds)\n")
#     for key, title, scale in metrics:
#         print(title)
#         print(f"  {'error':>6s}  " + "  ".join(f"{LABELS[m][:22]:>24s}" for m in MODELS))
#         for e in errors:
#             cells = []
#             for m in MODELS:
#                 vals = [to_num(r[key]) * scale for r in by[(m, e)]]
#                 mu, ci, n = mean_ci(vals)
#                 n_never = sum(np.isnan(v) for v in vals)
#                 extra = f" ({n_never} never)" if key == 'steps_to_reliable_success' and n_never else ""
#                 cells.append(f"{mu:9.2f} +- {ci:6.2f} n={n}{extra}")
#                 out_rows.append({'metric': key, 'model': m, 'radius_error_pct': e,
#                                  'mean': mu, 'ci95': ci, 'n': n})
#             print(f"  {e:+5.0f}%  " + "  ".join(f"{c:>24s}" for c in cells))
#         print()

#     with open(os.path.join(RESULTS_DIR, 'table_summary.csv'), 'w', newline='') as f:
#         w = csv.DictWriter(f, fieldnames=['metric', 'model', 'radius_error_pct',
#                                           'mean', 'ci95', 'n'])
#         w.writeheader()
#         w.writerows(out_rows)

#     # ----------------------------------- paired hybrid vs physics_only
#     print("Paired difference, hybrid minus physics_only (same seed), mean training return")
#     for e in errors:
#         p = {r['seed']: to_num(r['mean_return']) for r in by[('physics_only', e)]}
#         h = {r['seed']: to_num(r['mean_return']) for r in by[('hybrid', e)]}
#         diffs = [h[s] - p[s] for s in h if s in p]
#         mu, ci, n = mean_ci(diffs)
#         print(f"  {e:+5.0f}%   {mu:+7.2f} +- {ci:5.2f}  (n={n})")
#     print("Paired difference in steps to 3-in-a-row (incl. warmup), hybrid minus physics_only")
#     print("  negative = hybrid needed fewer real steps.")
#     for e in errors:
#         p = {r['seed']: to_num(r['steps_to_reliable_success']) for r in by[('physics_only', e)]}
#         h = {r['seed']: to_num(r['steps_to_reliable_success']) for r in by[('hybrid', e)]}
#         diffs = [h[s] - p[s] for s in h if s in p]
#         mu, ci, n = mean_ci(diffs)
#         print(f"  {e:+5.0f}%   {mu:+7.1f} +- {ci:5.1f}  (n={n})")

#     # -------------------------------------- paired hybrid vs black_box
#     # black_box has no physics prior, so the same black_box run (same seed)
#     # is the partner for the hybrid at every error level. Same seed means
#     # same warmup data and same network initialization.
#     bb = {r['seed']: r for r in by[('black_box', errors[0])]}
#     for key, title, scale in [
#             ('mean_return', 'mean training return', 1),
#             ('eval_success_rate', 'frozen-eval success on new tasks (%)', 100),
#             ('eval_mean_return', 'frozen-eval mean return', 1)]:
#         print(f"\nPaired difference, hybrid minus black_box (same seed), {title}")
#         print("  positive = hybrid better.  'wins' = seeds where hybrid was strictly better.")
#         for e in errors:
#             h = {r['seed']: to_num(r[key]) * scale for r in by[('hybrid', e)]}
#             diffs = [h[s] - to_num(bb[s][key]) * scale for s in h if s in bb]
#             mu, ci, n = mean_ci(diffs)
#             wins = sum(d > 0 for d in diffs)
#             sig = 'significant' if not np.isnan(ci) and abs(mu) > ci else 'NOT significant'
#             print(f"  {e:+5.0f}%   {mu:+7.2f} +- {ci:5.2f}  (n={n}, wins {wins}/{n})  {sig}")

#     # ------------------------------------------- learning-curve figure
#     curves = defaultdict(lambda: defaultdict(list))   # (model, err) -> ep -> [returns]
#     for r in episodes:
#         m = r['model']
#         targets = errors if m == 'black_box' else [err_key(r)]
#         for e in targets:
#             curves[(m, e)][int(r['episode'])].append(to_num(r['return']))

#     n_e = len(errors)
#     ncol = min(4, n_e)
#     nrow = math.ceil(n_e / ncol)
#     fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow),
#                              sharey=True, squeeze=False)
#     for idx, e in enumerate(errors):
#         ax = axes[idx // ncol][idx % ncol]
#         for m in MODELS:
#             eps = sorted(curves[(m, e)])
#             if not eps:
#                 continue
#             stats = [mean_ci(curves[(m, e)][k]) for k in eps]
#             mu = np.array([s[0] for s in stats])
#             ci = np.nan_to_num(np.array([s[1] for s in stats]))
#             ax.plot(eps, mu, color=COLORS[m], lw=2, label=LABELS[m])
#             ax.fill_between(eps, mu - ci, mu + ci, color=COLORS[m], alpha=0.15, lw=0)
#         ax.set_title(f"radius error {e:+.0f}%", fontsize=10)
#         ax.grid(alpha=0.25, lw=0.5)
#         for s in ('top', 'right'):
#             ax.spines[s].set_visible(False)
#         if idx % ncol == 0:
#             ax.set_ylabel('episode return')
#         if idx // ncol == nrow - 1:
#             ax.set_xlabel('training episode')
#     for j in range(n_e, nrow * ncol):
#         axes[j // ncol][j % ncol].axis('off')
#     handles, labels = axes[0][0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
#     fig.tight_layout(rect=(0, 0, 1, 0.94))
#     fig.savefig('fig_learning_curves.png', dpi=150)
#     plt.close(fig)

#     # ---------------------------------------------- metrics vs error
#     panels = [('mean_return', 'mean training return', 1),
#               ('eval_success_rate', 'frozen-eval success (%)', 100),
#               ('final_kstep_pos_err', '8-step error, random actions (mm)', 1000),
#               ('eval_onpolicy_kstep_pos_err', '8-step error, on-policy (mm)', 1000)]
#     fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
#     for ax, (key, ylabel, scale) in zip(axes, panels):
#         for m in MODELS:
#             stats = [mean_ci([to_num(r[key]) * scale for r in by[(m, e)]]) for e in errors]
#             mu = np.array([s[0] for s in stats])
#             ci = np.nan_to_num(np.array([s[1] for s in stats]))
#             if m == 'black_box':      # same at every error: a flat band
#                 ax.axhspan(mu[0] - ci[0], mu[0] + ci[0], color=COLORS[m], alpha=0.12, lw=0)
#                 ax.axhline(mu[0], color=COLORS[m], lw=2, ls='--', label=LABELS[m])
#                 continue
#             ax.errorbar(errors, mu, yerr=ci, color=COLORS[m], lw=2,
#                         marker='o', ms=5, capsize=3, label=LABELS[m])
#         ax.set_xlabel('radius error (%)   [+ = wheel assumed smaller]')
#         ax.set_ylabel(ylabel)
#         ax.grid(alpha=0.25, lw=0.5)
#         for s in ('top', 'right'):
#             ax.spines[s].set_visible(False)
#     axes[0].legend(frameon=False, fontsize=8)
#     fig.tight_layout()
#     fig.savefig('fig_vs_error.png', dpi=150)
#     plt.close(fig)

#     print("\nSaved: fig_learning_curves.png, fig_vs_error.png, "
#           f"{RESULTS_DIR}/table_summary.csv")


# if __name__ == '__main__':
#     main()


"""
analyze_study1.py

Reads results_study1/*.csv (written by run_study1.py) and produces:

  results_study1/table_summary.csv   mean and 95% CI per model x error
  fig_learning_curves.png            return per episode, mean +- 95% CI,
                                     one panel per radius error
  fig_vs_error.png                   training return, frozen-eval success
                                     and 8-step model error vs radius error
  + a printed table and a paired hybrid-vs-physics_only comparison

Uses only numpy + matplotlib (no pandas / scipy needed).
95% CIs use the t-distribution over seeds.

Usage:
    python analyze_study1.py
"""

import csv
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = 'results_study1'
MODELS = ['physics_only', 'hybrid', 'black_box']
COLORS = {'physics_only': '#2a78d6', 'hybrid': '#eb6834', 'black_box': '#1baf7a'}
LABELS = {'physics_only': 'Physics only', 'hybrid': 'Hybrid (physics + residual)',
          'black_box': 'Black box'}

# two-sided 95% t critical values, df = 1..30 (df > 30 -> 1.96)
_T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
        2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
        2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def mean_ci(values):
    v = np.array([x for x in values if x is not None and not np.isnan(x)], float)
    n = len(v)
    if n == 0:
        return np.nan, np.nan, 0
    if n == 1:
        return v[0], np.nan, 1
    t = _T95[n - 2] if n - 1 <= 30 else 1.96
    return v.mean(), t * v.std(ddof=1) / math.sqrt(n), n


def to_num(x):
    if x in ('', 'None', 'nan', None):
        return np.nan
    if x in ('True', 'False'):
        return 1.0 if x == 'True' else 0.0
    return float(x)


def read_csv(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        rows = list(csv.DictReader(f))
    # keep the LAST row for any duplicated key (safety for interrupted runs)
    return rows


def err_key(r):
    e = to_num(r['radius_error_pct'])
    return None if np.isnan(e) else e


def main():
    summary = read_csv('summary.csv')
    episodes = read_csv('episodes.csv')

    dedup = {}
    for r in summary:
        dedup[(r['model'], r['radius_error_pct'], r['seed'])] = r
    summary = list(dedup.values())

    errors = sorted({err_key(r) for r in summary if err_key(r) is not None})
    by = defaultdict(list)                     # (model, err) -> rows
    for r in summary:
        if r['model'] == 'black_box':
            for e in errors:                   # black_box is the same at every error
                by[('black_box', e)].append(r)
        else:
            by[(r['model'], err_key(r))].append(r)

    # ------------------------------------------------------------ table
    metrics = [
        ('steps_to_reliable_success', 'Steps to 3-in-a-row (incl. warmup)', 1),
        ('mean_return', 'Mean training return (all 15 ep.)', 1),
        ('success_rate_all', 'Training success rate', 100),
        ('eval_success_rate', 'Frozen-eval success, new tasks (%)', 100),
        ('eval_mean_return', 'Frozen-eval mean return', 1),
        ('final_onestep_pos_err', 'Final 1-step position error (mm)', 1000),
        ('final_kstep_pos_err', 'Final 8-step position error (mm)', 1000),
        ('eval_onpolicy_onestep_pos_err', 'On-policy 1-step position error (mm), eval paths', 1000),
        ('eval_onpolicy_kstep_pos_err', 'On-policy 8-step position error (mm), eval paths', 1000),
        ('mean_epochs_per_retrain', 'Mean epochs per retrain', 1),
    ]
    out_rows = []
    print("\nmean +- 95% CI over seeds   (n = number of seeds)\n")
    for key, title, scale in metrics:
        print(title)
        print(f"  {'error':>6s}  " + "  ".join(f"{LABELS[m][:22]:>24s}" for m in MODELS))
        for e in errors:
            cells = []
            for m in MODELS:
                vals = [to_num(r[key]) * scale for r in by[(m, e)]]
                mu, ci, n = mean_ci(vals)
                n_never = sum(np.isnan(v) for v in vals)
                extra = f" ({n_never} never)" if key == 'steps_to_reliable_success' and n_never else ""
                cells.append(f"{mu:9.2f} +- {ci:6.2f} n={n}{extra}")
                out_rows.append({'metric': key, 'model': m, 'radius_error_pct': e,
                                 'mean': mu, 'ci95': ci, 'n': n})
            print(f"  {e:+5.0f}%  " + "  ".join(f"{c:>24s}" for c in cells))
        print()

    with open(os.path.join(RESULTS_DIR, 'table_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'model', 'radius_error_pct',
                                          'mean', 'ci95', 'n'])
        w.writeheader()
        w.writerows(out_rows)

    # ------------------------------------------- the three comparisons
    # Paired by seed: same seed = same warmup data, same network init,
    # same planner noise stream. black_box has no physics prior, so its
    # run for a given seed is the partner at every error level.
    COMPARISONS = [
        ('hybrid', 'black_box',
         'Q1 (MAIN): hybrid vs black_box -- is adding (possibly wrong) physics '
         'better than learning from scratch?'),
        ('hybrid', 'physics_only',
         'Q2: hybrid vs physics_only -- is learning a correction better than '
         'trusting the nominal physics as-is?'),
        ('physics_only', 'black_box',
         'Q3 (reference): physics_only vs black_box -- is a wrong model better '
         'than no model at all?'),
    ]
    PAIRED_METRICS = [
        ('mean_return', 'training return (mean over 15 episodes)', 1, +1),
        ('eval_mean_return', 'frozen-eval return on new tasks', 1, +1),
        ('eval_success_rate', 'frozen-eval success on new tasks (%)', 100, +1),
        ('steps_to_reliable_success', 'steps to 3-in-a-row incl. warmup '
         '(lower is better)', 1, -1),
    ]

    def paired(a, b, key, scale, sign):
        """Returns per-error (mean, ci, n, wins, verdict) for a minus b,
        oriented so that positive always means 'a is better'."""
        rows = {}
        for e in errors:
            A = {r['seed']: to_num(r[key]) * scale for r in by[(a, e)]}
            B = {r['seed']: to_num(r[key]) * scale for r in by[(b, e)]}
            diffs = [sign * (A[s] - B[s]) for s in A
                     if s in B and not np.isnan(A[s]) and not np.isnan(B[s])]
            mu, ci, n = mean_ci(diffs)
            wins = sum(d > 0 for d in diffs)
            if n < 2 or np.isnan(ci):
                verdict = 'n/a'
            elif mu - ci > 0:
                verdict = f'{a} better'
            elif mu + ci < 0:
                verdict = f'{b} better'
            else:
                verdict = 'no significant difference'
            rows[e] = (mu, ci, n, wins, verdict)
        return rows

    print("=" * 78)
    print("PAIRED COMPARISONS (same seed).  Positive = first model better.")
    print("A difference is called significant only if its 95% CI excludes 0.")
    print("=" * 78)
    for a, b, question in COMPARISONS:
        print(f"\n{question}")
        for key, title, scale, sign in PAIRED_METRICS:
            print(f"  {title}")
            for e, (mu, ci, n, wins, verdict) in paired(a, b, key, scale, sign).items():
                print(f"    {e:+6.0f}%  {mu:+8.2f} +- {ci:6.2f}  "
                      f"(n={n:2d}, {a} won {wins:2d}/{n:<2d})  {verdict}")

    # One-line verdict table for Q1 across metrics (the crossover view)
    print("\nCROSSOVER VIEW for Q1 (hybrid vs black_box): verdict per error level")
    q1 = {key: paired('hybrid', 'black_box', key, scale, sign)
          for key, _, scale, sign in PAIRED_METRICS}
    print(f"  {'error':>7s}  " + "  ".join(f"{k[:22]:>26s}" for k in q1))
    for e in errors:
        print(f"  {e:+6.0f}%  " + "  ".join(f"{q1[k][e][4][:26]:>26s}" for k in q1))

    # ------------------------------------------- learning-curve figure
    curves = defaultdict(lambda: defaultdict(list))   # (model, err) -> ep -> [returns]
    for r in episodes:
        m = r['model']
        targets = errors if m == 'black_box' else [err_key(r)]
        for e in targets:
            curves[(m, e)][int(r['episode'])].append(to_num(r['return']))

    n_e = len(errors)
    ncol = min(4, n_e)
    nrow = math.ceil(n_e / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow),
                             sharey=True, squeeze=False)
    for idx, e in enumerate(errors):
        ax = axes[idx // ncol][idx % ncol]
        for m in MODELS:
            eps = sorted(curves[(m, e)])
            if not eps:
                continue
            stats = [mean_ci(curves[(m, e)][k]) for k in eps]
            mu = np.array([s[0] for s in stats])
            ci = np.nan_to_num(np.array([s[1] for s in stats]))
            ax.plot(eps, mu, color=COLORS[m], lw=2, label=LABELS[m])
            ax.fill_between(eps, mu - ci, mu + ci, color=COLORS[m], alpha=0.15, lw=0)
        ax.set_title(f"radius error {e:+.0f}%", fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        if idx % ncol == 0:
            ax.set_ylabel('episode return')
        if idx // ncol == nrow - 1:
            ax.set_xlabel('training episode')
    for j in range(n_e, nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig('fig_learning_curves.png', dpi=150)
    plt.close(fig)

    # ---------------------------------------------- metrics vs error
    panels = [('mean_return', 'mean training return', 1),
              ('eval_success_rate', 'frozen-eval success (%)', 100),
              ('final_kstep_pos_err', '8-step error, random actions (mm)', 1000),
              ('eval_onpolicy_kstep_pos_err', '8-step error, on-policy (mm)', 1000)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, (key, ylabel, scale) in zip(axes, panels):
        for m in MODELS:
            stats = [mean_ci([to_num(r[key]) * scale for r in by[(m, e)]]) for e in errors]
            mu = np.array([s[0] for s in stats])
            ci = np.nan_to_num(np.array([s[1] for s in stats]))
            if m == 'black_box':      # same at every error: a flat band
                ax.axhspan(mu[0] - ci[0], mu[0] + ci[0], color=COLORS[m], alpha=0.12, lw=0)
                ax.axhline(mu[0], color=COLORS[m], lw=2, ls='--', label=LABELS[m])
                continue
            ax.errorbar(errors, mu, yerr=ci, color=COLORS[m], lw=2,
                        marker='o', ms=5, capsize=3, label=LABELS[m])
        ax.set_xlabel('radius error (%)   [+ = wheel assumed smaller]')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, lw=0.5)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig('fig_vs_error.png', dpi=150)
    plt.close(fig)

    print("\nSaved: fig_learning_curves.png, fig_vs_error.png, "
          f"{RESULTS_DIR}/table_summary.csv")


if __name__ == '__main__':
    main()