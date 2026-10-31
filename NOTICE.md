# Attribution and release terms

Original study code: copyright 2026 Tianyu Shi, Hanqing Qian and Chengyi Tu; MIT license. Study-generated data and documentation: CC-BY-4.0, https://creativecommons.org/licenses/by/4.0/ .

The empirical source data are from Koster, Pislár and colleagues, *Deep reinforcement learning can promote sustainable human behaviour in a common-pool resource problem*, Nature Communications (2025), https://doi.org/10.1038/s41467-025-58043-7 . Source: https://github.com/google-deepmind/sustainable_behavior . Copyright 2024 DeepMind Technologies Limited. Upstream non-software materials use Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/legalcode . The upstream software uses Apache-2.0; a copy of its license is retained in `third_party/APACHE-2.0.txt` for attribution context.

`code/emp_train_all.npz`, `code/emp_train_nointerp.npz`, `code/emp_train_base.npz`, `results_revision/out/emp_all_gameids.npz`, `code/human_moments.json` and the human-observation portions of the comparison tables are derived from those empirical data. Transformations include condition filtering, per-round state feature construction, normalization, aggregation and case-library construction. These materials retain CC-BY-4.0 attribution and are excluded from the scope of the MIT software license. No endorsement by Google, DeepMind or the original authors is implied.

The public upstream CSV is obtained through the upstream notebook rather than bundled here. Derived libraries are supplied to run the evaluations without downloading the original participant-level file. The full upstream README, including its license distinction and disclaimer, is in `third_party/UPSTREAM_README.md` (read on 31 October 2026).
