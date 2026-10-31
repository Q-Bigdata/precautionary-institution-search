# Precautionary proposal bias in institutional program search

Research code and recorded results for **A precautionary proposal bias limits language-model discovery of institutions for a shared resource**, by Tianyu Shi, Hanqing Qian and Chengyi Tu. This repository provides the code and recorded data released on 31 October 2026.

## Contents

- `code/`: simulator, SR/SR2/AL/EV/EMP behavioural families, reference rules, program validation, search driver, calibration, black-box planner and evaluation modules; calibrated parameters and three EMP feature libraries.
- `code_revision/rjobs.py`: revision evaluation task runner, with package-relative paths repaired.
- `code_revision/rev_figures.py`: all five main and three Extended Data figures of the revised manuscript.
- `source_data_revision/`: the 24 supplied figure and table source-data files, including frozen validation and diagnostics.
- `results_original/`, `results_revision/`: recorded search candidates, task specifications and evaluation outputs; duplicate executable snapshots and progress/error logs are omitted.
- `prompts_revision/`: recorded revision prompts and samples.
- `examples/`: a reproducible subset of the recorded frozen SR tasks and a small AL/EV/EMP execution check.
- `third_party/`: upstream licensing and attribution sources for the empirical data.

Manuscript sources/PDFs, cover letters, author contact lists, reviews, internal session logs, credentials, caches and original server launchers are excluded. The previous figure-source snapshot is superseded by `source_data_revision/`.

## Environment

The checked environment used Python 3.13.5. The exact Python and package versions used for the accompanying checks are recorded in `VERIFICATION.json`; `requirements.txt` pins the four packages used in that environment.

```sh
python -m venv .venv
# Activate the environment using the command appropriate to your shell.
python -m pip install -r requirements.txt
```

## Recreate the figures

Run from the repository root:

```sh
python code_revision/rev_figures.py source_data_revision outputs/figures
```

This writes `Fig1`–`Fig5` and `EDFig1`–`EDFig3` as PDF and 300-dpi PNG. It uses recorded source data and does not retrain the models. Fonts can differ between operating systems. A final comma-separated argument restricts the figures, for example `Fig2,Fig5`.

## Run evaluations

The revision runner accepts `task_file output_prefix shard nshards`. Create the output directory first, for example with `python -c "from pathlib import Path; Path('outputs').mkdir(exist_ok=True)"`.

```sh
python code_revision/rjobs.py examples/frozen_sr_checks.json outputs/frozen_sr 0 1
python code_revision/rjobs.py examples/small_family_smoke.json outputs/smoke 0 1
```

The first command runs the proportional rule, published interpolating rule and selected share program against SR on the original four frozen seeds, 1,024 games per seed. The second uses reduced training/evolution budgets and is an execution check, not a reproduction of the reported AL/EV results. Output files have `_0.jsonl` appended to the prefix. Check each record's `ok` value; the runner records task failures inside JSON rather than raising a nonzero shell exit status for every failed task.

The complete recorded revision task lists are in `results_revision/out/t_*.json`. They include exact rule source or black-box weights, seeds, library names and any configuration overrides. Running all tasks can require substantial computation. The runner resumes a prefix by skipping previously recorded task IDs; use a fresh prefix for a fresh run.

The 560-game empirical library is `code/emp_train_nointerp.npz`; its results are named `EMP_noint` in the recorded task outputs and frozen-validation CSV. The `EMP` field is the 680-game library, while `EMP_base` is the 280-game baseline library. Read these names explicitly when comparing reported columns.

## Search and calibration

`code/driver.py::run_search` accepts a caller-supplied `gen_fn(prompts)`; each response must contain the generated source under its `text` key. An LLM provider, credentials and model access are not bundled. `code/search.py` builds the original prompts and scores candidates through `evaluate.py` worker processes. Use an absolute `srcdir` pointing to `code/` and load `calib_params.json` and `seed_sets.json` from that directory. No API request was made in the checks supplied here. Replaying recorded candidates and obtaining the same outputs from a newly sampled LLM are different tasks.

Legacy calibration, black-box and final-battery scripts read parameter files relative to the current working directory; run them from `code/`, using explicit paths to the intended inputs and outputs. `build_human_data.py` rebuilds the all-game and no-interpolation libraries from the upstream per-round CSV; the supplied baseline-only library is an additional archived input.

## Protocol note

The legacy `code/final_eval.py` defaults to EV `N=1000`, `gens=600`, `tail=200`, with an optional `FINAL_CONF` environment override. The manuscript describes the original final battery with EV `N=600`, `gens=400`. These defaults are preserved. The original manuscript specifies two EV seeds. Using `N=600`, `gens=400`, `tail=150` and seeds 301 and 302 reproduces the archived proportional-rule EV mean to floating-point precision. The recovered override is supplied in `configs/legacy_final_battery.json`; set the `FINAL_CONF` environment variable to its JSON contents when using the legacy final-battery script. The revision runner defaults to `N=600`, `gens=400`, `tail=150` and uses the seeds supplied by each revision task. Consult `PROTOCOL_CHECK.json` for the limited historical-configuration check, and use the recorded revision task files for frozen validation. The example commands do not assert that all original experiments have been rerun.

## Data, attribution and licensing

The human observations originate from Koster, Pislár and colleagues, *Deep reinforcement learning can promote sustainable human behaviour in a common-pool resource problem* (Nature Communications, 2025), https://doi.org/10.1038/s41467-025-58043-7 . The upstream repository is https://github.com/google-deepmind/sustainable_behavior . Its README distinguishes Apache-2.0 software from CC-BY-4.0 non-software materials; see `third_party/UPSTREAM_README.md` and `NOTICE.md`. The derived feature libraries retain the upstream data attribution and CC-BY-4.0 terms.

Release terms: MIT for this study's own code, CC-BY-4.0 for this study's generated data and documentation, subject to the named third-party terms. See `LICENSE` and `NOTICE.md`.

Corresponding author: Chengyi Tu. Use the manuscript title and author list above when citing this unpublished study; no DOI for this study is claimed.

## Archived software version

Version **v1.0.0** is archived on Zenodo: https://doi.org/10.5281/zenodo.23836483 . The fixed version refers to Git commit `08fa430e8989e7602710aeca845e08fb4b4c7d72` and the `v1.0.0` release. Please cite the version DOI; the repository citation metadata records the same three authors, software title, MIT software license and version. Study data and third-party materials retain the terms in `NOTICE.md`.
