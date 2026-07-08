# MECO Reading Time Evaluation

Psychometric evaluation for the BabyLM multilingual track: how well a model's
per-word **surprisal** predicts human reading times on the
[MECO](https://meco-read.com/) eye-tracking corpus, measured as the delta
log-likelihood of a mixed-effects model with surprisal over a matched baseline
without it.

Two evaluations:

- **MECO L1** — native reading (English, Dutch, Chinese), one score per language
  in the model's training mixture.
- **MECO L2** — reading English as a second language, scored per L1 reader group
  (Dutch / German / Chinese), run when English is in the mixture.

For each language:

```
normalized_loglik = logLik(surprisal model) − logLik(baseline model)
```

both fit by maximum likelihood (`REML = FALSE`) with by-subject random
intercepts and uncorrelated random slopes for surprisal, its 1- and 2-word
spillover, word length, log frequency, and word position.

This is a faithful, dependency-light reimplementation of the original R/lme4
pipeline: **no R, lme4, tidyverse, or reticulate required.** The mixed-model fit
is solved directly (profiled-deviance ML with a blockwise dense Cholesky and
BOBYQA), reproducing `lme4`'s `logLik` to a small tolerance.

## Install

```bash
pip install -r requirements.txt
```

## Run

Same ergonomics as the other multilingual tasks — track language codes
`eng` / `nld` / `zho`. Run from the parent `multilingual/` directory (the driver
script now lives in the shared `multilingual/scripts/` folder alongside the other
eval scripts):

```bash
bash scripts/meco_model.sh --model_name ORG/YOUR_MODEL --langs "eng nld zho"
# or directly:
python -m meco.meco_py.cli --model_name ORG/YOUR_MODEL --langs "eng nld zho"
```

Per requested language it computes native reading (`meco_l1`) and, when English
is in the mixture, English-as-L2 reading (`meco_l2`) for the non-English reader
groups. Incomplete evaluation is allowed: languages the model does not cover are
skipped.

Key options: `--revision` (checkpoint), `--langs "eng nld"`, `--l2 auto|on|off`,
`--backend auto|causal|mlm|seq2seq|mamba`, `--device cpu|cuda|mps`, `--data_dir`,
`--out_dir`, `--output`.

### Backends (model architectures)

`--backend auto` (default) detects the right minicons scorer from the model
config, so all common BabyLM architectures work:

| backend | scorer | covers |
|---|---|---|
| `causal` | `IncrementalLMScorer` | GPT-2, Llama, Mistral, **MoE** decoders (Mixtral / OLMoE / Qwen-MoE) |
| `mlm` | `MaskedLMScorer` | BERT-style, GPT-BERT masked head |
| `seq2seq` | `Seq2SeqScorer` | T5 / BART / mT5 |
| `mamba` | `MambaScorer` | Mamba / state-space models |

A mixture-of-experts model is not a separate backend — it loads through the same
causal (or masked) path as its dense counterpart.

### Output

Raw per-word surprisals are written to
`<out_dir>/<revision>/meco_<org__model>/predictions_meco.json`. The multilingual
collator includes them in the predictions upload, and the leaderboard computes
the official delta log-likelihood scores server-side.

For local diagnostics, the script also writes
`results_meco.json` in `{task: {lang: score}}` format (higher
`normalized_loglik` is better):

```json
{
  "meco_l1": {"eng": 5225.28, "nld": 425.91, "zho": 643.40},
  "meco_l2": {"nld": 88.15, "zho": 95.31}
}
```

A `meco_delta_loglik_<model>.csv` with the full logLik / baseline detail is
written alongside it.

## Data

The MECO reading-time `.rda` files and the stimulus TSVs are under `--data_dir` as:

```
<data_dir>/meco_l1/{wave1_joint_l1_data_trimmed_version2.1.rda,
                    wave2_joint_data_trimmed_wave2_version2.1.rda}
<data_dir>/meco_l2/{joint_data_l2_trimmed_version2.2.rda,
                    joint_data_trimmed_L2_wave2_version2.2.rda}
<data_dir>/meco_l1_stims.tsv
<data_dir>/meco_l2_stims.tsv
```

```
meco_py/
  frame.py       front-end: .rda -> per-language fit (wordfreq, spillover, filters)
  lmm.py         Python Implementation of Linear Mixed Effects Model 
  fit.py         scale + na.omit + delta log-likelihood per language
  surprisal.py   per-word surprisal via minicons (causal / mlm / seq2seq / mamba)
  cli.py         orchestrator: model -> surprisal -> submission JSON + detail CSV
```




Please cite the following:

Implementation
```
@inproceedings{salhan2026structured,
  title     = {Structured Exposure Pretraining in Bilingual Language Models for Modelling L2 Language Processing},
  author    = {Salhan, Suchir and Arnett, Catherine and Michaelov, James A. and Buttery, Paula},
  year      = {2026},
}
```

MECO (Wave 1 and Wave 2)
```
@article{siegelman2022meco,
  title   = {Expanding horizons of cross-linguistic research on reading: The Multilingual Eye-movement Corpus (MECO)},
  author  = {Siegelman, Noam and Schroeder, Sascha and Acart{\"u}rk, Cengiz and Ahn, Hyeonjeong and Alexeeva, Svetlana and Amenta, Simona and Bertram, Raymond and Bonandrini, Romina and Brysbaert, Marc and Chernova, Daria and Da Fonseca, Sara M. and Dirix, Nicolas and Duyck, Wouter and Fella, Anna and Frost, Ram and Gattei, Cristina A. and Kalaitzi, Alexandra and Kwon, Nayoung and L{\~o}o, Kristiina and Marelli, Marco and Papadopoulos, Timothy C. and Protopapas, Athanassios and Savo, Sara and Shalom, Diego E. and Slioussar, Natalia and Stein, Rina and Sui, Ling and Taboh, Alain and T{\o}nnesen, Vera and Usal, K{\"u}bra A. and Kuperman, Victor},
  journal = {Behavior Research Methods},
  volume  = {54},
  number  = {6},
  pages   = {2843--2863},
  year    = {2022},
  doi     = {10.3758/s13428-021-01772-6}
}
```

```
@article{siegelman2025meco2,
  title   = {Wave 2 of the Multilingual Eye-Movement Corpus (MECO): New text reading data across languages},
  author  = {Siegelman, Noam and Schroeder, Sascha and Bao, Yu B. and Acart{\"u}rk, Cengiz and Agrawal, Niharika and Bolliger, Lea S. and Brasser, Judith and Campos-Rojas, Cristian and Drieghe, Denis and Filipovi{\'c} \DJ ur{\dj}evi{\'c}, Du{\v{s}}ica and Goldina, Svetlana and Ib{\'a}{\~n}ez Orellana, Rodrigo and J{\"a}ger, Lena A. and J{\'o}hannesson, {\'O}lafur I. and Khare, Anjali and Kharlamov, Nikita and Knudsen, Hanne B. S. and Kristj{\'a}nsson, {\'A}rni and Lee, Chi Eon and Lee, Jiyeon R. and Leite, Marcos P. T. and Mancini, Simona and Mihajlovi{\'c}, Nikola and Mi{\v{s}}i{\'c}, Katarina and Orekhova, Maria and Parshina, Olga and Popovi{\'c} Stija{\v{c}}i{\'c}, Milica and Protopapas, Athanassios and Reich, Daniel R. and Rimzhim, Ariuna and Rothe-Neves, Rui and S{\'a}, Thiago M. M. and Santana-Covarrubias, Andr{\'e}s and Sekerina, Irina and Sigurdardottir, Hanna M. and Smirnova, Anastasia and Srivastava, Priyanka and Teixeira, Eduardo N. and Ugrinic, Ivana and Usal, K{\"u}bra A. and Vakulya, Katalin and Verma, Ananya and Vieira, Jo{\~a}o M. M. and Wu, Daniel H. and Xue, Jing and Zdravkovi{\'c}, Slobodanka and Zhuo, Jun and Ziaka, Lida and Kuperman, Victor},
  journal = {Scientific Data},
  volume  = {12},
  number  = {1},
  year    = {2025},
}
```

```
@article{kuperman2025meco2l2,
  title   = {New data on text reading in English as a second language: The Wave 2 expansion of the Multilingual Eye-Movement Corpus (MECO)},
  author  = {Kuperman, Victor and Schroeder, Sascha and Acart{\"u}rk, Cengiz and Agrawal, Niharika and Alexandre, D. M. and Bolliger, Lea S. and Brasser, Judith and Campos-Rojas, Cristian and Drieghe, Denis and \DJ ur{\dj}evi{\'c}, Du{\v{s}}ica F. and Gadelha De Freitas, L. V. and Goldina, Svetlana and Orellana, Rodrigo I. and J{\"a}ger, Lena A. and J{\'o}hannesson, {\'O}lafur I. and Khare, Anjali and Kharlamov, Nikita and Knudsen, Hanne B. S. and Kristj{\'a}nsson, {\'A}rni and Lee, Chi Eon and Lee, Jiyeon R. and Leite, Marcos P. T. and Mancini, Simona and Mihajlovi{\'c}, Nikola and Mi{\v{s}}i{\'c}, Katarina and Orekhova, Maria and Parshina, Olga and Popovi{\'c} Stija{\v{c}}i{\'c}, Milica and Protopapas, Athanassios and Reich, Daniel R. and Rimzhim, Ariuna and Rothe-Neves, Rui and S{\'a}, Thiago M. M. and Santana-Covarrubias, Andr{\'e}s and Sekerina, Irina and Sigurdardottir, Hanna M. and Smirnova, Anastasia and Srivastava, Priyanka and Teixeira, Eduardo N. and Ugrinic, Ivana and Usal, K{\"u}bra A. and Vakulya, Katalin and Vieira, Jo{\~a}o M. M. and Verma, Ananya and Wu, Daniel H. and Xue, Jing and Zdravkovi{\'c}, Slobodanka and Zhuo, Jun and Ziaka, Lida and Siegelman, Noam},
  journal = {Studies in Second Language Acquisition},
  pages   = {1--19},
  year    = {2025},
}
```
