"""Thin lm-eval CLI wrapper that registers the in-repo `hf-bos` model before dispatching.

`python -m lm_eval` does not import our custom model, so `--model hf-bos` would not resolve. Importing
bos_hf_model here triggers its @register_model decorator; we then hand off to lm-eval's stock CLI.
Use exactly like `python -m lm_eval ...` (same flags). Run from the multilingual/ directory so that
`scripts/` is on sys.path and `tasks/` is reachable via --include_path.
"""
import bos_hf_model  # noqa: F401  (registers the hf-bos model)
from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
