#!/bin/bash
cd /home/nwh/nips_15/github_repos/nips-noisepo
git add -A
git -c trailer.sign=false commit -m "feat: NeurIPS paper draft + code fixes

- Fix tokenizer mismatch: use chat template format for DPO preference data
- Add cosine and cyclic noise schedules to complete PLAN.md design space
- Fix eval_nacpo.py: add Qwen3.5 compatibility patch
- Fix run_nacpo_sweep.sh: move log() definition before first call
- Create paper/main.tex: full NeurIPS draft with theory, method, experiments
- Create paper/references.bib: placeholder citations for verification
- Add generate_paper_figures.py: automated figure generation from results"
