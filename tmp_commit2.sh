#!/bin/bash
cd /home/nwh/nips_15/github_repos/nips-noisepo
rm -f tmp_commit.sh
git add -A
git -c trailer.sign=false commit -m "fix: resilient data prep with retry logic for rate-limited HF downloads"
git push origin main
