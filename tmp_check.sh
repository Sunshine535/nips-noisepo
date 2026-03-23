#!/bin/bash
sleep 20
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p 27055 -i /home/nwh/.ssh/kun_ed25519 root@216.81.245.83 'cd /workspace/nips-noisepo && echo "=== LOG TAIL ===" && tail -60 run_full.log 2>/dev/null && echo "=== PROCESSES ===" && ps aux | grep -E "python|train_nacpo|run_all" | grep -v grep | head -5 && echo "=== GPU ===" && nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
