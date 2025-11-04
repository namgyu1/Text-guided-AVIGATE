#!/bin/bash
# Quick push to update server code

echo "=== Committing changes ==="
git add modules/modeling.py main_task_retrieval.py run_eval.sh
git commit -m "Fix: Aggressive audio chunking (chunk_size=2) and CPU offloading"

echo -e "\n=== Pushing to remote ==="
git push

echo -e "\n=== Done! ==="
echo "Now SSH to your server and run:"
echo "  cd /data03/namgyu/AVIGATE_2"
echo "  git pull"
echo "  bash run_eval.sh"
