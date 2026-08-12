#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
sudo -v
git pull --ff-only origin cluster
COMMIT=$(git rev-parse HEAD)
HEAD_POD=$(sudo k3s kubectl get pods -n ray-system -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}')
mapfile -t PODS < <(sudo k3s kubectl get pods -n ray-system -l ray.io/cluster=raycluster-kuberay -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')
for POD in "${PODS[@]}"; do
  CONTAINER=ray-worker; [[ "$POD" == *-head-* ]] && CONTAINER=ray-head
  sudo k3s kubectl cp python/hrllc_tsn_trainer.py "ray-system/$POD:/workspace/python/hrllc_tsn_trainer.py" -c "$CONTAINER"
  sudo k3s kubectl cp ns3/scratch/hrllc_tsn_mappo.cc "ray-system/$POD:/workspace/ns3/hrllc_tsn_mappo.cc" -c "$CONTAINER"
  sudo k3s kubectl exec -n ray-system "$POD" -c "$CONTAINER" -- sh -lc 'cp /workspace/ns3/hrllc_tsn_mappo.cc /workspace/external/ns-3.44/scratch/hrllc_tsn_mappo.cc && cd /workspace/external/ns-3.44 && ./ns3 build hrllc_tsn_mappo'
done
for FILE in configs/default_config.json configs/extension_qos.json cluster/submit_ns3_campaign_ray.py cluster/run_paper_figures_campaign.py cluster/aggregate_paper_figures.py python/paper_reference_full_suite.py; do
  sudo k3s kubectl cp "$FILE" "ray-system/$HEAD_POD:/workspace/$FILE" -c ray-head
done
for POD in "${PODS[@]}"; do
  [[ "$POD" == *-head-* ]] && continue
  for FILE in configs/default_config.json configs/extension_qos.json cluster/submit_ns3_campaign_ray.py cluster/run_paper_figures_campaign.py python/paper_reference_full_suite.py; do
    sudo k3s kubectl cp "$FILE" "ray-system/$POD:/workspace/$FILE" -c ray-worker
  done
done
ROOT="/cluster-results/paper-figures/$COMMIT-$(date +%Y%m%d-%H%M%S)"
LOG="/tmp/paper-figures-$COMMIT.log"
echo "RESULT_ROOT=$ROOT"
sudo k3s kubectl exec -n ray-system "$HEAD_POD" -- env PROJECT_GIT_COMMIT="$COMMIT" python3 /workspace/cluster/run_paper_figures_campaign.py --address auto --results-root "$ROOT" --rounds 1000 --iterations 256 --validation-seeds 20 --final-test-seeds 100 --seeds 42,43,44 --reference-scenarios 100 --max-in-flight 3 --timeout-seconds 14400 2>&1 | tee "$LOG"
sudo k3s kubectl exec -n ray-system "$HEAD_POD" -- python3 /workspace/cluster/aggregate_paper_figures.py --results-root "$ROOT"
test -f "$ROOT/campaign_manifest.json"
test -f "$ROOT/figures/combined_report.json"
test "$(find "$ROOT/figures" -maxdepth 1 -type f -name 'figure*.png' | wc -l)" -eq 7
test "$(find "$ROOT/figures" -maxdepth 1 -type f -name 'figure*.pdf' | wc -l)" -eq 7
DEST="results/paper-figures/$COMMIT"
mkdir -p "$DEST"
cp -a "$ROOT/figures/." "$DEST/"
cp "$ROOT/campaign_manifest.json" "$DEST/campaign_manifest.json"
git add -f "$DEST"
git commit -m "Add native ns-3 Figures 7-13 [skip cluster campaign]"
git push origin cluster
echo "DONE: Figures 7-13 validated and pushed from commit $COMMIT"
