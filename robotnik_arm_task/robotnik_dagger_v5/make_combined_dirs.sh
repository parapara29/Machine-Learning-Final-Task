#!/usr/bin/env bash
# Rebuild the aggregated DAgger folders used to train V5.3-A and V5.4 (relative symlinks).
#   arm_dagger_v5_3_combined = round 1 (arm_dagger_v5) + round 2 (arm_dagger_v5_3)       -> V5.3-A
#   arm_dagger_v5_4_combined = the above + round 3 (arm_dagger_v5_3_randomized)          -> V5.4
# Round-2 and round-3 files share names (same episode IDs and seeds), so round-3 links get
# a "_rand" suffix. Run from robotnik_dagger_v5/.
set -euo pipefail
cd "$(dirname "$0")"
link_all() {  # link_all <src_dir> <dst_dir> [suffix]
  local src=$1 dst=$2 suffix=${3:-}
  [ -d "$src" ] || { echo "missing $src (skipped)"; return; }
  for f in "$src"/dagger_*.csv; do
    ln -sfn "../$f" "$dst/$(basename "$f" .csv)$suffix.csv"
  done
}
rm -rf arm_dagger_v5_3_combined arm_dagger_v5_4_combined
mkdir -p arm_dagger_v5_3_combined arm_dagger_v5_4_combined
for d in arm_dagger_v5_3_combined arm_dagger_v5_4_combined; do
  link_all arm_dagger_v5 "$d"; link_all arm_dagger_v5_3 "$d"
done
link_all arm_dagger_v5_3_randomized arm_dagger_v5_4_combined _rand
echo "v5_3_combined: $(ls arm_dagger_v5_3_combined | wc -l) rollouts (expect 161)"
echo "v5_4_combined: $(ls arm_dagger_v5_4_combined | wc -l) rollouts (expect 1561)"
