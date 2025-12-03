#!/usr/bin/env bash
set -euo pipefail

# Size guard for critical modules. Fails if any target file exceeds MAX LOC.
MAX=${MAX_LOC:-500}

paths=(
  "src/core/alignment/banded_swa*.rs"
  "src/core/alignment/banded_swa/**/*.rs"
  "src/pipelines/linear/batch_extension*.rs"
  "src/pipelines/linear/batch_extension/**/*.rs"
)

fail=0

while IFS= read -r -d '' file; do
  # Only check files that actually exist in the repository
  if [[ -f "$file" ]]; then
    loc=$(wc -l < "$file")
    if (( loc > MAX )); then
      echo "SizeGuard: $file has $loc LOC (max $MAX)" >&2
      fail=1
    fi
  fi
done < <(git ls-files -z ${paths[@]})

exit $fail
