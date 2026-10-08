#!/bin/bash
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
uv_bin="${UV_BIN:-/home/sagara/.local/bin/uv}"
search_dir="${SEARCH_DIR:-$repo_root/outputs/real_cv/framingham/lasso_klambda_5fold_seed1234}"
action="${1:-}"
if [ -z "$action" ]; then
    echo "Usage: bash $0 {prepare|submit|run|aggregate|refit|submit-refit} [options]" >&2
    exit 2
fi
shift
cd "$repo_root"
case "$action" in
    prepare)
        "$uv_bin" run python scripts/real_cv/klambda_search.py prepare \
            --search-dir "$search_dir" --config "${CONFIG_PATH:-$repo_root/config.toml}" \
            --grid "${SEARCH_GRID:-$repo_root/framingham_klambda_grid.json}" \
            --input "${FRAMINGHAM_INPUT:-$repo_root/data/real/framingham/framingham.csv}" \
            --splits "${SPLITS_FILE:-$repo_root/data/real/cv/splits/framingham/framingham_5fold_seed1234.csv}" \
            --n-folds "${N_FOLDS:-5}" "$@"
        ;;
    submit-refit)
        "$uv_bin" run python scripts/real_cv/klambda_search.py submit \
            --search-dir "$search_dir" --refit "$@"
        ;;
    submit|run|aggregate|refit)
        "$uv_bin" run python scripts/real_cv/klambda_search.py "$action" \
            --search-dir "$search_dir" "$@"
        ;;
    *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
