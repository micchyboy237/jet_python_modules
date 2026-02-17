#!/usr/bin/env bash
# Script: whereami.sh
# Prints absolute path of this script + its directory (even when called via symlink / relative path)

set -euo pipefail

# ───────────────────────────────────────────────
# Get script's absolute path (resolves symlinks)
# ───────────────────────────────────────────────

# Prefer BASH_SOURCE (works when sourced or in functions too)
script="${BASH_SOURCE[0]:-$0}"

# Try best available realpath/readlink fallback chain
if command -v realpath >/dev/null 2>&1; then
    fullpath=$(realpath -- "$script" 2>/dev/null)
elif command -v readlink >/dev/null 2>&1 && readlink -f . >/dev/null 2>&1; then
    fullpath=$(readlink -f -- "$script" 2>/dev/null)
else
    # Pure POSIX fallback — no symlink resolution but correct dir
    cd -P -- "$(dirname -- "$script")" >/dev/null 2>&1 || exit 1
    fullpath="$PWD/$(basename -- "$script")"
fi

# Make sure we have something sensible
fullpath="${fullpath:-$script}"

# ───────────────────────────────────────────────
# Derive directory — use dirname on resolved path
# ───────────────────────────────────────────────
script_dir=$(dirname -- "$fullpath")

# ───────────────────────────────────────────────
# Output — clean and structured
# ───────────────────────────────────────────────

printf '%s\n' "Script path : $fullpath"
printf '%s\n' "Directory   : $script_dir"


echo "Running basic usage example..."
python "$script_dir/basic_usage.py"

echo -e "\nRunning metadata filter example..."
python "$script_dir/with_metadata_filter.py"

echo -e "\nRunning realistic documents example..."
python "$script_dir/more_realistic_documents.py"
