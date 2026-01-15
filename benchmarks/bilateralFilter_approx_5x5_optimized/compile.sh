#!/usr/bin/env bash
set -euo pipefail
exec make -C "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" "$@"
