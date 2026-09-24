#!/usr/bin/env bash
# Compatibility entry point.
exec bash "$(dirname "${BASH_SOURCE[0]}")/train_MPDesigner_debug.sh" "$@"
