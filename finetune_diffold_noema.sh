#!/usr/bin/env bash
# Compatibility entry point.
exec bash "$(dirname "${BASH_SOURCE[0]}")/finetune_MPDesigner_noema.sh" "$@"
