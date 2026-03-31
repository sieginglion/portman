#!/bin/bash

set -euo pipefail

poetry install --only=main --no-root
