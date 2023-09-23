#!/usr/bin/env bash

set -euo pipefail

pkill -f uvicorn && sleep 1
poetry run -- uvicorn --port 8080 backend.main:app 1>backend.log 2>&1 &
