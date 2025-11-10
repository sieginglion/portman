#!/bin/bash

set -euo pipefail

pkill -f uvicorn && sleep 1
poetry run -- uvicorn --host 0.0.0.0 --port 8080 --workers 1 backend.main:app 1>backend.log 2>&1 &
