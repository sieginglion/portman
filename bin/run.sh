#!/bin/bash

set -euo pipefail

poetry run -- uvicorn --host 0.0.0.0 --port 8080 --workers 1 backend.main:app
