#!/bin/bash

pkill -f 'src\/main\.py'
.venv/bin/python src/main.py 1>main.log 2>&1 &
