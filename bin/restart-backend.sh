#!/bin/bash

grep -q TWT48U.json /etc/crontab ||
    (echo "0 9 * * *	root	curl -o $(pwd)/TWT48U.json https://www.twse.com.tw/rwd/zh/exRight/TWT48U?response=json" | sudo tee -a /etc/crontab)

pkill -f uvicorn
sleep 1
poetry run -- uvicorn --host 0.0.0.0 --port 8080 --workers 2 backend.main:app 1>backend.log 2>&1 &
