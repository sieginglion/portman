#!/bin/bash

grep -q TWT48U.json /etc/crontab ||
    (echo "0 9 * * *	root	curl -o $(pwd)/TWT48U.json https://www.twse.com.tw/rwd/zh/exRight/TWT48U?response=json" | sudo tee -a /etc/crontab)

pkill -f uvicorn
poetry run -- uvicorn --host 0.0.0.0 --port 8080 portman.main:app 1>main.log 2>&1 &
