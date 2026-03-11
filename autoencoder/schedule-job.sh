#!/bin/bash

API_URL="http://0.0.0.0:8000/submit"

CNETID=$1
SCRIPT_PATH=$2
VENV_PATH=$3

curl -X POST "$API_URL" \
     -H "Content-Type: application/json" \
     -d "{\"script_path\": \"$SCRIPT_PATH\", \"venv_path\": \"$VENV_PATH\", \"cnetid\": \"$CNETID\"}"
printf "\n"
