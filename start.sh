#!/bin/bash
echo "🏀 Starting Titans Auto Tracker..."
cd "$(dirname "$0")"
pip3 install -q -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
