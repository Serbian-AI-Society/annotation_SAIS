#!/usr/bin/env bash
# Quick start: creates the dataset and loads 5 sample records.
# Run this from the argilla_setup directory.
#
# Usage:
#   chmod +x quickstart.sh
#   ./quickstart.sh

set -e

export ARGILLA_API_URL="https://serbian-ai-society-argilla-annotation.hf.space"
export ARGILLA_API_KEY="_GhUG7luAkNFuBOMzprOVbdyW_xIX4MMOQVxCQ5zQIZqnKkM75HZDJy5_duBgtAFpOXUPuwytbxTLEHwi-MlqrVd9k3sPGC04pVG45lM0kE"

echo "=== Installing argilla SDK ==="
pip install argilla -q

echo ""
echo "=== Testing connection ==="
python -c "
import argilla as rg
client = rg.Argilla(api_url='$ARGILLA_API_URL', api_key='$ARGILLA_API_KEY')
me = client.me
print(f'Logged in as: {me.username} (role: {me.role})')
workspaces = [w.name for w in client.workspaces]
print(f'Available workspaces: {workspaces}')
"

echo ""
echo "=== Creating dataset schema ==="
python setup_dataset.py

echo ""
echo "=== Loading 5 sample records ==="
python load_data.py --sample 5

echo ""
echo "=== Done! ==="
echo "Open your Argilla UI and you should see 'translation-annotation-sr' in the dataset list."
echo "URL: $ARGILLA_API_URL"
