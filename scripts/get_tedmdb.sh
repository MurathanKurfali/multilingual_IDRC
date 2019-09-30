#!/usr/bin/env bash
# Self-contained script to download and parse Ted-MDB annotations

git clone https://github.com/MurathanKurfali/Ted-MDB-Annotations.git
echo "TED-MDB raw annotations cloned"
python utils/extract_tedmdb.py --dir Ted-MDB-Annotations
echo "Annotations are extracted to \"data/text\""
rm -rf Ted-MDB-Annotations
echo "Raw annotations are deleted"
./scripts/prepare_laser_embeddings.sh ted
echo "Laser embeddings for TED-MDB annotations are created"