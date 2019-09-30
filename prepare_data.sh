#!/usr/bin/env bash

# prepare data: extract and create laser embeddings for Ted- MDB and PDTB3 annotations.
# Ted_MDB script is self-contained, therefore, you only need to provide the directory for PDTB3 annotations
# the annotations/embeddings will be saved under data/text data/embed respectively.

./scripts/get_tedmdb.sh
python utils/extract_pdtb_annotations.py --dir $1
echo "PDTB3 annotations are extracted"
./scripts/prepare_laser_embeddings.sh pdtb3
echo "Laser embeddings for PDTB3 annotations are created"
