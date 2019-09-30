# Zero-shot transfer for implicit discourse relation classification

This repository contains the implementation of the model presented at the "Zero-shot transfer for implicit discourse relation classification" paper.

## USAGE

#### Preparing the Data: 
You can prepare the data for the model via "prepare_data.sh" script. This scripts only needs the directory of PDTB3 annotations.
./prepare_data.sh pdtb3_annotations #  where pdtb3 annotations are saved under the directory "pdtb3_annotations"

As a result, the extracted sentences as well as their laser embeddings will be saved to data/text data/embed respectively. Note: Since Ted-MDB annotations are publicly available, that part of the script is self-contained

#### Training the model
To train the model, you can simply run the "run.sh" script as follows:

./run.sh pdtb3 saved_models

This script will
<ol type="a">
<li> train a separate "one vs. all" classifier for each sense using the PDTB3 annotations </li>
<li> save the models under "saved_models/" dir </li>
<li> report the performance of the model on all Ted-MDB languages as well as PDTB3 test set. </li>
</ol>

