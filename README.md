# Ads-Decoder

## Prerequisites
We use Anaconda virtual environment to run the code and install all required packages.

1. python 3.x
2. pytorch

    `conda install pytorch torchvision torchaudio -c pytorch`

3. sklearn
    
    `pip install -U scikit-learn`

4. gensim

    `pip install -U gensim`

5. pycocotools

    `pip install pycocotools`

## Download Data
Download the data by running the command below.

    sh download_data.sh

## Preprocess
Preprocess on the symbols boxes annotation to remove the invalid bounding boxes (all boxes should have a positive width and height).

    python preprocess/boxes.py --symbol_annotation_filename "data/annotations/Symbols.json"

Preprocess on the symbols labels annotation to reduce the label of a box into one word and build a label encoder.

    python preprocess/labels.py --symbol_annotation_filename "data/annotations/Symbols.json"
