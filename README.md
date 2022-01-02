# Ads-Decoder

## Prerequisites

## Download Data
Download the data by running the command below.

`sh download_data.sh`

## Preprocess
Preprocess on the symbols boxes annotation to remove the invalid bounding boxes (all boxes should have a positive width and height).

`python preprocess/boxes.py --symbol_annotation_filename "data/annotations/Symbols.json"`

Preprocess on the symbols labels annotation to reduce the label of a box into one word and build a label encoder.

`python preprocess/labels.py --symbol_annotation_filename "data/annotations/Symbols.json"`