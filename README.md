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
    
6. textblob

    ```
    pip install -U textblob
    python -m textblob.download_corpora
    ```

## Download Data
Download the data by running the command below.

    sh download_data.sh

## Preprocess
Preprocess on the symbols boxes annotation to remove the invalid bounding boxes (all boxes should have a positive width and height).

    python preprocess/boxes.py --symbol_annotation_filename "data/annotations/Symbols.json"

Preprocess on the symbols labels annotation to reduce the label of a box into one word and build a label encoder.

    python preprocess/labels.py --symbol_annotation_filename "data/annotations/Symbols.json"

## Train & Evaluate the Model

To train the Faster R-CNN model, run the following command.

    python fasterrcnn_train.py

To train the Text Faster R-CNN model, run the following command after finished training the Faster R-CNN model.

    python text_rcnn_train.py

## Detection

After training and evaluating the model, a checkpoint file will be created and stored at `outputs` directory.

Before detection, make sure that the checkpoint file for both Faster R-CNN and Text Faster R-CNN model exist in the `outputs` directory.

You can choose to download the pretrained models using the link below.

* Faster R-CNN (6 epochs): https://drive.google.com/file/d/1grz1hLD2C03j7DPhr42kDiOQUBFbqCS7/view?usp=sharing
* Text Faster R-CNN (3 epochs): 

Upload your image under `detect_input` directory and run the following command.

    python detect.py --files "detect_input/<image_name>.jpg" --descriptor "sentiments" --phrase "active"