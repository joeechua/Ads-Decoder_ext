# Ads-Decoder: Generating Symbolic Bounding Boxes and Labels for the given Images and Descriptors

This project focuses on combining the image features with the text descriptor and predict the symbolic regions with labels that draws the audience's attention. The idea of combining text embeddings with image features extracted by the image classifier is inspired from the [VQA](https://github.com/Shivanshu-Gupta/Visual-Question-Answering) model.

## Prerequisites
We use Anaconda virtual environment to run the code and install all required packages.

1. python 3.x
2. pytorch

```
conda install pytorch torchvision torchaudio -c pytorch
```

3. sklearn
    
```
pip install -U scikit-learn
```

4. gensim

```
pip install -U gensim
```

5. pycocotools

```
pip install pycocotools
```
    
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

To train the Text Faster R-CNN model, run the following command after finished training the Faster R-CNN model or you can choose to download the pretrained models using the [link](https://drive.google.com/file/d/1grz1hLD2C03j7DPhr42kDiOQUBFbqCS7/view?usp=sharing) and put file in the `outputs` directory.

    python text_rcnn_train.py

## Detection

After training and evaluating the model, a checkpoint file will be created and stored at `outputs` directory.

Before detection, make sure that the checkpoint file for Text Faster R-CNN model exist in the `outputs` directory or you can download from the links below.

* [Sentiments + Faster R-CNN](https://drive.google.com/file/d/1--5efB6h6qS3E-9kXEeiaMJBsLs4Mo4R/view?usp=sharing)
* [Strategies + Faster R-CNN](https://drive.google.com/file/d/1WvmPzvwKCqTCB3UqUqmETg-v9GxvQdR4/view?usp=sharing)

Upload your image(s) under `detect_input` directory and run the following command and replace the image name(s), descriptor, and phrase with your own choice.

    python detect.py 
     --files "detect_input/<image_1>.jpg detect_input/<image_2>.jpg" 
     --descriptor "<sentiments/topics/strategies>" --phrase "<any_phrase>" 
     --threshold "<any_float_value_between_0_and_1>"

Some examples of the detection result.

![alt text](detect_output/1.jpg?raw=true)

![alt text](detect_output/2.jpg?raw=true)

![alt text](detect_output/7.jpg?raw=true)

## References

```
Hussain, Zaeem, et al. "Automatic understanding of image and video
advertisements." 2017 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR). IEEE, 2017.

Ye, K., & Kovashka, A. (2018). Advise: Symbolism and external knowledge for decoding advertisements. 
Proceedings of the European Conference on Computer Vision (ECCV), 837-855.

Torchvision object detection finetuning tutorial. (n.d.). 
Retrieved January 16, 2022, from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html 
```
