# Ads-Decoder_ext: Generating Symbolic Bounding Boxes and Labels for the given Images and Descriptors

This project focuses on combining the image features with the text descriptor and predict the symbolic regions with labels that draws the audience's attention. The idea of combining text embeddings with image features extracted by the image classifier is inspired from the [VQA](https://github.com/Shivanshu-Gupta/Visual-Question-Answering) model. This is a continuation of [Ads-Decoder](https://github.com/yuhueilee/Ads-Decoder).

## Prerequisites
We use Anaconda virtual environment to run the code and install all required packages.

1. python >= 3.7
2. pytorch 1.10

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 
```
OR

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

3. sklearn 1.0
    
```
pip install -U scikit-learn
```

4. gensim 4.2

```
pip install -U gensim
```

5. pycocotools 2.0

```
pip install pycocotools
```
    
6. textblob 0.17.1

```
pip install -U textblob
python -m textblob.download_corpora
```

7. transformer 4.19

```
pip install transformers
```

8. sentence-transofmers 2.2

```
pip install -U sentence-transformers
```

9. nltk 3.4

```
pip install --user -U nltk
```


## Download Data
Download the data by running the command below in the terminal window at the `Ads-Decoder_ext` directory.

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
    
### MASSIVE M3

In this project, we used [MASSIVE M3](https://massive.org.au/index.html), a high performance computing platform, to train and evaluate the models. This section will provide guidance on how to run code on M3.

Note: Users need a registered HPC ID account (used to log in to MonARCH) or [request an account](https://docs.massive.org.au/M3/requesting-an-account.html) for M3.

1. Login to your account using `ssh` command. Please replace the username with your own.
```
ssh <username>@monarch.erc.monash.edu
```

2. Change directory to the folder where you want to run your code.
```
cd path/to/destinated/directory
```

3. Load the Anaconda module. To see the available modules, run `module avail anaconda`.
```
module load anaconda/2019.03-Python3.7-gcc5
```

4. Activate Anaconda virtual environment. To see how to create one, please visit this [link](https://docs.massive.org.au/M3/software/pythonandconda/python-anaconda.html#python-anaconda).
```
source activate path/to/your/virtual/environment
```

5. Clone this Git repository.
```
git clone https://github.com/joeechua/Ads-Decoder_ext.git
```

6. Change directory to `Ads-Decoder_ext`
```
cd Ads-Decoder_ext
```

7. Submit the job script. We provided a [sample job script](https://github.com/joeechua/Ads-Decoder_ext/blob/main/sentiments_fasterrcnn.job) in this repo as well. Please modify line 12 to the path of your Anaconda virtual environment.
```
sbatch <job_script_name>.job
```

Once the job has completed, the checkpoint files for the model will be created and stored under `outputs` directory. An output file will be created that logs the execution of the code.


## Detection

After training and evaluating the model, a checkpoint file will be created and stored at `outputs` directory.

Before detection, make sure that the checkpoint file for Text Faster R-CNN model and the fine-tuned sentence embedding model exist in the `outputs` directory or you can download from the links below.

* [QARBET](https://drive.google.com/file/d/1eR7fyBdK9y4jUhUwRm45mPYIf8jyClY3/view?usp=sharing)
* [Slogans + Faster R-CNN](https://drive.google.com/file/d/1Ac2BC5UjD3xH8ipBzou0H9GaK8R1ieti/view?usp=sharing)
* [Sentiments + Faster R-CNN](https://drive.google.com/file/d/171zmGcMgvabyNmebXqin98PIMk6oMpwp/view?usp=sharing)
* [Strategies + Faster R-CNN](https://drive.google.com/file/d/1c7w3qN4lXUAkgUz_0nvjDdRgJYrSTS8N/view?usp=sharing)
* [Topics + Faster R-CNN](https://drive.google.com/file/d/1wXylLpZR-Dd42C60vmwCN5Z1Ipg_EJXj/view?usp=sharing)
* [Sentence Embedding Model](https://drive.google.com/drive/folders/172xZs6xbRuBR8P6hmBXcvoOGXY4AtEE3?usp=sharing)

Upload your image(s) under `detect_input` directory and run the following command and replace the image name(s), descriptor, and phrase as well as threshold with your own choice.

    python detect.py \
     --files "detect_input/<image_1>.jpg detect_input/<image_2>.jpg" \
     --descriptor "<sentiments/topics/strategies>" --phrase "<any_phrase>" \
     --threshold "<any_float_value_between_0_and_1>"

The output image will be created and stored under the `detect_output` directory.

Some examples of the detection result using the sentiments as the descriptor.

<p align="center">
    <img src="detect_output/2.jpg?raw=true" height="350">
    <img src="detect_output/7.jpg?raw=true" height="350">
    <img src="detect_output/10.jpg?raw=true" height="350">
    <img src="detect_output/11.jpg?raw=true" height="350">
    <img src="detect_output/13.jpg?raw=true" height="350">
</p>

## Logical Backing for Model Choice

After evaluating each of the 5 improved descriptor models. We went with QARBERT (QA + RoBERTa) as our final model. This choice was made after choosing
to prioritise precision over recall for the purposes of this model. In the table below, we have listed the precision and recall for each model.

| Descriptor Model | Precision    | Recall        |
|------------------|--------------|---------------|
| QARBERT          | 0.4176706827 | 0.03455149502 |
| Sentiments       | 0.144278607  | 0.02511692361 |
| Slogans          | 0.2754491018 | 0.05492537313 |
| Topics           | 0.1368821293 | 0.02712545523 |
| Strategies       | 0.113110539  | 0.02505694761 |

## References

```
Gupta, S. (2020). Shivanshu-Gupta/Visual-Question-Answering. 
Retrieved from https://github.com/Shivanshu-Gupta/Visual-Question-Answering

Hussain, Zaeem, et al. "Automatic understanding of image and video
advertisements." 2017 IEEE Conference on Computer Vision and Pattern Recognition
(CVPR). IEEE, 2017.

Torchvision object detection finetuning tutorial. (n.d.). 
Retrieved January 16, 2022, from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html 

Ye, K., & Kovashka, A. (2018). Advise: Symbolism and external knowledge for decoding advertisements. 
Proceedings of the European Conference on Computer Vision (ECCV), 837-855.
```
