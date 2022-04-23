#%%
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import AdsDataset
from fasterrcnn_train import create_train_test_dataset
import tools.transforms as T
import tools.utils as utils
"""
- my understanding is that i can take the boxes output and compare with the boxes values given by the data.
- if +- some amount punya distance then ok? - ok turns out this is IOU 
-  Final report says IOU not goof enough, need to know if the symbol is detected correctly so that means 
   we'll need to get the word from the back and compare to the topic we detected and see how similar they are? WordEmbedModel??

- I think the evaluation can just do with the IOU first at least for the week 8 meeting.
- Ms Sailaja said we shld do precision, recall, and accuracy
- precision = true positive/ all positive predictions
- recall = true positive/ actual positive
- accuracy is accuracy

- but they have coco_eval and evaluate??? - ask abt this ig

Seniors Suggested
- add SentenceEmbreddeinModel to be able to handle Q&A + Slogans
- k-clustering for labels (they have a file /preprocess/clustered_symbol_lists)
"""
#%%
from dataset import AdsDataset

#%% i hv no idea what they used as train and test set so.... Random select 50% and check on all of the trained models first
ads_dataset = AdsDataset()

#%%
train, test = create_train_test_dataset(ads_dataset)
test.dataset.image_path

# %%
print(test.dataset[0])

#%%
from detect import detect
from preprocess.boxes import load_symbols_annotation

#%%
from preprocess.descriptors import load_annotation_json, SentimentPreProcessor

#%%
load_annotation_json()
# %% get the annotations for said files.

def testing():
   data = load_symbols_annotation()
   phrases = load_annotation_json("data/annotations/Sentiments.json")
   phrase_id = int(phrases["10/170489.png"][0][0])
   prep = SentimentPreProcessor()
   phrase = prep.id_to_word[phrase_id]
   results = detect(["data/10/170489.png"], phrase, "sentiments", "0", True)
   print(results)
   

testing()
   # for file in test.dataset.image_path:
   #    filepath = "data/" + file
   #    truth = data[file]
   #    phrase = phrases[file]
   #    pred = detect([filepath],phrase,"sentiments", "0", True ) #make it return list of values like it is in the symbols.json
   #    #compare bounding boxes ONLY(? if i do bbox only then how to do eval other than ) with a +- some value
   #    #if onz then save true?
# %%
