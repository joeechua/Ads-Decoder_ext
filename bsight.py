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
from fasterrcnn_train import create_train_test_dataset
from dataset import AdsDataset
ads_dataset = AdsDataset()
from detect import detect
from preprocess.boxes import load_symbols_annotation
from preprocess.descriptors import load_annotation_json, SentimentPreProcessor

#%%
train, test = create_train_test_dataset(ads_dataset)

# %% raises an error
print(test.dataset[0])

#%% first 10 files for testing the function
files = test.dataset.image_path[:10]

# %% get the annotations for said files.
def testing(files):
   data = load_symbols_annotation() #loads the symbols.json file
   phrases = load_annotation_json() #loads the sentiments.json file
   prep = SentimentPreProcessor() #loads a dictionary for indexing phrase
   #get phrase, truth, and predicted bbox for each file
   for file in files:
      phrase_id = int(phrases[file][0][0])
      phrase = prep.id_to_word[phrase_id]
      filepath = "data/" + file
      preds = detect([filepath],  phrase, "sentiments", "0", True)
      truths = data[file]
      #function to compare coords for IOU intersection/matching
   
#%%
testing(files)

# %%
data = load_symbols_annotation()
for file in files:
   truths = data[file]
   print(data[file])
# %%

   
   