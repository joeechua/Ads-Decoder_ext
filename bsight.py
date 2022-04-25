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
      print(preds)
      #function to compare coords for IOU intersection/matching
   
#%%
testing(files)

# %%
data = load_symbols_annotation()
# for file in files:
#    truths = data[file]
#    print(data[file])

#%%
data[files[2]]

#%%


bbox1 = [100, 171, 220, 345, 'violence']
bbox2 = [106.0, 347.0, 228.0, 491.0, 'use condoms']

# %% from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation#:~:text=You%27re%20calculating%20the%20area,%2F%20(union_area%20-%20intersection_area)%20.
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ---------

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    #topleftx, toplefty, bototmrightx, bottomrighty
    #x1,y1,x2,y2
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
# %%
