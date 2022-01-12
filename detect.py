"""
Usage:
    python tools/detect.py --directory "image.jpeg" --descriptor "sentiments" --phrase "active" --threshold "0"
"""
import numpy as np
import cv2
import torch
import glob as glob
import pickle
import argparse

import text_rcnn as text
from preprocess import descriptors as desc

# Remove this when pushing
# from google.colab.patches import cv2_imshow

# Add arguments to parser
parser = argparse.ArgumentParser(description="Detection of symbolic bounding boxes and labels")
parser.add_argument('--directory', dest='directory', help='Input image directory', default='None', type=str)
parser.add_argument('--descriptor', dest='descriptor', help='Descriptor', default='None', type=str)
parser.add_argument('--phrase', dest='phrase', help='Image phrase', default='None', type=str)
parser.add_argument('--threshold', dest='threshold', help='Detection threshold', default='None', type=str)

# set the computation device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def detect(directory, phrase, descriptor="None", threshold="None"):
  if threshold == "None":
    detection_threshold = 0
  else:
    detection_threshold = float(threshold)
  
  print("Threshold:", detection_threshold)
    


  # TODO: change directory with trained model
  if descriptor == "strategies":
    model = 'outputs/strategies_model.pth.tar'
  elif descriptor == "topics":
    model = 'outputs/topics_model.pth.tar'
  # Default: sentiment model
  else:
    model = 'outputs/checkpoint_sentiments_tfasterrcnn.pth.tar'
    
    model = torch.load(model)['model']
    model = model.to(device)

  # Evaluate model
  model.eval()
  
  # Test image directory
  test_images = glob.glob(f"{directory}/*")
  print(f"Test instances: {len(test_images)}")

  # Label classes
  le = pickle.loads(open("outputs/le.pickle", "rb").read())
  CLASSES = le.classes_
  print("Classes:", len(CLASSES))

  text_embed = desc.TextEmbedModel()
  phrase_embed = text_embed.get_vector_rep(phrase)
  phrase_embed = [torch.from_numpy(phrase_embed).float()] 

  # for i in range(len(test_images)):
  for i in range(500, 507):
      # get the image file name for saving output later on
      image_name = test_images[i].split('/')[-1]

      image = cv2.imread(test_images[i])
      orig_image = image.copy()
      # BGR to RGB
      image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
      # make the pixel range between 0 and 1
      image /= 255.0
      # bring color channels to front
      image = np.transpose(image, (2, 0, 1)).astype(np.float)
      # convert to tensor
      # image = torch.tensor(image, dtype=torch.float).cuda()
      image = torch.tensor(image, dtype=torch.float)
      # add batch dimension
      image = torch.unsqueeze(image, 0)

      with torch.no_grad():
          outputs = model(image, phrase_embed)
      
      # load all detection to CPU for further operations
      outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
      # carry further only if there are detected boxes
      if len(outputs[0]['boxes']) != 0:
          boxes = outputs[0]['boxes'].data.numpy()
          scores = outputs[0]['scores'].data.numpy()
          # filter out boxes according to `detection_threshold`
          boxes = boxes[scores >= detection_threshold].astype(np.int32)
          draw_boxes = boxes.copy()

          # get all the predicted class names
          pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
          
          # draw the bounding boxes and write the class name on top of it
          for j, box in enumerate(draw_boxes):
              cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 0, 255), 2)
              cv2.putText(orig_image, pred_classes[j], 
                          (int(box[0]), int(box[1]-5)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                          2, lineType=cv2.LINE_AA)

          # cv2.imshow('Prediction', orig_image)
          # cv2_imshow(orig_image)
          cv2.waitKey(1)
          cv2.imwrite(f"detect_output/{image_name}", orig_image)
      print(f"Image {i+1}: {image_name} done...")
      print('-'*50)

  print('TEST PREDICTIONS COMPLETE')
  cv2.destroyAllWindows()


if __name__ == "__main__":
  from text_rcnn import *
  args = parser.parse_args()
  detect(directory=args.directory, phrase=args.phrase, descriptor=args.descriptor,
        threshold=args.threshold)
