"""
Usage:
    python detect.py --files "detect_input/1.jpg" --descriptor "sentiments" --phrase "active"
"""
import numpy as np
import cv2
import torch
import pickle
import argparse
from preprocess import descriptors as desc

# Add arguments to parser
parser = argparse.ArgumentParser(
    description="Detection of symbolic bounding boxes and labels"
)
parser.add_argument(
    "--files",
    dest="files",
    help="List of image files",
    default="None",
    type=lambda s: [item for item in s.split()],
)
parser.add_argument(
    "--descriptor", dest="descriptor", help="Descriptor", default="None", type=str
)
parser.add_argument(
    "--phrase", dest="phrase", help="Image phrase", default="None", type=str
)
parser.add_argument(
    "--threshold", dest="threshold", help="Detection threshold", default="0", type=str
)

# set the computation device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def detect(filelist, phrase, descriptor="None", detection_threshold="0"):

    detection_threshold = float(detection_threshold)

    # TODO: change directory with trained model
    if descriptor == "strategies":
        model = "outputs/strategies_model.pth.tar"
    elif descriptor == "topics":
        model = "outputs/topics_model.pth.tar"
    # Default: sentiment model
    else:
        model = "outputs/checkpoint_sentiments_tfasterrcnn.pth.tar"

    # Load model
    model = torch.load(model)["model"]
    model = model.to(device)

    # Evaluate model
    model.eval()

    # Label classes
    le = pickle.loads(open("outputs/le.pickle", "rb").read())
    CLASSES = le.classes_

    # Generate random color
    np.random.seed(10)  # seed value
    COLORS = [tuple(np.random.randint(256, size=3)) for _ in range(len(CLASSES))]
    COLORS = [(int(c[0]), int(c[1]), int(c[2])) for c in COLORS]

    text_embed = desc.TextEmbedModel()
    phrase_embed = text_embed.get_vector_rep(phrase)
    phrase_embed = [torch.from_numpy(phrase_embed).float()]

    for i in range(len(filelist)):
        # get the image file name for saving output later on
        image_name = filelist[i].split("/")[-1]

        image = cv2.imread(filelist[i])
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float64)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float)
        # add batch dimension
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            outputs = model(image, phrase_embed)

        # load all detection to CPU for further operations
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]["boxes"]) != 0:
            boxes = outputs[0]["boxes"].data.numpy()
            scores = outputs[0]["scores"].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()

            # get all the predicted class names
            pred_classes = [CLASSES[i] for i in outputs[0]["labels"].cpu().numpy()]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                # find the index of the predicted label class
                index = np.where(CLASSES == pred_classes[j])[0][0]

                cv2.rectangle(
                    orig_image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    COLORS[index],
                    2,
                )
                draw_text(
                    img=orig_image,
                    text=pred_classes[j],
                    pos=(int(box[0]), int(box[1])),
                    text_color_bg=COLORS[index],
                )

            # cv2.imshow('Prediction', orig_image)
            cv2.waitKey(1)
            print(f"Writing {image_name} to file...")
            cv2.imwrite(f"detect_output/{image_name}", orig_image)
        print(f"Image {i+1}: {image_name} done...")
        print("-" * 50)

    print("TEST PREDICTIONS COMPLETE")
    cv2.destroyAllWindows()


def draw_text(
    img,
    text,
    pos,
    text_color_bg,
    text_color=(255, 255, 255),
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
    font_thickness=1,
):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(
        img,
        text,
        (x, y + text_h),
        font,
        font_scale,
        text_color,
        font_thickness
    )


if __name__ == "__main__":
    args = parser.parse_args()
    detect(filelist=args.files, phrase=args.phrase, descriptor=args.descriptor, detection_threshold=args.threshold)
