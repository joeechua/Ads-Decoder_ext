import pickle
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from dataset import AdsDataset
from tools.evaluate import evaluate
from tools.engine import train_one_epoch
from tools.evaluate import evaluate
import tools.transforms as T
import tools.utils as utils
from model import create_model

def get_transform(train: bool):
    """Return the transform function

    Args:
        train (bool): whether the transform is applied on training dataset

    Returns:
        func: transform function on image and target
    """
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def create_train_test_dataset(dataset: AdsDataset):
    """Split the dataset into training and testing

    Args:
        dataset (AdsDataset): a Pytorch Dataset

    Returns:
        (AdsDataset, AdsDataset): train dataset, test dataset
    """
    # randomly select the training and testing indices
    #indices = list(range(len(dataset)))
    indices = list(range(16))
    print(indices)
    train_indices, test_indices = train_test_split(
        indices, train_size=0.85, shuffle=True, random_state=24)

    # split the dataset into train and test
    train_dataset = torch.utils.data.Subset(AdsDataset(
        transforms=get_transform(train=True)), train_indices)
    test_dataset = torch.utils.data.Subset(AdsDataset(
        transforms=get_transform(train=False)), test_indices)

    return train_dataset, test_dataset


def train(num_classes: int, num_epochs: int, checkpoint=None, batch_size=8, num_workers=1):
    """Train the model

    Args:
        num_classes (int): number of label classes
        num_epochs (int): number of epochs to train the model
        checkpoint (str, optional): path to the checkpoint file. Defaults to None.
        batch_size (int, optional): batch size. Defaults to 8.
        num_workers (int, optional): number of workers. Defaults to 1.
    """
    # create training & testing dataset
    ads_dataset = AdsDataset()
    text_embed_size = ads_dataset.descriptor_preprocessor.embed_size
    train_dataset, test_dataset = create_train_test_dataset(ads_dataset)

    # define training data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=utils.collate_fn)

    # define testing data loaders
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        # get the model using our helper function
        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # # get number of input features for the classifier
        # in_features = model.roi_heads.box_predictor.cls_score.in_features
        # # replace the pre-trained head with a new one
        # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        model = create_model(num_classes)
        # specify text embedding size
        model.text_embed_size = text_embed_size

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # move model to the right device
    model.to(device)

    # construct a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

    # create lists to store statistics
    metric_logs, coco_evals = dict(), dict()

    print(metric_logs, coco_evals)

    # training
    for epoch in range(start_epoch, start_epoch + num_epochs):

        # train for one epoch, printing every 10 iterations
        curr_log = train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=len(train_dataset))

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        curr_eval = evaluate(model, test_dataloader, device=device, print_freq=len(test_dataset))

        # save checkpoint
        utils.save_checkpoint(epoch, model, optimizer)

        print(curr_log)
        print(curr_eval)
        print("Current epoch done")
    print("_________DONE_________")


if __name__ == "__main__":
    le = pickle.loads(open("outputs/le.pickle", "rb").read())
    train(num_classes=len(le.classes_), num_epochs=2)

