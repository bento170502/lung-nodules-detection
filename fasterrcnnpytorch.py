import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch.utils.model_zoo as model_zoo
model_urls = {
    'fasterrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth',
}


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = self.annotations.iloc[index, 0]
        # Convert to grayscale
       
        # Convert to tensor
        image = Image.open(img_path).convert("L")

        classes = ['node']
        class_dict = {c: i for i, c in enumerate(classes)}

        # Convert class label to one-hot vector
        class_label = self.annotations.iloc[index, 1]
        one_hot_label = [0] * len(classes)
        one_hot_label[class_dict[class_label]] = 1
        y_label = torch.tensor(one_hot_label)

        xmin = self.annotations.iloc[index, 2]
        ymin = self.annotations.iloc[index, 3]
        xmax = self.annotations.iloc[index, 4]
        ymax = self.annotations.iloc[index, 5]
        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        labels = torch.tensor([y_label], dtype=torch.float32).squeeze(0)
        target = {"boxes": boxes, "labels": labels}

        return image, target
"""def get_faster_rcnn(num_classes):
    # Load a pre-trained model for the COCO dataset
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace the box predictor with a new one for our dataset with a single class
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the backbone to handle single channel input
    backbone = torchvision.models.resnet50(weights='fasterrcnn_resnet50_fpn_coco')
    #removes the last two layers of the ResNet50 model using Sequential() and list()
    # The last two layers are the global average pooling layer and the fully connected layer. 
    # This is because the Faster R-CNN model expects the backbone network to output feature maps 
    # instead of class probabilities.
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    #This is because the ResNet50 backbone has a final output feature map of size 
    #7x7 with 2048 channels. The output feature map is used as input to the Region Proposal Network (RPN) 
    #to generate proposals for object regions.
    backbone.out_channels = 2048

    # Create the anchorboxes generator with custom sizes
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # Create the ROI align head
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # Replace the model backbone, rpn_anchor_generator, and box_roi_pool with the new ones
    model.backbone.body = backbone
    model.rpn.anchor_generator = anchor_generator
    model.roi_heads.box_roi_pool = roi_pooler

    return model
"""


def get_faster_rcnn(num_classes):
    # Load a pre-trained model for the COCO dataset
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False)

    state_dict = model_zoo.load_url(model_urls['fasterrcnn_resnet50_fpn_coco'])
    model.load_state_dict(state_dict)

    # Replace the box predictor with a new one for our dataset with a single class
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the backbone to handle single channel input
    backbone = torchvision.models.resnet50(pretrained=False, progress=True)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
    backbone.out_channels = 2048

    # Create the anchorboxes generator with custom sizes
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # Create the ROI align head
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    # Replace the model backbone, rpn_anchor_generator, and box_roi_pool with the new ones
    model.backbone.body = backbone
    model.rpn.anchor_generator = anchor_generator
    model.roi_heads.box_roi_pool = roi_pooler

    return model





csv_file = "path/to/your/csv_file.csv"
root_dir = "path/to/your/images"
model_save_path = "path/to/save/model.pth"
def train_model(csv_file,  num_classes, model_save_path):
    # Create dataset and dataloader
    dataset = CustomDataset(
        csv_file=csv_file,  transform=ToTensor())
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True,
                            num_workers=4)

    # Set up the model, optimizer, and learning rate scheduler
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"):
    This line of code sets the device to run the model on to either the GPU 
    (if available) or the CPU (if not). This is because running the model on 
    a GPU can significantly speed up the computation, especially 
    when working with large datasets.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 2  # Background class (0) + your single class (1)
    model = get_faster_rcnn(num_classes)
    # This line moves the model to the device specified earlier (GPU or CPU).
    model.to(device)
    #list of all the parameters in the model that require gradient computation during backpropagation
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    """
    This creates a learning rate scheduler that adjusts the learning rate 
    during training. In this case, it reduces the learning rate by a factor of 
    0.1 every 3 epochs. 
    """
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1,verbose=True)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            print(type(targets))
            images = [image.to(device) for image in images]
           
           

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # Update the learning rate
        lr_scheduler.step()

        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {losses.item()}")

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    return data_loader, device
    

# Evaluation loop
def evaluate_function(model, dataset,data_loader,device):
    model.eval()
    iou_threshold = 0.5
    num_test_samples = len(dataset)
    correct = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)

            for idx, prediction in enumerate(predictions):
                pred_boxes = prediction["boxes"].cpu()
                pred_labels = prediction["labels"].cpu()

                gt_boxes = targets[idx]["boxes"].cpu()

                for pred_box, pred_label in zip(pred_boxes, pred_labels):
                    ious = torchvision.ops.box_iou(gt_boxes, pred_box.unsqueeze(0))
                    max_iou, _ = ious.max(dim=1)
                    if max_iou >= iou_threshold:
                        correct += 1

    accuracy = correct / num_test_samples
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    while True:
        print('1. Train with Enhanced Dataset')
        print('2. Train with Original Dataset')
        print('3. Quit')
        choice = int(input('Enter choice: '))
        if choice == 1:
            """
            """
        elif choice == 2:
            csv_file = r"C:\Users\ASUS\Desktop\draft\annotation.csv"
           
            num_classes = 2
        elif choice == 3:
            break
        else:
            print("Invalid choice")
            continue

        model_save_path = "path/to/save/model.pth"
        # Train the model
        data_loader, device = train_model(csv_file,  num_classes, model_save_path)
        # Evaluate the model
        dataset = CustomDataset(csv_file=csv_file,  transform=ToTensor())
        evaluate_function(model=get_faster_rcnn(num_classes), dataset=dataset, data_loader=data_loader, device=device)

