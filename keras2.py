import os
import numpy as np
import pandas as pd
import cv2
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Add, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def load_annotation_file(filepath):
    return pd.read_csv(filepath, header=None, names=["filename", "class", "xmin", "ymin", "xmax", "ymax"])

def preprocess_data(df, img_folder):
    data = []
    for idx, row in df.iterrows():
       
        img = cv2.imread(df['filename'].values[idx], cv2.IMREAD_GRAYSCALE).astype('float32')
        img = img / 255.0
        bbox = [row["xmin"] / 50, row["ymin"] / 50, row["xmax"] / 50, row["ymax"] / 50]
        data.append((img,bbox))
    return data

annotation_file = r"C:\Users\ASUS\Desktop\draft\annotation.csv"
img_folder = r'C:\Users\ASUS\Desktop\draft\processed_img'

df = load_annotation_file(annotation_file)
data = preprocess_data(df, img_folder)

train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

def faster_rcnn(input_shape=(50, 50, 1)):
    input_layer = Input(shape=input_shape)

    # Base network
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(input_layer)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)

    x =     Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(x)

    # Region Proposal Network (RPN)
    rpn = Conv2D(512, (3, 3), activation='relu', padding='same', name='rpn_conv1')(x)
    rpn_cls = Conv2D(2, (1, 1), activation='softmax', name='rpn_cls')(rpn)
    rpn_reg = Conv2D(4, (1, 1), activation='linear', name='rpn_reg')(rpn)

    # Bounding box regression head
    bbox_head = Conv2D(256, (3, 3), activation='relu', padding='same', name='bbox_conv1')(x)
    bbox_reg = Conv2D(4, (1, 1), activation='linear', name='bbox_reg')(bbox_head)

    model = Model(inputs=input_layer, outputs=[rpn_cls, rpn_reg, bbox_reg])
    return model

model = faster_rcnn()
model.compile(optimizer=Adam(lr=1e4), loss=[BinaryCrossentropy(), 'mse', 'mse'], loss_weights=[1.0, 1.0, 1.0])

def data_generator(data, batch_size=32):
    while True:
        batch_data = random.sample(data, batch_size)
        batch_images = np.array([item[0] for item in batch_data]).reshape(-1, 50, 50, 1)
        batch_bboxes = np.array([item[1] for item in batch_data])
        batch_rpn_cls = np.zeros((batch_size, 2))
        batch_rpn_reg = np.zeros((batch_size, 4))
        yield batch_images, [batch_rpn_cls, batch_rpn_reg, batch_bboxes]

batch_size = 32
train_gen = data_generator(train_data, batch_size=batch_size)
val_gen = data_generator(val_data, batch_size=batch_size)

callbacks = [
    ModelCheckpoint("faster_rcnn_model.h5", monitor='val_loss',
                    save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
]

model.fit(
    train_gen,
    steps_per_epoch=len(train_data) // batch_size,
    epochs=100,
    validation_data=val_gen,
    validation_steps=len(val_data) // batch_size,
    callbacks=callbacks
)


model.save("faster_rcnn_final_model.h5")

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    xi1, yi1, xi2, yi2 = max(x1, x1_), max(y1, y1_), min(x2, x2_), min(y2, y2_)
    intersection = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union = box1_area + box2_area - intersection

    return intersection / union

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5):
    average_precisions = []

    for class_id in range(1):  
        detections = []
        ground_truths = []

        for pred_box in pred_boxes:
                        detections.append([*pred_box, class_id])

        for true_box in true_boxes:
            ground_truths.append([*true_box, class_id])

        detections = sorted(detections, key=lambda x: x[-1], reverse=True)
        ground_truth_count = len(ground_truths)

        true_positives = [0] * len(detections)
        false_positives = [0] * len(detections)

        for det_idx, detection in enumerate(detections):
            iou_max = 0
            gt_idx_max = -1
            for gt_idx, ground_truth in enumerate(ground_truths):
                current_iou = iou(detection[:4], ground_truth[:4])
                if current_iou > iou_max:
                    iou_max = current_iou
                    gt_idx_max = gt_idx

            if iou_max >= iou_threshold:
                true_positives[det_idx] = 1
                ground_truths.pop(gt_idx_max)
            else:
                false_positives[det_idx] = 1

        tp_cumsum = np.cumsum(true_positives).astype(np.float64)
        fp_cumsum = np.cumsum(false_positives).astype(np.float64)
        recalls = tp_cumsum / ground_truth_count
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        precisions = np.concatenate(([0], precisions, [0]))
        recalls = np.concatenate(([0], recalls, [1]))

        for idx in range(len(precisions) - 2, -1, -1):
            precisions[idx] = max(precisions[idx], precisions[idx + 1])

        indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
        average_precisions.append(np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices]))

    return sum(average_precisions) / len(average_precisions)


           




