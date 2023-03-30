import os
import cv2
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Conv2D, TimeDistributed, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.applications import VGG16
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import keras.backend as K
import tensorflow as tf
from keras.layers import Reshape,Layer


class RoiPoolingConv(Layer):
    def __init__(self, pool_size, **kwargs):
        self.pool_size = pool_size
        super(RoiPoolingConv, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return None, input_shape[1], self.pool_size, self.pool_size, input_shape[0][3]

    def call(self, inputs, **kwargs):
        img, rois = inputs

        # Extract box coordinates
        boxes = rois[0, :, :4]

        # Normalize box coordinates
        height, width = K.int_shape(img)[1:3]
        if height is None or width is None:
            normalized_boxes = K.zeros_like(boxes)
        else:
            normalized_boxes = K.concatenate([
                boxes[:, 1:2] / height,
                boxes[:, 0:1] / width,
                boxes[:, 3:4] / height,
                boxes[:, 2:3] / width
            ], axis=1)

        # Perform crop and resize
        output = tf.image.crop_and_resize(img, normalized_boxes, tf.range(tf.shape(rois)[1]), [self.pool_size, self.pool_size])

        return output

    def get_config(self):
        config = {'pool_size': self.pool_size}
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def rpn(base_layers, num_anchors):
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)

    return [x_class, x_regr, base_layers]


def classifier(base_layers, input_rois, num_rois, nb_classes):
    pooling_regions = 7
    input_shape = (num_rois, 7, 7, 512)

    # Compute the shape of the input image tensor
    _, height, width, _ = K.int_shape(base_layers)

    out_roi_pool = RoiPoolingConv(pooling_regions, name='roi_pooling_conv')([base_layers, input_rois])
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    out_class = TimeDistributed(Dense(1, activation='sigmoid', kernel_initializer='zero'), name='dense_class_{}'.format(nb_classes))(out)
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'), name='dense_regress_{}'.format(nb_classes))(out)

    return [out_class, out_regr]


    
class Config:
    def __init__(self):
        self.verbose = True
        self.img_size = (50, 50, 1)
        self.num_rois = 32
        self.num_classes = 1  # Only "nodule" class
        self.rpn_stride = 16
        self.anchor_box_scales = [32, 64, 128]
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

def get_model(input_shape_img, input_shape_rois, num_rois, num_classes):
    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=input_shape_rois)
    roi_input_reshaped = Reshape((-1, 4))(roi_input)
    base_layers = VGG16(include_top=False, input_tensor=img_input)

    for layer in base_layers.layers:
        layer.trainable = False

    num_anchors = 9
    rpn_layers = rpn(base_layers.output, num_anchors)
    classifier_layers = classifier(base_layers.output, roi_input_reshaped, num_rois, num_classes)

    model_rpn = Model(inputs=[img_input], outputs=rpn_layers[:2])
    model_classifier = Model(inputs=[img_input, roi_input], outputs=classifier_layers)
    model_all = Model(inputs=[img_input, roi_input], outputs=rpn_layers[:2] + classifier_layers)

    return model_rpn, model_classifier, model_all

def load_data(annotations_file, img_folder, config):
    df = pd.read_csv(annotations_file)
    X = []
    y = []
    for idx, row in df.iterrows():
        img = cv2.imread(os.path.join(img_folder, row['filename']), cv2.IMREAD_GRAYSCALE)
        #img = np.stack([img] * 3, axis=-1)  
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        bbox = np.array([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        bbox_padded = np.pad(bbox, (0, 4 - len(bbox)), mode='constant')
        X.append(img)
        y.append(bbox_padded)

    X = np.array(X)
    y = np.array(y)

    return X, y

    
input_shape_img = (None, None, 3)
input_shape_rois = (None, 4)
num_rois = 32
num_classes = 2 # Background + "node" class

model_rpn, model_classifier, model_all = get_model(input_shape_img, input_shape_rois, num_rois, num_classes)

model_rpn.compile(optimizer=Adam(lr=0.001), loss=['binary_crossentropy', 'mse'])
model_classifier.compile(optimizer=Adam(lr=0.001), loss=['binary_crossentropy', 'mse'])
model_all.compile(optimizer=Adam(lr=0.001), loss=['binary_crossentropy', 'mse'])


# Create the model
config = Config()
model_rpn, model_classifier, model_all = get_model((None, None, 3), (None, 4), config.num_rois, config.num_classes)

# Compile the model
model_rpn.compile(optimizer=Adam(lr=0.001), loss=['binary_crossentropy', 'mse'])
model_classifier.compile(optimizer=Adam(lr=0.001), loss=['binary_crossentropy', 'mse'])
model_all.compile(optimizer=Adam(lr=0.001), loss=['binary_crossentropy', 'mse'])

# Load the data
X, y = load_data("annotation.csv", "processed_img", config)

print("x",X.shape)
print("y",y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Callbacks
checkpoint = ModelCheckpoint('model_weights.h5', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, mode='min')
reduce_lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, mode='min')

y_train_reshaped = np.reshape(y_train, (-1,1, 4))
y_test_reshaped = np.reshape(y_test, (-1, 1, 4))
#print y_train_reshaped [0]

print("y_train_reshaped [0]",y_train_reshaped[0])

#X_train = np.vstack(X_train)
#y_train_reshaped = np.vstack(y_train_reshaped)
#X_test = np.vstack(X_test)
#y_test_reshaped = np.vstack(y_test_reshaped)

print("x_train",X_train.shape)
print("y_train_reshaped",y_train_reshaped.shape)
print("x_test",X_test.shape)
print("y_test_reshaped",y_test_reshaped.shape)


history = model_all.fit([X_train, y_train_reshaped], epochs=100, batch_size=16, validation_data=([X_test, y_test_reshaped]), callbacks=[checkpoint, early_stopping, reduce_lr_on_plateau])

# Train the model
#history = model_all.fit([X_train, *y_train], epochs=100, batch_size=16, validation_data=([X_test, *y_test]), callbacks=[checkpoint, early_stopping, reduce_lr_on_plateau])

# Save the trained model
model_all.save('trained_model.h5')
