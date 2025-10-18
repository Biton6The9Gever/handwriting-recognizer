
import numpy as np
import sys
import os
import tensorflow
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D # type: ignore
from tensorflow.keras.utils import to_categorical                          # type: ignore
from tensorflow.keras.applications import VGG19                            # type: ignore
from tensorflow.keras.optimizers import Adam                               # type: ignore
from tensorflow.keras.models import Model                                  # type: ignore
from tensorflow.keras import layers, models                                # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
data_creation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_creation'))
sys.path.append(data_creation_path)
from utils import CHAR_AMOUNT ,AUGMENTATIONS_AMOUNT ,IMAGES_AMOUNT
def build_model(X,y):
    le=LabelEncoder()
    y_labels = le.fit_transform(X.ravel())
    y_labels =tensorflow.keras.utils.to_categorical(y_labels, num_classes=CHAR_AMOUNT)
    
    X_train, X_test, y_train, y_test = train_test_split(
    y, y_labels, test_size=0.2, random_state=67, stratify=y_labels
    )
    base_model= VGG19(weights ='imagenet', include_top=False, input_shape=(64,64,3))
    base_model.trainable = False  # freeze tge conv base
    
    # model header
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(26, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
    
    #TODO fine tuning the model
    