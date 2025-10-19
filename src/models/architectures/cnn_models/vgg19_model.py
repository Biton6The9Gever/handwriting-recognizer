import numpy as np
import sys
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D  # type: ignore
from tensorflow.keras.utils import to_categorical                           # type: ignore
from tensorflow.keras.applications import VGG19                             # type: ignore
from tensorflow.keras.optimizers import Adam                                # type: ignore
from tensorflow.keras.models import Model                                   # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import save_model_with_metadata

# Local imports
data_creation_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_creation'))
sys.path.append(data_creation_path)
from utils import CHAR_AMOUNT, AUGMENTATIONS_AMOUNT, IMAGES_AMOUNT


def build_model(X, y):
    # encode labels
    le = LabelEncoder()
    y_labels = le.fit_transform(X.ravel())
    y_labels = tf.keras.utils.to_categorical(y_labels, num_classes=CHAR_AMOUNT)

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        y, y_labels, test_size=0.2, random_state=67, stratify=y_labels
    )

    # base model (VGG19)
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    base_model.trainable = False  # freeze convolutional base

    # add classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(CHAR_AMOUNT, activation='softmax')(x) 

    model = Model(inputs=base_model.input, outputs=output)

    # compile model
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # train first stage (frozen base)
    print("[INFO] Training head layers...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,
        batch_size=64,
        verbose=1
    )

    # evaluate after first training
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Stage 1 Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    # unfreeze top layers
    print("[INFO] Fine-tuning top layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-5]:  
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # fine-tune
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,
        batch_size=32,
        verbose=1
    )

    #evaluation
    final_loss, final_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Final Accuracy: {final_acc:.4f}, Loss: {final_loss:.4f}")
    save_model_with_metadata(model, final_acc, model_name="VGG19_letters")

    return model, le
