import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Masking    # type: ignore
from tensorflow.keras.utils import to_categorical                                   # type: ignore
from tensorflow.keras.optimizers import Adam                                        # type: ignore
from tensorflow.keras.models import Sequential                                      # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import save_model, CHAR_AMOUNT


def build_model_rnn(X, y):

    le = LabelEncoder()
    y_labels = le.fit_transform(X.ravel())
    y_labels = tf.keras.utils.to_categorical(y_labels, num_classes=CHAR_AMOUNT)

    # Preprocess input for RNN

    # reshape (n_samples, 64, 64, 3) -> (n_samples, 64, 64*3)
    X_seq = y.reshape(y.shape[0], 64, 64 * 3).astype("float32") / 255.0

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_labels, test_size=0.2, random_state=67, stratify=y_labels
    )

    # build model

    model = Sequential([
        Masking(mask_value=0.0, input_shape=(64, 64 * 3)),
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.3),
        LSTM(64),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(CHAR_AMOUNT, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # train 
    print("[INFO] Training RNN model...")
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=64,
        verbose=1
    )

    # Evaluate after training
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Accuracy: {acc:.4f}, Loss: {loss:.4f}")

    # save model
    save_model(model, acc, model_name="RNN_letters")

    return model, le