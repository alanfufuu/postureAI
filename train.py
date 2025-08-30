import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import math
import joblib

df = pd.read_csv("posture_data.csv")
X_raw = df.drop("class", axis=1).values
y = df["class"].values

def compute_features(row):
    nose_x, nose_y, ls_x, ls_y, rs_x, rs_y, le_x, le_y, re_x, re_y = row

    shoulder_width = abs(ls_x - rs_x) + 1e-6

    shoulder_slope = (ls_y - rs_y) / shoulder_width
    le_dist = np.sqrt((le_x - ls_x) ** 2 + (le_y - ls_y) ** 2) / shoulder_width
    re_dist = np.sqrt((re_x - rs_x) ** 2 + (re_y - rs_y) ** 2) / shoulder_width

    ear_line_slope = (le_y - re_y) / (le_x - re_x + 1e-6)
    shoulder_line_slope = (ls_y - rs_y) / (ls_x - rs_x + 1e-6)
    neck_tilt = math.atan(abs(ear_line_slope - shoulder_line_slope))

    coords = np.array([
        nose_x, nose_y, ls_x, ls_y, rs_x, rs_y, le_x, le_y, re_x, re_y
    ]) / shoulder_width

    return np.concatenate([coords, [shoulder_slope, le_dist, re_dist, neck_tilt]])

X_features = np.apply_along_axis(compute_features, 1, X_raw)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_features)

joblib.dump(scaler, "scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical,
    test_size=0.2, random_state=42,
    stratify=y_categorical
)

model = Sequential([
    Dense(256, activation="relu", input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(y_categorical.shape[1], activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
    ModelCheckpoint("posture_classifier.h5", monitor="val_accuracy", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)
]

print("Training Model with Engineered Features...")
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

model.save("posture_classifier.h5")
print("Model trained and saved as posture_classifier.h5")

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"🎯 Test Accuracy: {accuracy*100:.2f}%")


