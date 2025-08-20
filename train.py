#trains and saves tensorflow model
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

df = pd.read_csv('posture_data.csv')

X = df.drop('class', axis=1)
y = df['class']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state = 42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), 
    Dropout(0.5), 
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(y_categorical.shape[1], activation='softmax') #output
])

model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training Model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

model.save('posture_classifier.h5')
print(f'Model trained and saved as posture_classifier.h5')

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'\n Test Accuracy: {accuracy*100:.2f}%')





