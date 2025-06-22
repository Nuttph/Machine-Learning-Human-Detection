import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# โหลดข้อมูลจากไฟล์ CSV
df1 = pd.read_csv("Dataset/dataset_fighting.csv")
df2 = pd.read_csv("Dataset/dataset_hello.csv")

# รวม dataset เข้าด้วยกัน
df = pd.concat([df1, df2], ignore_index=True)

# แยก features และ labels
X = df.drop(columns=["label"]).values  # landmark values
y = df["label"].values                 # class labels (เช่น fighting, hello)

# แปลง label เป็นตัวเลข และ one-hot encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# แบ่งข้อมูลเป็น train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# สร้างโมเดล Neural Network ด้วย Keras
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ฝึกสอนโมเดล
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# ประเมินโมเดลกับ test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# บันทึกโมเดลและ label encoder
model.save("Model/pose_classifier_model.h5")
joblib.dump(label_encoder, "Model/label_encoder.pkl")
