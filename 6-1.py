import os
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from PIL import Image
import requests
from io import BytesIO

# Step 1: 加載預訓練的 VGG16 模型
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 添加自定義的分類層
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

# 完整模型
model = Model(inputs=base_model.input, outputs=predictions)

# 冷凍 VGG16 的預訓練層
for layer in base_model.layers:
    layer.trainable = False

# 編譯模型
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print("Model successfully built!")

# Step 2: 加載和準備本地數據集
# 設定數據集路徑
dataset_path = os.path.join("Face-Mask-Detection", "dataset")

# 調整數據集路徑
data_gen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# 訓練數據生成器
train_generator = data_gen.flow_from_directory(
    dataset_path,  # 本地數據集的主目錄
    target_size=(224, 224),  # 圖片調整大小
    batch_size=32,
    class_mode='categorical',
    subset='training'  # 用於訓練的子集
)

# 驗證數據生成器
val_generator = data_gen.flow_from_directory(
    dataset_path,  # 本地數據集的主目錄
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # 用於驗證的子集
)

print("Data successfully prepared!")

# Step 3: 訓練模型
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=train_generator.samples // 32,
    validation_steps=val_generator.samples // 32
)

print("Model training complete!")

# Step 4: 定義 URL 圖片分類函數
def classify_image(image_url, model, class_names):
    try:
        # 從 URL 加載圖片
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        
        # 預測
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][class_idx]
        return class_names[class_idx], confidence
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

# Step 5: 測試 URL 圖片分類
class_names = ['With Mask', 'Without Mask']  # 修改為你模型的類別名稱
test_image_url = input("請輸入圖片 URL: ")

result, confidence = classify_image(test_image_url, model, class_names)
if result:
    print(f"該圖片分類為: {result}，信心度: {confidence:.2f}")
else:
    print("圖片分類失敗！")
