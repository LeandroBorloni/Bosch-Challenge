import os
import subprocess
from ultralytics import YOLO
import pandas as pd
import cv2
from matplotlib import pyplot as plt

# Check if the trained model exists
model_path = "yolov8n_trained.pt"
if not os.path.exists(model_path):
    print(f"{model_path} not found. Training the model...")
    subprocess.run(["python", "train.py"])

# Load the trained model
model = YOLO(model_path)

# Predict on an image
img_url = "https://ultralytics.com/images/bus.jpg"
results = model(img_url) 

# Process prediction results
predictions = results[0]  # assuming we're interested in the first image's results
df = pd.DataFrame(predictions.boxes.xyxy.cpu().numpy(), columns=['xmin', 'ymin', 'xmax', 'ymax'])
df['confidence'] = predictions.boxes.conf.cpu().numpy()
df['class'] = predictions.boxes.cls.cpu().numpy()

# Filter for bus class (class id 5 for bus in COCO dataset)
bus_class_id = 5
df = df[df['class'] == bus_class_id]

# Print predictions dataframe
print(df)

# Optionally, display the image with bounding boxes
def plot_predictions(image, df):
    for _, row in df.iterrows():
        cv2.rectangle(image, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), (0, 255, 0), 2)
        label = f"Bus: {row['confidence']:.2f}"
        cv2.putText(image, label, (int(row['xmin']), int(row['ymin']) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load the image using OpenCV
image = cv2.imread("bus.jpg")
plot_predictions(image, df)



