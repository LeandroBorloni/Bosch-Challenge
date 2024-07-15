from ultralytics import YOLO

# Load the pre-trained model
model = YOLO("yolov8n.pt")  # (recommended for training)

# Train the model
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Save the model
model.save("yolov8n_trained.pt")
print("Model trained and saved as yolov8n_trained.pt")
