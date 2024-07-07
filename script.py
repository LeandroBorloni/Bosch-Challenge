from ultralytics import YOLO

# Load
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Validate
metrics = model.val()  # no arguments, dataset and settings remembered
metrics.box.map  # map 50-95
metrics.box.map50  # map 50
metrics.box.map75  # map 75
metrics.box.maps  # a list contains map 50-95 of each category

# Predict
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Export the model
print(model.export())
