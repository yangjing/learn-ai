from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# results = model.predict(source="images/5430905.jpg")

results = model("images")

for result in results:
  result.show()
