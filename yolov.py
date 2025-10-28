from ultralytics import YOLO

if __name__ == "__main__":
    print("王熠辉 2023213406")
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    model.train(data="data.yaml", epochs=10)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    results = model("./valid/images/1-10-_jpeg.rf.86e12c0f6fdf687c1bb073423017a026.jpg")  # predict on an image
    path = model.export(format="torchscript")

    print("王熠辉 2023213406")

