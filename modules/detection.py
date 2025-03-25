"""
Target detection using YOLOv8 and TensorRT.
"""
def yolov8_detection(model, image):
    """
    Perform detection using a YOLOv8 model.
    """
    results = model.predict(image, imgsz=640, conf=0.5, verbose=False)
    result = results[0].cpu()

    # Get information from result
    box = result.boxes.xyxy.numpy()
    conf = result.boxes.conf.numpy()
    cls = result.boxes.cls.numpy().astype(int)

    return cls, conf, box

def detect_targets(model, frame, file_extension):
    """
    Single entry-point for detection, deciding which detection function to use
    based on the file extension (".engine" -> YOLOv8 or tensorrt if needed).
    """
    if frame is None:
        return None, None, None

    cls, conf, box = yolov8_detection(model, frame)
    return cls, conf, box 