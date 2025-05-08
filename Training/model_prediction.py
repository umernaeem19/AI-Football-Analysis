from ultralytics import YOLO
import cv2


def test_model():
    # Load the trained model
    model = YOLO(
        "best.pt"
    )  # adjust path if needed

    # Run inference on an image
    results = model.predict(
        source="football-players-detection-1/test/images/40cd38_7_6_png.rf.68ef7fcd663cdf0f5b96bacdbcd94e07.jpg",  # path to your test image
        save=True,  # save output image(s)
        save_txt=True,  # save detections to labels file
        conf=0.25,  # confidence threshold
        device=0,  # set device
        imgsz=640,  # image size
    )

    # Output saved automatically to runs/predict/ directory


if __name__ == "__main__":
    test_model()
