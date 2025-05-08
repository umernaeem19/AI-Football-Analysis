from ultralytics import YOLO


def main():
    # Load pre-trained model
    model = YOLO("yolo12l.pt")

    # Train the model
    results = model.train(
        data="football-players-detection-1/data.yaml",  # Local dataset
        imgsz=640,
        epochs=100,
        batch=2,
        name="football_yolov12l_DONE_LAST",
        pretrained=True,
        project="runs/train",
        patience=50,
        save=True,
        save_period=1,
        device=0,
    )


if __name__ == "__main__":
    main()
