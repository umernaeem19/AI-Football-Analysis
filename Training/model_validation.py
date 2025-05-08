from ultralytics import YOLO

def main():
    # Load pre-trained model
    model = YOLO("best.pt")


    Results=model.val(data="football-players-detection-1/data.yaml", split='val')



if __name__ == "__main__":
    main()
