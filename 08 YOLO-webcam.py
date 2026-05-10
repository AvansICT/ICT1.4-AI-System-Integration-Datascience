# 🧪 YOLO Webcam Detectie Script
# Run dit script in een terminal (niet in Jupyter)

#pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO

def main():
    print("Start YOLO webcam detectie")

    # Laad model
    model = YOLO("yolov8n.pt")
    #model = YOLO("runs/detect/train/weights/best.pt")

    # Open webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Webcam kon niet geopend worden")
        return

    print("Webcam gestart - druk op 'q' om te stoppen")

    while True:
        # Lees frame
        ret, frame = cap.read()
        if not ret:
            print("Geen frame ontvangen")
            break

        # YOLO detectie
        results = model(frame)

        # Teken bounding boxes
        annotated = results[0].plot()

        # Toon beeld
        cv2.imshow("YOLO Webcam", annotated)

        # Check toets
        key = cv2.waitKey(1)
        if key == ord("q"):
            print("Gestopt door gebruiker")
            break

    # Opruimen
    cap.release()
    cv2.destroyAllWindows()
    print("Programma afgesloten")


if __name__ == "__main__":
    main()