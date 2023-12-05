from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_PATH = Path('data') / '160929_124_London_Buses_1080p.mp4'
OUTPUT_PATH = Path('output') / '160929_124_London_Buses_1080p_out.mp4'
if __name__ == '__main__':
    # load model
    model = YOLO('yolov8n.pt')
    model.fuse()
    print(model.names)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    # get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # create video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))

    # loop through frames
    while True:
        # read frame
        ret, frame = cap.read()

        # check if frame was read successfully
        if not ret:
            break

        # perform inference
        detections = model.predict(frame, classes=[0])[0]

        # visualize detections
        annotated_frame = detections.plot()

        # write annotated frame to video
        out.write(annotated_frame)

        # display annotated frame
        cv2.imshow('YOLOv8 Inference', annotated_frame)

        # wait for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release video capture object and video writer object, and close all windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
