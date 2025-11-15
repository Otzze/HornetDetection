import cv2
from ultralytics import YOLO

"""
    Runs YOLO inference on a video file and saves the output with detected boxes.
"""
def run_yolo_video_inference(video_path, weights_path, output_path):
    model = YOLO(weights_path)
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run inference
        results = model.predict(frame, stream=False, device='cpu')
        result = results[0]

        # Draw boxes on the frame
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = box.cls[0]
            label = f"{int(cls)} {conf:.2f}"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Done, saved to {output_path}")
