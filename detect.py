"""
RUN FOOTBALL KIT DETECTION

This script loads the custom-trained YOLO model and runs real-time
detection on a webcam feed or video file.

Usage:
    - To use webcam 0:
      python detect.py --model models/best.pt --source 0
    
    - To use a video file:
      python detect.py --model models/best.pt --source path/to/video.mp4
"""
import cv2
import argparse
from ultralytics import YOLO

# Define your custom colors for each class ID
# (MUST match the order in your data.yaml)
KIT_COLORS = {
    0: (255, 255, 0),    # 0: Europa (Cyan)
    1: (255, 0, 0),      # 1: First (Blue)
    2: (128, 128, 128),  # 2: Jacket (Grey)
    3: (0, 255, 255),    # 3: Second (Yellow)
    4: (0, 255, 0),      # 4: Third (Green)
    5: (230, 216, 173)   # 5: Training (Light Blue)
}

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8 Football Kit Detection")
    parser.add_argument(
        "--model", 
        required=True, 
        help="Path to the trained YOLO model (e.g., 'models/best.pt')"
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Video source. '0' for webcam 0, or path to a video file."
    )
    parser.add_argument(
        "--resolution",
        default=None,
        help="Output resolution (e.g., '1280x720'). Default is source resolution."
    )
    return parser.parse_args()

def main():
    """Main function to run the detection."""
    args = parse_arguments()

    output_res = None
    if args.resolution:
        try:
            w, h = map(int, args.resolution.split('x'))
            output_res = (w, h)
            print(f"Setting output resolution to: {output_res}")
        except ValueError:
            print("Error: Resolution must be in 'WxH' format (e.g., '1280x720').")
            return
            
    # Load
    print(f"Loading model from {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get class names from the model
    class_names = model.names
    print(f"Model loaded. Detecting classes: {class_names}")

    # Convert '0' to 0 for cv2.VideoCapture
    source = args.source
    if source == '0':
        source = 0
    
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise Exception("Could not open video source.")
    except Exception as e:
        print(f"Error opening video source '{args.source}': {e}")
        return

    print("Starting detection... Press 'q' to quit.")

    # Run detection
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video stream ended or frame could not be read.")
            break

        # --- Run inference ---
        results = model(frame, verbose=False)

        # --- Process results ---
        for result in results:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Get class ID and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Get class name and color
                class_name = class_names.get(class_id, "Unknown")
                color = KIT_COLORS.get(class_id, (255, 255, 255)) # Default to white

                # --- Draw on frame ---
                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Create label text
                label = f"{class_name}: {confidence:.2f}"
                
                # Draw the label background
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label) * 9, y1), color, -1)
                
                # Draw the label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if output_res:
            frame = cv2.resize(frame, output_res)
            
        # --- Display the frame ---
        cv2.imshow("Football Kit Detector", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    main()