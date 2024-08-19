import os
import cv2
from ultralytics import YOLO

# Paths provided by you
model_weights = r"C:\Users\JIYA\Desktop\Violence_Detetction_Backend\best.pt"
source_path = r"C:\Users\JIYA\Desktop\Violence_Detetction_Backend\INPUT_VID.mp4"
output_path = r"C:\Users\JIYA\Desktop\Violence_Detetction_Backend\OUTPUT_VID.mp4"

# Function to run inference on a video and save the output
def run_inference(model_weights, source, output):
    # Load the model
    model = YOLO(model_weights)
    
    # Open the input video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        violence_detected = False
        for result in results:
            for box in result.boxes:
                confidence = box.conf.item()  # Get confidence score
                
                if box.cls == 1 and confidence > 0.60:  # Only consider violence with confidence > 65%
                    violence_detected = True
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Overlay the label "Violence detected" with accuracy
                    label = f"Violence detected ({confidence * 100:.2f}%)"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        if not violence_detected:
            # Overlay the label "Non-violence detected" if no violence is detected
            cv2.putText(frame, "Non-violence detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Write the frame to the output video
        out.write(frame)
    
    # Release everything when job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Inference completed. Output saved to {output}")

if __name__ == "__main__":
    run_inference(model_weights, source_path, output_path)
