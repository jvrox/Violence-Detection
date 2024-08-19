from flask import Flask, request, send_file
import os
import uuid
from model import run_inference  # Import the function from model.py

app = Flask(__name__)

# Define paths
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
MODEL_WEIGHTS = 'C:/Users/JIYA/Desktop/Violence_Detetction_Backend/best.pt'  # Update this to your actual model weights path

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    if file:
        # Generate unique file names
        unique_id = str(uuid.uuid4())
        input_filename = f"{unique_id}_input.mp4"
        input_filepath = os.path.join(UPLOAD_FOLDER, input_filename)
        output_filename = f"{unique_id}_output.mp4"
        output_filepath = os.path.join(PROCESSED_FOLDER, output_filename)
        
        # Save the uploaded file
        file.save(input_filepath)
        
        # Run inference
        run_inference(MODEL_WEIGHTS, input_filepath, output_filepath)
        
        # Return the processed video file
        return send_file(output_filepath, mimetype='video/mp4', as_attachment=True, attachment_filename='output_video.mp4')

if __name__ == "__main__":
    app.run(debug=True)
