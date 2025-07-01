from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png'}

# Load the model
model2 = load_model('my_model.keras')

# Utility function to check allowed files
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the image
        processed_image_path = process_image(file_path, filename)
        return redirect(url_for('show_image', filename=processed_image_path))
    return redirect(request.url)

# Process the uploaded image
def process_image(file_path, filename):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width, _ = img.shape

    plt.figure(figsize=(15, 15))
    ax = plt.subplot(1, 1, 1)
    ax.imshow(img)

    stride = 5
    for h in range(0, height - 20, stride):
        for w in range(0, width - 20, stride):
            img_box = []
            img_box.append(img[h:h + 20, w:w + 20])
            img_box = np.array(img_box, dtype=np.int64)
            prediction = model2.predict(img_box, verbose=False)
            prediction = np.argmax(prediction)

            if prediction == 1:
                ax.add_patch(patches.Rectangle((w, h), 20, 20, edgecolor='r', facecolor='none'))

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return f'processed_{filename}'

# Route to display processed image
@app.route('/uploads/<filename>')
def show_image(filename):
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
