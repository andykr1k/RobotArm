from flask import Flask, request, jsonify
import os
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './db/data/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

csv_file = app.config['UPLOAD_FOLDER'] + 'training.csv'

if not os.path.isfile(csv_file):
    df = pd.DataFrame(columns=['timestamp', 'file_name', 'reward', 'angles'])
    df.to_csv(csv_file, index=False)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "Image not found in request"}), 400
    if 'reward' not in request.form:
        return jsonify({"error": "Reward data not found in request"}), 400
    if 'angles' not in request.form:
        return jsonify({"error": "Angles data not found in request"}), 400

    image = request.files['image']
    reward = request.form['reward']
    angles = request.form['angles']

    if image.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(file_path)

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        reward = float(reward)
        angles = str(angles)

        new_data = pd.DataFrame([[timestamp, filename, reward, angles]], columns=['timestamp', 'file_name', 'reward', 'angles'])
        new_data.to_csv(csv_file, mode='a', header=False, index=False)

        return jsonify({
            'message': 'File uploaded successfully!',
            'image_url': file_path,
            'timestamp': timestamp,
            'file_name': filename,
            'reward': reward,
            'angles': angles
        })

    return jsonify({"error": "Invalid file format"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
