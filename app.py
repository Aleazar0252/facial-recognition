from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import cv2
import numpy as np
from face_utils import FaceUtils
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'faces'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

face_utils = FaceUtils()

# Global variables for continuous recognition
continuous_recognition_active = False
frame_count = 0

os.makedirs('faces', exist_ok=True)
os.makedirs('static/uploads', exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train')
def train():
    return render_template('train.html')


@app.route('/list')
def list_faces():
    people = []
    if os.path.exists('faces'):
        for person_name in os.listdir('faces'):
            person_dir = os.path.join('faces', person_name)
            if os.path.isdir(person_dir):
                images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                if images:
                    people.append({
                        'name': person_name,
                        'image_count': len(images),
                        'sample_image': os.path.join(person_name, images[0])
                    })
    return render_template('list.html', people=people)


@app.route('/api/train', methods=['POST'])
def api_train():
    try:
        name = request.form['name']
        files = request.files.getlist('images')

        if not name or not files:
            return jsonify({'success': False, 'error': 'Name and images are required'})

        saved_count = face_utils.save_training_images(name, files)
        face_utils.train_model()

        return jsonify({
            'success': True,
            'message': f'Successfully trained with {saved_count} images for {name}'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})

        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = face_utils.recognize_faces(image)

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/continuous_recognition/start', methods=['POST'])
def start_continuous_recognition():
    """Start continuous recognition"""
    global continuous_recognition_active
    continuous_recognition_active = True
    return jsonify({'success': True, 'message': 'Continuous recognition started'})


@app.route('/api/continuous_recognition/stop', methods=['POST'])
def stop_continuous_recognition():
    """Stop continuous recognition"""
    global continuous_recognition_active
    continuous_recognition_active = False
    return jsonify({'success': True, 'message': 'Continuous recognition stopped'})


@app.route('/api/continuous_recognition/frame', methods=['POST'])
def continuous_recognition_frame():
    """Process a single frame for continuous recognition"""
    global frame_count
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'})

        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'})

        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = face_utils.recognize_faces_continuous(image, frame_count)
        frame_count += 1

        return jsonify({'success': True, 'results': results})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/delete_person/<name>', methods=['DELETE'])
def delete_person(name):
    try:
        person_dir = os.path.join('faces', name)
        if os.path.exists(person_dir):
            import shutil
            shutil.rmtree(person_dir)
            face_utils.train_model()
            return jsonify({'success': True, 'message': f'Deleted {name}'})
        else:
            return jsonify({'success': False, 'error': 'Person not found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    if os.path.exists('faces') and any(os.path.isdir(os.path.join('faces', d)) for d in os.listdir('faces')):
        face_utils.train_model()
    app.run(debug=True, host='0.0.0.0', port=5000)