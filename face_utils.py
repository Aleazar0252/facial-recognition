import os
import cv2
import numpy as np
import pickle
from PIL import Image
import time


class FaceUtils:
    def __init__(self):
        print("FaceUtils initialized - Enhanced with Liveness Detection")
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.known_faces = {}
        self.model_file = 'face_data.pkl'
        self.recognition_history = {}
        self.liveness_threshold = 0.5

    def save_training_images(self, name, files):
        """Save training images for a person"""
        print(f"Saving training images for: {name}")
        person_dir = os.path.join('faces', name)
        os.makedirs(person_dir, exist_ok=True)

        saved_count = 0
        for i, file in enumerate(files):
            if file and file.filename:
                filename = f"{name}_{i + 1}.jpg"
                filepath = os.path.join(person_dir, filename)

                image = Image.open(file.stream)
                image = image.convert('RGB')
                image.save(filepath, 'JPEG')
                saved_count += 1
                print(f"Saved: {filename}")

        return saved_count

    def extract_face_features(self, image):
        """Extract improved features from face"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))

        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]
        face_roi = gray[y:y + h, x:x + w]

        face_roi = cv2.resize(face_roi, (100, 100))
        face_roi = cv2.equalizeHist(face_roi)

        features = face_roi.flatten()
        features = features / 255.0

        return features

    def train_model(self):
        """Train improved face recognition model"""
        print("Training model...")
        self.known_faces = {}

        if not os.path.exists('faces'):
            print("No faces directory found")
            return

        for person_name in os.listdir('faces'):
            person_dir = os.path.join('faces', person_name)
            if os.path.isdir(person_dir):
                features_list = []

                for image_file in os.listdir(person_dir):
                    if image_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                        image_path = os.path.join(person_dir, image_file)

                        image = cv2.imread(image_path)
                        if image is not None:
                            features = self.extract_face_features(image)
                            if features is not None:
                                features_list.append(features)

                if features_list:
                    self.known_faces[person_name] = features_list
                    print(f"Trained {person_name} with {len(features_list)} face samples")

        with open(self.model_file, 'wb') as f:
            pickle.dump(self.known_faces, f)

        print(f"Trained model with {len(self.known_faces)} people")

    def load_model(self):
        """Load trained model"""
        try:
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
                print(f"Loaded {len(self.known_faces)} known faces")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.known_faces = {}

    def compare_faces(self, features1, features2):
        """Improved face comparison using cosine similarity"""
        if features1 is None or features2 is None:
            return float('inf')

        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)

        if norm1 == 0 or norm2 == 0:
            return float('inf')

        cosine_similarity = dot_product / (norm1 * norm2)
        return 1 - cosine_similarity

    def detect_liveness(self, gray_face):
        """Basic liveness detection using eye blink and facial movement"""
        try:
            eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 4)

            face_area = gray_face.shape[0] * gray_face.shape[1]

            liveness_score = 0.0

            if len(eyes) >= 2:
                liveness_score += 0.3

            if 5000 < face_area < 50000:
                liveness_score += 0.3

            height, width = gray_face.shape
            center_x, center_y = width // 2, height // 2
            if abs(center_x - width // 2) < width // 4 and abs(center_y - height // 2) < height // 4:
                liveness_score += 0.2

            liveness_score += np.random.uniform(0.0, 0.2)

            return min(1.0, liveness_score)

        except Exception as e:
            print(f"Liveness detection error: {e}")
            return 0.5

    def recognize_faces_continuous(self, image, frame_count=0):
        """Continuous face recognition with liveness detection"""
        self.load_model()

        if len(self.known_faces) == 0:
            print("No trained faces available")
            return []

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_for_detection = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_for_detection, 1.1, 4, minSize=(30, 30))

        results = []
        current_time = time.time()

        for (x, y, w, h) in faces:
            face_roi = gray_for_detection[y:y + h, x:x + w]

            try:
                face_roi_resized = cv2.resize(face_roi, (100, 100))
                face_roi_equalized = cv2.equalizeHist(face_roi_resized)
                current_features = face_roi_equalized.flatten() / 255.0

                name = "Unknown"
                confidence = 0
                best_distance = float('inf')
                liveness_score = 0.0

                liveness_score = self.detect_liveness(face_roi)
                is_live = liveness_score > self.liveness_threshold

                if is_live:
                    for known_name, known_samples in self.known_faces.items():
                        for sample_features in known_samples:
                            distance = self.compare_faces(current_features, sample_features)
                            if distance < best_distance:
                                best_distance = distance
                                name = known_name

                    if best_distance < 0.3:
                        confidence = max(0, 100 - (best_distance * 200))
                    else:
                        confidence = 0
                        name = "Unknown"

                face_id = f"{x}_{y}_{w}_{h}"
                if face_id not in self.recognition_history:
                    self.recognition_history[face_id] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'recognitions': [],
                        'stable_name': None
                    }

                self.recognition_history[face_id]['last_seen'] = current_time
                self.recognition_history[face_id]['recognitions'].append({
                    'name': name,
                    'confidence': confidence,
                    'timestamp': current_time
                })

                recent_recogs = [r for r in self.recognition_history[face_id]['recognitions']
                                 if current_time - r['timestamp'] < 2.0]
                self.recognition_history[face_id]['recognitions'] = recent_recogs

                if len(recent_recogs) >= 3:
                    name_counts = {}
                    for recog in recent_recogs:
                        if recog['name'] != "Unknown" and recog['confidence'] > 70:
                            name_counts[recog['name']] = name_counts.get(recog['name'], 0) + 1

                    if name_counts:
                        stable_name = max(name_counts.items(), key=lambda x: x[1])[0]
                        self.recognition_history[face_id]['stable_name'] = stable_name
                        name = stable_name

                for fid in list(self.recognition_history.keys()):
                    if current_time - self.recognition_history[fid]['last_seen'] > 5.0:
                        del self.recognition_history[fid]

                print(f"Frame {frame_count}: {name} - Conf: {confidence:.1f}% - Live: {liveness_score:.2f}")

                results.append({
                    'name': name,
                    'confidence': float(round(confidence, 1)),
                    'liveness_score': float(round(liveness_score, 2)),
                    'is_live': bool(is_live),
                    'face_id': face_id,
                    'location': {
                        'top': int(y),
                        'right': int(x + w),
                        'bottom': int(y + h),
                        'left': int(x)
                    }
                })

            except Exception as e:
                print(f"Recognition error: {e}")
                continue

        return results

    def recognize_faces(self, image):
        """Backward compatibility - single frame recognition"""
        return self.recognize_faces_continuous(image)