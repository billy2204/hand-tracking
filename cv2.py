import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from collections import deque
import time
import json

class HandTrackingAlternative:
    """
    Alternative hand tracking using OpenCV's built-in features
    Compatible with Apple Silicon và các hệ thống không hỗ trợ MediaPipe
    """
    
    def __init__(self):
        # Hand cascade classifier (tải từ OpenCV)
        self.hand_cascade = None
        self.load_hand_cascade()
        
        # Contour-based hand detection
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        
        # Model classification
        self.model = None
        self.model_trained = False
        self.training_data = []
        self.training_labels = []
        self.alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.prediction_buffer = deque(maxlen=10)
        
        # Load model nếu có
        self.load_model()
    
    def load_hand_cascade(self):
        """
        Load hand cascade classifier
        """
        try:
            # Thử load hand cascade (có thể cần tải riêng)
            self.hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
        except:
            print("Hand cascade không có sẵn, sử dụng face cascade thay thế")
            self.hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_hand_contour(self, frame):
        """
        Detect hand using contour detection
        """
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for skin color
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (assumed to be hand)
            largest_contour = max(contours, key=cv2.contourArea)
            
            if cv2.contourArea(largest_contour) > 5000:  # Minimum area threshold
                return largest_contour, mask
        
        return None, mask
    
    def extract_hand_features(self, contour, frame):
        """
        Extract features from hand contour
        """
        if contour is None:
            return None
        
        features = []
        
        # 1. Contour area
        area = cv2.contourArea(contour)
        features.append(area)
        
        # 2. Perimeter
        perimeter = cv2.arcLength(contour, True)
        features.append(perimeter)
        
        # 3. Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        features.append(aspect_ratio)
        
        # 4. Extent (ratio of contour area to bounding rectangle area)
        rect_area = w * h
        extent = float(area) / rect_area
        features.append(extent)
        
        # 5. Solidity (ratio of contour area to convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area) / hull_area
            features.append(solidity)
        else:
            features.append(0)
        
        # 6. Convexity defects (fingers detection)
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        if len(hull_indices) > 3:
            defects = cv2.convexityDefects(contour, hull_indices)
            if defects is not None:
                defect_count = len(defects)
                features.append(defect_count)
                
                # Average defect depth
                avg_depth = np.mean([defect[0][3] for defect in defects])
                features.append(avg_depth)
            else:
                features.extend([0, 0])
        else:
            features.extend([0, 0])
        
        # 7. Moments-based features
        moments = cv2.moments(contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Hu moments
            hu_moments = cv2.HuMoments(moments)
            features.extend(hu_moments.flatten())
        else:
            features.extend([0] * 7)
        
        # 8. Bounding box features
        features.extend([x, y, w, h])
        
        # 9. Centroid distance features
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Distance from centroid to extreme points
            leftmost = tuple(contour[contour[:,:,0].argmin()][0])
            rightmost = tuple(contour[contour[:,:,0].argmax()][0])
            topmost = tuple(contour[contour[:,:,1].argmin()][0])
            bottommost = tuple(contour[contour[:,:,1].argmax()][0])
            
            distances = [
                np.sqrt((cx - leftmost[0])**2 + (cy - leftmost[1])**2),
                np.sqrt((cx - rightmost[0])**2 + (cy - rightmost[1])**2),
                np.sqrt((cx - topmost[0])**2 + (cy - topmost[1])**2),
                np.sqrt((cx - bottommost[0])**2 + (cy - bottommost[1])**2)
            ]
            features.extend(distances)
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def collect_training_data(self, letter, features):
        """
        Thu thập dữ liệu training
        """
        if features is not None and len(features) > 0:
            self.training_data.append(features)
            self.training_labels.append(letter)
            print(f"Đã thu thập {len([l for l in self.training_labels if l == letter])} mẫu cho chữ '{letter}'")
    
    def train_model(self):
        """
        Training model
        """
        if len(self.training_data) < 10:
            print("Cần ít nhất 10 mẫu để training!")
            return False
        
        X = np.array(self.training_data)
        y = np.array(self.training_labels)
        
        # Handle NaN values
        X = np.nan_to_num(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Test accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model accuracy: {accuracy:.2f}")
        self.model_trained = True
        
        self.save_model()
        return True
    
    def predict_letter(self, features):
        """
        Dự đoán chữ cái
        """
        if not self.model_trained or self.model is None or features is None:
            return None, 0.0
        
        features = np.nan_to_num(features)
        features = features.reshape(1, -1)
        
        try:
            prediction = self.model.predict(features)[0]
            probability = np.max(self.model.predict_proba(features))
            
            self.prediction_buffer.append((prediction, probability))
            
            if len(self.prediction_buffer) >= 5:
                recent_predictions = [p[0] for p in list(self.prediction_buffer)[-5:]]
                most_common = max(set(recent_predictions), key=recent_predictions.count)
                avg_confidence = np.mean([p[1] for p in list(self.prediction_buffer)[-5:] if p[0] == most_common])
                return most_common, avg_confidence
            
            return prediction, probability
        except:
            return None, 0.0
    
    def save_model(self):
        """
        Lưu model
        """
        model_data = {
            'model': self.model,
            'training_data': self.training_data,
            'training_labels': self.training_labels
        }
        
        with open('hand_tracking_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        print("Đã lưu model!")
    
    def load_model(self):
        """
        Load model
        """
        try:
            with open('hand_tracking_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.training_data = model_data['training_data']
                self.training_labels = model_data['training_labels']
                self.model_trained = True
                print("Đã load model thành công!")
        except:
            print("Không tìm thấy model. Cần training mới.")
    
    def start_recognition(self):
        """
        Bắt đầu nhận dạng
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        training_mode = False
        current_letter = ""
        collect_count = 0
        target_count = 30
        
        print("=== HƯỚNG DẪN SỬ DỤNG (Phiên bản tương thích) ===")
        print("- Nhấn 't' để bắt đầu/kết thúc training mode")
        print("- Trong training mode, nhấn A-Z để thu thập dữ liệu")
        print("- Nhấn 'space' để train model")
        print("- Nhấn 'q' để thoát")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect hand contour
            contour, mask = self.detect_hand_contour(frame)
            
            # Draw contour if found
            if contour is not None:
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                
                # Extract features
                features = self.extract_hand_features(contour, frame)
                
                if training_mode and current_letter:
                    self.collect_training_data(current_letter, features)
                    collect_count += 1
                    
                    cv2.putText(frame, f"Collecting '{current_letter}': {collect_count}/{target_count}", 
                               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    
                    if collect_count >= target_count:
                        print(f"Đã thu thập đủ {target_count} mẫu cho chữ '{current_letter}'")
                        current_letter = ""
                        collect_count = 0
                
                elif not training_mode and self.model_trained:
                    prediction, confidence = self.predict_letter(features)
                    
                    if prediction and confidence > 0.6:
                        cv2.putText(frame, f"Letter: {prediction}", 
                                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display mode
            mode_text = "TRAINING" if training_mode else "RECOGNITION"
            color = (0, 0, 255) if training_mode else (0, 255, 0)
            cv2.putText(frame, f"Mode: {mode_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Show mask in corner
            mask_small = cv2.resize(mask, (160, 120))
            frame[10:130, frame.shape[1]-170:frame.shape[1]-10] = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            
            cv2.imshow('Hand Recognition (Compatible Version)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('t'):
                training_mode = not training_mode
                current_letter = ""
                collect_count = 0
            elif key == ord(' ') and training_mode:
                print("Bắt đầu training model...")
                self.train_model()
            elif training_mode and chr(key).upper() in self.alphabet:
                current_letter = chr(key).upper()
                collect_count = 0
                print(f"Bắt đầu thu thập dữ liệu cho chữ '{current_letter}'")
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = HandTrackingAlternative()
    tracker.start_recognition()