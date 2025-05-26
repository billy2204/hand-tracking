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
    
    def draw_ui_panel(self, frame, training_mode, current_letter, collect_count, target_count):
        """
        Vẽ panel hướng dẫn sử dụng đẹp mắt
        """
        height, width = frame.shape[:2]
        
        # Tạo panel nền trong suốt
        overlay = frame.copy()
        
        # Panel chính (góc trái)
        panel_width = 380
        panel_height = 280
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (100, 100, 100), 2)
        
        # Blend với frame gốc
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Header
        header_color = (0, 255, 255) if training_mode else (0, 255, 0)
        header_text = "🎯 TRAINING MODE" if training_mode else "👁️ RECOGNITION MODE"
        cv2.putText(frame, header_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, header_color, 2)
        
        # Đường phân cách
        cv2.line(frame, (20, 50), (panel_width-10, 50), (100, 100, 100), 1)
        
        if training_mode:
            # Training mode instructions
            instructions = [
                "📝 TRAINING INSTRUCTIONS:",
                "• Press A-Z: Select letter to collect",
                "• Hold pose: Collect samples",
                "• Press SPACE: Train model",
                "• Press T: Exit training mode"
            ]
            
            y_pos = 75
            for i, instruction in enumerate(instructions):
                color = (255, 255, 255) if i == 0 else (200, 200, 200)
                font_scale = 0.6 if i == 0 else 0.5
                cv2.putText(frame, instruction, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
                y_pos += 25
            
            # Current collection status
            if current_letter:
                # Progress bar
                progress_width = 200
                progress_height = 20
                progress_x = 20
                progress_y = 200
                
                # Background
                cv2.rectangle(frame, (progress_x, progress_y), (progress_x + progress_width, progress_y + progress_height), (50, 50, 50), -1)
                
                # Progress fill
                if target_count > 0:
                    fill_width = int((collect_count / target_count) * progress_width)
                    cv2.rectangle(frame, (progress_x, progress_y), (progress_x + fill_width, progress_y + progress_height), (0, 255, 0), -1)
                
                # Progress text
                progress_text = f"Collecting '{current_letter}': {collect_count}/{target_count}"
                cv2.putText(frame, progress_text, (20, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Percentage
                percentage = int((collect_count / target_count) * 100) if target_count > 0 else 0
                cv2.putText(frame, f"{percentage}%", (progress_x + progress_width + 10, progress_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            # Recognition mode instructions
            instructions = [
                "👁️ RECOGNITION MODE:",
                "• Show hand gesture to camera",
                "• Letter will be detected automatically",
                "• Green = High confidence",
                "• Press T: Enter training mode"
            ]
            
            y_pos = 75
            for i, instruction in enumerate(instructions):
                color = (255, 255, 255) if i == 0 else (200, 200, 200)
                font_scale = 0.6 if i == 0 else 0.5
                cv2.putText(frame, instruction, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
                y_pos += 25
        
        # Hotkeys panel (góc phải trên)
        hotkey_panel_width = 200
        hotkey_panel_height = 120
        hotkey_x = width - hotkey_panel_width - 10
        hotkey_y = 10
        
        # Nền hotkeys
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (hotkey_x, hotkey_y), (hotkey_x + hotkey_panel_width, hotkey_y + hotkey_panel_height), (0, 0, 0), -1)
        cv2.rectangle(overlay2, (hotkey_x, hotkey_y), (hotkey_x + hotkey_panel_width, hotkey_y + hotkey_panel_height), (100, 100, 100), 2)
        cv2.addWeighted(overlay2, 0.8, frame, 0.2, 0, frame)
        
        # Hotkeys
        cv2.putText(frame, "⌨️ HOTKEYS", (hotkey_x + 10, hotkey_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.line(frame, (hotkey_x + 10, hotkey_y + 30), (hotkey_x + hotkey_panel_width - 10, hotkey_y + 30), (100, 100, 100), 1)
        
        hotkeys = [
            "T - Toggle Mode",
            "SPACE - Train",
            "Q - Quit",
            "A-Z - Select Letter"
        ]
        
        y_pos = hotkey_y + 50
        for hotkey in hotkeys:
            cv2.putText(frame, hotkey, (hotkey_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            y_pos += 18
        
        # Status bar (dưới cùng)
        status_height = 40
        status_y = height - status_height
        cv2.rectangle(frame, (0, status_y), (width, height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, status_y), (width, status_y + 2), (100, 100, 100), -1)
        
        # Training data info
        if len(self.training_data) > 0:
            unique_letters = len(set(self.training_labels))
            total_samples = len(self.training_data)
            status_text = f"📊 Dataset: {unique_letters} letters, {total_samples} samples"
        else:
            status_text = "📊 No training data yet"
        
        cv2.putText(frame, status_text, (10, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Model status
        model_status = "🤖 Model: Ready" if self.model_trained else "🤖 Model: Not trained"
        model_color = (0, 255, 0) if self.model_trained else (0, 0, 255)
        cv2.putText(frame, model_status, (width - 200, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, model_color, 1)
    
    def draw_prediction_display(self, frame, prediction, confidence):
        """
        Vẽ kết quả prediction đẹp mắt
        """
        if prediction and confidence > 0.6:
            height, width = frame.shape[:2]
            
            # Tạo box hiển thị prediction ở giữa màn hình
            box_width = 300
            box_height = 150
            box_x = (width - box_width) // 2
            box_y = (height - box_height) // 2
            
            # Nền với gradient effect
            overlay = frame.copy()
            
            # Main box
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 50, 0), -1)
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), (0, 255, 0), 3)
            
            # Confidence bar background
            conf_bar_width = 200
            conf_bar_height = 10
            conf_bar_x = box_x + (box_width - conf_bar_width) // 2
            conf_bar_y = box_y + box_height - 30
            
            cv2.rectangle(overlay, (conf_bar_x, conf_bar_y), (conf_bar_x + conf_bar_width, conf_bar_y + conf_bar_height), (100, 100, 100), -1)
            
            # Confidence bar fill
            fill_width = int(confidence * conf_bar_width)
            color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 0, 255)
            cv2.rectangle(overlay, (conf_bar_x, conf_bar_y), (conf_bar_x + fill_width, conf_bar_y + conf_bar_height), color, -1)
            
            # Blend
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Letter text (lớn và đẹp)
            letter_size = 4.0
            letter_thickness = 8
            text_size = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, letter_size, letter_thickness)[0]
            letter_x = box_x + (box_width - text_size[0]) // 2
            letter_y = box_y + 80
            
            # Shadow effect
            cv2.putText(frame, prediction, (letter_x + 3, letter_y + 3), cv2.FONT_HERSHEY_SIMPLEX, letter_size, (0, 0, 0), letter_thickness)
            cv2.putText(frame, prediction, (letter_x, letter_y), cv2.FONT_HERSHEY_SIMPLEX, letter_size, (255, 255, 255), letter_thickness)
            
            # Confidence text
            conf_text = f"Confidence: {confidence:.1%}"
            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            conf_x = box_x + (box_width - conf_size[0]) // 2
            cv2.putText(frame, conf_text, (conf_x, box_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    def start_recognition(self):
        """
        Bắt đầu nhận dạng với UI đẹp
        """
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        training_mode = False
        current_letter = ""
        collect_count = 0
        target_count = 30
        
        print("🚀 Starting Sign Language Recognition...")
        print("📱 UI controls are displayed on screen")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect hand contour
            contour, mask = self.detect_hand_contour(frame)
            
            # Draw enhanced contour
            if contour is not None:
                # Draw filled contour with transparency
                overlay = frame.copy()
                cv2.fillPoly(overlay, [contour], (0, 255, 0))
                cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
                
                # Draw contour outline
                cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                
                # Draw convex hull
                hull = cv2.convexHull(contour)
                cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
                
                # Extract features
                features = self.extract_hand_features(contour, frame)
                
                if training_mode and current_letter:
                    self.collect_training_data(current_letter, features)
                    collect_count += 1
                    
                    if collect_count >= target_count:
                        print(f"✅ Completed collecting {target_count} samples for '{current_letter}'")
                        current_letter = ""
                        collect_count = 0
                
                elif not training_mode and self.model_trained:
                    prediction, confidence = self.predict_letter(features)
                    self.draw_prediction_display(frame, prediction, confidence)
            
            # Draw main UI panel
            self.draw_ui_panel(frame, training_mode, current_letter, collect_count, target_count)
            
            # Show processed mask in corner (smaller)
            mask_small = cv2.resize(mask, (120, 90))
            mask_colored = cv2.applyColorMap(mask_small, cv2.COLORMAP_JET)
            frame[frame.shape[0]-100:frame.shape[0]-10, frame.shape[1]-130:frame.shape[1]-10] = mask_colored
            
            # Add corner label for mask
            cv2.putText(frame, "Hand Mask", (frame.shape[1]-130, frame.shape[0]-105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow('🤟 Sign Language Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("👋 Goodbye!")
                break
            elif key == ord('t'):
                training_mode = not training_mode
                current_letter = ""
                collect_count = 0
                mode_name = "TRAINING" if training_mode else "RECOGNITION"
                print(f"🔄 Switched to {mode_name} mode")
            elif key == ord(' ') and training_mode and len(self.training_data) > 0:
                print("🧠 Training model... Please wait...")
                success = self.train_model()
                if success:
                    print("✅ Model trained successfully!")
                else:
                    print("❌ Training failed. Need more data.")
            elif training_mode and chr(key).upper() in self.alphabet:
                current_letter = chr(key).upper()
                collect_count = 0
                print(f"📝 Started collecting data for letter '{current_letter}'")
        
        cap.release()
        cv2.destroyAllWindows()
        print("🔚 Application closed")

if __name__ == "__main__":
    tracker = HandTrackingAlternative()
    tracker.start_recognition()