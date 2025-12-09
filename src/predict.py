"""
Predict menggunakan model YOLOv8 yang sudah ditraining
"""
import os
import cv2
from ultralytics import YOLO
from config import *

def load_model(model_path=None):
    if model_path is None:
        # Cek hasil_colab dulu, baru checkpoint
        colab_model = os.path.join(BASE_DIR, 'hasil_colab', 'best.pt')
        checkpoint_model = os.path.join(CHECKPOINT_DIR, 'cheating_detector', 'weights', 'best.pt')
        
        if os.path.exists(colab_model):
            model_path = colab_model
            print(f"✅ Loading model dari Colab: {colab_model}")
        elif os.path.exists(checkpoint_model):
            model_path = checkpoint_model
            print(f"✅ Loading model dari checkpoint: {checkpoint_model}")
        else:
            print(f"❌ Model tidak ditemukan di:")
            print(f"   - {colab_model}")
            print(f"   - {checkpoint_model}")
            return None
    
    if not os.path.exists(model_path):
        print(f"Model tidak ditemukan: {model_path}")
        return None
    
    return YOLO(model_path)

def analyze_detections(detections):
    """Analisis hasil deteksi untuk menentukan status kecurangan"""
    detected_classes = set()
    
    for det in detections:
        class_id = int(det[5])
        detected_classes.add(class_id)
    
    # Cek indikator kecurangan
    cheating_reasons = []
    
    # Head pose tidak center
    head_cheating = detected_classes.intersection(set(CHEATING_INDICATORS['head']))
    if head_cheating:
        for c in head_cheating:
            cheating_reasons.append(f"Head: {CLASS_NAMES[c]}")
    
    # Eye gaze tidak center
    eye_cheating = detected_classes.intersection(set(CHEATING_INDICATORS['eye']))
    if eye_cheating:
        for c in eye_cheating:
            cheating_reasons.append(f"Eye: {CLASS_NAMES[c]}")
    
    # Lip movement (speaking)
    lip_cheating = detected_classes.intersection(set(CHEATING_INDICATORS['lip']))
    if lip_cheating:
        cheating_reasons.append("Lip: Speaking detected")
    
    is_cheating = len(cheating_reasons) > 0
    
    return {
        'is_cheating': is_cheating,
        'reasons': cheating_reasons,
        'detected_classes': [CLASS_NAMES[c] for c in detected_classes]
    }

def predict_image(model, image_path):
    """Prediksi single image"""
    results = model(image_path, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            detections.append([x1, y1, x2, y2, conf, cls])
    
    analysis = analyze_detections(detections)
    
    return detections, analysis

def predict_video(model, video_path, output_path=None):
    """Prediksi video"""
    cap = cv2.VideoCapture(video_path)
    
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    frame_count = 0
    cheating_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Predict
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Analyze
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            detections.append([x1, y1, x2, y2, conf, cls])
        
        analysis = analyze_detections(detections)
        
        if analysis['is_cheating']:
            cheating_frames += 1
            status = "CHEATING DETECTED"
            color = (0, 0, 255)
        else:
            status = "NORMAL"
            color = (0, 255, 0)
        
        cv2.putText(annotated_frame, status, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if output_path:
            out.write(annotated_frame)
        
        cv2.imshow('Cheating Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\nTotal frames: {frame_count}")
    print(f"Cheating frames: {cheating_frames} ({cheating_frames/frame_count*100:.1f}%)")

def predict_webcam(model):
    """Real-time prediction dari webcam"""
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Predict
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Analyze
        detections = []
        for box in results[0].boxes:
            detections.append([
                *box.xyxy[0].tolist(),
                box.conf[0].item(),
                box.cls[0].item()
            ])
        
        analysis = analyze_detections(detections)
        
        # Display status
        if analysis['is_cheating']:
            status = "CHEATING!"
            color = (0, 0, 255)
            reasons = ", ".join(analysis['reasons'][:2])
            cv2.putText(annotated_frame, reasons, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            status = "NORMAL"
            color = (0, 255, 0)
        
        cv2.putText(annotated_frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow('Cheating Detection - Press Q to Quit', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    model = load_model()
    if model is None:
        sys.exit(1)
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if path == 'webcam':
            predict_webcam(model)
        elif path.endswith(('.mp4', '.avi', '.mov')):
            predict_video(model, path)
        else:
            detections, analysis = predict_image(model, path)
            print(f"\nDetected: {analysis['detected_classes']}")
            print(f"Cheating: {analysis['is_cheating']}")
            if analysis['reasons']:
                print(f"Reasons: {', '.join(analysis['reasons'])}")
    else:
        print("Usage:")
        print("  python predict.py <image_path>")
        print("  python predict.py <video_path>")
        print("  python predict.py webcam")
