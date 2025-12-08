"""
GUI untuk testing model deteksi kecurangan
"""
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import cv2
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(__file__))
from predict import load_model, predict_image, analyze_detections
from config import CLASS_NAMES, CONFIDENCE_THRESHOLD, CHEATING_INDICATORS

class CheatingDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem Deteksi Kecurangan Ujian - YOLOv8")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        self.model = None
        self.cap = None
        self.running = False
        
        self.setup_ui()
        self.load_model_async()
    
    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#1976D2', height=60)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title = tk.Label(header, text="🎓 Sistem Deteksi Kecurangan Ujian", 
                        font=("Arial", 18, "bold"), bg='#1976D2', fg='white')
        title.pack(expand=True)
        
        # Main container
        main = tk.Frame(self.root, bg='#f0f0f0')
        main.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left - Image/Video display
        left_frame = tk.Frame(main, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas_frame = tk.Frame(left_frame, bg='#333', width=640, height=480)
        self.canvas_frame.pack(pady=10)
        self.canvas_frame.pack_propagate(False)
        
        self.canvas_label = tk.Label(self.canvas_frame, text="Pilih gambar atau aktifkan webcam",
                                     bg='#333', fg='#888', font=("Arial", 12))
        self.canvas_label.pack(expand=True)
        
        # Buttons
        btn_frame = tk.Frame(left_frame, bg='#f0f0f0')
        btn_frame.pack(pady=10)
        
        self.btn_image = tk.Button(btn_frame, text="📷 Pilih Gambar", command=self.select_image,
                                   font=("Arial", 11), padx=15, pady=8, cursor="hand2")
        self.btn_image.pack(side=tk.LEFT, padx=5)
        
        self.btn_webcam = tk.Button(btn_frame, text="🎥 Webcam", command=self.toggle_webcam,
                                    font=("Arial", 11), padx=15, pady=8, cursor="hand2")
        self.btn_webcam.pack(side=tk.LEFT, padx=5)
        
        self.btn_video = tk.Button(btn_frame, text="🎬 Pilih Video", command=self.select_video,
                                   font=("Arial", 11), padx=15, pady=8, cursor="hand2")
        self.btn_video.pack(side=tk.LEFT, padx=5)
        
        # Right - Results panel
        right_frame = tk.Frame(main, bg='#fff', width=250, relief=tk.RIDGE, bd=1)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        right_frame.pack_propagate(False)
        
        # Status
        status_frame = tk.Frame(right_frame, bg='#fff', pady=15)
        status_frame.pack(fill=tk.X)
        
        tk.Label(status_frame, text="STATUS", font=("Arial", 10, "bold"), 
                bg='#fff', fg='#666').pack()
        
        self.status_label = tk.Label(status_frame, text="READY", 
                                     font=("Arial", 24, "bold"), bg='#fff', fg='#4CAF50')
        self.status_label.pack(pady=10)
        
        # Separator
        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, padx=10)
        
        # Detections
        det_frame = tk.Frame(right_frame, bg='#fff', pady=15)
        det_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(det_frame, text="DETEKSI", font=("Arial", 10, "bold"),
                bg='#fff', fg='#666').pack()
        
        self.det_listbox = tk.Listbox(det_frame, font=("Arial", 10), height=8,
                                      selectmode=tk.NONE, relief=tk.FLAT)
        self.det_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Separator
        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, padx=10)
        
        # Reasons
        reason_frame = tk.Frame(right_frame, bg='#fff', pady=15)
        reason_frame.pack(fill=tk.X)
        
        tk.Label(reason_frame, text="ALASAN", font=("Arial", 10, "bold"),
                bg='#fff', fg='#666').pack()
        
        self.reason_label = tk.Label(reason_frame, text="-", font=("Arial", 10),
                                     bg='#fff', fg='#333', wraplength=220, justify=tk.LEFT)
        self.reason_label.pack(pady=5, padx=10)
        
        # Footer status
        self.footer = tk.Label(self.root, text="⏳ Loading model...", 
                              font=("Arial", 9), bg='#f0f0f0', fg='#666', anchor='w')
        self.footer.pack(fill=tk.X, padx=20, pady=5)
    
    def load_model_async(self):
        def load():
            self.model = load_model()
            if self.model:
                self.footer.config(text="✅ Model loaded successfully", fg='green')
            else:
                self.footer.config(text="❌ Model not found. Train first!", fg='red')
        
        threading.Thread(target=load, daemon=True).start()
    
    def select_image(self):
        if self.running:
            self.stop_webcam()
        
        path = filedialog.askopenfilename(
            title="Pilih gambar",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if path and self.model:
            self.process_image(path)
    
    def process_image(self, path):
        # Load and display image
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Predict
        results = self.model(path, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        # Get annotated image
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Resize for display
        h, w = annotated_rgb.shape[:2]
        max_h, max_w = 480, 640
        scale = min(max_w/w, max_h/h)
        new_w, new_h = int(w*scale), int(h*scale)
        display = cv2.resize(annotated_rgb, (new_w, new_h))
        
        # Show image
        photo = ImageTk.PhotoImage(Image.fromarray(display))
        self.canvas_label.config(image=photo, text="")
        self.canvas_label.image = photo
        
        # Analyze
        detections = []
        for box in results[0].boxes:
            detections.append([
                *box.xyxy[0].tolist(),
                box.conf[0].item(),
                box.cls[0].item()
            ])
        
        analysis = analyze_detections(detections)
        self.update_results(analysis)
    
    def update_results(self, analysis):
        # Update status
        if analysis['is_cheating']:
            self.status_label.config(text="⚠️ CHEATING", fg='#F44336')
        else:
            self.status_label.config(text="✅ NORMAL", fg='#4CAF50')
        
        # Update detections
        self.det_listbox.delete(0, tk.END)
        for cls in analysis['detected_classes']:
            self.det_listbox.insert(tk.END, f"  • {cls}")
        
        # Update reasons
        if analysis['reasons']:
            self.reason_label.config(text="\n".join(analysis['reasons']))
        else:
            self.reason_label.config(text="Tidak ada indikasi kecurangan")
    
    def toggle_webcam(self):
        if self.running:
            self.stop_webcam()
        else:
            self.start_webcam()
    
    def start_webcam(self):
        if not self.model:
            self.footer.config(text="❌ Model belum loaded!", fg='red')
            return
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.footer.config(text="❌ Webcam tidak tersedia!", fg='red')
            return
        
        self.running = True
        self.btn_webcam.config(text="⏹️ Stop")
        self.footer.config(text="🎥 Webcam aktif", fg='green')
        
        self.process_webcam()
    
    def process_webcam(self):
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if ret:
            # Predict
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
            
            # Get annotated frame
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            # Resize
            h, w = annotated_rgb.shape[:2]
            scale = min(640/w, 480/h)
            new_w, new_h = int(w*scale), int(h*scale)
            display = cv2.resize(annotated_rgb, (new_w, new_h))
            
            # Show
            photo = ImageTk.PhotoImage(Image.fromarray(display))
            self.canvas_label.config(image=photo, text="")
            self.canvas_label.image = photo
            
            # Analyze
            detections = []
            for box in results[0].boxes:
                detections.append([
                    *box.xyxy[0].tolist(),
                    box.conf[0].item(),
                    box.cls[0].item()
                ])
            
            analysis = analyze_detections(detections)
            self.update_results(analysis)
        
        if self.running:
            self.root.after(30, self.process_webcam)
    
    def stop_webcam(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_webcam.config(text="🎥 Webcam")
        self.footer.config(text="✅ Model loaded successfully", fg='green')
    
    def select_video(self):
        if self.running:
            self.stop_webcam()
        
        path = filedialog.askopenfilename(
            title="Pilih video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov")]
        )
        
        if path and self.model:
            self.process_video(path)
    
    def process_video(self, path):
        self.cap = cv2.VideoCapture(path)
        self.running = True
        self.btn_video.config(text="⏹️ Stop")
        self.footer.config(text=f"🎬 Playing: {os.path.basename(path)}", fg='green')
        self.process_webcam()  # Same logic as webcam
    
    def on_closing(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CheatingDetectorGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
