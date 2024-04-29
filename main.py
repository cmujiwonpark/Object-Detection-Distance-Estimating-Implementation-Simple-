import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import cv2
import cvzone
import math
import time
from PIL import Image, ImageTk

window = tk.Tk()

# window size
width = 1920
height = 1080
window.geometry(f"{width}x{height}")

# guide text
notifier  = """
Watch out for obstacles! 
"""

denotifier = """You're fine for now"""

warning_sign = Image.open('warning_sign.png')
warning_sign = ImageTk.PhotoImage(warning_sign)

# style for buttons
style = ttk.Style()
style.configure("Custom.TButton", foreground="black", background="black", font=("Arial", 12, "bold"))

def notify():
    text_label.config(state=tk.NORMAL)  # Enable editing the text widget
    text_label.delete("1.0", tk.END)  # Clear existing content
    text_label.insert(tk.END, notifier)  # Insert the guide text
    # Insert the image at the end and tag it for easy removal
    text_label.image_create(tk.END, image=warning_sign, padx=10, pady=10, align='baseline')
    text_label.config(state=tk.DISABLED)  # Disable editing the text widget
    text_label.configure(font=("Arial", 30, "bold"))  # Set the font to bold

def denotify():
    text_label.config(state=tk.NORMAL)  # Enable editing the text widget
    text_label.delete("1.0", tk.END)  # Clear existing content
    text_label.insert(tk.END, denotifier)  # Insert the guide text
    text_label.config(state=tk.DISABLED)  # Disable editing the text widget
    text_label.configure(font=("Arial", 30, "bold"))  # Set the font to bold

# creating all buttons
button1 = ttk.Button(window, text="Waymake", style="Custom.TButton")
button2 = ttk.Button(window, text="Stop Waymakers", style="Custom.TButton")

# arranging
button1.grid(row=1, column=1, padx=10, pady=10)
button2.grid(row=1, column=2, padx=10, pady=10)

# Create a text widget for displaying guide
text_label = tk.Text(window, font=("Arial", 10), width=160, height=35, state=tk.DISABLED)
text_label.grid(row=2, column=1, columnspan=4, padx=10, pady=10)


cap = cv2.VideoCapture(0)

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

person_width = 50
person_d = 200
car_width = 300
car_d = 400


prevframe = 0
newframe = 0

distance_dict = {} 

def process_frame():
    global prevframe, newframe
    newframe = time.time()
    good, img = cap.read()
    res = model(img, stream=True)

    current_distances = {} 
    notify_flag = False
    for r in res:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            class_index = int(box.cls[0])
            # Check if the detected class is a person

            if classNames[class_index] == "person": 
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0]*100)) / 100
                person_focal_length = (person_width * person_d) / w 
                person_apparent_width = w
                person_distance = ((person_width * person_focal_length) / person_apparent_width)*100
                cvzone.putTextRect(img, f'person, distance: {int(person_distance)}cm', (max(0, x1), max(35, y1)),
                                scale=1, thickness=1)
                distance = person_distance
                identifier = (x1, y1, x2, y2)
                current_distances[identifier] = distance
                if distance<100: 
                    notify_flag = True   
            elif classNames[class_index] == "car": 
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                car_focal_length = (car_width*car_d)/w
                car_apparent_width = w
                car_distance = ((car_width * car_focal_length) / car_apparent_width) * 901 
                cvzone.putTextRect(img, f'car, distance: {int(car_distance)}cm', (max(0, x1), max(35, y1)),
                                   scale=1, thickness=1)
                identifier = (x1, y1, x2, y2)
                current_distances[identifier] = car_distance 
                if car_distance<200: 
                    notify_flag = True 
            else:
                cls = int(box.cls[0])
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{classNames[cls]}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    distance_dict.update(current_distances)
    for key in list(distance_dict.keys()): 
        if key not in current_distances: 
            del distance_dict[key]

    if notify_flag: 
        notify()
    else: 
        denotify() 

    fps = 1 / (newframe - prevframe)
    prevframe = newframe
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    window.after(1, process_frame)

def start_webcam():
    process_frame()

def stop_webcam():
    cap.release()
    cv2.destroyAllWindows()


def display_list():
    text_label.config(state=tk.NORMAL)  # Enable editing the text widget
    text_label.delete("1.0", tk.END)  # Clear existing content
    text_label.insert(tk.END, notifier)  # Insert the guide text
    text_label.config(state=tk.DISABLED)  # Disable editing the text widget
    text_label.configure(font=("Arial", 12, "bold"))  # Set the font to bold

# Button actions
button1.configure(command=start_webcam)
button2.configure(command=stop_webcam)


window.mainloop()