from tkinter import *
import cv2 as cv
from PIL import Image, ImageTk
import mediapipe as mp
import os 

VideoCapture = 1
fireimage_path = os.path.join(os.path.dirname(__file__), "placeholder.png")
capture_x = 1300
capture_y = 800


class App:
    def __init__(self, root):
        self.cap = None        
        self.prev_frame_time = 0
        self.frame_count = 0

        # GUI Setup
        self.root = root
        self.root.title("Python Study")
        self.root.geometry("1300x800")
        self.root.resizable(True, True)
    
        self.frame = Frame(self.root)
        self.frame.pack()

        self.canvas = Canvas(self.frame, width=capture_x, height=capture_y, bg="black")
        self.canvas.pack()

        self.coords_Label = Label(self.frame, text="Coords: 0, 0", font=("Arial", 10))
        self.coords_Label.place(x=20, y=30)

        self.handstatus_Label = Label(self.frame, text="Hand Status", font=("Arial", 10))
        self.handstatus_Label.place(x=20, y=60)

        fire_image = Image.open(fireimage_path)
        fire_image = fire_image.resize((100, 100), Image.LANCZOS)
        self.fire_photo = ImageTk.PhotoImage(fire_image)
        self.fireimage_Label = Label(self.frame, image=self.fire_photo)

        
        landmark_frame_Button = Button(self.frame, text="Landmark Frame", command=self.toggle_hand_skeleton)
        landmark_frame_Button.place(x=20, y=90)
        
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.10,
            min_tracking_confidence=0.10,
            model_complexity=0 
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.cap = cv.VideoCapture(VideoCapture)  
        if not self.cap.isOpened():
            print("No Camera found")
            self.cleanup()
            return
            
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, capture_x)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, capture_y)
        self.delay = 15 
        self.show_skeleton = True
        
        self.update()

    def toggle_hand_skeleton(self):
        self.show_skeleton = not self.show_skeleton
        
        if self.show_skeleton:
            print("Hand skeleton shown")
        else:
            print("Hand skeleton hidden")

    def update(self):
        ret, frame = self.cap.read()
        frame = cv.flip(frame, 1) 
        if not ret:
            print("Failed to capture image from camera.")
            self.cleanup()
            return

        left_hand_status = "not detected"
        right_hand_status = "not detected"

        
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_classification in zip(results.multi_hand_landmarks, results.multi_handedness):

                hand_label = hand_classification.classification[0].label
                hand_confidence = hand_classification.classification[0].score

                if hand_label == "Left":
                    print(f"Left Hand: {hand_confidence:.2f}")
                else:
                    print(f"Right Hand: {hand_confidence:.2f}")


                if self.show_skeleton:
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)     
                h, w, _ = frame.shape

                index_finger_tip = hand_landmarks.landmark[8]
                index_finger_tip_px = int(index_finger_tip.x * w)
                index_finger_tip_py = int(index_finger_tip.y * h)

                middle_finger_tip = hand_landmarks.landmark[12]
                middle_finger_tip_py = int(middle_finger_tip.y * h)

                ring_finger_tip = hand_landmarks.landmark[16]
                ring_finger_tip_py = int(ring_finger_tip.y * h)

                pinky_finger_tip = hand_landmarks.landmark[20]
                pinky_finger_tip_py = int(pinky_finger_tip.y * h)

                middle_finger_mcp = hand_landmarks.landmark[9]
                middle_finger_mcp_py = int(middle_finger_mcp.y * h)
                middle_finger_mcp_px = int(middle_finger_mcp.x * w)

                if (index_finger_tip_py > middle_finger_mcp_py and 
                    middle_finger_tip_py > middle_finger_mcp_py and 
                    ring_finger_tip_py > middle_finger_mcp_py and 
                    pinky_finger_tip_py > middle_finger_mcp_py):

                
                    self.fireimage_Label.place_forget()
                    self.handstatus_Label.config(text="Hand Status: Closed")
                else:
                    self.handstatus_Label.config(text="Hand Status: Opened")

                    self.fireimage_Label.place(x=middle_finger_mcp_px - 20, y=middle_finger_mcp_py + 50)
                    self.coords_Label.config(text=f"Coords: {middle_finger_mcp_px - 20}, {middle_finger_mcp_py + 50}")

        display_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img = Image.fromarray(display_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=NW, image=imgtk)
        self.canvas.imgtk = imgtk 

        self.root.after(self.delay, self.update)



    def cleanup(self):
        if hasattr(self, 'cam') and self.cam:
            self.cam.release()
        if hasattr(self, 'root'):
            self.root.quit()
            print("Application closed.")

    def __del__(self):
        self.cleanup()

if __name__ == "__main__":
    root = Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()