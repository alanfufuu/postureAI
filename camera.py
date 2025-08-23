#check to see which camera index to actually use if multiple cameras connected
import cv2

def find_cameras():
    print("Searching for available cameras...")
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera found at index: {index}")
            cap.release()
        else:
            print(f"No camera at index: {index}")

if __name__ == '__main__':
    find_cameras()