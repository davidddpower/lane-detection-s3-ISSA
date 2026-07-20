
import cv2
import numpy as np
import os

video_file = 'Lane Detection Test Video 01.mp4'

# Verific dacă fișierul video există
if not os.path.exists(video_file):
    exit()

cam = cv2.VideoCapture(video_file)

while True:
    ret, frame = cam.read()

    # ret (bool): Return code of the `read` operation. Did we get an image or not?
    #             (if not maybe the camera is not detected/connected etc.)

    # frame (array): The actual frame as an array.
    #                Height x Width x 3 (3 colors, BGR) if color image.
    #                Height x Width if Grayscale
    #                Each element is 0-255.
    #                You can slice it, reassign elements to change pixels, etc.

    if ret is False:
        break

    # Exercitiul 2
    height, width, _ = frame.shape
    frame = cv2.resize(frame, (width // 2, height // 2))
    height, width, _ = frame.shape

    # Exercitiul 3
    gray_frame = np.zeros((height, width), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            b, g, r = frame[row, col]
            gray_frame[row, col] = (int(b) + int(g) + int(r)) // 3
    cv2.imshow('Grayscale', gray_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
