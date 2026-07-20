
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

    # Exercitiul 4
    upper = [(int(width * 0.45), int(height * 0.75)), (int(width * 0.55), int(height * 0.75))]
    lower = [(0, height), (width, height)]

    trapezoid_bounds = np.array([upper[0], upper[1], lower[0], lower[1]], dtype=np.int32)
    trapezoid_points = np.zeros((height, width), dtype=np.uint8)

    cv2.fillConvexPoly(trapezoid_points, trapezoid_bounds, 1)
    road = gray_frame * trapezoid_points

    # Exercitiul 5
    screen_upper = [(0, 0), (width, 0)]
    screen_lower = [(0, height), (width, height)]

    frame_bounds = np.array([screen_upper[1], screen_upper[0], screen_lower[0], screen_lower[1]],
                            dtype=np.float32)
    trapezoid_bounds_float = np.float32(trapezoid_bounds)
    matrix = cv2.getPerspectiveTransform(trapezoid_bounds_float, frame_bounds)
    top_down = cv2.warpPerspective(road, matrix, (width, height))
    cv2.imshow('Top-Down Version', top_down)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
