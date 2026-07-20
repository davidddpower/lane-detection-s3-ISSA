
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

    #Exercitiul 6
    blur_frame = cv2.blur(top_down, ksize=(3, 3))

    #Exercitiul 7
    sobel_matrix = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])

    sobel_horizontal = np.transpose(sobel_matrix)
    blur_frame_float = np.float32(blur_frame)

    sobelV_result = cv2.filter2D(blur_frame_float, -1, sobel_matrix)
    sobelH_result = cv2.filter2D(blur_frame_float, -1, sobel_horizontal)

    cv2.imshow('Sobel Vertical', cv2.convertScaleAbs(sobelV_result))
    cv2.imshow('Sobel Horizontal', cv2.convertScaleAbs(sobelH_result))
    combined = np.sqrt(sobelV_result ** 2 + sobelH_result ** 2)
    cv2.imshow('Sobel Version', cv2.convertScaleAbs(combined))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
