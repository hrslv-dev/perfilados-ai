
from camera.capture import CameraCapture
from vision.preprocessemt import Preprocessor
from vision.contours import ContourDetector

import cv2; 

camera = CameraCapture()
preprocessor = Preprocessor()
contour_detector = ContourDetector()

camera.connect()

while True: 
    frame = camera.read_frame()
    
    processed = preprocessor.process(frame)

    contours = contour_detector.find_contours(processed)

    frame_with_contours = (contour_detector.draw_contours(frame.copy(), contours))
    
    cv2.imshow("Contours", frame_with_contours)

    if cv2.waitKey(1) == 27: 
        break 

camera.release()