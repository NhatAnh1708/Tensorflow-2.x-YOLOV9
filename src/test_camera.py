import random
import time

import cv2

cap = cv2.VideoCapture(0)

while True:
    start_time = time.time()
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    ret, frame = cap.read()
    cv2.putText(
        frame,
        "FPS: {:.2f}".format(1.0 / (time.time() - start_time)),
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
