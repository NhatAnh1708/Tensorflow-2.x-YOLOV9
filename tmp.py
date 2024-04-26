# from ultralytics import YOLO
 
# model = YOLO('face_plateplate.pt')

 
# model.export(format="coreml", nms=True, imgsz=320)


# # import cv2
# # import time 
# import numpy as np

# def noise(frame):
#     h, w, c = frame.shape
#     kernel_size = (h//10, w//10)
#     new_frame = cv2.blur(frame, kernel_size)
#     return new_frame
# def medianfilter(img):
#     h, w, c = img.shape
#     img = cv2.boxFilter(img, -1, (h//10, w//10))
#     return img
# video = cv2.VideoCapture(0)

# while True:
#     ret, frame = video.read()
#     start_time = time.time()
#     frame = medianfilter(frame)
#     fps = 1/(time.time()-start_time)
#     fps = np.round(fps, 2)
#     cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow('Video', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video.release()


# def pixelate_image(image, grid_size):
# 	(h, w) = image.shape[:2]
# 	xGridLines = np.linspace(0, w, grid_size + 1, dtype="int")
# 	yGridLines = np.linspace(0, h, grid_size + 1, dtype="int")

# 	for i in range(1, len(xGridLines)):
# 		for j in range(1, len(yGridLines)):
# 			cell_startX = xGridLines[j - 1]
# 			cell_startY = yGridLines[i - 1]
# 			cell_endX = xGridLines[j]
# 			cell_endY = yGridLines[i]
# 			cell = image[cell_startY:cell_endY, cell_startX:cell_endX]
# 			(B, G, R) = [int(x) for x in cv2.mean(cell)[:3]]
# 			cv2.rectangle(image, (cell_startX, cell_startY), (cell_endX, cell_endY),
# 				(B, G, R), -1)

# 	return image


from moviepy.editor import VideoFileClip

def convert_webm_to_mp4(input_file, output_file):
    # Load the WebM video
    video = VideoFileClip(input_file)
    
    # Save the video in MP4 format
    video.write_videofile(output_file, codec="libx264", audio_codec="aac")
    
    # Close the video file
    video.close()

# Example usage
input_file = "data_test/testcase14.webm"
output_file = "output.mp4"
convert_webm_to_mp4(input_file, output_file)
