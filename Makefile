download:
	python src/download.py
export-onnx:
	# git clone https://github.com/WongKinYiu/yolov9.git
	# cd yolov9
	pip install -r yolov9/requirements.txt
	python yolov9/export.py --weights yolov9e.pt --img 640 --batch 1 --device 0 --include onnx

onnx-tf:
	onnx-tf convert -i yolov9e.onnx -o yolov9e_pb
tf-tflite:
	python src/export_tflite.py
test-cam:
	python src/test_camera.py 

help:
	@echo "export-onnx: export onnx model"
	@echo "onnx-tf: convert onnx model to tensorflow model"
	@echo "test-cam: test webcam"
	@echo "help: show help message"