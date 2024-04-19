export-onnx:
	python yolov9/export.py --weights model/yolov9-e.pt --img 640 --batch 1 --device 0 --include onnx

onnx-tf:
	onnx-tf convert -i model/yolov9-e.onnx -o yolov9e_pb