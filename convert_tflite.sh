echo "Starting convert pytorch to tflite" && \
make download && \
make export-onnx && \
make onnx-tf && \
make tf-tflite && \
echo "Finished convert pytorch to tf"