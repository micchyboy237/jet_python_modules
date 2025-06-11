# Use arm64 onnx quantized models
# Resolve symlinks

# For sentence-transformers/all-MiniLM-L12-v2
ls -l /Users/jethroestrada/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L12-v2/snapshots/c004d8e3e901237d8fa7e9fff12774962e391ce5/onnx/model_qint8_arm64.onnx
cp -L /Users/jethroestrada/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L12-v2/snapshots/c004d8e3e901237d8fa7e9fff12774962e391ce5/onnx/model_qint8_arm64.onnx /Users/jethroestrada/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L12-v2/snapshots/c004d8e3e901237d8fa7e9fff12774962e391ce5/onnx/model.onnx
rm /Users/jethroestrada/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L12-v2/snapshots/c004d8e3e901237d8fa7e9fff12774962e391ce5/onnx/model_qint8_arm64.onnx
ls -l /Users/jethroestrada/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L12-v2/snapshots/c004d8e3e901237d8fa7e9fff12774962e391ce5/onnx/
du -sh /Users/jethroestrada/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L12-v2/snapshots/c004d8e3e901237d8fa7e9fff12774962e391ce5/onnx/model.onnx

# For cross-encoder/ms-marco-MiniLM-L6-v2
ls -l /Users/jethroestrada/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2/snapshots/ce0834f22110de6d9222af7a7a03628121708969/onnx/model_qint8_arm64.onnx
cp -L /Users/jethroestrada/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2/snapshots/ce0834f22110de6d9222af7a7a03628121708969/onnx/model_qint8_arm64.onnx /Users/jethroestrada/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2/snapshots/ce0834f22110de6d9222af7a7a03628121708969/onnx/model.onnx
rm /Users/jethroestrada/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2/snapshots/ce0834f22110de6d9222af7a7a03628121708969/onnx/model_qint8_arm64.onnx
ls -l /Users/jethroestrada/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2/snapshots/ce0834f22110de6d9222af7a7a03628121708969/onnx/
du -sh /Users/jethroestrada/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L6-v2/snapshots/ce0834f22110de6d9222af7a7a03628121708969/onnx/model.onnx
