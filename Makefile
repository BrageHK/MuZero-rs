build:
	TRAIN_BACKEND=$$(yq -r '.training_backend' configs/config.yaml); \
	INFERENCE_BACKEND=$$(yq -r '.inference_backend' configs/config.yaml); \
	cargo build -r --features="$$TRAIN_BACKEND,$$INFERENCE_BACKEND"

train:
	TRAIN_BACKEND=$$(yq -r '.training_backend' configs/config.yaml); \
	INFERENCE_BACKEND=$$(yq -r '.inference_backend' configs/config.yaml); \
	cargo run -r -p mz-train --bin train --features="$$TRAIN_BACKEND,$$INFERENCE_BACKEND"
