.DEFAULT_GOAL := help

build: ## Build release binary with backends from config.yaml
	TRAIN_BACKEND=$$(yq -r '.training_backend' configs/config.yaml); \
	INFERENCE_BACKEND=$$(yq -r '.inference_backend' configs/config.yaml); \
	cargo build -r --features="$$TRAIN_BACKEND,$$INFERENCE_BACKEND"

train: ## Run training with backends from config.yaml
	TRAIN_BACKEND=$$(yq -r '.training_backend' configs/config.yaml); \
	INFERENCE_BACKEND=$$(yq -r '.inference_backend' configs/config.yaml); \
	cargo run -r -p mz-train --bin train --features="$$TRAIN_BACKEND,$$INFERENCE_BACKEND"

benchmark: ## Sweep inference batch sizes across all backends
	cargo run -r -p mz-train --bin resnet_infer_bench --features "tch,vulkan,rocm"
	cargo run -r -p mz-train --bin resnet_infer_bench --features "wgpu"

web-build: ## Compile mz-web to wasm into web/pkg
	wasm-pack build crates/mz-web --target web --out-dir web/pkg

web-serve: ## Serve crates/mz-web/web on localhost:8080
	python3 -m http.server -d crates/mz-web/web 8080

web: web-build web-serve ## Build wasm then serve web app

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "%-12s %s\n", $$1, $$2}'
