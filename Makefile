.PHONY: build run dev clean install fmt check help

MODEL ?= $(HOME)/Models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf

help:
	@echo "Oxide-rs - Fast AI Inference Library & CLI"
	@echo ""
	@echo "Usage:"
	@echo "  make build      Build release binary"
	@echo "  make run        Run oxide-rs with MODEL env var"
	@echo "  make dev        Build in dev mode (faster compile)"
	@echo "  make clean      Remove build artifacts"
	@echo "  make install    Install to ~/.local/bin"
	@echo "  make fmt        Format code"
	@echo "  make check      Run linter"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL=$(MODEL)"

build:
	cargo build --release

run: build
	./target/release/oxide-rs --model $(MODEL)

dev:
	cargo build

clean:
	cargo clean
	rm -rf target/

install: build
	@mkdir -p $(HOME)/.local/bin
	cp ./target/release/oxide-rs $(HOME)/.local/bin/
	@echo "Installed to ~/.local/bin/oxide-rs"

fmt:
	cargo fmt

check:
	cargo clippy -- -D warnings
