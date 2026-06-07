#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# Script: steps_ai_inter.sh
# Purpose: Archive & replicate successful OpenVINO AI Inference setup
#          for Intel GPU acceleration on Ubuntu 24.04 (Noble) using Qwen2.5-0.5B.
# Author: Antigravity AI Assistant
# -----------------------------------------------------------------------------

set -e # Exit immediately if a command exits with a non-zero status

# Text formatting helper functions
info() { echo -e "\e[34m[INFO]\e[0m $*"; }
success() { echo -e "\e[32m[SUCCESS]\e[0m $*"; }
warn() { echo -e "\e[33m[WARNING]\e[0m $*"; }
error() { echo -e "\e[31m[ERROR]\e[0m $*"; }

show_help() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  1  - Install Intel GPU NEO Drivers & OpenCL Runtimes (Ubuntu 24.04 Noble)"
    echo "  2  - Setup Virtual Environment & Python dependencies (CPU-only PyTorch + OpenVINO)"
    echo "  3  - Export Qwen2.5-0.5B-Instruct model to OpenVINO INT8 format"
    echo "  4  - Run Qwen benchmark script"
    echo "  5  - Run Qwen terminal chat script"
    echo "  6  - Run Web AI Chat Interface"
    echo "  all- Prepare environment completely (Steps 1, 2, 3)"
    echo "  help- Show this help message"
    echo ""
}

# -----------------------------------------------------------------------------
# Step 1: Install Intel GPU Drivers and OpenCL Runtime
# -----------------------------------------------------------------------------
install_intel_gpu_drivers() {
    info "Installing Intel GPU Drivers and OpenCL headers..."
    sudo apt update
    sudo apt install -y gpg-agent wget ocl-icd-opencl-dev opencl-headers clinfo

    info "Downloading Intel Compute Runtime (NEO) packages for Ubuntu 24.04..."
    # Specific Ubuntu 24.04 packages used in successful run
    wget -c https://github.com/intel/compute-runtime/releases/download/24.39.31294.12/intel-opencl-icd_24.39.31294.12_amd64.deb
    wget -c https://github.com/intel/compute-runtime/releases/download/24.39.31294.12/libigdgmm12_22.5.2_amd64.deb

    info "Installing package dependencies via dpkg..."
    sudo dpkg -i libigdgmm12_22.5.2_amd64.deb
    sudo dpkg -i intel-opencl-icd_24.39.31294.12_amd64.deb
    sudo apt install -f -y # Resolve any outstanding dependency issues

    info "Adding current user '$USER' to 'render' and 'video' groups..."
    sudo usermod -aG render,video $USER

    success "Intel GPU driver installation complete!"
    warn "IMPORTANT: You must log out and log back in, or run 'newgrp render' to apply group permissions."
    
    info "Checking if GPU is detected (clinfo):"
    clinfo | grep -E "Device Name|Platform Name|Device Type" || warn "GPU clinfo verification failed. Make sure you are in the render/video group."
}

# -----------------------------------------------------------------------------
# Step 2: Setup Python Virtual Environment and Install Dependencies
# -----------------------------------------------------------------------------
setup_python_env() {
    info "Setting up Python virtual environment 'ov_env'..."
    sudo apt install -y python3-venv python3.12-venv
    
    # Create the virtual environment in home directory if it doesn't exist
    if [ ! -d "$HOME/ov_env" ]; then
        python3 -m venv "$HOME/ov_env"
        info "Created virtual environment at $HOME/ov_env"
    else
        info "Virtual environment $HOME/ov_env already exists"
    fi

    # Activate environment
    source "$HOME/ov_env/bin/activate"

    info "Installing OpenVINO and huggingface dependencies..."
    # Uninstall conflicting CUDA libraries to keep it lightweight
    pip uninstall -y torch accelerate nvidia-cudnn-cu13 nvidia-cublas cuda-toolkit triton 2>/dev/null || true

    # Install specific core packages
    pip install transformers tokenizers huggingface_hub safetensors tqdm regex pyyaml nncf openvino

    # Install Optimum components without dragging heavy dependencies
    pip install optimum --no-deps
    pip install optimum-intel --no-deps
    pip install openvino-tokenizers
    
    # Install CPU-only PyTorch to minimize size (keeps install to ~200MB vs 2GB CUDA PyTorch)
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install optimum-onnx onnx

    # Upgrade/Pin versions for stability (settled on these versions)
    info "Pinning stable library versions..."
    pip install transformers==4.57.6 optimum-intel==1.27.0

    success "Python virtual environment & dependencies set up successfully!"
}

# -----------------------------------------------------------------------------
# Step 3: Export Model using optimum-cli
# -----------------------------------------------------------------------------
export_openvino_model() {
    # Ensure virtual environment is active
    if [ -z "$VIRTUAL_ENV" ]; then
        if [ -f "$HOME/ov_env/bin/activate" ]; then
            source "$HOME/ov_env/bin/activate"
        else
            error "Virtual environment not found! Please run Step 2 first."
            exit 1
        fi
    fi

    info "Exporting Qwen/Qwen2.5-0.5B-Instruct to OpenVINO INT8 format..."
    optimum-cli export openvino \
      --model Qwen/Qwen2.5-0.5B-Instruct \
      --task text-generation-with-past \
      --weight-format int8 \
      "$HOME/qwen-0.5b-ov"

    success "Qwen2.5-0.5B INT8 successfully exported to $HOME/qwen-0.5b-ov"

    # Move to chat application directory if it exists
    if [ -d "$HOME/ai-chat-interface" ]; then
        info "Moving model folder inside 'ai-chat-interface/'..."
        rm -rf "$HOME/ai-chat-interface/qwen-0.5b-ov"
        mv "$HOME/qwen-0.5b-ov" "$HOME/ai-chat-interface/"
        success "Model moved to $HOME/ai-chat-interface/qwen-0.5b-ov"
    fi
}

# -----------------------------------------------------------------------------
# Run commands helper
# -----------------------------------------------------------------------------
run_benchmark() {
    source "$HOME/ov_env/bin/activate"
    if [ -f "$HOME/benchmark_v2.py" ]; then
        info "Running benchmark_v2.py..."
        python3 "$HOME/benchmark_v2.py"
    else
        error "benchmark_v2.py not found in $HOME"
    fi
}

run_chat_qwen() {
    source "$HOME/ov_env/bin/activate"
    if [ -f "$HOME/chat_qwen.py" ]; then
        info "Running chat_qwen.py..."
        python3 "$HOME/chat_qwen.py"
    else
        error "chat_qwen.py not found in $HOME"
    fi
}

run_web_interface() {
    source "$HOME/ov_env/bin/activate"
    if [ -d "$HOME/ai-chat-interface" ]; then
        info "Starting Web AI Chat Interface..."
        cd "$HOME/ai-chat-interface"
        python3 app.py
    else
        error "ai-chat-interface directory not found in $HOME"
    fi
}

# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------
COMMAND=${1:-""}

case "$COMMAND" in
    1)
        install_intel_gpu_drivers
        ;;
    2)
        setup_python_env
        ;;
    3)
        export_openvino_model
        ;;
    4)
        run_benchmark
        ;;
    5)
        run_chat_qwen
        ;;
    6)
        run_web_interface
        ;;
    all)
        install_intel_gpu_drivers
        setup_python_env
        export_openvino_model
        success "Full setup complete!"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        # If no arguments, show menu
        echo "====================================================="
        echo "   OpenVINO AI Inference Archive Setup Script        "
        echo "====================================================="
        echo "1) Install Intel GPU NEO Drivers & OpenCL Runtimes"
        echo "2) Setup Virtual Env & Python Dependencies (CPU PyTorch + OV)"
        echo "3) Export Qwen2.5-0.5B-Instruct Model to OpenVINO INT8"
        echo "4) Run Benchmark Script"
        echo "5) Run Terminal Chat Script"
        echo "6) Run Web AI Chat Interface"
        echo "7) Exit"
        echo "====================================================="
        read -rp "Select an option [1-7]: " opt
        case $opt in
            1) install_intel_gpu_drivers ;;
            2) setup_python_env ;;
            3) export_openvino_model ;;
            4) run_benchmark ;;
            5) run_chat_qwen ;;
            6) run_web_interface ;;
            *) info "Exiting." ; exit 0 ;;
        esac
        ;;
esac
