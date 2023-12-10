# Use an NVIDIA CUDA base image with the desired CUDA version
FROM nvidia/cuda:11.5.2-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for Miniconda and other tools
RUN apt-get update && apt-get install -y wget git gcc ffmpeg libsm6 libxext6

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /miniconda && \
    rm /miniconda.sh

# Add Miniconda to PATH
ENV PATH="/miniconda/bin:${PATH}"

# Set the working directory in the container
WORKDIR /app
COPY animatediff_env.yml .
RUN git clone https://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@github.com/AlphaaDi/any_diff_core.git

# Copy the Conda environment file (if you have one)
RUN conda env create -f animatediff_env.yml

RUN chmod +777 /app/any_diff_core

# Activate the Conda environment

# SHELL ["conda", "run", "-n", "animatediff_pipe", "/bin/bash", "-c"]

# CMD  conda run --no-capture-output -n animatediff_pipe python /app/any_diff_core/core_video_editor_server.py

SHELL ["/bin/bash", "-c"]
CMD source /miniconda/etc/profile.d/conda.sh && conda activate animatediff_pipe && python /app/any_diff_core/core_video_editor_server.py