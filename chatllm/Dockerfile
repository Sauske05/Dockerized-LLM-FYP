FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set the working directory
WORKDIR /chatllm

# Copy all files into the container
COPY . /chatllm/

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
    python3-pip

RUN pip3 install --no-cache-dir -r requirements.txt

RUN pip3 install torch torchvision torchaudio

RUN pip3 uninstall triton -y

RUN pip3 install triton
# Expose the port FastAPI will run on
EXPOSE 8080

# Command to run the application
CMD ["python3", "inference.py"]
