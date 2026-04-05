FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# LLaVA library (for LLaVA-Med)
RUN pip install --no-cache-dir git+https://github.com/haotian-liu/LLaVA.git

# PMC-VQA repo (for MedVInT-TD)
RUN git clone https://github.com/xiaoman-zhang/PMC-VQA.git /tmp/PMC-VQA

# Copy project
COPY . .

CMD ["bash"]
