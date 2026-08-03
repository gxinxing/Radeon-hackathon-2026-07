# Track 3 (Physical AI) submission image — full env for reproducibility on AMD Radeon.
# Adjust the base image tag to the ROCm/PyTorch image available on Radeon Cloud.
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10

WORKDIR /app

# Install Genesis + RL deps (torch comes from the base ROCm image).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source.
COPY . .

# JupyterLab for Radeon Cloud "Open Notebook" access.
RUN pip install --no-cache-dir jupyter

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
