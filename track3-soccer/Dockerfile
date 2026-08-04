# Dockerfile for Track 3 — Humanoid Robot Soccer Policy Training on AMD Radeon GPU
# Base image: ROCm 6.2 + PyTorch
FROM rocm/pytorch:6.2

# Install Python dependencies for training, simulation, and deployment
RUN pip install --no-cache-dir \
    genesis-world \
    rsl_rl \
    tensordict \
    onnxruntime \
    matplotlib \
    imageio

# Clone Booster Robotics repositories (provides t1_walk.pt + URDF models)
RUN git clone https://github.com/BoosterRobotics/booster_deploy.git /workspace/booster_deploy && \
    git clone https://github.com/BoosterRobotics/booster_assets.git /workspace/booster_assets && \
    cd /workspace/booster_assets && pip install --no-cache-dir -e .

# Set working directory
WORKDIR /workspace/radeon-repo

# Copy project files
COPY . /workspace/radeon-repo

# Set environment variables for Booster assets
ENV BOOSTER_ASSETS_PATH=/workspace/booster_assets
ENV BOOSTER_DEPLOY_PATH=/workspace/booster_deploy

# Entry point: quick training test (256 envs, 100 iterations — ~5 minutes)
ENTRYPOINT ["python", "train_hierarchical.py", "--num_envs", "256", "--max_iterations", "100"]
