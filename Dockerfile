# DDI-GNN Dockerfile
# Drug-Drug Interaction Prediction using Graph Neural Networks

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch Geometric dependencies
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
RUN pip install torch-geometric

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY setup.py .
COPY README.md .

# Create directories
RUN mkdir -p data/raw data/processed models outputs

# Install package
RUN pip install -e .

# Expose ports
# FastAPI
EXPOSE 8000
# Streamlit
EXPOSE 8501

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_PATH=/app/models/best_model.pt
ENV CONFIG_PATH=/app/configs/best_model.yaml

# Default command - run FastAPI server
CMD ["python", "-m", "uvicorn", "src.api.serve:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative commands:
# Training: docker run ddi-gnn python scripts/train.py --config configs/best_model.yaml
# Streamlit: docker run -p 8501:8501 ddi-gnn streamlit run src/api/streamlit_app.py
# Prediction: docker run ddi-gnn python scripts/predict.py --drug1 "SMILES1" --drug2 "SMILES2"
