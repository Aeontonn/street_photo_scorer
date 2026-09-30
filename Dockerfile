# Container for the scoring API (Google Cloud Run). Listens on port 7860.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HF_HOME=/home/user/.cache/huggingface \
    ARTIFACTS_DIR=/home/user/app/data
WORKDIR /home/user/app

COPY --chown=user requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Download CLIP at build time so the first request does not have to.
RUN python -c "from transformers import CLIPModel, CLIPProcessor; n='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(n); CLIPProcessor.from_pretrained(n)"

COPY --chown=user src ./src
COPY --chown=user data ./data

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
