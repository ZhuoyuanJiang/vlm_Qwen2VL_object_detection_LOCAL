# Triton Inference Server for Qwen2-VL Nutrition Detection
#
# Build:
#   docker build -t qwen2vl-triton .
#
# Run GPTQ INT4 model:
#   docker run --gpus all --rm -d --shm-size=4G \
#       -p 8000:8000 -p 8001:8001 -p 8002:8002 \
#       -v /path/to/gptq-weights:/models/qwen2vl-nutrition-detection-r4-joint-merged-gptq-int4:ro \
#       qwen2vl-triton gptq
#
# Run BF16 model:
#   docker run --gpus all --rm -d --shm-size=4G \
#       -p 8000:8000 -p 8001:8001 -p 8002:8002 \
#       -v /path/to/bf16-weights:/models/qwen2vl-nutrition-detection-r4-joint-merged:ro \
#       qwen2vl-triton bf16

FROM nvcr.io/nvidia/tritonserver:26.01-vllm-python-py3

# Copy Triton model configs (config.pbtxt + model.json for each model)
COPY triton_model_repository/qwen2vl_nutrition_gptq_int4 /opt/triton_configs/qwen2vl_nutrition_gptq_int4
COPY triton_model_repository/qwen2vl_nutrition_bf16 /opt/triton_configs/qwen2vl_nutrition_bf16

# Copy entrypoint script
COPY docker/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh

EXPOSE 8000 8001 8002

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["gptq"]
