#!/bin/bash
# Launch all tool servers 
# Each tool requires its own conda environment and GPU

VERL_TOOL_DIR="Online-GRPO-new/verl-tool"

echo "🚀 Launching tool servers..."

# EasyOCR (conda: easyocr_env, GPU: 6, port: 6758)
conda run -n easyocr_env --no-banner \
    bash -c "cd ${VERL_TOOL_DIR} && VERL_APPLY_QWEN25VL_PATCH=0 CUDA_VISIBLE_DEVICES=8 \
    python -m verl_tool.servers.serve --host localhost --port 6758 --tool_type easyocr" &
echo "✅ EasyOCR server started (port 6758)"

# GLLaVA (conda: gllava, GPU: 7, port: 8690)
conda run -n gllava --no-banner \
    bash -c "cd ${VERL_TOOL_DIR} && VERL_APPLY_QWEN25VL_PATCH=0 CUDA_VISIBLE_DEVICES=9 \
    python -m verl_tool.servers.serve --host localhost --port 8690 --tool_type gllava" &
echo "✅ GLLaVA server started (port 8690)"

# GroundingDINO (conda: groundingdino, GPU: 7, port: 6569)
conda run -n groundingdino --no-banner \
    bash -c "cd ${VERL_TOOL_DIR} && VERL_APPLY_QWEN25VL_PATCH=0 CUDA_VISIBLE_DEVICES=9 \
    python -m verl_tool.servers.serve --host localhost --port 6569 --tool_type groundingdino" &
echo "✅ GroundingDINO server started (port 6569)"

# DiagramFormalizer (conda: diagramformalizer, GPU: 7, port: 7866)
conda run -n diagramformalizer --no-banner \
    bash -c "cd ${VERL_TOOL_DIR} && VERL_APPLY_QWEN25VL_PATCH=0 CUDA_VISIBLE_DEVICES=9 \
    python -m verl_tool.servers.serve --host localhost --port 7866 --tool_type diagramformalizer" &
echo "✅ DiagramFormalizer server started (port 7866)"

# MultiMath (conda: multimath, GPU: 6, port: 6582)
conda run -n multimath --no-banner \
    bash -c "cd ${VERL_TOOL_DIR} && VERL_APPLY_QWEN25VL_PATCH=0 CUDA_VISIBLE_DEVICES=8 \
    python -m verl_tool.servers.serve --host localhost --port 6582 --tool_type multimath" &
echo "✅ MultiMath server started (port 6582)"

# ChartMoE (conda: chartmoe-env, GPU: 0, port: 6658)
conda run -n chartmoe-env --no-banner \
    bash -c "cd ${VERL_TOOL_DIR} && CUDA_VISIBLE_DEVICES=8 \
    python -m verl_tool.servers.serve --host localhost --port 6658 --tool_type chartmoe" &
echo "✅ ChartMoE server started (port 6658)"

echo ""
echo "All tool servers launched. Use 'jobs' to check status or 'kill %N' to stop a specific server."
wait
