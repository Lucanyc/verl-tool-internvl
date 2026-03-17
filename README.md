# 📖verl-tool-internvl

verl-tool environment adapted for InternVL online GRPO with tool RL training.

## ⚙️ Installation

### Step 1: Create Conda Environment

```bash
conda create -n verl-tool-internvl python=3.10 -y
conda activate verl-tool-internvl
```

### Step 2: Install PyTorch

```bash
cd verl-tool-internvl/verl-tool
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Verify Verl Version

```bash
cd verl-tool-internvl/verl-tool
git log --oneline -1
```

Expected output:
```
8e0b9bd9 (HEAD) [recipe] chore: Remove the duplicate definition of `class Role` (#2503)
```

If not on this version, switch to it:
```bash
git checkout 8e0b9bd9
```

### Step 4: Install Verl Dependencies

```bash
cd verl
pip install -r requirements.txt
```

### Step 5: Install Verl and Verl-tool

```bash
# Install verl (in verl directory)
pip install -e .

# Go back and install verl-tool
cd ..
pip install -e .
```

### Step 6: Install Additional Packages

```bash
pip install flash-attn --no-build-isolation
pip install openai vllm==0.8.3
```

### Step 7: Verify Installation

```bash
python -c "
import flash_attn
print('✅ flash-attn installed:', flash_attn.__version__)
"
```

```bash
python -c "
from verl_tool.workers.rollout.async_server import VerlToolAsyncLLMServerManager
from verl.workers.rollout.async_server import AsyncLLMServerManager
print('✅ Installation successful!')
print('✅ verl-tool-internvl environment is ready!')
"
```


# 🛠️ Tool Environment Setup
All tool environment configs are located in the `environment_setting/` directory. To set up a tool environment, run:
```bash
conda env create -f environment_setting/.yaml
```

For example, to set up ChartMoE:
```bash
conda env create -f environment_setting/chartmoe_env.yaml
conda activate chartmoe
```


### Launch Tool Servers

After setting up all tool environments, launch all tool servers at once:
```bash
bash launch_tool_servers.sh
```

Or launch a single tool server manually:
```bash
conda activate 
cd verl-tool-internvl/verl-tool
VERL_APPLY_QWEN25VL_PATCH=0 CUDA_VISIBLE_DEVICES= \
    python -m verl_tool.servers.serve --host localhost --port  --tool_type 
```

| Tool | Conda Env | Port | GPU |
|------|-----------|------|-----|
| ChartMoE | chartmoe-env | 6658 | 8 |
| EasyOCR | easyocr_env | 6758 | 7 |
| MultiMath | multimath | 6582 | 7 |
| GLLaVA | gllava | 8690 | 7 |
| GroundingDINO | groundingdino | 6569 | 8 |
| DiagramFormalizer | diagramformalizer | 7866 | 8 |
