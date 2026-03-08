# 📖verl-tool-internvl

verl-tool environment adapted for InternVL training.

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

### Step 3: Verify verl Version

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

### Step 4: Install verl Dependencies

```bash
cd verl
pip install -r requirements.txt
```

### Step 5: Install verl and verl-tool

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


# 🛠️ Tool environment setup
