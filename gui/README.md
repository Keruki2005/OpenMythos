# OpenMythos GUI

A simple graphical user interface for interacting with the OpenMythos model.

## Features

### 📋 Model Configuration Tab
- **Quick Load Presets**: Load pre-configured model variants (1B, 3B, 10B)
- **Custom Configuration**: Fine-tune model parameters:
  - Vocabulary size
  - Model dimension
  - Number of attention heads
  - Sequence length
  - Loop iterations
  - Attention type (MLA or GQA)

### 🔍 Model Statistics
Real-time display of:
- Configuration parameters
- Architecture details
- Parameter counts (in millions and billions)
- Device information

### 🧠 Inference Tab
- **Token Input**: Enter space-separated token IDs
- **Forward Pass**: Run a forward pass and view output logits
- **Token Generation**: Generate new tokens with configurable parameters
- **Loop Control**: Adjust inference-time reasoning depth

### ℹ️ Info Tab
- Documentation about the OpenMythos architecture
- Model variants overview
- Configuration tips
- Technical references

## Installation

1. First, install OpenMythos:
```bash
pip install open-mythos
```

2. The GUI uses Python's built-in `tkinter`, which is usually pre-installed. If not:
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# macOS (usually comes with Python)
# Windows (usually comes with Python)
```

## Usage

Run the GUI:
```bash
python gui/mythos_gui.py
```

### Quick Start
1. **Load a Model**: Click one of the preset buttons (Mythos 1B, 3B, or 10B)
2. **View Stats**: Check the model statistics on the right
3. **Run Inference**: Go to the Inference tab
4. **Input Tokens**: Enter token IDs (e.g., `0 1 2 3 4 5`)
5. **Execute**: Click "Run Forward Pass" or "Generate Tokens"
6. **View Results**: Check the output display

### Custom Model Configuration
1. Adjust parameters in the left panel
2. Select your preferred attention type (MLA or GQA)
3. Click "Load Custom Model"
4. Wait for the model to load (watch the status label)

## Architecture Overview

The OpenMythos model consists of three stages:

```
Input Tokens
    ↓
[Prelude] — standard transformer layers
    ↓
[Recurrent Block] — looped transformer with input injection (runs N times)
    ↓
[Coda] — standard transformer layers
    ↓
Output Logits
```

Each loop iteration performs:
- Multi-head attention (MLA or GQA)
- Mixture of Experts FFN
- Residual connections with injection
- Layer normalization

## Model Variants

| Model | Parameters | Experts | Context | Use Case |
|-------|-----------|---------|---------|----------|
| 1B | 1 billion | 64 | 4k | Quick testing, development |
| 3B | 3 billion | 64 | 4k | Experimentation |
| 10B | 10 billion | 128 | 8k | Serious inference |
| 50B+ | Larger | More | Up to 1M | Production, research |

## Performance Tips

1. **Memory**: Start with smaller models (1B) if VRAM is limited
2. **Speed**: Fewer loops = faster inference but less reasoning depth
3. **Quality**: More loops = better reasoning but slower
4. **Attention**: MLA is more memory-efficient than GQA for longer sequences

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `vocab_size` | Vocabulary size | 1000 |
| `dim` | Model dimension (hidden size) | 256 |
| `n_heads` | Number of attention heads | 8 |
| `max_seq_len` | Maximum sequence length | 128 |
| `max_loop_iters` | Maximum loop iterations | 4 |
| `attn_type` | Attention type (`mla` or `gqa`) | `mla` |
| `prelude_layers` | Transformer layers before recurrence | 1 |
| `coda_layers` | Transformer layers after recurrence | 1 |
| `n_experts` | Number of MoE experts | 8 |
| `expert_dim` | Expert hidden dimension | 64 |

## Troubleshooting

### "Could not import OpenMythos"
```bash
pip install open-mythos
# For GPU support
pip install open-mythos[flash]
```

### "CUDA out of memory"
- Load a smaller model (start with 1B)
- Reduce `dim` and `n_heads`
- Reduce `max_seq_len`

### Model loading hangs
- Check that PyTorch is properly installed
- Ensure you have sufficient disk space
- Try loading a smaller preset

### Slow inference
- Reduce `max_loop_iters`
- Use a smaller model
- Check that GPU acceleration is available

## Advanced Usage

### Custom Model from Python
```python
from open_mythos.main import OpenMythos, MythosConfig

config = MythosConfig(
    vocab_size=2048,
    dim=512,
    n_heads=16,
    max_seq_len=256,
    max_loop_iters=8,
    attn_type="mla"
)

model = OpenMythos(config)
```

### Batch Inference
```python
import torch

ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
logits = model(ids, n_loops=4)  # Shape: (batch_size, seq_len, vocab_size)
```

## References

- **GitHub**: https://github.com/kyegomez/OpenMythos
- **Paper**: Loop, Think, & Generalize - Implicit Reasoning in Recurrent Depth Transformers
- **License**: MIT

## License

This GUI is part of the OpenMythos project, licensed under the MIT License.
