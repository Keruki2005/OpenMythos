"""
OpenMythos GUI - Simple graphical interface for the OpenMythos model

A tkinter-based GUI for loading, configuring, and running inference with
the OpenMythos recurrent-depth transformer model.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import torch
from typing import Optional
import traceback

try:
    from open_mythos.main import OpenMythos, MythosConfig
    from open_mythos import (
        mythos_1b,
        mythos_3b,
        mythos_10b,
    )
    OPENMYTHOS_AVAILABLE = True
except ImportError:
    OPENMYTHOS_AVAILABLE = False


class MythosGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("OpenMythos GUI - Recurrent Depth Transformer")
        self.root.geometry("1400x900")
        
        self.model: Optional[OpenMythos] = None
        self.config: Optional[MythosConfig] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main UI with tabs."""
        if not OPENMYTHOS_AVAILABLE:
            messagebox.showerror(
                "Import Error",
                "OpenMythos is not installed.\n\n"
                "Please install it with:\npip install open-mythos"
            )
            self.root.destroy()
            return
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Configuration
        self.config_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.config_tab, text="Configuration")
        self.setup_config_tab()
        
        # Tab 2: Inference
        self.inference_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.inference_tab, text="Inference")
        self.setup_inference_tab()
        
        # Tab 3: Info
        self.info_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.info_tab, text="Info")
        self.setup_info_tab()
        
    def setup_config_tab(self):
        """Setup the model configuration tab."""
        # Left panel for controls
        left_panel = ttk.LabelFrame(self.config_tab, text="Model Configuration", padding=10)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Preset buttons
        preset_frame = ttk.LabelFrame(left_panel, text="Quick Load Presets", padding=10)
        preset_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            preset_frame,
            text="Load Mythos 1B",
            command=lambda: self.load_preset("1b")
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            preset_frame,
            text="Load Mythos 3B",
            command=lambda: self.load_preset("3b")
        ).pack(fill=tk.X, pady=5)
        
        ttk.Button(
            preset_frame,
            text="Load Mythos 10B",
            command=lambda: self.load_preset("10b")
        ).pack(fill=tk.X, pady=5)
        
        # Custom configuration
        custom_frame = ttk.LabelFrame(left_panel, text="Custom Configuration", padding=10)
        custom_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Parameter inputs
        params = [
            ("Vocab Size", "vocab_size", 1000),
            ("Model Dimension (dim)", "dim", 256),
            ("Attention Heads", "n_heads", 8),
            ("Max Sequence Length", "max_seq_len", 128),
            ("Max Loop Iterations", "max_loop_iters", 4),
            ("Prelude Layers", "prelude_layers", 1),
            ("Coda Layers", "coda_layers", 1),
            ("Experts", "n_experts", 8),
            ("Expert Dimension", "expert_dim", 64),
            ("LoRA Rank", "lora_rank", 8),
            ("KV Heads (GQA)", "n_kv_heads", 2),
        ]
        
        self.config_inputs = {}
        for label, key, default in params:
            ttk.Label(custom_frame, text=label).pack(anchor=tk.W)
            entry = ttk.Entry(custom_frame, width=20)
            entry.insert(0, str(default))
            entry.pack(fill=tk.X, pady=2)
            self.config_inputs[key] = entry
        
        # Attention type
        ttk.Label(custom_frame, text="Attention Type").pack(anchor=tk.W, pady=(10, 0))
        self.attn_type = ttk.Combobox(
            custom_frame,
            values=["mla", "gqa"],
            state="readonly",
            width=18
        )
        self.attn_type.set("mla")
        self.attn_type.pack(fill=tk.X, pady=2)
        
        # Load custom button
        ttk.Button(
            custom_frame,
            text="Load Custom Model",
            command=self.load_custom_model
        ).pack(fill=tk.X, pady=10)
        
        # Status
        self.status_label = ttk.Label(custom_frame, text="Ready", foreground="blue")
        self.status_label.pack(anchor=tk.W, pady=10)
        
        # Right panel for statistics
        right_panel = ttk.LabelFrame(self.config_tab, text="Model Statistics", padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.stats_text = scrolledtext.ScrolledText(
            right_panel,
            width=50,
            height=40,
            state=tk.DISABLED
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
    def setup_inference_tab(self):
        """Setup the inference tab."""
        # Input frame
        input_frame = ttk.LabelFrame(self.inference_tab, text="Input", padding=10)
        input_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(input_frame, text="Token IDs (space-separated):").pack(anchor=tk.W)
        self.token_input = ttk.Entry(input_frame, width=80)
        self.token_input.pack(fill=tk.X, pady=5)
        self.token_input.insert(0, "0 1 2 3 4 5")
        
        # Parameters frame
        params_frame = ttk.LabelFrame(self.inference_tab, text="Parameters", padding=10)
        params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(params_frame, text="Loop Iterations:").pack(side=tk.LEFT, padx=5)
        self.n_loops = ttk.Spinbox(params_frame, from_=1, to=32, width=5)
        self.n_loops.set(4)
        self.n_loops.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(params_frame, text="Max New Tokens:").pack(side=tk.LEFT, padx=5)
        self.max_tokens = ttk.Spinbox(params_frame, from_=1, to=256, width=5)
        self.max_tokens.set(8)
        self.max_tokens.pack(side=tk.LEFT, padx=5)
        
        # Action buttons
        button_frame = ttk.Frame(self.inference_tab)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(
            button_frame,
            text="Run Forward Pass",
            command=self.run_forward_pass
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame,
            text="Generate Tokens",
            command=self.generate_tokens
        ).pack(side=tk.LEFT, padx=5)
        
        # Output frame
        output_frame = ttk.LabelFrame(self.inference_tab, text="Output", padding=10)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            width=100,
            height=25,
            state=tk.DISABLED
        )
        self.output_text.pack(fill=tk.BOTH, expand=True)
        
    def setup_info_tab(self):
        """Setup the information tab."""
        info_text = scrolledtext.ScrolledText(
            self.info_tab,
            width=100,
            height=40,
            state=tk.NORMAL
        )
        info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        info_content = """
OPENMYTHOS - RECURRENT DEPTH TRANSFORMER GUI
=============================================

ARCHITECTURE OVERVIEW:
The OpenMythos model implements a Recurrent-Depth Transformer (RDT) with three stages:

1. PRELUDE: Standard transformer layers (run once)
2. RECURRENT BLOCK: Looped transformer with input injection (run N times)
3. CODA: Standard transformer layers (run once)

The recurrent update rule at each loop step t is:
    h_{t+1} = A·h_t + B·e + Transformer(h_t, e)

Where:
  - h_t is the hidden state after loop t
  - e is the encoded input (from the Prelude)
  - A and B are learned injection parameters
  - The Transformer applies attention and MLP

KEY FEATURES:
• Systematic Generalization: Better composition of knowledge
• Depth Extrapolation: More loops = deeper reasoning
• Parameter Efficiency: Share weights across loops
• Stable Training: LTI-constrained injection parameters

ATTENTION IMPLEMENTATIONS:
• MLA (Multi-Latent Attention): Compressed KV cache, more memory efficient
• GQA (Grouped Query Attention): Fewer KV heads than Q heads

MODEL VARIANTS:
┌─────────────┬──────────────┬─────────┬──────────────┬─────────────┐
│ Model       │ Parameters   │ Experts │ Context      │ Loop Iters  │
├─────────────┼──────────────┼─────────┼──────────────┼─────────────┤
│ mythos_1b   │ 1B           │ 64      │ 4k tokens    │ 16          │
│ mythos_3b   │ 3B           │ 64      │ 4k tokens    │ 16          │
│ mythos_10b  │ 10B          │ 128     │ 8k tokens    │ 24          │
│ mythos_50b  │ 50B          │ 256     │ 8k tokens    │ 32          │
│ mythos_100b │ 100B         │ 256     │ 1M tokens    │ 32          │
│ mythos_500b │ 500B         │ 512     │ 1M tokens    │ 48          │
│ mythos_1t   │ 1T           │ 512     │ 1M tokens    │ 64          │
└─────────────┴──────────────┴─────────┴──────────────┴─────────────┘

CONFIGURATION TIPS:
• vocab_size: Size of the token vocabulary
• dim: Model dimension (hidden size) - larger = more capacity
• n_heads: Number of attention heads - typically dim/64
• max_seq_len: Maximum input sequence length
• max_loop_iters: Maximum reasoning depth at inference
• attn_type: Choose between 'mla' (memory efficient) or 'gqa' (standard)
• n_experts: Number of mixture-of-experts units
• expert_dim: Dimension of each expert's hidden layer

INFERENCE TIPS:
• More loops = deeper reasoning but slower
• Fewer loops = faster but less capable
• Memory usage scales with: batch_size × seq_len × dim
• For limited VRAM: use smaller models or reduce seq_len

PERFORMANCE MONITORING:
The GUI displays:
• Total parameters in millions/billions
• Architecture breakdown
• Configuration summary
• Spectral radius (should be < 1 for stability)

REFERENCES:
• GitHub: https://github.com/kyegomez/OpenMythos
• Paper: "Loop, Think, & Generalize - Implicit Reasoning in RDT"
• License: MIT

For more information, visit the project README.
"""
        
        info_text.insert(tk.END, info_content)
        info_text.config(state=tk.DISABLED)
        
    def load_preset(self, size):
        """Load a preset model configuration."""
        self.status_label.config(text="Loading...", foreground="orange")
        self.root.update()
        
        def load():
            try:
                if size == "1b":
                    self.config = mythos_1b()
                elif size == "3b":
                    self.config = mythos_3b()
                elif size == "10b":
                    self.config = mythos_10b()
                
                self.model = OpenMythos(self.config).to(self.device)
                self.update_stats()
                
                self.status_label.config(
                    text=f"✓ Mythos {size.upper()} loaded successfully!",
                    foreground="green"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
                self.status_label.config(text="Failed to load", foreground="red")
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
        
    def load_custom_model(self):
        """Load a custom model with user-specified parameters."""
        self.status_label.config(text="Loading custom model...", foreground="orange")
        self.root.update()
        
        def load():
            try:
                params = {}
                for key, entry in self.config_inputs.items():
                    try:
                        params[key] = int(entry.get())
                    except ValueError:
                        raise ValueError(f"Invalid value for {key}")
                
                params["attn_type"] = self.attn_type.get()
                
                # Handle MLA-specific parameters
                if params["attn_type"] == "mla":
                    params["n_kv_heads"] = params.get("n_kv_heads", 8)
                    params["kv_lora_rank"] = 32
                    params["q_lora_rank"] = 64
                    params["qk_rope_head_dim"] = 16
                    params["qk_nope_head_dim"] = 16
                    params["v_head_dim"] = 16
                else:
                    # GQA
                    params["n_kv_heads"] = int(self.config_inputs.get("n_kv_heads", ttk.Entry()).get() or 2)
                
                self.config = MythosConfig(**params)
                self.model = OpenMythos(self.config).to(self.device)
                self.update_stats()
                
                self.status_label.config(
                    text="✓ Custom model loaded!",
                    foreground="green"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{str(e)}\n\n{traceback.format_exc()}")
                self.status_label.config(text="Failed to load", foreground="red")
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
        
    def update_stats(self):
        """Update the statistics display."""
        if self.model is None or self.config is None:
            return
        
        total_params = sum(p.numel() for p in self.model.parameters())
        total_params_m = total_params / 1e6
        total_params_b = total_params / 1e9
        
        stats = f"""
MODEL STATISTICS
================

Configuration:
  Attention Type: {self.config.attn_type}
  Vocab Size: {self.config.vocab_size:,}
  Model Dimension: {self.config.dim}
  Attention Heads: {self.config.n_heads}
  Max Sequence Length: {self.config.max_seq_len:,}
  Max Loop Iterations: {self.config.max_loop_iters}

Architecture:
  Prelude Layers: {self.config.prelude_layers}
  Coda Layers: {self.config.coda_layers}
  Experts: {self.config.n_experts}
  Expert Dimension: {self.config.expert_dim}
  Experts per Token: {self.config.n_experts_per_tok}
  LoRA Rank: {self.config.lora_rank}

Parameters:
  Total: {total_params:,}
  Total (M): {total_params_m:,.2f}M
  Total (B): {total_params_b:,.4f}B

Hardware:
  Device: {self.device}
  CUDA Available: {torch.cuda.is_available()}
  {f'GPU Memory: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f}GB' if torch.cuda.is_available() else ''}

"""
        
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, stats)
        self.stats_text.config(state=tk.DISABLED)
        
    def run_forward_pass(self):
        """Run a forward pass on the model."""
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
        
        try:
            token_ids = list(map(int, self.token_input.get().split()))
            n_loops = int(self.n_loops.get())
            
            ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                logits = self.model(ids, n_loops=n_loops)
            
            output = f"""
FORWARD PASS RESULTS
====================

Input Tokens: {token_ids}
Number of Loops: {n_loops}

Output Shape: {tuple(logits.shape)}
  Batch: 1
  Sequence Length: {logits.shape[1]}
  Vocab Size: {logits.shape[2]}

Top Predictions (position 0):
"""
            top_k = 5
            top_logits, top_indices = torch.topk(logits[0, 0, :], top_k)
            for i, (logit, idx) in enumerate(zip(top_logits, top_indices)):
                output += f"\n  {i+1}. Token {idx.item()}: {logit.item():.4f}"
            
            # Compute spectral radius if available
            if hasattr(self.model, 'recurrent') and hasattr(self.model.recurrent, 'injection'):
                try:
                    A = self.model.recurrent.injection.get_A()
                    rho = torch.linalg.eigvals(A).abs().max().item()
                    output += f"\n\nSpectral Radius ρ(A): {rho:.6f}"
                    output += f"\nStability: {'✓ STABLE (ρ < 1)' if rho < 1 else '✗ UNSTABLE (ρ ≥ 1)'}"
                except:
                    pass
            
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, output)
            self.output_text.config(state=tk.DISABLED)
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error during inference:\n{str(e)}\n\n{traceback.format_exc()}")
    
    def generate_tokens(self):
        """Generate new tokens."""
        if self.model is None:
            messagebox.showwarning("Warning", "Please load a model first!")
            return
        
        try:
            token_ids = list(map(int, self.token_input.get().split()))
            n_loops = int(self.n_loops.get())
            max_tokens = int(self.max_tokens.get())
            
            ids = torch.tensor([token_ids], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    ids,
                    max_new_tokens=max_tokens,
                    n_loops=n_loops
                )
            
            output = f"""
TOKEN GENERATION RESULTS
========================

Input Tokens: {token_ids}
Number of Loops: {n_loops}
Max New Tokens: {max_tokens}

Generated Token IDs: {output_ids[0].tolist()}
Generated Length: {len(output_ids[0])}
New Tokens: {len(output_ids[0]) - len(token_ids)}
"""
            
            self.output_text.config(state=tk.NORMAL)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, output)
            self.output_text.config(state=tk.DISABLED)
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error during generation:\n{str(e)}\n\n{traceback.format_exc()}")


def main():
    root = tk.Tk()
    gui = MythosGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
