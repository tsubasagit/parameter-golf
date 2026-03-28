# QAT Mixed Int5/Int6 + All Proven Techniques

**val_bpb: TBD** (pending first run on 8xH100)

## Key Innovation: STE QAT with Mixed Precision

While the current SOTA (1.1428) uses post-training quantization, this submission adds **Quantization-Aware Training (QAT)** using the Straight-Through Estimator:

- **Int5 [-16,15] QAT** for MLP weights during forward pass
- **Int6 [-32,31] QAT** for attention weights during forward pass
- **FP16 passthrough** for embeddings (no QAT)

The model learns weight configurations inherently robust to quantization, reducing the gap between training and post-quantization performance.

### QAT Implementation

```python
def _ste_fake_quantize(w, clip_range):
    scale = w.detach().abs().amax(dim=-1, keepdim=True) / clip_range
    w_q = torch.clamp(torch.round(w / scale), -(clip_range+1), clip_range) * scale
    return w + (w_q - w).detach()  # STE: forward=quantized, backward=identity
```

QAT is activated after 500 warmup steps, allowing the model to first learn general representations.

## Architecture (inherits all proven techniques)

- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x expansion (hidden=1536), relu^2
- SmearGate + BigramHash(10240, dim=128)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings

## Training

- Muon optimizer: matrix_lr=0.02, WD=0.04, momentum 0.92->0.99
- AdamW for embeddings/scalars: WD=0.04
- warmdown=3000 iters, warmup=20 steps
- seq_len=2048, batch=786K tokens
- grad_clip=0.3, 3% magnitude pruning
- SWA: start_frac=0.4, every=50 steps
- **QAT: enabled at step 500, int5 MLP / int6 attention**

## Evaluation

- Sliding window eval: stride=64
- Mixed int5/int6 + zstd-22 compression

## Run Command

```bash
RUN_ID=qat_mixed \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Expected Improvement

QAT addresses the quantization gap directly. The SmearGate submission (PR #162) showed QAT with uniform int6 was effective. By combining mixed-precision QAT (int5 MLP / int6 attention) with all other SOTA techniques, we expect to reduce the quantization gap and potentially improve BPB by ~0.001-0.003.

## Hardware

8x NVIDIA H100 80GB HBM3 SXM (RunPod).
