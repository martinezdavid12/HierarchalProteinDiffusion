#!/usr/bin/env python3
"""
Hierarchical Protein Diffusion Model - Fixed Implementation

Masked Diffusion Language Model (MDLM) for protein sequences with:
- Correct positive hazard weighting w(t) = -alpha'(t)/(1-alpha(t))
- Proper unmask fraction in ancestral sampling
- tqdm-based rich training/debug metrics
- Validation split + masked-token accuracy @ fixed mask rate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import EsmTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
import time
import math
import random
from typing import Dict, List, Tuple, Optional

# -------------------------
# Small helpers
# -------------------------
def grad_global_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += (p.grad.data.norm(2).item()) ** 2
    return total ** 0.5

def cuda_mem_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    alloc = torch.cuda.memory_allocated() / (1024 ** 2)
    reserv = torch.cuda.memory_reserved() / (1024 ** 2)
    return alloc, reserv


class ProteinDiffusionModel:
    """Main class for the protein diffusion model with fixed implementation."""

    def __init__(self,
                 d_model: int = 512,
                 n_layers: int = 12,
                 n_heads: int = 8,
                 max_len: int = 2048,
                 dropout: float = 0.1,
                 learning_rate: float = 1e-4,  # Reduced for stability
                 weight_decay: float = 0.01):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.MASK_ID = self.tokenizer.mask_token_id
        self.PAD_ID  = self.tokenizer.pad_token_id
        self.BOS_ID  = self.tokenizer.bos_token_id
        self.EOS_ID  = self.tokenizer.eos_token_id

        # Canonical AA allowlist (sampling-time clamp only)
        self.CANON = list("ACDEFGHIKLMNPQRSTVWY")
        self.CANON_IDS = self.tokenizer.convert_tokens_to_ids(self.CANON)

        # Model
        self.model = MDLMTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            max_len=max_len,
            dropout=dropout
        ).to(self.device)
        # Tie weights for stability
        self.model.output_head.weight = self.model.token_emb.weight

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # State
        self.training_history = []
        self.train_loader = None
        self.val_loader = None

    # -------------------------
    # Data
    # -------------------------
    def load_data(self,
                  dataset_name: str = "dahvid12/uniprot50-sequences-subsample100",
                  batch_size: int = 32,
                  max_len: int = 1024,
                  num_workers: int = 2,
                  val_frac: float = 0.1,
                  seed: int = 42):
        """Load dataset and build train/val loaders."""
        print(f"Loading dataset: {dataset_name}")
        ds = load_dataset(dataset_name, split="train")
        ds = ds.filter(lambda ex: ex.get("sequence") is not None and len(ex["sequence"]) > 0)
        df = ds.to_pandas()
        seq_series = df["sequence"]

        # Simple random split
        N = len(seq_series)
        idx = torch.randperm(N, generator=torch.Generator().manual_seed(seed)).tolist()
        n_val = int(round(val_frac * N))
        val_idx = set(idx[:n_val])
        train_seqs = seq_series[[i for i in range(N) if i not in val_idx]]
        val_seqs   = seq_series[[i for i in range(N) if i in val_idx]]

        train_ds = ProteinSeqDataset(train_seqs)
        val_ds   = ProteinSeqDataset(val_seqs)

        self.train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=lambda b: collate_batch(b, max_len=max_len, tokenizer=self.tokenizer)
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=lambda b: collate_batch(b, max_len=max_len, tokenizer=self.tokenizer)
        )

        print(f"Dataset loaded: train={len(train_ds)} | val={len(val_ds)}")

    # -------------------------
    # Schedules (MDLM)
    # -------------------------
    def alpha_cosine(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(0.5 * math.pi * t.clamp(0, 1)).pow(2)

    def alpha_prime_cosine(self, t: torch.Tensor) -> torch.Tensor:
        a = 0.5 * math.pi
        return -a * torch.sin(2 * a * t)

    def weight_w(self, t: torch.Tensor) -> torch.Tensor:
        # Positive hazard
        a  = self.alpha_cosine(t)
        ap = self.alpha_prime_cosine(t)
        return (-ap) / (1.0 - a).clamp_min(1e-6)

    def t_for_mask_frac(self, p_mask: float) -> float:
        """
        Pick t so that (1 - alpha(t)) ~= p_mask for cosine alpha.
        alpha(t) = cos^2(pi/2 * t) => t = (2/pi) * arccos(sqrt(alpha))
        """
        a = max(1e-6, min(1 - 1e-6, 1.0 - p_mask))
        return (2.0 / math.pi) * math.acos(math.sqrt(a))

    # -------------------------
    # Training
    # -------------------------
    def train(self, epochs: int = 5, clip_grad_norm: float = 1.0,
              val_mask_frac: float = 0.35, val_batches: int = 20):
        """Train with tqdm debug + per-epoch validation on masked-token accuracy."""
        assert self.train_loader is not None and self.val_loader is not None, "Call load_data() first."
        print(f"Starting training for {epochs} epochs...")
        self.model.train()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            epoch_steps = 0
            epoch_tokens = 0
            epoch_masked = 0
            t0 = time.time()

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
            for batch in pbar:
                core_ids  = batch["core_ids"].to(self.device, non_blocking=True)
                core_attn = batch["core_attn"].to(self.device, non_blocking=True)

                # Avoid extreme hazards during training
                t_scalar = random.uniform(0.05, 0.95)

                loss, metr = self.mdlm_loss_step(core_ids, core_attn, t_scalar, return_metrics=True)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                self.optimizer.step()

                # Stats
                epoch_steps  += 1
                epoch_loss   += float(loss.detach().item())
                epoch_tokens += metr["valid_tokens"]
                epoch_masked += metr["masked_tokens"]

                # Debug
                gn = grad_global_norm(self.model)
                alloc, reserv = cuda_mem_mb()
                elapsed = max(1e-6, time.time() - t0)
                toks_per_s = epoch_tokens / elapsed
                pbar.set_postfix({
                    "loss": f"{(epoch_loss/epoch_steps):.4f}",
                    "mask%": f"{(100.0*metr['masked_frac']):.1f}",
                    "len": f"{metr['mean_len']:.0f}",
                    "α(t)": f"{metr['alpha_t']:.3f}",
                    "w(t)": f"{metr['w_t']:.3f}",
                    "g||": f"{gn:.2f}",
                    "tok/s": f"{toks_per_s:.0f}",
                    "CUDA(MB)": f"{alloc:.0f}/{reserv:.0f}" if torch.cuda.is_available() else "CPU",
                })

            # Epoch summary
            epoch_time = time.time() - t0
            avg_loss = epoch_loss / max(1, epoch_steps)
            masked_frac_epoch = epoch_masked / max(1, epoch_tokens)

            # ----- Validation: masked-token accuracy @ fixed mask rate -----
            val_acc = self.masked_accuracy(self.val_loader, mask_frac=val_mask_frac, max_batches=val_batches)

            print(f"Epoch {epoch}/{epochs}: "
                  f"loss={avg_loss:.4f} | masked={masked_frac_epoch*100:.1f}% | "
                  f"val@{int(val_mask_frac*100)}%={val_acc*100:.1f}% | time={epoch_time:.1f}s")

            self.training_history.append({
                'epoch': epoch,
                'loss': avg_loss,
                'masked_frac': masked_frac_epoch,
                'val_acc_maskfrac': val_acc,
                'time': epoch_time
            })

    def mdlm_loss_step(self,
                       core_ids: torch.Tensor,
                       core_attn: torch.Tensor,
                       t_scalar: float,
                       return_metrics: bool = False) -> Tuple[torch.Tensor, Optional[Dict]]:
        """MDLM loss with correct hazard weighting and metrics."""
        z_t, masked = self.forward_mask(core_ids, core_attn, t_scalar)
        logits = self.model(z_t, t_scalar)  # [B,L,V]

        valid_tokens  = int(core_attn.sum().item())
        masked_tokens = int(masked.sum().item())
        masked_frac   = masked_tokens / max(1, valid_tokens)

        if masked_tokens == 0:
            loss = logits.new_tensor(0.0, requires_grad=True)
        else:
            targets  = core_ids[masked]
            logits_m = logits[masked]
            ce = F.cross_entropy(logits_m, targets, reduction="mean")
            w  = self.weight_w(torch.tensor([t_scalar], device=logits.device)).item()
            if w < 0:
                raise RuntimeError(f"w(t) must be >= 0, got {w}")
            loss = ce * float(w)

        if not return_metrics:
            return loss, None

        a_t  = self.alpha_cosine(torch.tensor([t_scalar], device=core_ids.device)).item()
        ap_t = self.alpha_prime_cosine(torch.tensor([t_scalar], device=core_ids.device)).item()
        metrics = {
            "t": t_scalar,
            "alpha_t": a_t,
            "alpha_prime_t": ap_t,
            "w_t": -ap_t / max(1e-6, 1 - a_t),
            "valid_tokens": valid_tokens,
            "masked_tokens": masked_tokens,
            "masked_frac": float(masked_frac),
            "mean_len": float(core_attn.sum(dim=1).float().mean().item()),
        }
        return loss, metrics

    def forward_mask(self, core_ids: torch.Tensor, core_attn: torch.Tensor, t_scalar: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Absorbing forward masking; only mask valid tokens (ignore PAD)."""
        B, L = core_ids.shape
        valid = core_attn.bool()
        a = self.alpha_cosine(torch.tensor([t_scalar], device=core_ids.device))
        p_mask = (1.0 - a).item()
        p_mask = max(0.1, min(0.9, p_mask))  # soft curriculum
        U = torch.rand(B, L, device=core_ids.device)
        masked = (U < p_mask) & valid

        # ensure at least one masked per sequence with any valid tokens
        for b in range(B):
            if masked[b].sum() == 0 and valid[b].sum() > 0:
                idx = torch.where(valid[b])[0][0]
                masked[b, idx] = True

        z_t = core_ids.clone()
        z_t[masked] = self.MASK_ID
        return z_t, masked

    # ---- Fixed-rate forward mask for validation ----
    def forward_mask_fixed(self, core_ids: torch.Tensor, core_attn: torch.Tensor, p_mask: float):
        """Mask with a fixed probability p_mask (ignores alpha schedule)."""
        B, L = core_ids.shape
        valid = core_attn.bool()
        p = max(1e-3, min(0.999, p_mask))
        U = torch.rand(B, L, device=core_ids.device)
        masked = (U < p) & valid
        for b in range(B):
            if masked[b].sum() == 0 and valid[b].sum() > 0:
                idx = torch.where(valid[b])[0][0]
                masked[b, idx] = True
        z_t = core_ids.clone()
        z_t[masked] = self.MASK_ID
        return z_t, masked

    # ---- Validation metric: masked-token accuracy at fixed mask rate ----
    @torch.no_grad()
    def masked_accuracy(self, data_loader: DataLoader, mask_frac: float = 0.35, max_batches: int = 20) -> float:
        """Argmax accuracy on masked tokens with a fixed mask fraction (fast, no sampling loop)."""
        self.model.eval()
        t_eval = self.t_for_mask_frac(mask_frac)  # choose t consistent with mask rate
        correct, total = 0, 0
        for i, batch in enumerate(data_loader):
            if i >= max_batches:
                break
            core_ids  = batch["core_ids"].to(self.device)
            core_attn = batch["core_attn"].to(self.device)
            z_t, masked = self.forward_mask_fixed(core_ids, core_attn, mask_frac)
            logits = self.model(z_t, t_eval)              # [B,L,V]
            preds  = logits.argmax(dim=-1)                # [B,L]
            corr = ((preds == core_ids) & masked).sum().item()
            tot  = masked.sum().item()
            correct += corr
            total   += tot
        self.model.train()
        return 0.0 if total == 0 else correct / total

    # -------------------------
    # Sampling (ancestral SUBS)
    # -------------------------
    def alpha_cosine_val(self, t: float) -> float:
        return math.cos(0.5 * math.pi * max(0, min(1, t))) ** 2

    def sample(self,
               sequence: str,
               mask_positions: Optional[List[int]] = None,
               mask_fraction: float = 0.3,
               T: int = 64,
               temperature: float = 1.0,
               top_p: float = 0.9) -> str:
        """Sample/generate a sequence with the trained model."""
        self.model.eval()
        tokens = self.tokenizer(
            sequence, add_special_tokens=True, padding=True, truncation=True,
            max_length=1024, return_tensors="pt"
        )
        core_ids  = tokens["input_ids"][0, 1:-1]
        core_attn = tokens["attention_mask"][0, 1:-1]

        if mask_positions is None:
            valid_positions = torch.where(core_attn.bool())[0]
            num_mask = max(1, int(len(valid_positions) * mask_fraction))
            sel = valid_positions[torch.randperm(len(valid_positions))[:num_mask]].tolist()
            mask_positions = sel

        edit_mask = torch.zeros_like(core_ids, dtype=torch.bool)
        for pos in mask_positions:
            if 0 <= pos < len(edit_mask):
                edit_mask[pos] = True

        with torch.no_grad():
            reconstructed, _ = self.mdlm_infill_sample(
                core_ids.to(self.device), edit_mask.to(self.device),
                T=T, temperature=temperature, top_p=top_p
            )
        return self.core_ids_to_str(reconstructed)

    def mdlm_infill_sample(self,
                           orig_core: torch.Tensor,
                           edit_mask: torch.Tensor,
                           T: int = 64,
                           temperature: float = 1.0,
                           top_p: float = 0.9) -> Tuple[torch.Tensor, Dict]:
        """Ancestral sampling with SUBS; keeps unmasked tokens fixed; unmask fraction > 0."""
        x = orig_core.clone().to(self.device)
        x[edit_mask] = self.MASK_ID

        for step in range(T, 0, -1):
            t = step / T
            s = (step - 1) / T

            a_t = self.alpha_cosine_val(t)
            a_s = self.alpha_cosine_val(s)
            p_unmask = max(0.0, min(1.0, (a_s - a_t) / max(1e-6, 1.0 - a_t)))

            logits = self.model(x.unsqueeze(0), t).squeeze(0)  # [L,V]
            self.clamp_to_canonical(logits)

            probs = F.softmax(logits / max(1e-6, temperature), dim=-1)
            probs = self.nucleus_sampling(probs, top_p)

            masked_now = (x == self.MASK_ID) & edit_mask
            if masked_now.any():
                sampled = torch.multinomial(probs[masked_now], 1).squeeze(-1)
                max_conf = probs.max(-1).values
                k = max(1, int(p_unmask * masked_now.sum().item()))
                if k > 0:
                    cand_idx = torch.where(masked_now)[0]
                    if len(cand_idx) >= k:
                        topk = torch.topk(max_conf[cand_idx], k=k).indices
                        sel_idx = cand_idx[topk]
                        x[sel_idx] = sampled[topk]

        if (x == self.MASK_ID).any():
            final_logits = self.model(x.unsqueeze(0), 0.0).squeeze(0)
            self.clamp_to_canonical(final_logits)
            x = torch.where(x == self.MASK_ID, final_logits.argmax(-1), x)

        return x, {"steps": T, "final_mask_left": int((x == self.MASK_ID).sum().item())}

    # -------------------------
    # Sampling utilities
    # -------------------------
    def clamp_to_canonical(self, logits: torch.Tensor):
        """Clamp logits to canonical amino acids (for generation)."""
        V = logits.size(-1)
        allowed = torch.tensor(self.CANON_IDS, device=logits.device)
        ban = torch.ones(V, dtype=torch.bool, device=logits.device)
        ban[allowed] = False
        logits[..., ban] = float("-inf")

    def nucleus_sampling(self, probs: torch.Tensor, top_p: float) -> torch.Tensor:
        """Top-p (nucleus) sampling filter; numerically safe."""
        sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
        cdf = sorted_probs.cumsum(dim=-1)
        to_remove = cdf > top_p
        to_remove[..., 1:] = to_remove[..., :-1].clone()
        to_remove[..., 0] = 0
        sorted_probs = torch.where(to_remove, torch.zeros_like(sorted_probs), sorted_probs)
        denom = sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        sorted_probs = sorted_probs / denom
        inv = torch.argsort(sorted_idx, dim=-1)
        probs = torch.gather(sorted_probs, dim=-1, index=inv)
        return probs

    def core_ids_to_str(self, ids: torch.Tensor) -> str:
        tokens = self.tokenizer.convert_ids_to_tokens(ids.tolist())
        tokens = [t if t in self.CANON else "A" for t in tokens]
        return "".join(tokens)

    # -------------------------
    # I/O
    # -------------------------
    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        print(f"Model loaded from {path}")


class TimeEmbed(nn.Module):
    """Time embedding module with improved stability."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.lin = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, t_scalar: float, device: torch.device) -> torch.Tensor:
        t = torch.tensor([t_scalar], device=device).float()
        half = self.d_model // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=device) / half)
        angles = t[:, None] * freqs[None, :]
        time_emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # [1,d_model]
        return self.lin(time_emb)  # [1,d_model]


class MDLMTransformer(nn.Module):
    """Improved transformer model for protein diffusion."""

    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 12,
                 n_heads: int = 8, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.time_emb  = TimeEmbed(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, 4 * d_model, dropout=dropout,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, input_ids: torch.Tensor, t_scalar: float) -> torch.Tensor:
        B, L = input_ids.shape
        tok  = self.token_emb(input_ids)                      # [B,L,d]
        pos  = self.pos_emb.weight[:L][None].expand(B, L, -1) # [B,L,d]
        temb = self.time_emb(t_scalar, input_ids.device)      # [1,d]
        temb = temb.unsqueeze(1).expand(B, L, -1)             # [B,L,d]
        h = tok + pos + temb
        h = self.transformer(h)
        logits = self.output_head(h)                          # [B,L,V]
        return logits


class ProteinSeqDataset(Dataset):
    """Dataset class for protein sequences."""

    def __init__(self, sequences):
        self.sequences = [str(seq).strip().upper() for seq in sequences.tolist()]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def collate_batch(batch_seqs: List[str], max_len: int, tokenizer) -> Dict[str, torch.Tensor]:
    """Collate function for batching sequences (dynamic padding)."""
    tokens = tokenizer(
        batch_seqs,
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    # Remove BOS/EOS → core
    core_ids  = input_ids[:, 1:-1].contiguous()
    core_attn = attention_mask[:, 1:-1].contiguous()

    # Fill PAD in padded positions for robustness
    PAD_ID = tokenizer.pad_token_id
    core_ids = torch.where(core_attn.bool(), core_ids, torch.full_like(core_ids, PAD_ID))
    return {"core_ids": core_ids, "core_attn": core_attn}


def main():
    print("Initializing Protein Diffusion Model...")

    model = ProteinDiffusionModel(
        d_model=512,
        n_layers=16,
        n_heads=8,
        learning_rate=1e-4
    )

    # Build train/val loaders (10% val)
    model.load_data(batch_size=32, max_len=512, val_frac=0.1)

    # Train + per-epoch validation (masked-token accuracy @ 35% mask)
    model.train(epochs=10, val_mask_frac=0.35, val_batches=20)

    # Save
    model.save_model("protein_diffusion_model.pth")

    # Example sampling
    print("\nExample sampling:")
    test_sequence = "MKLLILTCLVAVALARPKHPIKHQGLPQEVLNENLLRFFVAPFPEVFGKEKVNEL"
    result = model.sample(test_sequence, mask_fraction=0.3)
    print(f"Original: {test_sequence}")
    print(f"Sampled:  {result}")

    print("Training completed successfully!")


if __name__ == "__main__":
    main()
