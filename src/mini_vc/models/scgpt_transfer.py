"""Load the official scGPT whole-human encoder for K562 transfer experiments.

The released checkpoint uses FlashMHA's fused Wqkv tensors. PyTorch's
MultiheadAttention stores the same Q, K, V projections as in_proj tensors.
This module renames those tensors and verifies that every encoder parameter
comes from the checkpoint before any experiment is run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from mini_vc.scgpt_core import TransformerModel


def map_perturbation_genes(
    gene_names: list[str], vocab: dict[str, int]
) -> tuple[list[int], list[int], dict]:
    """Map measured gene symbols to checkpoint tokens, retaining explicit OOVs."""
    aliases = {"ELMSAN1": "MIDEAS"}
    token_ids = []
    oov_indices = []
    oov_genes = []
    used_aliases = {}
    for gene in gene_names:
        symbol = gene if gene in vocab else aliases.get(gene, gene)
        if symbol in vocab:
            token_ids.append(int(vocab[symbol]))
            oov_indices.append(-1)
            if symbol != gene:
                used_aliases[gene] = symbol
        else:
            token_ids.append(int(vocab["<pad>"]))
            oov_indices.append(len(oov_genes))
            oov_genes.append(gene)
    return token_ids, oov_indices, {
        "mapped_perturbation_genes": len(gene_names) - len(oov_genes),
        "total_perturbation_genes": len(gene_names),
        "aliases": used_aliases,
        "oov_genes": oov_genes,
    }


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_scgpt_encoder(
    checkpoint_dir: Path,
) -> tuple[TransformerModel, dict[str, int], dict]:
    """Load all gene, value, and transformer encoder weights from scGPT."""
    args = json.loads((checkpoint_dir / "args.json").read_text(encoding="utf-8"))
    vocab = json.loads((checkpoint_dir / "vocab.json").read_text(encoding="utf-8"))
    if args.get("input_emb_style") != "continuous" or args.get("n_bins") != 51:
        raise ValueError("checkpoint input style differs from the supported scGPT protocol")
    if "<pad>" not in vocab:
        raise ValueError("scGPT vocabulary lacks <pad>")
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=int(args["embsize"]),
        nhead=int(args["nheads"]),
        d_hid=int(args["d_hid"]),
        nlayers=int(args["nlayers"]),
        nlayers_cls=int(args["n_layers_cls"]),
        vocab=vocab,
        dropout=float(args["dropout"]),
        pad_token=str(args["pad_token"]),
        pad_value=int(args["pad_value"]),
        do_mvc=bool(args["MVC"]),
        input_emb_style=str(args["input_emb_style"]),
        cell_emb_style="avg-pool",  # the released checkpoint was trained without CLS
        use_fast_transformer=False,
    )
    # PyTorch's NestedTensor inference shortcut cannot be followed by trainable
    # transformer layers in this partial fine-tuning setup.
    model.transformer_encoder.use_nested_tensor = False
    raw = torch.load(
        checkpoint_dir / "best_model.pt", map_location="cpu", weights_only=True
    )
    converted = {}
    for name, value in raw.items():
        name = name.replace(".self_attn.Wqkv.weight", ".self_attn.in_proj_weight")
        name = name.replace(".self_attn.Wqkv.bias", ".self_attn.in_proj_bias")
        converted[name] = value
    target = model.state_dict()
    required = (
        "encoder.",
        "value_encoder.",
        "transformer_encoder.",
    )
    missing_required = [
        name for name in target if name.startswith(required) and name not in converted
    ]
    mismatched_required = [
        name
        for name in target
        if name.startswith(required)
        and name in converted
        and target[name].shape != converted[name].shape
    ]
    if missing_required or mismatched_required:
        raise ValueError(
            f"scGPT encoder is incomplete: missing={missing_required}, "
            f"shape_mismatch={mismatched_required}"
        )
    matching = {
        name: value
        for name, value in converted.items()
        if name in target and target[name].shape == value.shape
    }
    model.load_state_dict(matching, strict=False)
    audit = {
        "checkpoint_sha256": checkpoint_sha256(checkpoint_dir / "best_model.pt"),
        "vocab_sha256": checkpoint_sha256(checkpoint_dir / "vocab.json"),
        "loaded_parameter_tensors": len(matching),
        "loaded_encoder_tensors": sum(name.startswith(required) for name in matching),
        "encoder_parameter_tensors": sum(name.startswith(required) for name in target),
        "n_vocab": len(vocab),
        "embedding_dim": int(args["embsize"]),
        "n_transformer_layers": int(args["nlayers"]),
        "attention_mapping": "FlashMHA Wqkv -> PyTorch MultiheadAttention in_proj",
    }
    return model, vocab, audit


class ScGPTPerturbationRegressor(nn.Module):
    """Predict pseudobulk delta from true control cells and gene action tokens."""

    def __init__(
        self,
        backbone: TransformerModel,
        perturbation_token_ids: list[int],
        oov_indices: list[int],
        n_output_genes: int,
        train_backbone: bool,
        n_unfrozen_layers: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        if train_backbone:
            if not 1 <= n_unfrozen_layers <= len(backbone.transformer_encoder.layers):
                raise ValueError("invalid number of unfrozen transformer layers")
            for layer in backbone.transformer_encoder.layers[-n_unfrozen_layers:]:
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
        self.train_backbone = train_backbone
        self.n_unfrozen_layers = n_unfrozen_layers if train_backbone else 0
        self.register_buffer(
            "perturbation_token_ids",
            torch.tensor(perturbation_token_ids, dtype=torch.long),
        )
        dimension = backbone.d_model
        if len(oov_indices) != len(perturbation_token_ids):
            raise ValueError("OOV indices do not match perturbation tokens")
        self.register_buffer("oov_indices", torch.tensor(oov_indices, dtype=torch.long))
        n_oov = max(oov_indices, default=-1) + 1
        self.oov_embeddings = nn.Parameter(torch.empty(n_oov, dimension))
        nn.init.normal_(self.oov_embeddings, mean=0.0, std=0.02)
        self.action_adapter = nn.Linear(dimension, 128)
        self.head = nn.Sequential(
            nn.Linear(n_output_genes + dimension + 128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_output_genes),
        )

    def train(self, mode: bool = True) -> "ScGPTPerturbationRegressor":
        super().train(mode)
        self.backbone.eval()
        if self.train_backbone and mode:
            for layer in self.backbone.transformer_encoder.layers[-self.n_unfrozen_layers:]:
                layer.train(True)
        return self

    def encode_control(
        self, control_gene_ids: torch.Tensor, control_values: torch.Tensor
    ) -> torch.Tensor:
        mask = torch.zeros_like(control_gene_ids, dtype=torch.bool)
        encoded = self.backbone._encode(control_gene_ids, control_values, mask)
        return encoded.mean(dim=1).mean(dim=0)

    def forward(
        self,
        reference: torch.Tensor,
        perturbation_features: torch.Tensor,
        control_gene_ids: torch.Tensor,
        control_values: torch.Tensor,
        cached_control_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context = (
            cached_control_embedding
            if cached_control_embedding is not None
            else self.encode_control(control_gene_ids, control_values)
        )
        gene_embeddings = self.backbone.encoder(self.perturbation_token_ids)
        is_oov = self.oov_indices >= 0
        if bool(is_oov.any()):
            gene_embeddings = gene_embeddings.clone()
            gene_embeddings[is_oov] = self.oov_embeddings[self.oov_indices[is_oov]]
        action = perturbation_features @ gene_embeddings
        action = self.action_adapter(action)
        state = context.unsqueeze(0).expand(len(reference), -1)
        delta = self.head(torch.cat([reference, state, action], dim=-1))
        return reference + delta


def load_scgpt_adapter(
    adapter_path: Path,
    pretrained_dir: Path,
    dataset,
    device: torch.device,
) -> ScGPTPerturbationRegressor:
    """Rebuild a trained transfer model from its adapter and official weights."""
    saved = torch.load(adapter_path, map_location="cpu", weights_only=True)
    backbone, vocab, audit = load_scgpt_encoder(pretrained_dir)
    if saved["checkpoint_sha256"] != audit["checkpoint_sha256"]:
        raise ValueError("the pretrained checkpoint differs from the trained adapter")
    if saved["gene_vocabulary"] != dataset.perturbation_vocabulary:
        raise ValueError("perturbation vocabulary differs from the training dataset")
    if saved["output_genes"] != dataset.genes["gene"].astype(str).tolist():
        raise ValueError("output genes differ from the training dataset")
    token_ids, oov_indices, _ = map_perturbation_genes(
        dataset.perturbation_vocabulary, vocab
    )
    mode = saved["mode"]
    if mode not in {"frozen", "finetuned"}:
        raise ValueError(f"unsupported saved mode: {mode}")
    model = ScGPTPerturbationRegressor(
        backbone,
        token_ids,
        oov_indices,
        len(dataset.genes),
        train_backbone=mode == "finetuned",
        n_unfrozen_layers=int(saved["config"]["training"]["n_unfrozen_layers"]),
    )
    expected = {name for name, param in model.named_parameters() if param.requires_grad}
    if set(saved["state_dict"]) != expected:
        raise ValueError("adapter checkpoint is missing or adding trainable tensors")
    incompatible = model.load_state_dict(saved["state_dict"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected adapter tensors: {incompatible.unexpected_keys}")
    return model.to(device).eval()
