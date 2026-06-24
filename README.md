# Multi-Turn Composed Image Retrieval with Conversation-Aware Query Conditioning

A composed image retrieval (CIR) system that lets a user refine a fashion search across several turns. Instead of forcing every modification into one query, the model keeps a running memory of earlier feedback so each new instruction sharpens the existing intent rather than replacing it.

> Composed Image Retrieval takes a **reference image** plus a **textual description of desired changes** and returns a target image that keeps the reference's visual identity while applying the requested edits. This project extends that setup from a single query to an ongoing conversation.

---

## Why multi-turn

Search rarely happens in one shot. A shopper might ask for a longer dress, look at the results, then ask for a darker colour, then decide on shorter sleeves. A single-turn system treats each request as a fresh query, so the user has to restate earlier constraints every time. Here, the earlier feedback stays in a conversation memory, and a new instruction refines the accumulated query instead of starting over.

---

## How it works

The pipeline runs in six stages, from raw input to retrieved results:

**1. Input encoding.** The reference image is passed through a frozen vision encoder. Each feedback turn is encoded by a frozen text encoder, producing one embedding per turn.

**2. Conversation-aware query conditioning.** Past feedback embeddings are held in a sliding-window history buffer (max window size `T = 8`). A `MultiTurnAttentionBlock` cross-attends with the current turn as the query and the history buffer as keys and values, producing a refined context embedding that summarises what the user has asked for so far. The block uses residual connections, layer normalisation, and a feed-forward network. The buffer is updated with detached gradients to keep memory usage bounded.

**3. Gated injection into query tokens.** The refined context is projected and added to the model's learnable query tokens, scaled by a learnable memory parameter `α`:

```
Q_k = Q + α · φ(h̃_k)
```

`α` is initialised to **zero**, so the model starts close to the single-turn behaviour and gradually learns how much historical context to fold in. With an empty buffer on the first turn, the conditioning term vanishes and the single-turn pathway is preserved.

**4. Multimodal fusion.** A Querying Transformer (Q-Former) receives the conditioned query tokens, the reference visual features, and the current feedback text, and produces a joint representation that captures both the reference's visual attributes and the constraints accumulated across turns.

**5. Prompt generation.** A sentence-level prompt is generated from the fused representation to express the accumulated modifications.

**6. Retrieval.** Candidate target images are encoded with the same frozen vision encoder, and the history-aware query representation is matched against them by normalised inner product to return the top results.

### Training objective

Optimisation combines three complementary losses:

- **Image–Text Contrastive (ITC)** — aligns image and text representations in the batch.
- **Relative Text Contrastive (RTC)** — aligns the fused reference-plus-feedback query with the target image feature.
- **Alignment** — pulls the mean fusion-query representation toward the mean learnable prompt-token representation.

```
L = L_itc + L_rtc + L_align
```

In the multi-turn setting, the ITC term is computed on the final-turn fused representation. Each FashionIQ caption pair is reformulated as a two-turn dialogue so the model receives sequential supervision during training.

---

## Repository structure

```
multiturn-cir/
├── src/
│   ├── blip_fine_tune_2.py      # main training entry point (use --multi-turn)
│   ├── blip_validate.py         # validation on a trained checkpoint
│   ├── validate_blip.py         # metric computation (Recall@K)
│   ├── validate_blip_rerank.py  # validation with candidate re-ranking
│   ├── cirr_test_submission.py  # CIRR test-split submission helper
│   ├── data_utils.py            # FashionIQ / CIRR dataset classes and transforms
│   ├── utils.py                 # shared helpers
│   └── lavis/                   # vision-language backbone components
├── fashionIQ_dataset/           # dataset (see below) — not tracked in the repo
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/Shan81111/multiturn-cir.git
cd multiturn-cir
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for both training and evaluation.

### Dataset

Experiments use the [FashionIQ](https://github.com/XiaoxiaoGuo/fashion-iq) benchmark, covering three categories: **Dress**, **Shirt**, and **Toptee**. Place the data at the repository root so the loaders can find it:

```
fashionIQ_dataset/
├── captions/        # cap.{dress|shirt|toptee}.{train|val|test}.json
├── image_splits/    # split.{dress|shirt|toptee}.{train|val|test}.json
└── images/          # downloaded FashionIQ images
```

---

## Training

Train the multi-turn model on FashionIQ:

```bash
python src/blip_fine_tune_2.py \
  --dataset fashionIQ \
  --multi-turn \
  --blip-model-name blip2_cir_cat \
  --backbone pretrain \
  --num-epochs 30 \
  --batch-size 128 \
  --learning-rate 2e-5 \
  --weight-decay 0.05 \
  --memory-lr-mult 5.0 \
  --save-best \
  --save-memory
```

Key flags:

| Flag | Purpose |
| --- | --- |
| `--multi-turn` | Enables the conversation memory and multi-turn training recipe. Omit it for the single-turn setting. |
| `--memory-lr-mult` | Learning-rate multiplier for the memory module (default `5.0`). |
| `--backbone` | `pretrain` for ViT-g, `pretrain_vitL` for ViT-L. |
| `--blip-model-name` | `blip2_cir_cat` or `blip2_cir`. |
| `--save-best` / `--save-memory` | Save the best checkpoint and the memory-module weights. |

Training details: AdamW optimiser, OneCycle learning-rate schedule, mixed-precision, and gradient clipping (`max_norm = 1.0`). The memory scale `α` is initialised to zero and frozen for the first two epochs, then unfrozen; bias and LayerNorm parameters are excluded from weight decay. The history window is `T = 8`.

---

## Evaluation

Evaluate a trained checkpoint on the FashionIQ validation set:

```bash
python src/blip_validate.py \
  --dataset fashionIQ \
  --backbone pretrain \
  --model-path path/to/checkpoint.pt
```

Following standard protocol, performance is reported as **Recall@10** and **Recall@50**, averaged across the Dress, Shirt, and Toptee categories.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{omar2026multiturn,
  title     = {Multi-Turn Composed Image Retrieval with Conversation-Aware Query Conditioning},
  author    = {Omar, Shanzay and Saqib, Yamsheen and Shakeel, M. Haroon and Taj, Murtaza},
  year      = {2026},
  note      = {Under review}
}
```

---

## Acknowledgements

Built on the BLIP-2 vision-language backbone (via LAVIS) and evaluated on the FashionIQ benchmark. Thanks to the authors of these resources for releasing them to the community.
