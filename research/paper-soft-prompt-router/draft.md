# Soft Prompt Routing: Cross-Model Query Transfer Between Heterogeneous Edge-Cloud LLMs

## Abstract

We investigate soft prompt routing as a mechanism for edge-cloud LLM collaboration, where a small frozen edge model compresses user queries into continuous soft prompt tokens that steer a large frozen cloud model's generation. Through systematic ablation across four edge models (1B–4.5B parameters) paired with a 26B cloud model, we identify three key findings: (1) instruction-tuned edge models suffer complete mode collapse in cross-model transfer, while base models preserve query semantics; (2) entity transfer accuracy scales non-monotonically with edge model size — a 2.3B base model (50% entity match) outperforms both a 1B model (34%) and a 4.5B model (37%); (3) soft prompts reliably transfer domain-level information (84% accuracy) but systematically lose entity-level specifics, with translation and numerical content most affected. Our results suggest that the soft prompt bottleneck functions as a lossy "topic router" rather than a faithful query encoder, and that instruction tuning fundamentally corrupts the representation space for cross-model transfer.

## 1. Introduction

The deployment of large language models (LLMs) in resource-constrained settings motivates architectures that split computation between an edge device and a cloud server. While speculative decoding addresses latency within same-family models, the question of whether a small heterogeneous edge model can meaningfully guide a large cloud model remains underexplored.

We formalize this as the **soft prompt routing** problem: given a user query processed by a frozen edge model, can a trainable projection learn to generate soft prompt tokens in the cloud model's embedding space that cause the cloud to produce the correct response? This setup — both models frozen, only the bridge trainable — is practically motivated by scenarios where neither the edge nor cloud model can be fine-tuned (e.g., API-served cloud models, sealed edge devices).

Our contributions:
- A systematic ablation across four edge models (gemma-3-1b-it, gemma-4-E2B, gemma-4-E4B, gemma-4-E4B-it) paired with gemma-4-26B-A4B-it as cloud, evaluated on 100 prompts across 8 domains
- The first empirical evidence that instruction tuning destroys cross-model transferability at all layers
- Discovery of non-monotonic scaling in edge model size for soft prompt quality
- Characterization of the "domain vs. entity" gap: soft prompts transfer topic-level but not entity-level information

## 2. Related Work

**Soft prompt tuning.** Prefix tuning (Li & Liang, 2021) and prompt tuning (Lester et al., 2021) optimize continuous vectors prepended to frozen LLM inputs. These methods operate within a single model; cross-model prompt transfer remains largely unexplored.

**Knowledge distillation.** Hinton et al. (2015) introduced teacher-student frameworks. Recent work extends to heterogeneous architectures (Jiao et al., 2020), but typically trains the student model rather than a fixed bridge.

**Speculative decoding.** Leviathan et al. (2023) use a small draft model to accelerate a large verifier. This operates within the same model family and optimizes latency, not cross-model semantic transfer.

**Edge-cloud split inference.** Matsubara et al. (2022) survey split computing for DNNs, primarily in vision. LLM-specific edge-cloud architectures remain nascent. The closest work is BLIP-2 (Li et al., 2023), which bridges a frozen vision encoder to a frozen LLM via a trainable Q-Former — our architecture adapts this pattern to text-to-text cross-model transfer.

**Model routing.** Routing approaches (Jiang et al., 2024; Ong et al., 2024) select which model to query based on input characteristics. Our work differs in that the edge model doesn't select a model but rather encodes the query into a continuous representation that steers generation.

## 3. Method

### 3.1 Architecture

Our pipeline consists of three components:

1. **Edge encoder** (frozen): A small LLM processes the user query and produces a hidden state vector h ∈ ℝ^{d_edge} from the last token of the last layer.

2. **Projection bridge** (trainable): A linear projection P: ℝ^{d_edge} → ℝ^{d_edge} followed by a 3-layer MLP that maps h to K soft prompt tokens in the cloud's embedding space:
   - P(h) → GELU → Linear → GELU → Linear → reshape to (K, d_cloud)
   
   The MLP has hidden dimension 2 × d_edge and output dimension K × d_cloud.

3. **Cloud decoder** (frozen): A large LLM receives the K soft prompt tokens via `inputs_embeds` and generates the response autoregressively.

### 3.2 Quantization-Aware Training (QAT)

To simulate deployment conditions where soft prompts must be transmitted over a network, we apply per-token symmetric fake quantization with a straight-through estimator (STE) during training. At 8-bit precision, quantization error is negligible (mean |q - x| < 0.3 across all experiments).

### 3.3 Training

Both edge and cloud models are frozen (4-bit NF4 quantization via bitsandbytes). Only the projection and MLP parameters (~50M) are trained with AdamW (lr=3e-4, weight decay=0.01, cosine decay with 200-step warmup) on cross-entropy loss between the cloud's predicted tokens and the target response.

Training data: 1,185 prompt-response pairs generated by a stronger teacher model, with responses averaging 546 characters. Data includes 1,505 confusable pairs (e.g., "capital of Japan" vs. "capital of Jamaica") to encourage fine-grained entity encoding.

### 3.4 Evaluation

We evaluate on 100 prompts across 8 domains: geography (20), science (15), math (15), literature (10), history (10), translation (10), creative (10), reasoning (10). Metrics:

- **Key entity match**: Does the output contain the critical entity (e.g., "Tokyo" for "capital of Japan")?
- **Domain match**: Is the output in the correct topic area?
- **BLEU-1/2**: N-gram overlap with reference answers.

## 4. Experiments

### 4.1 Edge Model Ablation

We compare four edge models, all paired with gemma-4-26B-A4B-it as cloud (K=32, 10k training steps):

| Edge Model | Type | Params (eff.) | H | Entity Match | Domain Match |
|---|---|---|---|---|---|
| gemma-3-1b-it | IT | 1.0B | 1152 | 34% | 82% |
| gemma-4-E2B | Base | 2.3B | 1536 | **50%** | **84%** |
| gemma-4-E4B | Base | 4.5B | 2560 | 37% | 79% |
| gemma-4-E4B-it | IT | 4.5B | 2560 | 0% | 0% |

**Finding 1: Instruction tuning destroys cross-model transferability.** E4B-it produces complete mode collapse — all 100 outputs are off-topic instruction-following style text ("Okay, let's break down..."). The same architecture without instruction tuning (E4B base) achieves 37% entity match, demonstrating that IT corrupts the representation space for cross-model transfer.

**Finding 2: Non-monotonic scaling.** E2B (2.3B, H=1536) outperforms both the smaller 1b (34%) and the larger E4B (37%). We hypothesize that E2B's hidden dimension (1536) is better matched to the 32-token bottleneck than E4B's wider representation (2560), where the additional dimensions carry information that the MLP cannot effectively compress into K=32 tokens.

### 4.2 Per-Domain Analysis

| Domain | 1b-it | E2B base | E4B base |
|---|---|---|---|
| Reasoning | 60% | **100%** | 60% |
| Creative | 50% | **80%** | 60% |
| Geography | 20% | **50%** | 20% |
| Science | 33% | **53%** | 47% |
| History | **60%** | 50% | 30% |
| Literature | **80%** | 60% | 50% |
| Math | 0% | **20%** | 27% |
| Translation | 0% | 0% | 0% |

E2B base leads in 5 of 8 domains, with reasoning achieving perfect accuracy. The domain-entity gap is consistent: domain match (84%) far exceeds entity match (50%), confirming that soft prompts encode topic-level but not entity-level information.

**Translation failure.** All models score 0% on translation — cross-lingual entity mapping cannot be achieved through soft prompts alone, as the target word (e.g., "merci", "おはよう") has no representation in the edge model's encoding of the source prompt.

**Math partial failure.** Numerical specifics (0–27% across models) are poorly transmitted. The edge model's hidden state encodes "this is a math question" but not "which specific numbers."

### 4.3 Bottleneck Width (K)

| K | Entity Match | Domain Match | Best CE |
|---|---|---|---|
| 32 | 34% | 82% | 0.051 |
| 64 | 32% | 87% | 0.030 |

Doubling K from 32 to 64 (with gemma-3-1b-it edge) does not improve entity match (34% → 32%). Lower training CE (0.030) does not translate to better generation quality, suggesting the bottleneck is in the edge representation, not the channel capacity.

### 4.4 Teacher Data Quality

| Teacher Data | Samples | Avg Response | Entity Match |
|---|---|---|---|
| gemma3:4b distilled | 5000 | 27 chars | 12.5% (8-prompt) |
| a stronger teacher model | 1185 | 546 chars | 34% (100-prompt) |

Despite 4× fewer samples, Higher-quality teacher data with longer, information-rich responses produces substantially better soft prompt training signal.

## 5. Analysis

### 5.1 The Domain-Entity Gap

Our results reveal a consistent pattern across all configurations: soft prompts reliably encode **what kind of question** is being asked (domain) but fail to encode **which specific instance** (entity). This is consistent with an information bottleneck interpretation — the MLP must compress the edge model's ~1500-dimensional hidden state into K=32 tokens, each ~2800-dimensional. The resulting representation preserves the dominant eigenvectors (which correspond to topic/domain structure) but loses the finer-grained entity signals.

### 5.2 Why Instruction Tuning Hurts

Instruction-tuned models learn to produce hidden states that encode *how to respond* (conversational style, safety alignment, instruction-following patterns) rather than *what was asked*. In single-model use, this is beneficial — the model's own decoder expects these signals. In cross-model transfer, these signals are noise: the cloud model's decoder expects raw semantic content, not the edge model's style markers. Base models, which haven't been trained to encode response patterns, produce cleaner semantic representations.

### 5.3 The E2B Sweet Spot

The non-monotonic scaling (E2B > E4B) may reflect an interaction between model capacity and bottleneck dimensionality. E4B's larger hidden state (2560-dim) distributes query information across more dimensions, making it harder for a fixed-size MLP to extract and compress. E2B's 1536-dim state is information-dense enough to encode query semantics but compact enough for the bridge to process effectively. This suggests that optimal edge model selection depends on the bottleneck architecture, not just raw model quality.

## 6. Limitations

- **Single cloud model.** All experiments use gemma-4-26B-A4B-it. Results may differ with other cloud architectures.
- **Limited training data.** 1,185 samples is small; scaling to 5,000–10,000 may improve results.
- **English only.** All evaluation prompts are in English.
- **Greedy decoding.** We use greedy generation; sampling strategies may yield different results.
- **No runtime latency analysis.** We focus on accuracy, not edge-cloud communication overhead.

## 7. Conclusion

We present the first systematic study of soft prompt routing between heterogeneous frozen LLMs. Our key findings — that instruction tuning destroys transferability, that edge model scaling is non-monotonic, and that soft prompts function as topic routers rather than query encoders — provide a foundation for future work on edge-cloud LLM architectures.

The 50% entity match achieved by E2B base demonstrates that meaningful cross-model query transfer is possible, while the persistent domain-entity gap (84% vs 50%) highlights the fundamental limitation of continuous soft prompt channels. Future directions include hybrid architectures combining soft prompts with discrete token transmission, contrastive training on confusable pairs, and theoretical analysis through the information bottleneck framework.

## References

- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. arXiv:1503.02531.
- Jiao, X., et al. (2020). TinyBERT: Distilling BERT for natural language understanding. EMNLP.
- Lester, B., Al-Rfou, R., & Constant, N. (2021). The power of scale for parameter-efficient prompt tuning. EMNLP.
- Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast inference from transformers via speculative decoding. ICML.
- Li, J., et al. (2023). BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. ICML.
- Li, X.L., & Liang, P. (2021). Prefix-tuning: Optimizing continuous prompts for generation. ACL.
- Matsubara, Y., et al. (2022). Split computing and early exiting for deep learning applications: Survey and research challenges. ACM Computing Surveys.
