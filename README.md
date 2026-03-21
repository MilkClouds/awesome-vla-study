# 🦾 Awesome VLA Study

**Getting started with VLA?** This guide takes you from the foundations to the frontier — diffusion and flow matching, state-of-the-art robot foundation model architectures, data scaling, RL fine-tuning, and world models. Papers in reading order.

### 📋 Prerequisites
- Basic probability & optimization (enough to follow ELBO, score matching derivations)
- Deep learning fundamentals (Transformers, attention, tokenization)
  - 💡 *Starting from scratch?* [MIT 6.S191 — Intro to Deep Learning](http://introtodeeplearning.com) covers CNNs, Transformers, and generative models in a 1-week intensive bootcamp. More courses [below](#-recommended-courses).

### 💬 Weekly Format (Recommended)
- **Paper presentation**: 1–2 participants per week, 30 min/paper — architecture, training, key results
- **Discussion**: Compare design choices across the week's papers, discuss limitations and open questions (15–20 min)

| Phase | Weeks | Topic | Readings |
|-------|-------|-------|----------|
| **Phase 1** | W1–3 | Generative Model Foundations | MIT 6.S184 course |
| **Phase 2** | W4–5 | Early Foundation RFMs & Robot Policy | RT-1, RT-2, Octo, OpenVLA, BeT, Diffusion Policy, ACT |
| **Phase 3** | W6–7 | Current RFM Architectures | CogACT, GR00T N1, X-VLA, π0, InternVLA-M1 |
| **Phase 4** | W8–9 | Data Scaling | OXE, AgiBot World, UMI, VITRA, Human to Robot Transfer |
| **Phase 5** | W10–11 | Efficient Inference & Dual-System | RTC, SmolVLA, Helix, Fast-in-Slow |
| **Phase 6** | W12–14 | RL Fine-tuning, Reasoning & World Model | HIL-SERL, SimpleVLA-RL, π\*0.6, CoT-VLA, ThinkAct, Fast-ThinkAct, UniVLA, Cosmos Policy, DreamZero |

---

## Phase 1: Generative Model Foundations (Weeks 1–3)

**📚 Core Material**: [MIT 6.S184 — Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/2025/index.html) (Holderrieth & Erives, MIT CSAIL, 2025) | [Course notes paper](https://arxiv.org/abs/2506.02070)

### Week 1: ODE/SDE Foundations & Diffusion Models
| Material | Topic |
|----------|-------|
| Lectures 1–2 | ODE/SDE basics, forward/reverse processes, conditional/marginal probability paths |
| Lab 1 | Hands-on SDE simulation |

### Week 2: Flow Matching, Score Matching & Training
| Material | Topic |
|----------|-------|
| Lectures 3–4 | Flow Matching, Score Matching, guidance, classifier-free guidance |
| Labs 2–3 | Building a toy diffusion model from scratch |

### Week 3: Generative Robotics & Review
| Material | Topic |
|----------|-------|
| Lecture 5 | Guest lecture by Benjamin Burchfiel (Toyota Research): diffusion models for robotics |
| Lecture 6 | Generative protein design (optional) |



---

## Phase 2: Early Foundation Robot Models & Robot Policy (Weeks 4–5)

### Week 4: Early Foundation Robot Models — RT-1, RT-2, Octo, OpenVLA
| # | Paper | Link | Key Topic |
|---|-------|------|-----------|
| 1 | **RT-1: Robotics Transformer** — Brohan et al. (2022) | [2212.06817](https://arxiv.org/abs/2212.06817) | First large-scale Robotics Transformer (no VLM) |
| 2 | **RT-2: Vision-Language-Action Models** — Brohan et al. (2023) | [2307.15818](https://arxiv.org/abs/2307.15818) | VLM backbone → VLA paradigm |
| 3 | **Octo** — Ghosh et al. (2024) | [2405.12213](https://arxiv.org/abs/2405.12213) | Open-source generalist policy, modular design, pretrained on OXE (no VLM) |
| 4 | **OpenVLA** — Kim et al. (2024) | [2406.09246](https://arxiv.org/abs/2406.09246) | First open-source VLM-based VLA |

> 📎 Supplementary video: [Stanford CS25 V3 — Low-level Embodied Intelligence](https://www.youtube.com/watch?v=fz8wf9hN20c)

<!-- TODO: Decision Transformer (Chen et al., 2021, 2106.01345) and Trajectory Transformer (Janner et al., 2021, 2106.02039) — RL as sequence modeling; the conceptual basis for "action as tokens". Consider adding as a supplementary reference. -->

**Key points**: RT-1 (35M, no VLM) → RT-2 (55B VLM, action as text tokens) establishes the VLA concept. Octo (27M–93M, diffusion head, no VLM) and OpenVLA (7B, VLM + 256-bin discretization) are the first open-source generalist robot policies enabling community iteration.

### Week 5: Core Robot Policies — Diffusion Policy, ACT, BeT
| # | Paper | Link | Key Topic |
|---|-------|------|-----------|
| 5 | **Behavior Transformers (BeT)** — Shafiullah et al. (2022) | [2206.11251](https://arxiv.org/abs/2206.11251) | Multimodal action discretization, k-means + offset |
| 6 | **Diffusion Policy** — Chi et al. (2023) | [2303.04137](https://arxiv.org/abs/2303.04137) | Diffusion for robot control, action sequence prediction |
| 7 | **ACT/ALOHA** — Zhao et al. (2023) | [2304.13705](https://arxiv.org/abs/2304.13705) | Action Chunking Transformer, CVAE, bimanual |

**Key points**: Three approaches to the multimodal action problem. Action chunking (predicting K future actions at once) is foundational for later VLA work.

---

## Phase 3: Current RFM Architectures (Weeks 6–7)

### Week 6: VLM + Action Head — CogACT, GR00T N1, X-VLA
| # | Paper | Link | Key Topic |
|---|-------|------|-----------|
| 8 | **CogACT** — Li et al. (2024) | [2411.19650](https://arxiv.org/abs/2411.19650) | VLM + DiT action head, action token learning |
| 9 | **GR00T N1** — Bjorck et al. (2025) | [2503.14734](https://arxiv.org/abs/2503.14734) | 2B diffusion transformer, whole-body humanoid control |
| 10 | **X-VLA** — Zheng et al. (2025) | [2510.10274](https://arxiv.org/abs/2510.10274) | Soft prompts for cross-embodiment, Florence-Large + flow matching |

**Key points**: All three use only the VLM's last hidden state to drive a separate action head.

### Week 7: VLM + Action Expert — π0, InternVLA-M1
| # | Paper | Link | Key Topic |
|---|-------|------|-----------|
| 11 | **π0** — Black et al. (2024) | [2410.24164](https://arxiv.org/abs/2410.24164) | Flow matching + action expert accessing VLM intermediate features |
| 12 | **InternVLA-M1** — Chen et al. (2025) | [2510.13778](https://arxiv.org/abs/2510.13778) | Spatial grounding → action generation, AR-based |

> 📎 Background: **Transfusion** — Zhou et al. (2024) | [2408.11039](https://arxiv.org/abs/2408.11039) — AR + diffusion in one transformer; π0's architectural basis

**Key points**: Unlike Week 6's action heads that only see the VLM's last hidden state, these action experts access VLM internal hidden states.

---

## Phase 4: Data Scaling (Weeks 8–9)

### Week 8: Large-Scale Robot Datasets — OXE, AgiBot World
| # | Paper | Link | Key Topic |
|---|-------|------|-----------|
| 13 | **Open X-Embodiment (OXE)** — Open X-Embodiment Collaboration (2023) | [2310.08864](https://arxiv.org/abs/2310.08864) | 1M+ trajectories, 22 embodiments, standardized data format |
| 14 | **AgiBot World** — Bu et al. (2025) | [2503.06669](https://arxiv.org/abs/2503.06669) | 1M+ trajectories, 217 tasks, 5 deployment scenarios |

> 📎 **Data formats** — Recording-oriented: [rosbag](http://wiki.ros.org/rosbag) (ROS 1), [mcap](https://mcap.dev/) (vendor-neutral, ROS 2 default). Training-oriented: [RLDS](https://github.com/google-research/rlds) (TensorFlow/OXE standard), [LeRobotDataset](https://github.com/huggingface/lerobot) (HuggingFace, Parquet + video).  
> 📎 [From the Evolution of Rosbag to the Future of AI Tooling](https://rerun.io/blog/rosbag) — by the original rosbag author; covers rosbag V1→V2 → rosbag2 (sqlite3) → MCAP evolution

**Key points**: Large-scale multi-embodiment datasets that enable generalist robot policy pretraining. OXE standardized the data format across 22 robot embodiments via RLDS; AgiBot World provides high-quality data at scale.

### Week 9: Data Collection Methods — UMI, VITRA, Human to Robot Transfer
| # | Paper | Link | Key Topic |
|---|-------|------|-----------|
| 15 | **UMI** — Chi et al. (2024) | [2402.10329](https://arxiv.org/abs/2402.10329) | Robot-free SE(3) data collection via handheld gripper |
| 16 | **VITRA** — Li et al. (2025) | [2510.21571](https://arxiv.org/abs/2510.21571) | Human video → VLA training data (1M episodes from egocentric human videos) |
| 17 | **Human to Robot Transfer** — Kareer et al. (2025) | [2512.22414](https://arxiv.org/abs/2512.22414) | Human video → robot transfer emerges with VLA scaling |

**Key points**: Three data sources beyond robot teleoperation — UMI (embodiment-agnostic physical demos, <$200 hardware), egocentric video, and exocentric video.

---

## Phase 5: Efficient Inference & Dual-System (Weeks 10–11)

### Week 10: Fast-Acting VLA — SmolVLA & RTC
| # | Paper | Link | Key Topic |
|---|-------|------|-----------|
| 18 | **SmolVLA** — Shukor et al. (2025) | [2506.01844](https://arxiv.org/abs/2506.01844) | 450M params (~1/7 of π0), model compression + async inference |
| 19 | **RTC** — Black et al. (2025) | [2506.07339](https://arxiv.org/abs/2506.07339) | Async inference — freezing + inpainting, no retraining needed |

**Key points**: Two complementary approaches — SmolVLA compresses the model itself, RTC optimizes the inference pipeline. Can be combined.

### Week 11: Dual-System VLA — Helix & Fast-in-Slow
| # | Paper | Link | Key Topic |
|---|-------|------|-----------|
| 20 | **Helix** — Figure AI (2025) | [figure.ai/news/helix](https://www.figure.ai/news/helix) | S2: 7B VLM @7-9Hz, S1: 80M @200Hz, humanoid |
| 21 | **Fast-in-Slow** — Chen et al. (2025) | [2506.01953](https://arxiv.org/abs/2506.01953) | Integrated dual-system, end-to-end trainable |

**Key points**: Dual-System separates slow reasoning (VLM) from fast execution (lightweight policy) at different frequencies. Helix (separately trained) vs Fast-in-Slow (end-to-end trainable).

---

## Phase 6: RL Fine-tuning, Reasoning & World Model (Weeks 12–14)

### Week 12: RL Fine-tuning & Human-in-the-Loop — HIL-SERL, SimpleVLA-RL, π*0.6
| # | Paper | Link | Key Topic |
|---|-------|------|-----------|
| 22 | **HIL-SERL** — Luo et al. (2024) | [2410.21845](https://arxiv.org/abs/2410.21845) | Human-in-the-loop RL, sample-efficient real-world training |
| 23 | **SimpleVLA-RL** — Li et al. (2025) | [2509.09674](https://arxiv.org/abs/2509.09674) | RL fine-tuning for autoregressive VLA, outcome-based rewards |
| 24 | **π\*0.6 / Recap** — Physical Intelligence (2025) | [2511.14759](https://arxiv.org/abs/2511.14759) | RL for flow-based VLA, advantage-conditioned, learns from suboptimal data |

**Key points**: Three RL approaches — HIL-SERL (human-in-the-loop, sample-efficient), SimpleVLA-RL (outcome rewards), π\*0.6 (advantage-conditioned, learns from suboptimal data).

### Week 13: Reasoning VLA — CoT-VLA, ThinkAct, Fast-ThinkAct
| # | Paper | Link | Key Topic |
|---|-------|------|-----------|
| 25 | **CoT-VLA** — Zhao et al. (2025) | [2503.22020](https://arxiv.org/abs/2503.22020) | Visual chain-of-thought reasoning (future image prediction) before action |
| 26 | **ThinkAct** — Huang et al. (2025) | [2507.16815](https://arxiv.org/abs/2507.16815) | Decouple reasoning from execution; RL grounds plan quality in task success, not language supervision |
| 27 | **Fast-ThinkAct** — Huang et al. (2026) | [2601.09708](https://arxiv.org/abs/2601.09708) | Text-level CoT dispensable — latent distillation preserves planning capacity at ~10× speed |

> 📎 Fast-ThinkAct's reasoning compression is orthogonal to Week 10's model compression (SmolVLA, RTC) — the two can stack.

**Key points**: Reasoning representation — image tokens (CoT-VLA) vs. visual latent (ThinkAct) vs. compressed latent tokens (Fast-ThinkAct). ThinkAct grounds reasoning in task-outcome RL instead of language supervision. Fast-ThinkAct shows planning structure, not verbosity, carries the signal (~10× faster, performance preserved).

### Week 14: World Model — UniVLA, Cosmos Policy, DreamZero
| # | Paper | Link | Key Topic |
|---|-------|------|-----------|
| 28 | **UniVLA** — Wang et al. (2025) | [2506.19850](https://arxiv.org/abs/2506.19850) | Unified AR VLA with world modeling as training objective |
| 29 | **Cosmos Policy** — Kim et al. (2026) | [2601.16163](https://arxiv.org/abs/2601.16163) | Pretrained video foundation model as robot policy backbone |
| 30 | **DreamZero** — Ye et al. (2026) | [dreamzero0.github.io](https://dreamzero0.github.io/) | World Action Model, joint world+action generation in latent space |

**Key points**: Three ways to leverage world knowledge — training regularizer (UniVLA, no world prediction at inference), pretrained video FM as policy backbone (Cosmos Policy), joint world+action generation in latent space (DreamZero).

---

## Contributing

Suggestions for papers, resources, or structural improvements are welcome — please open an issue or PR.

## See Also

- 🔥 **[vla0-trl](https://github.com/MilkClouds/vla0-trl)** — A complete VLA in ~1,200 lines of Python. Fine-tunes Qwen2.5-VL with TRL's SFTTrainer to predict actions as text, scoring ~90% on LIBERO. Read the entire codebase in an afternoon.
- 🔥 **[vla-eval](https://github.com/allenai/vla-evaluation-harness)** — One framework to evaluate any VLA model on any robot simulation benchmark.
- [Awesome-RL-VLA](https://github.com/Denghaoyuan123/Awesome-RL-VLA) — RL for VLA models
- [Awesome-VLA-Robotics](https://github.com/Jiaaqiliu/Awesome-VLA-Robotics) — Large-scale VLA paper collection
- [awesome-physical-ai](https://github.com/keon/awesome-physical-ai) — A curated list of academic papers and resources on Physical AI

---

## 📚 Recommended Courses

Courses covering the prerequisites for this study guide — only those with recent (2023+) video lectures freely available on YouTube. Pick what you need.

| Area | Course | Instructor | Link | Notes |
|------|--------|------------|------|-------|
| **DL Fundamentals** | MIT 6.S191: Intro to Deep Learning | Alexander Amini | [introtodeeplearning.com](http://introtodeeplearning.com) · [YouTube '25](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI) | 1-week bootcamp (10 lectures) — CNN, Transformer, generative models, RL |
| | Andrej Karpathy: Neural Networks: Zero to Hero | Andrej Karpathy | [karpathy.ai/zero-to-hero.html](https://karpathy.ai/zero-to-hero.html) · [YouTube](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | Backprop → GPT, build everything from scratch in code |
| **Vision** | Stanford CS231n: DL for Computer Vision | Fei-Fei Li et al. | [cs231n.stanford.edu](https://cs231n.stanford.edu) · [YouTube '25](https://www.youtube.com/playlist?list=PLoROMvodv4rOmsNzYBMe0gJY2XS8AQg16) | The canonical CV course — backprop to detection/segmentation/video |
| **NLP / Transformers** | Stanford CS224n: NLP with Deep Learning | Christopher Manning | [web.stanford.edu/class/cs224n](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/) · [YouTube '24](https://www.youtube.com/playlist?list=PLoROMvodv4rOaMFbaqxPDoLWjDaRAdP9D) | Word vectors → Transformers → LLMs |
| **RL** | UC Berkeley CS285: Deep RL | Sergey Levine | [rail.eecs.berkeley.edu/deeprlcourse](http://rail.eecs.berkeley.edu/deeprlcourse/) · [YouTube '23](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVYGEGiAOMaOzzv41Jfm_Ps) | Policy gradients, Q-learning, model-based & offline RL — by a leading robotics RL researcher |
