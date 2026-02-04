# Learning Guide: AI Inference Engineering

## Purpose

A curated collection of resources for engineers working on AI inference systems — covering LLM serving, GPU kernel programming, attention mechanisms, quantization, distributed inference, and production deployment. Compiled from the AER Labs community.

## How to read

Recommended reading order:

1. Read "Tier 1" for all topics first (foundational concepts)
2. Read "Tier 2" for all topics (intermediate depth)
3. Read "Tier 3" for all topics (advanced / cutting-edge)

## Table of contents

- [1. LLM Inference Fundamentals](#1-llm-inference-fundamentals)
- [2. Inference Engines & Serving Systems](#2-inference-engines--serving-systems)
- [3. Attention Mechanisms & Memory Optimization](#3-attention-mechanisms--memory-optimization)
- [4. Quantization & Model Compression](#4-quantization--model-compression)
- [5. CUDA & GPU Kernel Programming](#5-cuda--gpu-kernel-programming)
- [6. Structured Output & Guided Decoding](#6-structured-output--guided-decoding)
- [7. Distributed & Multi-GPU Inference](#7-distributed--multi-gpu-inference)
- [8. Post-Training & Fine-Tuning](#8-post-training--fine-tuning)
- [9. Hardware Architecture & Co-Design](#9-hardware-architecture--co-design)
- [10. State-Space Models & Alternative Architectures](#10-state-space-models--alternative-architectures)
- [11. Compiler & DSL Approaches](#11-compiler--dsl-approaches)
- [12. Confidential & Secure Inference](#12-confidential--secure-inference)
- [13. AI Agents & LLM Tooling](#13-ai-agents--llm-tooling)
- [14. Production Inference at Scale](#14-production-inference-at-scale)
- [15. Benchmarking & Profiling](#15-benchmarking--profiling)
- [16. Courses & Comprehensive Guides](#16-courses--comprehensive-guides)
- [17. Tools & Libraries](#17-tools--libraries)
- [18. Reference Collections](#18-reference-collections)

---

## 1. LLM Inference Fundamentals

#### Tier 1

- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/) - kipply. Breaks down the compute and memory costs of transformer inference, essential for understanding bottlenecks in LLM serving.

- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) - Maarten Grootendorst. Visual walkthrough of quantization techniques for LLMs, covering the core concepts behind memory-efficient inference.

- [KV Cache in LLM Inference](https://pub.towardsai.net/kv-cache-in-llm-inference-7b904a2a6982) - Towards AI. Explanation of KV cache mechanics in LLM inference, covering how key-value caching reduces redundant computation during autoregressive generation.

- [Top 5 AI Model Optimization Techniques for Faster, Smarter Inference](https://developer.nvidia.com/blog/top-5-ai-model-optimization-techniques-for-faster-smarter-inference/) - Eduardo Alvarez, NVIDIA. Overview of key optimization techniques for improving inference performance and cost as models grow in size and complexity.

- [11 Production LLM Serving Engines: vLLM vs TGI vs Ollama](https://medium.com/@techlatest.net/11-production-llm-serving-engines-vllm-vs-tgi-vs-ollama-162874402840) - TechLatest. Comparative survey of 11 production LLM serving engines with trade-off analysis for different deployment scenarios.

- [How Fast Can We Perform a Forward Pass?](https://bounded-regret.ghost.io/how-fast-can-we-perform-a-forward-pass/) - Bounded Regret. Analysis of theoretical and practical limits on transformer forward pass speed, complementing kipply's Transformer Inference Arithmetic.

- [How Do MoE Models Compare to Dense Models in Inference?](https://epoch.ai/gradient-updates/moe-vs-dense-models-inference) - Epoch AI. Comparison of mixture-of-experts vs dense models focusing on inference costs, efficiency, and decoding dynamics.

- [LLM Routing](https://www.liuxunzhuo.com/llm-routing) - Xunzhuo Liu. Overview of LLM routing strategies for directing requests to optimal models based on task characteristics.

- [How LLM Inference Works](https://arpitbhayani.me/blogs/how-llm-inference-works) - Arpit Bhayani. End-to-end walkthrough of the LLM inference journey from prompt to response, covering tokenization, embedding, and autoregressive generation.

#### Tier 2

- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) - Thinking Machines Lab. Explores why LLM inference produces non-reproducible results and techniques to achieve deterministic outputs.

- [Densing Law of LLMs](https://www.nature.com/articles/s42256-025-01137-0) - Nature Machine Intelligence. Introduces "capability density" (capability per parameter) as a metric for evaluating LLMs, revealing an empirical scaling law for model efficiency.

- [Enabling Deterministic Inference for SGLang](https://lmsys.org/blog/2025-09-22-sglang-deterministic/) - LMSYS Org. Details the integration of batch-invariant kernels into SGLang to enable reproducible inference results.

- [The Next 1000x Cost Saving of LLM](https://ralphmao.github.io/token-cost/) - Ralph Mao. Analysis of how LLM per-token costs have dropped ~1000x and where the next wave of cost reductions will come from across the stack.

- [Rethinking Thinking Tokens: LLMs as Improvement Operators](https://arxiv.org/abs/2510.01123) - arXiv. Explores whether models can leverage metacognition to provide better reasoning without inflating context length and token costs.

- [Thinking Through How Pretraining vs RL Learn](https://www.dwarkesh.com/p/bits-per-sample) - Dwarkesh Patel. Analysis of how pretraining and reinforcement learning differ in their learning dynamics, with implications for RLVR progress.

- [Energy Use of AI Inference: Efficiency Pathways and Test-Time Compute](https://arxiv.org/abs/2509.20241) - arXiv. Analysis of per-query energy use in AI inference as scaling reaches billions of queries, providing estimates for capacity planning, emissions accounting, and efficiency prioritization.

- [Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Parallel Reasoning](https://openreview.net/forum?id=jAsr5GHt3P) - OpenReview. Analysis of how increasing inference-time computation through repeated sampling and adaptive parallel reasoning boosts LLM reasoning capabilities.

- [Souper Model: How Simple Arithmetic Unlocks State-of-the-Art LLM Performance](https://ai.meta.com/research/publications/souper-model-how-simple-arithmetic-unlocks-state-of-the-art-llm-performance/) - Meta AI. Research on how simple model merging arithmetic achieves state-of-the-art LLM performance.

#### Tier 3

- [Hyperparameters are all you need: Using five-step inference for an optimal diffusion schedule](https://zenodo.org/records/17180452) - Zenodo. Analysis of truncation error in diffusion ODE/SDE solvers and optimal inference scheduling with minimal hyperparameter tuning.

## 2. Inference Engines & Serving Systems

#### Tier 1

- [Understanding LLM Inference Engines: Inside Nano-vLLM (Part 1)](https://neutree.ai/blog/nano-vllm-part-1) - Neutree. Pedagogical walkthrough of LLM inference engine internals through a minimal vLLM reimplementation, covering scheduling, batching, and memory management.

- [vLLM Architectural Deep Dive](https://docs.google.com/presentation/d/1dMnZyXDff1zh1bfV0v5bku4e7iWBaOkaGNsegnpWe3w/edit?usp=sharing) - Ayush Satyam (Modus Labs). Presentation covering vLLM's architecture, high-throughput serving design, and key implementation decisions.

- [vLLM vs SGLang Benchmark Report](https://github.com/Cloud-Linuxer/qwen-8b/blob/main/BENCHMARK_REPORT_KR.md) - Cloud-Linuxer. Side-by-side performance comparison of vLLM and SGLang inference engines on Qwen-8B.

- [The Rise of vLLM: Building an Open Source LLM Inference Engine](https://www.youtube.com/watch?v=WLl8D1nyaW8) - Anyscale. Video on vLLM's evolution from research project to the dominant open-source LLM inference engine.

- [SGLang](https://www.sglang.io/) - SGLang. Official site for SGLang, a fast serving framework for large language and vision models with RadixAttention and structured generation.

- [SGLang: An Efficient Open-Source Framework for Large-Scale LLM Serving](https://www.youtube.com/watch?v=w9-AYqIhHRo) - Anyscale. Video presentation on SGLang's architecture and performance optimizations for large-scale LLM serving.

- [SGLang Cookbook](https://cookbook.sglang.io/docs/intro) - SGLang. Official cookbook with practical recipes and patterns for using SGLang in production.

- [mini-sglang](https://github.com/sgl-project/mini-sglang) - SGLang Project. Minimal reimplementation of SGLang for educational purposes, useful for understanding the core engine design.

#### Tier 2

- [vLLM-Style Fast Inference Engine: Building from Scratch on CPU](https://medium.com/@alaminibrahim433/vllm-style-fast-inference-engine-building-from-scratch-on-cpu-1f2a1f31f02a) - Al Amin Ibrahim. Hands-on guide to implementing PagedAttention and continuous batching from scratch, demystifying vLLM's core innovations.

- [Disaggregated Inference at Scale with PyTorch and vLLM](https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/) - PyTorch Blog. Explains how disaggregated inference separates prefill and decode stages for better resource utilization at scale.

- [Ray Serve: Reduce LLM Inference Latency by 60% with Custom Request Routing](https://www.anyscale.com/blog/ray-serve-faster-first-token-custom-routing) - Anyscale. Demonstrates prefix caching and cache-aware routing in Ray Serve for significant latency reduction in multi-turn LLM conversations.

- [vLLM Concurrency Demo](https://github.com/Regan-Milne/vllm-concurrency-demo) - Regan Milne. Single-GPU vLLM concurrency testing setup with Prometheus/Grafana monitoring on RTX 4090, useful for benchmarking serving performance.

- [vLLM - Why Requests Take Hours Under Load](https://blog.dotieuthien.com/posts/vllm) - dotieuthien. Analysis of why vLLM requests can take 2-3 hours under heavy load, diagnosing KV cache block exhaustion and queue starvation.

- [vLLM Semantic Router v0.1 Iris](https://blog.vllm.ai/2026/01/05/vllm-sr-iris.html) - vLLM Blog. System-level intelligence for Mixture-of-Models routing, combining model selection, safety filtering, semantic caching, and intelligent request routing.

- [vLLM KV Offloading Connector](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html) - vLLM Blog. Deep-dive into vLLM 0.11.0's KV cache offloading to CPU DRAM, covering host-to-device throughput optimization for improved inference throughput.

- [vLLM-Omni v0.12.0rc1](https://github.com/vllm-project/vllm-omni/releases/tag/v0.12.0rc1) - vLLM Project. Major release focused on multi-modal inference capabilities with 187 commits from 45 contributors.

- [vLLM Metal: Apple Silicon Plugin](https://github.com/vllm-project/vllm-metal) - vLLM Project. Community-maintained hardware plugin enabling vLLM on Apple Silicon GPUs.

- [vLLM Daily](https://github.com/vllm-project/vllm-daily) - vLLM Project. Daily summarization of merged PRs in the vLLM repository, useful for tracking development velocity and features.

- [MiMo-V2-Flash: Efficient Reasoning and Agentic Foundation Model](https://github.com/XiaomiMiMo/MiMo-V2-Flash) - Xiaomi. Efficient reasoning, coding, and agentic foundation model with [vLLM recipe](https://docs.vllm.ai/projects/recipes/en/latest/MiMo/MiMo-V2-Flash.html).

- [SGLang: Enable Return Routed Experts (PR #12162)](https://github.com/sgl-project/sglang/pull/12162) - ocss884. Feature enabling SGLang to return routed experts during forward pass for RL training integration, based on MiMo's R3 protocol.

- [optillm: Optimizing Inference Proxy for LLMs](https://github.com/algorithmicsuperintelligence/optillm) - Algorithmic Superintelligence. Inference optimization proxy that sits between clients and LLM endpoints for improved throughput and cost efficiency.

- [SGLang Diffusion: Accelerating Video and Image Generation](https://lmsys.org/blog/2025-11-07-sglang-diffusion/) - LMSYS Org. Bringing SGLang's state-of-the-art serving performance to diffusion model inference for image and video generation.

- [Turbocharging LinkedIn's Recommendation Systems with SGLang](https://www.linkedin.com/blog/engineering/ai/turbocharging-linkedins-recommendation-systems-with-sglang) - LinkedIn Engineering. How LinkedIn integrated SGLang to accelerate their recommendation systems at scale.

- [Advancing Low-Bit Quantization for LLMs: AutoRound x LLM Compressor](https://blog.vllm.ai/2025/12/09/intel-autoround-llmc.html) - vLLM Blog. Achieving faster, more efficient LLM serving with Intel's AutoRound and LLM Compressor integration.

- [Token-Level Truth: Real-Time Hallucination Detection (HaluGate)](https://blog.vllm.ai/2025/12/14/halugate.html) - vLLM Blog. Real-time extrinsic hallucination detection for production LLM systems at the token level.

- [NVIDIA Nemotron 3 Nano on vLLM](https://blog.vllm.ai/2025/12/15/run-nvidia-nemotron-3-nano.html) - vLLM Blog. Running highly efficient and accurate AI agents with NVIDIA Nemotron 3 Nano on vLLM.

- [AMD x vLLM Semantic Router](https://blog.vllm.ai/2025/12/16/vllm-sr-amd.html) - vLLM Blog. AMD's collaboration with vLLM to build system-level intelligence for LLM routing.

- [vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP](https://blog.vllm.ai/2025/12/17/large-scale-serving.html) - vLLM Blog. Achieving 2,200 tokens/s per H200 GPU serving DeepSeek with Wide Expert Parallelism at rack scale.

- [vLLM Semantic Router + NVIDIA Dynamo Integration Demo](https://www.youtube.com/watch?v=rRULSR9gTds) - Abdallah Samara. End-to-end demo of vLLM Semantic Router integrated with NVIDIA Dynamo for intelligent inference routing.

- [Intelligent LLM Inferencing via vLLM Semantic Router + LLM-D](https://youtu.be/dCxow80vgSc) - AI Cloud Clarity. Video on combining vLLM Semantic Router with LLM-D for intelligent large-scale LLM inference.

- [vLLM Router](https://github.com/vllm-project/router) - vLLM Project. High-performance, lightweight router for vLLM large-scale deployments.

- [vLLM-Omni Diffusion Acceleration](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion_acceleration) - vLLM. Guide to accelerating diffusion model inference with vLLM-Omni.

- [Awesome vLLM Plugins](https://github.com/BudEcosystem/Awesome-vLLM-plugins) - Bud Ecosystem. Curated list of plugins built on top of vLLM for extended functionality.

- [PowerInfer: High-Speed LLM Serving for Local Deployment](https://github.com/SJTU-IPADS/PowerInfer) - SJTU IPADS. Fast LLM inference engine optimized for consumer-grade GPUs through neuron-aware sparse computation.

- [Tokenflood: Load Testing for LLMs](https://github.com/twerkmeister/tokenflood) - twerkmeister. Load testing framework for simulating arbitrary loads on instruction-tuned LLMs.

- [LMCache: Efficient KV Cache Layer for Enterprise-Scale Inference](https://arxiv.org/abs/2510.09665) - arXiv. Moving KV caches outside GPU devices for cross-query and cross-engine cache reuse at enterprise scale.

- [Optimizing Inference with NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/) - Eduardo Alvarez, NVIDIA. Using FP4 quantization for KV cache to reduce memory footprint for long context and large batch inference.

- [Tensor Parallel (NanoVLLM)](https://liyuan24.github.io/writings/2025_12_18_nanovllm_tensor_parallel_kernel_fusion.html) - Liyuan. Tensor parallel implementation with kernel fusion in NanoVLLM, distributing model weights and KV cache across GPUs.

- [Prompt Caching](https://ngrok.com/blog/prompt-caching/) - ngrok. Practical guide to implementing prompt caching for reduced latency and cost in LLM applications.

- [vLLM-Omni: Omni-Modality Model Serving](https://github.com/vllm-project/vllm-omni) - vLLM Project. High-throughput and memory-efficient inference and serving engine for omni-modality models.

- [Announcing vLLM-Omni: Easy, Fast, and Cheap Omni-Modality Model Serving](https://blog.vllm.ai/2025/11/30/vllm-omni.html) - vLLM Blog. Official announcement of vLLM-Omni, a major extension of the vLLM ecosystem for next-generation omni-modality models.

- [FriendliAI Achieves 3x Faster Qwen3 235B Inference](https://friendli.ai/blog/Qwen3-235B-benchmark) - FriendliAI. Benchmark demonstrating up to 3x faster Qwen3-235B inference compared to standard vLLM through optimized MoE-aware infrastructure.

- [Together AI Delivers Fastest Inference for Top Open-Source Models](https://www.together.ai/blog/fastest-inference-for-the-top-open-source-models) - Together AI. Achieving up to 2x faster inference for Qwen, DeepSeek, and Kimi through GPU optimization, speculative decoding, and FP4 quantization on NVIDIA Blackwell.

- [vLLM Gaudi Documentation](http://vllm-gaudi.readthedocs.io/) - vLLM. Documentation for running vLLM on Intel Gaudi accelerators.

- [vLLM v0.12.0 Release](https://github.com/vllm-project/vllm/releases/tag/v0.12.0) - vLLM Project. Major release featuring 474 commits from 213 contributors, including PyTorch 2.9.0 upgrade, CUDA 12.9, and V0 deprecation.

- [DeepSeek-V3 Usage Tips on vLLM](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2.html#usage-tips) - vLLM. Practical usage tips for serving DeepSeek-V3 with vLLM.

- [SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications](https://suffix-decoding.github.io/) - NeurIPS 2025 Spotlight. Achieves up to 5.3x speedup in LLM inference for agentic applications using efficient suffix trees for training-free, CPU-based speculative drafting.

- [LMCache: Supercharge Your LLM with the Fastest KV Cache Layer](https://github.com/LMCache/LMCache) - LMCache. Open-source KV cache management library for accelerating LLM serving with cross-query and cross-engine cache reuse.

- [ArcticInference: vLLM Plugin for High-Throughput, Low-Latency Inference](https://github.com/snowflakedb/ArcticInference) - Snowflake. vLLM plugin optimized for high-throughput and low-latency inference workloads.

- [vLLM Production Stack: LLM Inference for Enterprises (Part 1)](https://cloudthrill.ca/vllm-production-stack-llm-inference-for-enterprises-p1) - CloudDude, Cloudthrill. Overview of the community-maintained vLLM production stack with Python-native router, LMCache-powered KV-cache network, autoscaling hooks, and Grafana dashboards.

- [vLLM Semantic Router: Improving Efficiency in AI Reasoning](https://developers.redhat.com/articles/2025/09/11/vllm-semantic-router-improving-efficiency-ai-reasoning) - Red Hat Developer. Open-source system for intelligent, cost-aware request routing that ensures every generated token adds value.

- [How to Make vLLM 13x Faster with LMCache + NVIDIA Dynamo](https://www.youtube.com/watch?v=iuoOpOQkURo) - Faradawn Yang. Hands-on tutorial demonstrating 13x vLLM performance improvement using LMCache and NVIDIA Dynamo.

- [Disaggregated Serving in TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.md) - NVIDIA. Technical deep-dive into disaggregated serving architecture in TensorRT-LLM, separating prefill and decode stages.

- [From Monolithic to Modular: Scaling Semantic Routing with Extensible Architecture](https://blog.vllm.ai/2025/10/27/semantic-router-modular.html) - vLLM Blog. How refactoring the vLLM Semantic Router's Rust-based classification layer addresses computational scaling challenges through extensible architecture.

- [vLLM Semantic Router](https://vllm-semantic-router.com/) - vLLM. Official site for the AI-powered intelligent Mixture-of-Models router with neural network processing.

- [Signal-Decision Driven Architecture: Reshaping Semantic Routing at Scale](https://blog.vllm.ai/2025/11/19/signal-decision.html) - vLLM Blog. Evolution of vLLM Semantic Router from classification-based to signal-decision driven architecture for production AI systems.

- [Cornserve: Easy, Fast, and Scalable Multimodal AI](https://github.com/cornserve-ai/cornserve) - Cornserve AI. Framework for easy, fast, and scalable multimodal AI serving.

- [Speculators: Speculative Decoding Library for vLLM](https://github.com/vllm-project/speculators) - vLLM Project. Unified library for building, evaluating, and storing speculative decoding algorithms for LLM inference in vLLM.

- [LPLB: MoE Load Balancer Based on Linear Programming](https://github.com/deepseek-ai/LPLB) - DeepSeek AI. Early research stage MoE load balancer using linear programming for efficient expert routing.

- [SGLang Development Roadmap 2025 Q4](https://github.com/sgl-project/sglang/issues/12780) - SGLang Project. Official development roadmap covering P/D disaggregation, feature compatibility, and reliability targets for Q4 2025.

- [DeepSpeed: Deep Learning Optimization Library](https://github.com/deepspeedai/DeepSpeed) - DeepSpeed AI. Deep learning optimization library that makes distributed training and inference easy, efficient, and effective.

- [Spec-Bench: Comprehensive Benchmark for Speculative Decoding](https://github.com/hemingkx/Spec-Bench) - hemingkx. Comprehensive benchmark and unified evaluation platform for speculative decoding methods. ACL 2024 Findings.

- [State of the Model Serving Communities — November 2025](https://inferenceops.substack.com/p/state-of-the-model-serving-communities-ea6) - InferenceOps. Monthly update on AI/ML model inference communities including contributions from Red Hat AI teams.

- [AutoRound Meets SGLang: Enabling Quantized Model Inference](https://lmsys.org/blog/2025-11-13-AutoRound/) - LMSYS Org. Official collaboration between SGLang and AutoRound enabling low-bit quantization for efficient LLM inference through signed-gradient optimization.

- [Building Clean, Maintainable vLLM Modifications Using the Plugin System](https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html) - vLLM Blog. Guide to building clean and maintainable vLLM modifications using the official plugin system.

- [Speculators: Standardized, Production-Ready Speculative Decoding](https://developers.redhat.com/articles/2025/11/19/speculators-standardized-production-ready-speculative-decoding) - Red Hat Developer. How the speculators library standardizes speculative decoding for production deployment.

- [Streamlined Multi-Node Serving with Ray Symmetric-Run](https://blog.vllm.ai/2025/11/22/ray-symmetric-run.html) - vLLM Blog. New Ray command for launching vLLM servers on every node in a cluster, simplifying multi-node model serving on HPC setups.

- [Introducing Serverless LoRA Inference](https://wandb.ai/wandb_fc/product-announcements-fc/reports/Introducing-Serverless-LoRA-Inference--VmlldzoxNTEyNDU4OQ) - W&B. Bring-your-own-LoRA serving for fine-tuned models on Weights & Biases Inference.

#### Tier 3

- [llm-d Architecture](https://llm-d.ai/docs/architecture) - llm-d. Overview of the llm-d distributed inference architecture, covering component design and system topology for large-scale LLM serving.

- [TensorRT-LLM: Combining Guided Decoding and Speculative Decoding](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs%2Fsource%2Fblogs%2Ftech_blog%2Fblog12_Combining_Guided_Decoding_and_Speculative_Decoding.md) - NVIDIA. Technical deep-dive into combining structured output generation with speculative decoding for faster constrained inference.

- [One Token to Corrupt Them All: A vLLM Debugging Tale](https://www.ai21.com/blog/vllm-debugging-mamba-bug/) - AI21. Deep-dive into debugging a critical vLLM state corruption bug in Mamba architectures, covering scheduler and memory management internals.

- [Inference OSS Ecosystem featuring vLLM](https://www.nvidia.com/en-us/on-demand/session/other25-dynamoday06/?playlistId=playList-e42aee58-4db9-4ce4-8a6f-c41d8e272d72) - NVIDIA. Session on large-scale LLM serving with vLLM, covering disaggregated inference, Wide-EP for sparse MoE models, and rack-scale deployments on GB200.

- [Inferact](https://inferact.ai/) - Founded by vLLM creators and core maintainers. Company building on vLLM as the world's AI inference engine.

- [SGLang: Add Suffix Decoding Speculative Algorithm (PR #13553)](https://github.com/sgl-project/sglang/pull/13553) - adityakamat2. Implementation of suffix decoding in SGLang for training-free speculative decoding with CPU-based drafting for structured output throughput improvement.

- [GPU Model Runner V2 (vLLM PR #25266)](https://github.com/vllm-project/vllm/pull/25266/files) - WoosukKwon. Major vLLM refactor removing persistent batch, using NumPy arrays for CPU states, and simplifying pre-/post-processing for the GPU model runner.

- [SGLang RFC: Block Diffusion Large Language Model (dLLM) Framework](https://github.com/sgl-project/sglang/pull/12766) - SGLang Project. RFC for supporting diffusion language models (dLLMs) in SGLang, following LLaDA's debut as the first diffusion language model.

## 3. Attention Mechanisms & Memory Optimization

#### Tier 1

- [Paged Attention from First Principles: A View Inside vLLM](https://hamzaelshafie.bearblog.dev/paged-attention-from-first-principles-a-view-inside-vllm/) - Hamza El Shafie. Ground-up explanation of PagedAttention, the virtual memory-inspired technique that enables efficient KV cache management in LLM serving.

- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/) - EleutherAI. Explanation of Rotary Positional Embedding (RoPE), which unifies absolute and relative position encoding approaches used in most modern LLMs.

- [Attention Normalizes the Wrong Norm](https://convergentthinking.sh/posts/attention-normalizes-the-wrong-norm/) - Convergent Thinking. Analysis of why softmax constrains the L1 norm to 1 when it should constrain the L2 norm, with implications for attention mechanism design.

- [LLM Optimization Lecture 4: Grouped Query Attention, Paged Attention](https://youtu.be/Myhz-whrq5Q?si=R4QXM0RzDfaAmAk4) - Faradawn Yang. Video lecture covering Grouped Query Attention and Paged Attention mechanisms for efficient LLM inference.

- [How Attention Got So Efficient: GQA, MLA, DSA](https://youtu.be/Y-o545eYjXM?si=ERqziyDvW5Bm3uJ4) - Jia-Bin Huang. Video explaining the evolution of efficient attention mechanisms from Grouped Query Attention to Multi-Latent Attention and DeepSeek Attention.

- [The Q, K, V Matrices](https://arpitbhayani.me/blogs/qkv-matrices) - Arpit Bhayani. Ground-up construction of Query, Key, and Value matrices at the core of the attention mechanism in transformers.

#### Tier 2

- [Long Context Attention](https://nrehiew.github.io/blog/long_context/) - nrehiew. Analysis of attention mechanisms for long-context scenarios, covering the computational and memory challenges of scaling sequence length.

- [QSInference: Fast and Memory Efficient Sparse Attention](https://github.com/yogeshsinghrbt/QSInference) - Yogesh Singh. Library for fast and memory-efficient sparse attention computation during inference.

- [Flash Attention from Scratch Part 4: Bank Conflicts & Swizzling](https://lubits.ch/flash/Part-4) - Sonny. Detailed walkthrough of GPU memory bank conflicts and swizzling techniques in Flash Attention kernel implementation.

- [Triton Flash Attention Kernel Walkthrough](https://nathanchen.me/public/Triton-Flash-Attention-Kernel-Walkthrough.html) - Nathan Chen. Step-by-step analysis of a Flash Attention kernel written in Triton, connecting high-level API to low-level GPU operations.

- [CUDA: Add GQA Ratio 4 for GLM 4.7 Flash (llama.cpp PR #18953)](https://github.com/ggml-org/llama.cpp/pull/18953) - am17an. Enabling Flash Attention for GLM 4.7 in llama.cpp with GQA ratio 4 support.

- [Autocomp Trainium Attention](https://charleshong3.github.io/blog/autocomp_trainium_attention.html) - Charles Hong. Attention kernel implementation on AWS Trainium, covering custom attention computation on non-NVIDIA hardware.

- [REFORM: A New Approach for Reasoning AI to Handle Ultra-Long Inputs](https://www.youtube.com/watch?v=pfLHtbT6cO4) - AER Labs. Video on REFORM, a method enabling reasoning AI models to process extremely long input sequences efficiently.

- [How to Reduce KV Cache Bottlenecks with NVIDIA Dynamo](https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/) - Amr Elmeleegy, NVIDIA. Techniques for reducing KV cache bottlenecks in LLM inference using NVIDIA Dynamo's intelligent caching and routing.

- [A User's Guide to FlexAttention in FlashAttention CuTe DSL](https://research.colfax-intl.com/a-users-guide-to-flexattention-in-flash-attention-cute-dsl/) - Reuben Stern, Colfax Research. Guide to implementing attention variants (causal, sliding window, etc.) using FlexAttention in FlashAttention's CuTe DSL.

- [Solve the GPU Cost Crisis with kvcached](https://yifanqiao.notion.site/Solve-the-GPU-Cost-Crisis-with-kvcached-289da9d1f4d68034b17bf2774201b141) - Yifan Qiao. How virtualized, elastic KV cache enables LLM serving on shared GPUs, reducing GPU costs.

- [PrisKV: A Colocated Tiered KVCache Store for LLM Serving](https://aibrix.github.io/posts/2025-11-26-priskv-intro/) - AIBrix. Tiered KV cache store architecture for LLM serving, addressing memory pressure through colocated multi-tier caching.

- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) - FLA Org. Efficient implementations of state-of-the-art linear attention models for fast sequence modeling.

- [Diffusers Attention Backends](https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends) - HuggingFace. Guide to attention backend options in the Diffusers library for optimizing diffusion model inference.

#### Tier 3

- [End-to-End Test-Time Training (TTT-E2E)](https://x.com/i/status/2009187297137446959) - Stanford, NVIDIA, UC Berkeley, Astera Institute. Method for compressing long contexts into weights, eliminating KV cache dependency for continuously learning LLMs.

- [DroPE: Extending Context by Dropping Positional Embeddings](https://pub.sakana.ai/DroPE/) - Sakana AI. Simple method for extending pretrained LLM context length without long-context fine-tuning by selectively dropping positional embeddings.

- [PQCache: Product Quantization for KV Cache Compression](https://sky-light.eecs.berkeley.edu/#/blog/pqcache) - UC Berkeley. Using product quantization to compress KV cache for memory-efficient long-context inference.

- [kvcached: Virtualized Elastic KV Cache for Dynamic GPU Sharing](https://github.com/ovg-project/kvcached) - OVG Project. Virtualized KV cache management enabling dynamic GPU sharing and elastic memory allocation across inference workloads.

- [Optimizing Long-Context Prefill on Multiple Older-Generation GPU Nodes](https://moreh.io/blog/optimizing-long-context-prefill-on-multiple-older-generation-gpu-nodes-251226/) - Moreh. Techniques for efficient long-context prefill computation distributed across older GPU hardware.

- [Cross-GPU KV Cache Marketplace](https://github.com/neelsomani/kv-marketplace) - Neel Somani. Cross-GPU KV cache marketplace enabling efficient KV cache sharing and trading across GPU devices.

- [Analog In-Memory Computing Attention Mechanism for Fast and Energy-Efficient Inference](https://www.nature.com/articles/s43588-025-00854-1) - Nature Computational Science. Leveraging in-memory computing with emerging gain-cell devices to accelerate the attention mechanism in large language models.

## 4. Quantization & Model Compression

#### Tier 1

- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization) - Maarten Grootendorst. Accessible visual introduction to quantization methods for reducing LLM memory footprint and compute requirements.

- [Product Quantization: Compressing High-Dimensional Vectors by 97%](https://www.pinecone.io/learn/series/faiss/product-quantization/) - Pinecone. How product quantization dramatically compresses high-dimensional vectors for 97% less memory and 5.5x faster nearest-neighbor search.

#### Tier 2

- [MS-AMP: Microsoft Automatic Mixed Precision Library](https://github.com/Azure/MS-AMP/tree/main) - Microsoft Azure. Automatic mixed precision library for efficient training and inference with advanced precision management.

- [torchao: Quantized Models and Quantization Recipes on HuggingFace Hub](https://pytorch.org/blog/torchao-quantized-models-and-quantization-recipes-now-available-on-huggingface-hub/) - PyTorch Blog. Official guide to PyTorch-native quantization workflows using torchao, with pre-quantized model availability on HuggingFace.

- [Quantization: CUDA vs Triton](https://www.dropbox.com/scl/fi/hzfx1l267m8gwyhcjvfk4/Quantization-Cuda-vs-Triton.pdf?rlkey=s4j64ivi2kpp2l0uq8xjdwbab&e=1&dl=0) - Comparison of CUDA and Triton implementations for quantization kernels, covering performance trade-offs and implementation strategies.

- [PyTorch ao MX Kernels (PTX)](https://github.com/pytorch/ao/blob/18dbe875a0ce279739dda06fda656e76845acaac/torchao/csrc/cuda/mx_kernels/ptx.cuh#L73) - PyTorch. Reference implementation of microscaling (MX) format kernels in PTX, showing low-level CUDA intrinsics for mixed-precision compute.

- [FlashInfer: Support for Advanced Quantization (HQQ)](https://github.com/flashinfer-ai/flashinfer/issues/2423) - FlashInfer. Discussion on extending FlashInfer's FP4 quantization to support HQQ and other advanced quantization algorithms beyond max-based scaling.

- [8-bit Rotational Quantization: Compress Vectors by 4x](https://weaviate.io/blog/8-bit-rotational-quantization) - Weaviate. Novel vector quantization algorithm using random rotations to improve the speed-quality tradeoff of vector search.

- [SmolLM-Smashed: Tiny Giants, Optimized for Speed](https://huggingface.co/blog/PrunaAI/smollm-tiny-giants-optimized-for-speed) - Pruna AI. Optimization techniques applied to small language models for maximum inference throughput.

- [Post Training Quantization](https://liyuan24.github.io/writings/2026_01_06_post_training_quantization.html) - Liyuan. Detailed explanation of post-training quantization techniques for compressing model weights without backpropagation.

- [Future Leakage in Block-Quantized Attention](https://matx.com/research/leaky_quantization) - MatX. Analysis of how block quantization in attention introduces future information leakage, and implications for quantized inference accuracy.

- [Why Stochastic Rounding is Essential for Modern Generative AI](https://cloud.google.com/blog/topics/developers-practitioners/why-stochastic-rounding-is-essential-for-modern-generative-ai?hl=en) - Google Cloud. Explains why stochastic rounding is critical for maintaining model quality in low-precision training and inference.

- [LLM Pruning Collection](https://github.com/zlab-princeton/llm-pruning-collection) - Princeton zLab. Collection of LLM pruning implementations, training code for GPUs & TPUs, and evaluation scripts.

- [torchao Float8 Training](https://github.com/pytorch/ao/blob/main/torchao%2Ffloat8%2FREADME.md) - PyTorch. Guide to FP8 training and inference with torchao, covering float8 precision for accelerated transformer workloads.

- [NVIDIA TransformerEngine](https://github.com/NVIDIA/TransformerEngine) - NVIDIA. Library for accelerating Transformer models using FP8 and FP4 precision on Hopper, Ada, and Blackwell GPUs.

- [Gemlite: Custom Low-Bit Fused CUDA Kernels](https://dropbox.github.io/gemlite_blogpost/) - Dropbox. CUDA kernels for building custom low-bit fused GEMV operations with sparse bitpacking and dequantization.

- [Survey of Quantization Formats](https://github.com/vipulSharma18/Survey-of-Quantization-Formats) - Vipul Sharma. Survey of modern quantization formats (MXFP8, NVFP4) and inference optimization tools (TorchAO, GemLite) with benchmarking results on RTX 4090.

- [INT vs FP: Comparing Low-Bit Integer and Float-Point Formats](https://github.com/ChenMnZ/INT_vs_FP) - ChenMnZ. Framework for comparing low-bit integer and floating-point quantization formats for LLM inference.

- [GPTQModel: LLM Quantization Toolkit](https://github.com/ModelCloud/GPTQModel) - ModelCloud. LLM model quantization toolkit with hardware acceleration support for NVIDIA CUDA, AMD ROCm, Intel XPU, and CPU via HuggingFace, vLLM, and SGLang.

#### Tier 3

- [AirLLM: 70B Inference with Single 4GB GPU](https://github.com/lyogavin/airllm) - lyogavin. Library enabling inference of 70B-parameter models on a single 4GB GPU through layer-wise loading and quantization techniques.

- [BitNet: Official Inference Framework for 1-bit LLMs](https://github.com/microsoft/BitNet) - Microsoft. Official framework for running 1-bit quantized LLMs, pushing the boundary of extreme compression for inference.

## 5. CUDA & GPU Kernel Programming

#### Tier 1

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM) - Simon Boehm. Iterative optimization of a CUDA matrix multiplication kernel, progressing from naive to near-cuBLAS performance with detailed explanations at each step.

- [Were RNNs All We Needed? A GPU Programming Perspective](https://dhruvmsheth.github.io/projects/gpu_pogramming_curnn/) - Dhruv Sheth. CUDA implementation of parallelizable GRUs and LSTMs, bridging classical sequence models with modern GPU programming techniques.

- [sparse-llm.c: LLM Training in Raw C/CUDA](https://github.com/WilliamZhang20/sparse-llm.c) - William Zhang. Minimal LLM training implementation in plain C and CUDA without frameworks, ideal for understanding low-level training mechanics.

- [Blocks, Threads, and Kernels: A Deeper Dive](https://vanshnawander.github.io/vansh/posts/blocks-threads-kernels.html) - Vansh. Foundational explanation of CUDA's execution model covering thread hierarchy, block organization, and kernel launch mechanics.

- [Intro to GPUs For the Researcher](https://hackbot.dad/writing/intro-to-gpus/) - Shane Caldwell. Practical guide to getting comfortable with GPU hardware, focused on achieving higher MFU (Model FLOPS Utilization).

- [Ten Years Later: Why CUDA Succeeded](https://parallelprogrammer.substack.com/p/ten-years-later-why-cuda-succeeded) - Parallel Programmer. Retrospective on CUDA's rise to dominance in GPU computing, covering the ecosystem and design decisions that drove adoption.

- [Inside the GPU SM: Understanding CUDA Thread Execution](https://medium.com/@bethe1tweets/inside-the-gpu-sm-understanding-cuda-thread-execution-9caf9ef9ffd8) - Medium. Explanation of streaming multiprocessor internals and how CUDA threads are actually executed on GPU hardware.

- [CPU-GPU Synchronization](https://tomasruizt.github.io/posts/08_cpu_gpu_synchronization/) - Tomas Ruiz. Guide to understanding and managing CPU-GPU synchronization, a common source of performance bottlenecks.

- [GPU Architecture Deep Dive: From HBM to Tensor Cores](https://www.youtube.com/watch?v=5UWphJWdAHY) - Parallel Routines. Visual explanation of GPU architecture from memory hierarchy (HBM, L2, shared memory) through tensor core operations.

- [How to Think About GPUs](https://jax-ml.github.io/scaling-book/gpus/) - JAX Scaling Book. Deep-dive chapter on GPU architecture — how each chip works, how they're networked, and what it means for LLMs, with NVIDIA GPU focus.

- [NVIDIA Parallel Thread Execution (PTX) ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) - NVIDIA. Official documentation for NVIDIA's Parallel Thread Execution ISA, the low-level virtual instruction set for CUDA GPU programming.

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) - NVIDIA. Official CUDA programming guide covering the CUDA programming model, API reference, and best practices for GPU kernel development.

- [Reduction (Sum) Series: Part 1 — Introduction](https://kathsucurry.github.io/cuda/2025/10/14/reduction_sum_part1.html) - kathsucurry. Introductory guide to implementing parallel sum reduction using CUDA, covering key concepts and kernel design.

#### Tier 2

- [fast.cu: Fastest Kernels Written from Scratch](https://github.com/pranjalssh/fast.cu) - Pranjal. Collection of performance-optimized CUDA kernels written from scratch, useful as reference implementations for kernel engineering.

- [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog) - CUDA for Fun. Detailed worklog of writing a CUDA matmul kernel that exceeds cuBLAS performance on H100, covering Hopper-specific optimizations.

- [Unweaving Warp Specialization](https://rohany.github.io/blog/warp-specialization/) - Rohan Yadav. Deep explanation of warp specialization techniques in CUDA kernels, covering how to assign different roles to warps for improved throughput.

- [Processing Strings 109x Faster than Nvidia on H100](https://ashvardanian.com/posts/stringwars-on-gpus/) - Ash Vardanian. Deep-dive into GPU string processing with StringZilla v4, demonstrating CUDA kernel design for non-numeric workloads.

- [CUTLASS Tutorial: Fast Matrix-Multiplication with WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) - Colfax Research. Tutorial on implementing high-performance GEMM using CUTLASS and WGMMA instructions on NVIDIA Hopper architecture.

- [Categorical Foundations for CuTe Layouts](https://research.colfax-intl.com/categorical-foundations-for-cute-layouts/) - Jay, Colfax Research. Mathematical foundations of CuTe's layout system using category theory, explaining how multi-dimensional data maps to linear GPU memory.

- [NVIDIA GEMM Optimization Notes](https://arseniivanov.github.io/blog.html#nvidia-gemm) - Arsenii Ivanov. Notes on NVIDIA GEMM optimization techniques and performance considerations.

- [HAT MatMul: GPU Matrix Multiplication via OpenJDK Babylon](https://openjdk.org/projects/babylon/articles/hat-matmul/hat-matmul) - OpenJDK. Exploration of GPU-accelerated matrix multiplication through the HAT (Heterogeneous Accelerator Toolkit) project in Project Babylon.

- [Agent-Assisted Kernel Optimization: From PyTorch to Hand-Written Assembly](https://www.wafer.ai/blog/topk-sigmoid-optimization) - Wafer AI. Using an AI agent with ISA analysis tools to achieve 10x speedup on AMD MI300X, compressing weeks of expert kernel optimization work.

- [magnetron: A Homemade PyTorch from Scratch](https://github.com/MarioSieg/magnetron/blob/d3ff3f5c50dbace90adf24e583f6d13a0ac8ee11/magnetron/magnetron_cpu_blas.inl#L3316) - Mario Sieg. WIP PyTorch reimplementation from scratch including CPU BLAS kernels, useful for understanding tensor library internals.

- [Blackwell Pipelining with CuTeDSL](https://veitner.bearblog.dev/blackwell-pipelining-with-cutedsl/) - Simon. Overlapping workloads on Blackwell GPUs using CuTeDSL's asynchronous pipeline primitives for maximum throughput.

- [Effective Transpose on Hopper GPU](https://github.com/simveit/effective_transpose) - simveit. Optimized matrix transpose implementation targeting NVIDIA Hopper architecture.

- [Numerics in World Models](https://0x00b1.github.io/blog/2025/12/25/numerics-in-world-models/) - Analysis of numerical precision considerations in world model implementations and their impact on model behavior.

- [Learn by Doing: TorchInductor Reduction Kernels](https://karthick.ai/blog/2025/Learn-By-Doing-Torchinductor-Reduction/) - Karthick Panner. Hands-on walkthrough of TorchInductor's reduction kernel generation pipeline.

- [Inside NVIDIA GPUs: Anatomy of High-Performance Matmul Kernels](https://www.aleksagordic.com/blog/matmul) - Aleksa Gordic. From GPU architecture and PTX/SASS to warp-tiling and deep asynchronous tensor core pipelines.

- [Low Latency Communication Kernels with NVSHMEM](https://www.youtube.com/live/rZBF2PuycLQ) - GPU MODE. Lecture on using NVSHMEM for low-latency GPU-to-GPU communication in distributed workloads.

- [NVIDIA CUDA Tile](https://developer.nvidia.com/cuda/tile) - NVIDIA. A tile-based GPU programming model targeting portability for NVIDIA Tensor Cores, representing the largest CUDA advancement since 2006.

- [Focus on Your Algorithm — NVIDIA CUDA Tile Handles the Hardware](https://developer.nvidia.com/blog/focus-on-your-algorithm-nvidia-cuda-tile-handles-the-hardware) - Jonathan Bentz, NVIDIA. Introduction to CUDA Tile in CUDA 13.1, a virtual instruction set for tile-based programming that abstracts hardware complexity.

- [Blackwell NVFP4 Kernel Hackathon Journey](https://yue-zhang-2025.github.io/2025/12/02/blackwell-nvfp4-kernel-hackathon-journey.html) - Yue Zhang. Personal account of optimizing NVFP4 kernels for NVIDIA Blackwell GPUs during a kernel hackathon.

- [Tracing Hanging and Complicated GPU Kernels Down to the Source Code](https://blog.vllm.ai/2025/12/03/improved-cuda-debugging.html) - vLLM Blog. Advanced CUDA debugging techniques for tracing hanging kernels back to source code, building on earlier CUDA core dump work.

- [Automating Algorithm Discovery: A Case Study in Kernel Generation](https://adrs-ucb.notion.site/datadog) - UC Berkeley / Datadog. Case study on using automated approaches for GPU kernel algorithm discovery and generation.

- [AMD GPU Debugging](https://thegeeko.me/blog/amd-gpu-debugging/) - thegeeko. Guide to debugging GPU programs on AMD hardware, covering ROCm debugging tools and techniques.

- [Reduction (Sum) Series: Part 2 — Implementation](https://kathsucurry.github.io/cuda/2025/10/27/reduction_sum_part2.html) - kathsucurry. Hands-on process for implementing and optimizing sum reduction kernels, based on Mark Harris's optimization techniques.

- [The GPU Observability Gap: Why We Need eBPF on GPU Devices](https://eunomia.dev/blog/2025/10/14/the-gpu-observability-gap-why-we-need-ebpf-on-gpu-devices/) - Eunomia. Analysis of the observability gap in GPU workloads and how eBPF can provide kernel-level visibility for GPU device monitoring.

- [Demystifying Numeric Conversions in CuTeDSL](https://veitner.bearblog.dev/demystifying-numeric-conversions-in-cutedsl/) - Simon. Guide to numeric type conversion primitives in CuTeDSL, covering the conversion hierarchy from high-level to low-level types.

- [CUTLASS Python DSL: Compile with TVM FFI](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/compile_with_tvm_ffi.html) - NVIDIA. Documentation on compiling CUTLASS Python DSL kernels using TVM FFI for deployment.

#### Tier 3

- [Rust GPU: The Future of GPU Programming](https://rust-gpu.github.io/) - Rust GPU Project. Toolchain for writing GPU shaders and compute kernels in Rust, offering memory safety guarantees for GPU code.

- [rust-cuda: Ecosystem for GPU Code in Rust](https://github.com/Rust-GPU/rust-cuda) - Rust GPU. Libraries and tools for writing and executing fast GPU code fully in Rust, providing an alternative to CUDA C++.

- [Stanford CS149 Assignment 5: Kernels](https://github.com/stanford-cs149/asst5-kernels) - Stanford. Course assignment on GPU kernel programming, useful for structured hands-on learning.

- [B200 Blockscaled GEMM: The Setup](https://veitner.bearblog.dev/b200-blockscaled-gemm-the-setup/) - Simon. Analysis of blockscaled GEMM kernel setup on NVIDIA B200, covering layout calculations, MMA operation setup, and kernel initialization.

- [cuTile Python](https://github.com/PeaBrane/cutile-python) - PeaBrane. Python bindings for cuTile, a programming model for writing parallel kernels for NVIDIA GPUs.

- [Inside VOLT: Designing an Open-Source GPU Compiler](https://arxiv.org/abs/2511.13751) - arXiv. Design of VOLT, an open-source GPU compiler for emerging open GPU architectures with custom ISAs, addressing the gap in open SIMT compiler infrastructure.

## 6. Structured Output & Guided Decoding

#### Tier 1

- [Guided Decoding Performance on vLLM and SGLang](https://blog.squeezebits.com/70642) - SqueezeBits. Comprehensive benchmark comparing XGrammar and LLGuidance guided decoding backends across vLLM and SGLang, with practical setup recommendations.

#### Tier 2

- [TensorRT-LLM: Combining Guided Decoding and Speculative Decoding](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs%2Fsource%2Fblogs%2Ftech_blog%2Fblog12_Combining_Guided_Decoding_and_Speculative_Decoding.md) - NVIDIA. Explores the intersection of structured output generation and speculative decoding for faster constrained inference in TensorRT-LLM.

- [TSON: Token-Efficient Structured Object Notation for LLMs](https://github.com/zenoaihq/tson) - Zeno AI. MIT-licensed token-efficient structured object notation designed to reduce token usage when generating structured data with LLMs.

- [Tool Calling in Inference](https://www.baseten.co/blog/tool-calling-in-inference/) - Baseten. Deep-dive into tool calling basics, why quality varies between providers, and how to build reliable and scalable tool calling for LLMs.

## 7. Distributed & Multi-GPU Inference

#### Tier 1

- [Meta AI Infrastructure Overview](https://iodized-hawthorn-94a.notion.site/Meta-AI-Infrastructure-Overview-1-27754c8e1f0a80359634c2e3c47d9e77) - Overview of Meta's AI infrastructure stack, covering GPU clusters, networking, and the systems powering large-scale model training and inference.

- [From Single GPU to Clusters: A Practical Journey into Distributed Training](https://debnsuma.github.io/my-blog/posts/distributed-training-from-scratch/) - Debnsuma. Hands-on guide to distributed training, breaking down core concepts and techniques for scaling deep learning across multiple GPUs and machines using PyTorch and Ray.

#### Tier 2

- [Visualizing Parallelism in Transformer](https://ailzhang.github.io/posts/distributed-compute-in-transformer/) - Ailing Zhang. Visual guide extending the JAX Scaling Book's "Transformer Accounting" diagram to multi-device parallelism, making tensor/pipeline/data parallelism intuitive.

- [RoCEv2 for Deep Learning](https://iodized-hawthorn-94a.notion.site/RoCEv2-26954c8e1f0a80b78bf1c6adc583e670) - Introduction to RDMA over Converged Ethernet v2 and its role in high-bandwidth GPU-to-GPU communication for distributed deep learning.

- [Distributed Inference on Heterogeneous Accelerators](https://moreh.io/blog/distributed-inference-on-heterogeneous-accelerators-including-gpus-rubin-cpx-and-ai-accelerators-250923/) - Moreh. MoAI Inference Framework for automatic distributed inference across heterogeneous hardware including AMD MI300X, MI308X, and NVIDIA Rubin CPX.

- [UCCL: Efficient Communication Library for GPUs](https://github.com/uccl-project/uccl) - UCCL Project. GPU communication library covering collectives, P2P (KV cache transfer, RL weight transfer), and expert parallelism with GPU-driven operations.

- [Parallel CPU-GPU Execution for LLM Inference on Constrained GPUs](https://arxiv.org/abs/2506.03296) - arXiv. Hybrid GPU-CPU execution for LLM inference that offloads KV cache and attention computation to CPU, addressing GPU memory constraints during autoregressive decoding.

- [DeepSeek R1 671B on AMD MI300X GPUs: Maximum Throughput](https://docs.moreh.io/benchmarking/deepseek_r1_671b_on_amd_mi300x_gpus_maximum_throughput/) - Moreh. Performance evaluation of DeepSeek R1 671B inference across 40 AMD MI300X GPUs (5 servers).

- [Learning to Love Mesh-Oriented Sharding](https://blog.ezyang.com/2025/12/learning-to-love-mesh-oriented-sharding/) - Edward Z. Yang. Deep explanation of mesh-oriented sharding for distributed tensor computation, covering the mental model shift from manual to mesh-based parallelism.

- [An Open Source AI Compute Stack: Kubernetes + Ray + PyTorch + vLLM](https://www.youtube.com/watch?v=4o2amJxMHUc) - CNCF. Video on building an open-source AI compute stack combining Kubernetes, Ray, PyTorch, and vLLM for production inference.

- [SkyPilot + NVIDIA Dynamo](https://github.com/skypilot-org/skypilot/tree/master/examples/serve/nvidia-dynamo) - SkyPilot. Example integration of SkyPilot with NVIDIA Dynamo for managing and scaling AI inference workloads across clouds.

- [Ray on TPUs with GKE: A More Native Experience](https://cloud.google.com/blog/products/containers-kubernetes/ray-on-tpus-with-gke-a-more-native-experience) - Google Cloud. New Ray on GKE features including label-based scheduling, atomic slice reservations, JaxTrainer, and built-in TPU awareness.

- [Disaggregated Inference: 18 Months Later](https://hao-ai-lab.github.io/blogs/distserve-retro/) - Hao AI Lab. Retrospective on disaggregated inference 18 months after DistServe, showing how splitting prefill and decode across separate compute pools became standard in NVIDIA Dynamo, llm-d, SGLang, vLLM, and more.

- [Power Up FSDP2 as a Flexible Training Backend for Miles](https://lmsys.org/blog/2025-12-03-miles-fsdp/) - LMSYS Org. Adding FSDP to Miles as a more flexible training backend for large-scale model training.

- [Shift Parallelism: Low-Latency, High-Throughput LLM Inference](https://arxiv.org/abs/2509.16495) - arXiv. Novel parallelism strategy combining benefits of tensor parallelism (low latency) and data parallelism (high throughput) for efficient LLM inference.

- [The vLLM MoE Playbook: A Practical Guide to TP, DP, PP and Expert Parallelism](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html) - ROCm Blogs. Practical guide to combining tensor, data, pipeline, and expert parallelism for MoE models on vLLM deployments.

#### Tier 3

- [How To Scale Your Model](https://jax-ml.github.io/scaling-book/) - JAX Team. Comprehensive book covering TPU/GPU architecture, inter-device communication, and parallelism strategies for training and inference at scale.

## 8. Post-Training & Fine-Tuning

#### Tier 1

- [Post-training 101](https://tokens-for-thoughts.notion.site/post-training-101) - Han Fang, Karthik A Sankararaman. Hitchhiker's guide to LLM post-training covering RLHF, DPO, and modern alignment techniques.

#### Tier 2

- [Self-Supervised Reinforcement Learning and Patterns in Time](https://www.youtube.com/watch?v=uU2fpNjJJBU) - Benjamin. Video lecture on self-supervised RL approaches and temporal pattern recognition, connecting reinforcement learning with representation learning.

- [Compute as Teacher: Turning Inference Compute Into Reference-Free Supervision](https://arxiv.org/abs/2509.14234) - arXiv. Proposes CaT, which converts a model's own exploration during inference into self-supervision by synthesizing references from parallel rollouts.

- [Shaping Capabilities with Token-Level Pretraining Data Filtering](https://github.com/neilrathi/token-filtering) - Neil Rathi. Research on token-level data filtering during pretraining to selectively shape model capabilities.

- [LLaMA-Factory: Unified Efficient Fine-Tuning of 100+ LLMs & VLMs](https://github.com/hiyouga/LLaMA-Factory) - hiyouga. Production-ready framework for fine-tuning over 100 language and vision models with LoRA, QLoRA, and full fine-tuning support. ACL 2024.

- [Mistral v0.3 (7B) Continued Pre-Training with Unsloth](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-CPT.ipynb) - Unsloth. Google Colab notebook demonstrating continued pre-training of Mistral 7B with Unsloth's optimized training pipeline.

- [Optimizing Large-Scale Pretraining at Character.ai (Squinch)](https://blog.character.ai/squinch/) - Character.AI. Techniques from Noam Shazeer's team for making large-scale transformer training faster and more efficient, now shared publicly.

- [Selective Gradient Masking](https://alignment.anthropic.com/2025/selective-gradient-masking/) - Anthropic. Technique for selectively masking gradients during training for improved alignment and capability control.

- [SAPO: Qwen Post-Training](https://qwen.ai/blog?id=sapo) - Qwen. Qwen's approach to post-training optimization for improved model quality and alignment.

- [Determinism and Scalability in Post-Training RL Systems](https://www.youtube.com/watch?v=3tlVeXp5xe8) - Ethan Su, AER Labs. Video on achieving deterministic and scalable reinforcement learning for post-training.

- [Code Walkthrough of AReaL](https://www.linkedin.com/pulse/code-walkthrough-areal-chenyang-zhao-barrc) - Chenyang Zhao. Detailed code walkthrough of AReaL, praised as having the most artistic code in the RL infrastructure community.

- [GPT OSS (20B) 500K Context Fine-Tuning with Unsloth](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt_oss_(20B)_500K_Context_Fine_tuning.ipynb) - Unsloth. Colab notebook for fine-tuning a 20B model with 500K context length using Unsloth's optimizations.

- [NVIDIA NeMo Emerging Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers) - NVIDIA NeMo. Collection of emerging optimizer implementations for efficient large-scale model training.

- [Efficient MoE Pre-Training at Scale with torchtitan](https://pytorch.org/blog/efficient-moe-pre-training-at-scale-with-torchtitan/) - PyTorch Blog. Guide to efficient Mixture-of-Experts pre-training using the torchtitan framework.

- [GRPO Fine-Tuning for Ministral3-VL](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/grpo_ministral3_vl.ipynb) - HuggingFace TRL. Colab notebook demonstrating Group Relative Policy Optimization fine-tuning for Ministral3 vision-language model.

- [SFT Fine-Tuning for Ministral3-VL](https://colab.research.google.com/github/huggingface/trl/blob/main/examples/notebooks/sft_ministral3_vl.ipynb) - HuggingFace TRL. Colab notebook for supervised fine-tuning of Ministral3 vision-language model.

- [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) - Thinking Machines Lab. Analysis showing LoRA matches full training performance more broadly than expected.

- [Agentic RL Systems](https://amberljc.github.io/blog/2025-09-05-agentic-rl-systems.html) - Amber Li. Overview of agentic reinforcement learning systems and their architecture for autonomous task completion.

- [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) - Thinking Machines Lab. Analysis showing on-policy, dense supervision is a useful and effective tool for knowledge distillation.

- [The Smol Training Playbook: Secrets to Building World-Class LLMs](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook) - HuggingFace TB. Comprehensive playbook covering secrets and best practices for training high-quality language models.

- [No More Train-Inference Mismatch: Bitwise Consistent On-Policy RL](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html) - vLLM Blog. Demonstrating bitwise consistent on-policy RL with TorchTitan as training engine and vLLM as inference engine, ensuring matching numerics.

- [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda) - Yingru Li et al. Deep analysis of a critical systems-level bottleneck causing RL training collapse due to training-inference mismatch.

- [Vision Reinforcement Learning (VLM RL) with Unsloth](https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl) - Unsloth. Guide to training vision and multimodal models via GRPO and reinforcement learning with Unsloth.

- [Evolution Strategies at the Hyperscale](https://eshyperscale.github.io/) - ES Hyperscale. Making general ML training as fast and easy as inference using evolution strategies.

- [Muon Optimizer Guide: Quick Start and Key Details](https://main-horse.github.io/translations/kexue/11416/#four-versions) - main-horse. Practical guide to the Muon optimizer covering four versions and key implementation details.

## 9. Hardware Architecture & Co-Design

#### Tier 1

- [Domain-Specific Architectures](https://fleetwood.dev/posts/domain-specific-architectures) - Fleetwood. Overview of domain-specific hardware design principles and their application to AI accelerator architectures.

#### Tier 2

- [The Case for Co-Designing Model Architectures with Hardware](https://arxiv.org/abs/2401.14489) - arXiv. Demonstrates that modifying DL model architectures to better match target GPU hardware can yield significant runtime improvements without sacrificing accuracy.

- [Jet-Nemotron: Efficient Language Model with Post Neural Architecture Search](https://hanlab.mit.edu/projects/jet-nemotron) - MIT HAN Lab. Hybrid attention model family achieving 47x generation throughput speedup at 64K context length compared to full-attention baselines through combined full and linear attention.

- [Maia 200: The AI Accelerator Built for Inference](https://blogs.microsoft.com/blog/2026/01/26/maia-200-the-ai-accelerator-built-for-inference/) - Microsoft. Breakthrough inference accelerator on TSMC 3nm with native FP8/FP4 tensor cores, 216GB HBM3e at 7 TB/s, and 272MB on-chip SRAM, designed to improve economics of AI token generation.

- [Inside the NVIDIA Rubin Platform: Six New Chips, One AI Supercomputer](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/) - Kyle Aubrey, NVIDIA. Technical overview of NVIDIA's next-generation Rubin platform architecture for AI factories.

- [A Close Look at SRAM for Inference in the Age of HBM Supremacy](https://www.viksnewsletter.com/p/a-close-look-at-sram-for-inference) - Vik's Newsletter. In-depth analysis of SRAM's specific performance benefits for inference and why HBM remains essential despite SRAM's advantages.

- [TinyTPU](https://www.tinytpu.com/) - TinyTPU Project. An attempt to understand and build a TPU by complete novices, with accompanying [GPU MODE lecture](https://www.youtube.com/watch?v=kccs9xk09rw).

- [NVIDIA Nemotron 3 Family](https://research.nvidia.com/labs/nemotron/Nemotron-3/) - NVIDIA Research. Technical details of the Nemotron 3 model family designed for efficient and accurate AI agent deployment.

- [The Chip Made for the AI Inference Era: The Google TPU](https://www.uncoveralpha.com/p/the-chip-made-for-the-ai-inference) - UncoverAlpha. Comprehensive deep-dive covering technical, strategic, and financial aspects of the Google TPU.

- [Google TPUv7: The 900lb Gorilla in the Room](https://t.co/Xh1ohGxVjB) - SemiAnalysis. Full stack review of TPUv7 Ironwood covering Anthropic's 1GW+ TPU usage, new customers, CUDA moat analysis, and next-generation TPUv8.

- [ECE298A-TPU: A Custom AI Chip](https://github.com/WilliamZhang20/ECE298A-TPU) - William Zhang. Custom AI chip project designed to be taped out, useful for understanding TPU architecture from scratch.

## 10. State-Space Models & Alternative Architectures

#### Tier 2

- [Cuthbert: State-Space Model Inference with JAX](https://github.com/state-space-models/cuthbert) - state-space-models. JAX-based inference implementation for state-space models, providing an alternative to transformer architectures for sequence modeling.

- [Trinity Large: An Open 400B Sparse MoE Model](https://www.arcee.ai/blog/trinity-large) - Arcee AI. Deep-dive into Trinity Large architecture, sparsity design, and training at scale, with Preview, Base, and TrueBase checkpoints.

- [The Spatial Blindspot of Vision-Language Models](https://arxiv.org/abs/2601.09954) - arXiv. Analysis of how CLIP-style image encoders flatten 2D structure into 1D patch sequences, degrading spatial reasoning in VLMs.

- [Recursive Language Models: The Paradigm of 2026](https://www.primeintellect.ai/blog/rlm) - Prime Intellect. Blueprint for "context folding": recursively compressing and reshaping an agent's own context to prevent context rot in ultra-long multi-step rollouts.

- [RLM: Inference Library for Recursive Language Models](https://github.com/alexzhang13/rlm) - Alex Zhang. General plug-and-play inference library for Recursive Language Models supporting various sandboxes.

- [PRIME: Scalable Reinforcement Learning for LLMs](https://arxiv.org/html/2512.23966v1) - arXiv. Research on scalable RL approaches for language model training and optimization.

- [nanochat Miniseries v1](https://github.com/karpathy/nanochat/discussions/420) - Andrej Karpathy. Discussion on optimizing LLM families controlled by a compute dial, covering training efficiency and model scaling principles.

- [Deep Sequence Models Tend to Memorize Geometrically](https://arxiv.org/abs/2510.26745) - arXiv. Contrasts associative vs geometric views of how transformers store parametric memory, revealing that memorization follows geometric rather than co-occurrence patterns.

- [Universal Reasoning Model](https://arxiv.org/abs/2512.14693) - arXiv. Systematic analysis of Universal Transformers showing that improvements on ARC-AGI arise from recurrent inductive bias and strong nonlinear computation.

- [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) - arXiv. Extension of residual connections via expanded stream width and diversified connectivity patterns while preserving identity mapping properties.

- [KerJEPA: Kernel Discrepancies for Euclidean Self-Supervised Learning](https://arxiv.org/abs/2512.19605) - arXiv. New family of self-supervised learning algorithms using kernel-based regularization for improved training stability and downstream generalization.

- [Introducing Mistral 3](https://mistral.ai/news/mistral-3) - Mistral AI. Announcement of the Mistral 3 family of frontier open-source multimodal models.

- [Qwen3-VL: Scanning Two-Hour Videos and Pinpointing Details](https://the-decoder.com/qwen3-vl-can-scan-two-hour-videos-and-pinpoint-nearly-every-detail/) - THE DECODER. Technical report analysis of Alibaba's Qwen3-VL, showing the open multimodal model excels at image-based math and can analyze hours of video footage.

- [DeltaNet Explained (Part II)](https://sustcsonglin.github.io/blog/2024/deltanet-2/) - Songlin Yang. Algorithm for parallelizing DeltaNet computation across the sequence length dimension.

- [Titans + MIRAS: Helping AI Have Long-Term Memory](https://research.google/blog/titans-miras-helping-ai-have-long-term-memory/) - Google Research. Research on Titans and MIRAS architectures for enabling long-term memory in AI models.

- [DeepSeek-V3.2 Technical Report](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf) - DeepSeek AI. Technical report for DeepSeek-V3.2, covering architecture improvements and training methodology.

- [PLEIADES: Building Temporal Kernels from Orthogonal Polynomials](https://www.youtube.com/watch?v=B5bzYl4zjPU) - PeaBrane. NeurIPS presentation on PLEIADES, constructing temporal kernels using orthogonal polynomials for efficient sequence modeling.

- [Loads and Loads of Fluffy Kittens](https://hazyresearch.stanford.edu/blog/2025-11-17-fluffy-kittens) - Stanford HazyResearch. Research on efficient model architectures and scaling approaches from the HazyResearch group.

- [Nemotron Elastic: Towards Efficient Many-in-One Reasoning LLMs](https://arxiv.org/abs/2511.16664) - arXiv. Training a single model that targets multiple scales and deployment objectives through elastic architecture, avoiding separate training runs for each size.

- [TiDAR: Think in Diffusion, Talk in Autoregression](https://www.alphaxiv.org/abs/2511.08923v1) - alphaXiv. Hybrid architecture combining diffusion-based thinking with autoregressive generation for improved reasoning.

- [Weight-Sparse Transformers Have Interpretable Circuits](https://arxiv.org/abs/2511.13653) - arXiv. Training models with sparse weights to produce more human-understandable circuits, advancing mechanistic interpretability.

## 11. Compiler & DSL Approaches

#### Tier 1

- [Helion: Python-Embedded DSL for ML Kernels](https://github.com/pytorch/helion) - PyTorch. A Python-embedded domain-specific language for writing fast, scalable ML kernels with minimal boilerplate, lowering the barrier to custom kernel development.

#### Tier 2

- [AOTInductor: Ahead-of-Time Compilation for PyTorch](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html) - PyTorch. Official documentation for AOTInductor, enabling ahead-of-time compilation of PyTorch models for deployment without Python runtime dependency.

- [Helion Flex Attention Example](https://github.com/pytorch/helion/blob/main/examples/flex_attention.py) - PyTorch. Reference implementation of flexible attention variants using Helion DSL, demonstrating how to write custom attention kernels with minimal code.

- [CUDA Tile IR](https://github.com/NVIDIA/cuda-tile) - NVIDIA. MLIR-based intermediate representation and compiler infrastructure for CUDA kernel optimization, focusing on tile-based computation patterns targeting NVIDIA tensor cores.

- [cuTile Python Samples](https://github.com/PeaBrane/cutile-python/blob/17927b4ea6a5db95eea38f6b72f5696f4c2fae09/samples/PeriodicConv1D.py#L63) - PeaBrane. Sample implementations using the cuTile programming model for writing parallel GPU kernels.

- [Intel ISPC: Implicit SPMD Program Compiler](https://ispc.github.io/index.html) - Intel. Open-source compiler for high-performance SIMD programming on CPU and GPU using an implicit SPMD model.

## 12. Confidential & Secure Inference

#### Tier 2

- [Confidential Compute for AI Inference with TEEs](https://chutes.ai/news/confidential-compute-for-ai-inference-how-chutes-delivers-verifiable-privacy-with-trusted-execution-environments) - Chutes. How Chutes delivers verifiable privacy for AI inference using Trusted Execution Environments in an adversarial, permissionless miner network.

## 13. AI Agents & LLM Tooling

#### Tier 1

- [AgentKernelArena](https://github.com/AMD-AGI/AgentKernelArena) - AMD AGI. End-to-end benchmarking environment for evaluating LLM-powered coding agents (Cursor, Claude Code, Codex, SWE-agent, GEAK) on CUDA kernel writing tasks.

#### Tier 2

- [agent-trace](https://github.com/yurekami/agent-trace) - yurekami. Tracing and observability tooling for LLM agent execution pipelines.

- [The ATOM Project: American Truly Open Models](https://atomproject.ai/) - ATOM. Initiative to build leading open AI models in the US, focusing on transparency and open research.

- [The Importance of Agent Harness in 2026](https://www.philschmid.de/agent-harness-2026) - Phil Schmid. Why Agent Harnesses are essential for building reliable AI systems capable of handling complex, multi-day tasks.

- [Async Coding Agents](https://benanderson.work/blog/async-coding-agents/) - Ben Anderson. Patterns and architecture for building asynchronous coding agents that can work on multiple tasks concurrently.

- [Shipping at Inference-Speed](https://steipete.me/posts/2025/shipping-at-inference-speed) - Peter Steinberger. Perspective on how AI inference speed changes software development workflows and shipping velocity.

- [Strands Agents SDK (Python)](https://github.com/strands-agents/sdk-python) - Strands. Model-driven approach to building AI agents in just a few lines of code.

- [open-ptc-agent: Programmatic Tool Calling with MCP](https://github.com/Chen-zexi/open-ptc-agent) - Chen Zexi. Open-source implementation of code execution with MCP (Programmatic Tool Calling).

- [Agents Towards Production](https://github.com/NirDiamant/agents-towards-production) - Nir Diamant. End-to-end, code-first tutorials covering every layer of production-grade GenAI agents with reusable blueprints.

- [Donating MCP and Establishing the Agentic AI Foundation](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation) - Anthropic. Announcement of MCP donation to the Agentic AI Foundation for open agentic AI infrastructure.

- [OML: AI-native Cryptography for Open-Model Attribution](https://www.youtube.com/watch?v=ygUOUnU6adE) - AER Labs. Talk on the Open Model Layer framework for security and attribution of open-source AI models.

- [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) - Manus. Practical principles for context engineering in AI agents, sharing local optima discovered while building the Manus agent.

- [Fara-7B: An Efficient Agentic Model for Computer Use](https://www.microsoft.com/en-us/research/blog/fara-7b-an-efficient-agentic-model-for-computer-use/) - Microsoft Research. Efficient 7B agentic small language model for computer use with robust safety measures.

## 14. Production Inference at Scale

#### Tier 2

- [Learn How Cursor Partnered with Together AI for Real-Time Inference](https://www.together.ai/blog/learn-how-cursor-partnered-with-together-ai-to-deliver-real-time-low-latency-inference-at-scale) - Together AI. How Together AI productionized NVIDIA Blackwell (B200/GB200) for Cursor's in-editor agents, covering ARM hosts, kernel tuning, and FP4/TensorRT quantization.

- [GPU (In)efficiency in AI Workloads](https://www.anyscale.com/blog/gpu-in-efficiency-in-ai-workloads) - Anyscale. Analysis of why GPUs are underutilized in production AI and how AI-native execution architectures improve GPU efficiency.

- [Achieving 4x LLM Performance Boost with KVCache (TensorMesh)](https://www.gmicloud.ai/blog/gmi-cloud-achieves-4x-llm-performance-boost-with-tensormesh) - GMI Cloud. 4x reduction in Time to First Token using SSD-augmented KVCache with TensorMesh prefix caching.

- [Building TensorMesh](https://www.youtube.com/watch?v=zHW4Zzd7pjI) - TensorMesh. Video walkthrough of TensorMesh's architecture and design for LLM inference optimization.

- [The Hidden Metric Destroying Your AI Agent's Performance](https://www.tensormesh.ai/blog-posts/hidden-metric-ai-agent-performance) - TensorMesh. How enterprise-grade AI-native caching cuts inference costs and latency by up to 10x.

- [Migrating from Slurm to dstack](https://github.com/dstackai/migrate-from-slurm/blob/main/guide.md) - dstack. Step-by-step guide for migrating AI workloads from Slurm to cloud-native dstack orchestration.

- [dstack 0.20 GA](https://dstack.ai/blog/0_20/) - dstack. Fleet-first UX and other major changes in dstack's GA release for cloud-native AI workload orchestration.

- [LMCache ROI Calculator: When KV Cache Storage Reduces Costs](https://www.tensormesh.ai/blog-posts/ai-inference-cost-calculator) - TensorMesh. Calculator and analysis for when KV cache storage-backed caching reduces overall AI inference costs.

- [LMCache Storage ROI Calculator](https://www.tensormesh.ai/tools/lmcache-storage-tco-calculator) - TensorMesh. Calculator for evaluating the ROI of adding storage-backed caching capacity with LMCache.

- [LMCache Context Engineering: 92% Prefix Reuse Rate](https://www.linkedin.com/posts/lmcache-lab_we-ran-a-tiny-one-shot-experiment-from-a-activity-7408688760395333632-pfvw) - LMCache Lab. Experiment showing 81% input cost reduction ($6.00 to $1.15) through prefix caching on a SWE-bench task with Claude Code.

- [AI Inference Costs in 2025: The $255B Market's Energy Crisis](https://www.tensormesh.ai/blog-posts/ai-inference-costs-2025-energy-crisis) - TensorMesh. Analysis of the energy and cost challenges facing the growing AI inference market.

- [Theseus: A Distributed GPU-Accelerated Query Processing Platform](https://medium.com/p/paper-summary-theseus-a-distributed-and-scalable-gpu-accelerated-query-processing-platform-c4b3e020252a) - Paper summary of Theseus, a distributed query processing system leveraging GPU acceleration for scalable data operations.

- [KV-Cache Wins You Can See: From Prefix Caching in vLLM to Distributed Caching in llm-d](https://llm-d.ai/blog/kvcache-wins-you-can-see) - llm-d. How llm-d enables smarter prefix-aware, load- and SLO-aware routing for better latency and throughput.

- [Host Overhead is Killing Your Inference Efficiency](https://modal.com/blog/host-overhead-inference-efficiency) - Modal. Analysis of how CPU-side host overhead blocks GPU utilization and techniques to eliminate it for better inference efficiency.

- [Democratizing AI Compute with AMD Using SkyPilot](https://rocm.blogs.amd.com/ecosystems-and-partners/democratizing-multicloud-skypi/README.html) - ROCm Blogs. How SkyPilot integrates with AMD's open AI stack for seamless multi-cloud deployment and NVIDIA-to-AMD GPU migration.

- [Revealing the Hidden Economics of Open Models in the AI Era](https://www.linuxfoundation.org/blog/revealing-the-hidden-economics-of-open-models-in-the-ai-era) - Linux Foundation. Research on the economic role of open models in the AI era from MIT and the Linux Foundation.

#### Tier 3

- [ANN v3: 200ms p99 Query Latency over 100 Billion Vectors](https://turbopuffer.com/blog/ann-v3) - Turbopuffer. ANN search at 100+ billion vector scale with 200ms p99 latency at 1k QPS and 92% recall, demonstrating extreme-scale vector search infrastructure.

## 15. Benchmarking & Profiling

#### Tier 1

- [Evaluation Guidebook](https://huggingface.co/spaces/OpenEvals/evaluation-guidebook) - OpenEvals / HuggingFace. Comprehensive guide to evaluating AI models, covering evaluation methodologies, metrics, and best practices.

- [AI Hardware Benchmarking & Performance Analysis](https://artificialanalysis.ai/benchmarks/hardware) - Artificial Analysis. Comprehensive benchmarking of AI accelerator systems for LLM inference across chip configurations, inference software, and concurrent load scaling.

#### Tier 2

- [FlashInfer-Bench](https://bench.flashinfer.ai/) - FlashInfer. Standardized benchmarking platform for AI infrastructure and kernel performance evaluation.

- [FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems](https://arxiv.org/abs/2601.00227v1) - arXiv. Framework connecting AI-generated kernel creation, benchmarking, and real-world inference system integration in a closed-loop workflow.

- [FlashInfer MLSys 2026 Tutorial](http://mlsys26.flashinfer.ai/) - FlashInfer. Tutorial materials from MLSys 2026 on FlashInfer's attention kernel library and LLM inference optimization.

- [Sorting-Free GPU Kernels for LLM Sampling](https://flashinfer.ai/2025/03/10/sampling.html) - FlashInfer. Technical blog on sorting-free GPU kernel implementations for efficient LLM sampling operations.

## 16. Courses & Comprehensive Guides

#### Tier 1

- [ML Hardware and Systems (ECE 5545)](https://abdelfattah-class.github.io/ece5545/) - University course covering machine learning hardware, systems design, and the intersection of algorithms with compute architectures.

- [MIT 6.5940: TinyML and Efficient Deep Learning Computing](https://hanlab.mit.edu/courses/2024-fall-65940) - MIT. Covers efficient AI computing techniques for deploying deep learning on resource-constrained devices and optimizing cloud infrastructure.

#### Tier 2

- [The Ultra-Scale Playbook](https://huggingface.co/spaces/nanotron/ultrascale-playbook) - Nanotron / HuggingFace. Interactive guide to scaling language model training and inference, covering parallelism strategies, communication patterns, and hardware utilization.

- [New AI Research Program](https://newlinesio.substack.com/p/new-ai-research-program-to-connect) - Aayush Saini. Open AI research program connecting researchers and practitioners for collaborative learning.

- [Physics of Language Models: Part 4.1a, How to Build a Versatile Synthetic Dataset](https://www.youtube.com/watch?v=x3G8knjPDbM) - Zeyuan Allen-Zhu. Lecture on constructing synthetic datasets for understanding language model capabilities and limitations.

- [Google's Year in Review: 8 Areas with Research Breakthroughs in 2025](https://blog.google/technology/ai/2025-research-breakthroughs/) - Google. Overview of Google's key AI research breakthroughs spanning models, products, science, and robotics.

- [Productionizing Diffusion Models](https://a-r-r-o-w.github.io/blog/3_blossom/00001_productionizing_diffusion-1/) - Arrow. Guide to bringing diffusion models from research to production deployment.

- [vLLM Internals Deep Dive (Thread)](https://x.com/archiexzzz/status/2005182120977989839) - Archie Sengupta. Visual thread diving deep into vLLM's internal architecture and design decisions.

- [Live from NeurIPS: Meet the Researchers — Nemotron Labs](https://www.youtube.com/live/tqA8Klv3MUQ?si=SkrvgM2HIRiL1iF4) - NVIDIA Developer. Live session from NeurIPS featuring NVIDIA Nemotron Labs researchers discussing their latest work.

- [Is Parallel Programming Practical? (perfbook)](https://mirrors.edge.kernel.org/pub/linux/kernel/people/paulmck/perfbook/perfbook.html) - Paul McKenney. Comprehensive book on parallel programming from the Linux kernel community, covering synchronization, memory ordering, and performance optimization.

- [Stanford AI Club: Jeff Dean on Important AI Trends](https://www.youtube.com/watch?v=AnTw_t21ayE) - Stanford AI Club. Jeff Dean's talk covering 15 years of ML progress and key AI trends.

- [Build an LLM from Scratch with MAX](https://llm.modular.com/) - Modular. Interactive tutorial for learning LLM internals by building from scratch using Modular's MAX Framework.

## 17. Tools & Libraries

#### Tier 1

- [HuggingFace Inference Providers with VS Code](https://huggingface.co/docs/inference-providers/en/guides/vscode) - HuggingFace. Guide to using HuggingFace inference providers directly within GitHub Copilot Chat in VS Code.

- [asxiv.org](https://asxiv.org/) - AI-powered interface for exploring and understanding arXiv research papers.

- [FlashInfer: Kernel Library for LLM Serving](https://github.com/flashinfer-ai/flashinfer) - FlashInfer. High-performance kernel library for LLM serving, providing optimized attention and decoding kernels.

#### Tier 2

- [AirLLM](https://github.com/lyogavin/airllm) - lyogavin. Run 70B parameter models on a single 4GB GPU through aggressive memory optimization and layer-wise inference.

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) - hiyouga. Unified fine-tuning framework supporting 100+ LLMs and VLMs with multiple training strategies.

- [Think-AI: Local AI Search on Your Computer](https://github.com/mrunalpendem123/Think-AI-) - mrunalpendem. Local AI search tool for running inference and search on your own machine.

- [Optimium: Next-Gen AI Inference Optimization Engine](https://optimium.enerzai.com/) - Enerzai. High-performance and flexible AI inference optimization engine.

- [HuggingFace Optimum](https://github.com/huggingface/optimum) - HuggingFace. Library for accelerating inference and training of Transformers, Diffusers, TIMM, and Sentence Transformers with hardware optimization tools.

- [HAMi: Heterogeneous AI Computing Virtualization Middleware](https://github.com/Project-HAMi/HAMi) - CNCF. GPU virtualization middleware enabling sharing and isolation of heterogeneous AI accelerators.

- [Inside the MAX Framework (Modular Meetup)](https://www.youtube.com/watch?v=WK5dVQ8vhbU) - Modular. Video walkthrough of Modular's MAX framework for unified AI inference and deployment.

- [miles](https://github.com/radixark/miles) - radixark. Inference tooling project from the AER Labs community.

- [Chronos-1.5B](https://huggingface.co/squ11z1/Chronos-1.5B) - squ11z1. 1.5B parameter model on HuggingFace.

- [Transformers v5.0.0rc0](https://github.com/huggingface/transformers/releases/tag/v5.0.0rc0) - HuggingFace. Major release candidate for Transformers v5 with significant API changes including dynamic weight loading and tokenization updates.

- [paperreview.ai](https://paperreview.ai/) - AI-powered platform for reviewing and understanding AI research papers.

## 18. Reference Collections

- [GPU Performance Engineering Resources](https://github.com/wafer-ai/gpu-perf-engineering-resources) - Wafer AI. Comprehensive tiered learning guide for GPU kernel programming and optimization, covering fundamentals through production deployment.

- [AER Labs Blog](https://aerlabs.tech/blogs) - AER Labs. Technical blog covering AI inference optimization, vLLM architecture, PagedAttention, KV cache systems, and LLM deployment strategies.

---

## Supplementary Materials

- [AI Inference Lab: Edge AI & HW Co-Design](https://discord.com/channels/aerlabs) - Marco Gonzalez. Presentation on edge AI inference and hardware co-design considerations.

---

## Contributing

Have a resource to share? Open a pull request or issue with the link, a brief description, and suggested category/tier placement.

## License

MIT
