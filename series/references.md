# 参考文献

## 基礎（必読）

- Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
- Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)
- Loshchilov et al., "nGPT: Normalized Transformer with Representation Learning on the Hypersphere" (arXiv:2410.01131, 2024)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (arXiv 2021; Neurocomputing 2024) - RoPEの原論文
- Nagata et al., "Variance Matters: Detecting Semantic Differences without Corpus/Word Alignment" (EMNLP 2023)
- Yamagiwa et al., "Revisiting Cosine Similarity via Normalized ICA-transformed Embeddings" (arXiv 2024)

## 離散と連続の界面

- Jang et al., "Categorical Reparameterization with Gumbel-Softmax" (ICLR 2017)
- Maddison et al., "The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables" (ICLR 2017)
- Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation" (arXiv 2013) - STEの原論文

## MoE・スパース性

- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (ICLR 2017) - MoEの基礎
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (JMLR 2022)
- Jiang et al., "Mixtral of Experts" (arXiv 2024)
- Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models" (arXiv 2024)

## 双曲幾何学

- Nickel & Kiela, "Poincaré Embeddings for Learning Hierarchical Representations" (NeurIPS 2017)
- Mathieu et al., "Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders" (NeurIPS 2019)
- Yang et al., "Hyperbolic Fine-Tuning for Large Language Models (HypLoRA)" (arXiv:2410.04010, 2024)
    - ※AQuAで最大13.0%向上。発展版のHoRA（適応的曲率）では17.30%向上との報告あり
- Sinha et al., "Learning Structured Representations with Hyperbolic Embeddings" (NeurIPS 2024)
- He et al., "Hyperbolic Deep Learning for Foundation Models: A Survey" (arXiv:2507.17787, 2025)

## 拡散モデル

- Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (ICLR 2021)

## 情報幾何

- Amari, "Information Geometry and Its Applications" (Springer 2016)
- Martens & Grosse, "Optimizing Neural Networks with Kronecker-factored Approximate Curvature" (ICML 2015)
- Liu et al., "Reconstructing Deep Neural Networks: Unleashing the Optimization Potential of Natural Gradient Descent" (NeurIPS 2024)
- 参考（査読未通過）: Hwang, "FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information" (arXiv:2405.12807)
    - ※ICLR 2025で取り下げ。Adamと自然勾配の関係についての興味深い視点を提供するが、確立された理論ではない点に注意

## Model Collapse

- Shumailov et al., "The Curse of Recursion: Training on Generated Data Makes Models Forget" (arXiv 2023)

## 古典

- 甘利俊一『情報幾何学の新展開』
- Bishop "Pattern Recognition and Machine Learning" (PRML)
- Goodfellow et al., "Deep Learning" (2016)

## 発展

- Carlsson, "Topology and Data" (Bulletin of the AMS, 2009) - TDA入門
- Lee, "Introduction to Riemannian Manifolds" (2018) - 数学的基礎
- Bronstein et al., "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges" (2021)

## 最新動向（2024-2025年）

- Chlenski et al., "Mixed-Curvature Decision Trees and Random Forests" (ICML 2025)
- Fein-Ashley et al., "Hyperbolic Vision Transformers (HVT)" (2024)
- Grover et al., "Spectro-Riemannian Graph Neural Networks" (ICLR 2025)

## オンライン資源

- 3Blue1Brown: "Neural Networks" シリーズ（幾何学的直感）
- Distill.pub: インタラクティブな可視化記事
- The Annotated Transformer: コード付き解説
- Awesome Hyperbolic Representation Learning (GitHub): 双曲表現学習の論文リスト
- MoE-Infinity (GitHub): MoE実装のリファレンス
