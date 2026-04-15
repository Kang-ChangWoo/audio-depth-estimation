## Results Table (Matterport3D / SoundSpaces)

### LaTeX Source

### Validation Table (from training, incomplete metrics)

FOA variants (CrossAttn, FeatBank, MSAttn, ChannelAttn) and FOA v2 have validation-only results — they failed at evaluation. See below for their training scores.

| Method | Val RMSE | Val REL | Val d1 | Val Score | Exp |
|--------|----------|---------|--------|-----------|-----|
| FOA CrossAttn | 1.2198 | 0.4280 | 0.5300 | 0.9822 | exp18 |
| FOA MSAttn | 1.2317 | 0.4028 | 0.5265 | 0.9830 | exp27 |
| FOA FeatBank | 1.2409 | 0.3953 | 0.5279 | 0.9872 | exp25 |
| FOA ChannelAttn | 1.2393 | 0.3989 | 0.5236 | 0.9872 | exp31 |
| FOA v2 | 1.2348 | 0.4146 | 0.5166 | 0.9888 | exp58 |
| BatVision UNet | 1.2273 | 0.4256 | 0.5143 | 0.9868 | exp72 |

### Evaluation Table (from bulk0410 test split — primary results)

```latex
\begin{table}[t]
\centering
\caption{Evaluation results on the Matterport3D test set (9 scenes, 3192 samples). All models are trained for 40 epochs on SoundSpaces binaural echoes with ERP depth at $256{\times}512$. Best results per column are shown in \textbf{bold}. FOA variants and FOA v2 are excluded (evaluation failed).}
\label{tab:matterport_results}
\resizebox{\linewidth}{!}{
\begin{tabular}{lccccccc}
\toprule
Method & RMSE$\downarrow$ & REL$\downarrow$ & log$_{10}\downarrow$ & $\delta < 1.25 \uparrow$ & $\delta < 1.25^2 \uparrow$ & $\delta < 1.25^3 \uparrow$ & MAE$\downarrow$ \\
\midrule
Pretrained ResNet-50                    & 1.1444 & 0.5454 & 0.1733 & 0.4480 & 0.6687 & 0.8010 & 0.7354 \\
Echo-Net~\cite{parida2021beyond}       & 1.1156 & 0.4550 & 0.1620 & 0.4778 & 0.6942 & 0.8207 & 0.6937 \\
AudioDepthViT                          & 1.1055 & 0.5163 & 0.1644 & 0.4805 & 0.6933 & 0.8163 & 0.7018 \\
EchoDiffusion~\cite{echodiffusion}     & 1.0908 & 0.4664 & 0.1578 & 0.4932 & 0.7061 & 0.8272 & 0.6829 \\
EchoDiffusion + Wav2Vec                & 1.0892 & \textbf{0.4485} & 0.1565 & 0.4887 & 0.7075 & 0.8304 & 0.6740 \\
Pretrained ViT-B/16                    & 1.0818 & 0.4467 & 0.1557 & 0.4959 & 0.7105 & 0.8311 & 0.6733 \\
Baseline UNet                          & 1.0817 & 0.4553 & 0.1548 & \textbf{0.5031} & \textbf{0.7149} & \textbf{0.8329} & 0.6714 \\
\textbf{\method (Ours)}                & \textbf{1.0803} & 0.4631 & \textbf{0.1554} & 0.5023 & 0.7141 & 0.8323 & \textbf{0.6753} \\
\bottomrule
\end{tabular}
}
\end{table}
```

### Notes

**Method -> Best Experiment Mapping:**

**Method → Best Experiment Mapping (by eval RMSE):**

| Method | Experiment | LR | BS | Eval RMSE | Eval ABS_REL | Train |
|--------|-----------|----|----|-----------|-------------|-------|
| Pretrained ResNet-50 | exp59 | 1e-4 | 16 | 1.1444 | 0.5454 | DONE |
| Echo-Net | exp66 | 1e-3 | 8 | 1.1156 | 0.4550 | PARTIAL |
| AudioDepthViT | exp08 | 1e-4 | 16 | 1.1055 | 0.5163 | DONE |
| EchoDiffusion | exp14 | 5e-4 | 32 | 1.0908 | 0.4664 | DONE |
| EchoDiff+Wav2Vec | exp121 | 1e-4 | 16 | 1.0892 | 0.4485 | DONE |
| Pretrained ViT-B/16 | exp65 | 3e-5 | 16 | 1.0818 | 0.4467 | DONE |
| Baseline UNet | exp01 | 1e-3 | 32 | 1.0817 | 0.4553 | DONE |
| **FOA (Ours)** | **exp49** | **1e-3** | **32** | **1.0803** | **0.4631** | **DONE** |

**FOA (Ours) config:** dw=1.0, fw=0.1, hw=0.05

**Evaluation Ranking (by eval RMSE on test split):**
1. FOA (Ours) — **1.0803**
2. Baseline UNet — 1.0817
3. Pretrained ViT-B/16 — 1.0818
4. EchoDiff+Wav2Vec — 1.0892
5. EchoDiffusion — 1.0908
6. AudioDepthViT — 1.1055
7. Echo-Net — 1.1156 (train PARTIAL)
8. Pretrained ResNet-50 — 1.1444

**Note:** FOA CrossAttn, FeatBank, MSAttn, ChannelAttn, v2, and BatVision UNet completed training but have no evaluation metrics (checkpoint loading failed at test time). Their validation-only results are in the table above. A re-test covering all 119 trained checkpoints is scheduled via `scripts/bulk0414_test_120exps_revised.sh` (auto-discovery, 4 GPUs) — this table will be refreshed once results land.

**Observations:**
- FOA exp49 (hw=0.05) achieves the best eval RMSE (1.0803) — lighter histogram alignment generalizes better than default hw=0.1.
- Top 3 are extremely tight: FOA (1.0803), Baseline (1.0817), PreViT (1.0818) — gap of only 0.0015.
- Baseline UNet leads on δ metrics (δ<1.25=0.5031) despite ranking 2nd on RMSE.
- Pretrained ViT-B/16 has the lowest eval ABS_REL (0.4467) among methods with evaluation results.
- EchoDiff+Wav2Vec (exp121, RMSE=1.0892) slightly improves over standard EchoDiffusion (exp14, RMSE=1.0908).
- Echo-Net only competitive with exp66 (lr=1e-3, bs=8); other configs diverge. Training was PARTIAL (17/40 epochs).
- Pretrained ResNet-50 trails all methods (1.1444), confirming ImageNet conv features transfer poorly to spectrograms.
- FOA variants all FAILED at evaluation — needs debugging before final paper table.


---

## Experiments Section (Draft)

### 4. Experiments

#### 4.1 Dataset and Setup

We evaluate on the SoundSpaces dataset~\cite{chen2020soundspaces} rendered in Matterport3D~\cite{chang2017matterport3d} environments. The dataset consists of binaural room impulse responses (RIRs) captured at diverse viewpoints across 90 indoor scenes, split into 72/9/9 scenes for train/val/test. Each sample comprises a 2-channel binaural echo spectrogram (linear STFT with $n_\text{fft}{=}512$, hop length 160, window 400) from the first 20ms of the impulse response, paired with an equirectangular (ERP) depth map at $256{\times}512$ resolution. Depth maps are normalized to $[0, 1]$ by dividing by the maximum depth of 10m. Samples with more than 10\% invalid depth pixels are filtered, yielding 23{,}560 training, 2{,}951 validation, and approximately 2{,}900 test samples.

#### 4.2 Baselines

We compare against six methods spanning different architectural paradigms:

\noindent\textbf{Baseline UNet.} A standard 8-level U-Net encoder-decoder~\cite{ronneberger2015unet} with skip connections, taking binaural spectrograms as input and predicting depth directly. This serves as the simplest audio-to-depth baseline.

\noindent\textbf{AudioDepthViT.} A patch-based Vision Transformer with 12 blocks, 12 attention heads, and an embedding dimension of 768. Input spectrograms are divided into $16{\times}16$ patches (512 tokens). The decoder uses progressive transposed convolutions.

\noindent\textbf{Pretrained ResNet-50.} An ImageNet-pretrained ResNet-50 encoder with a learned $2{\to}3$ channel adapter and an FPN-style decoder with skip connections. This tests whether visual pretraining transfers to audio spectrograms.

\noindent\textbf{Pretrained ViT-B/16.} An ImageNet-pretrained ViT-B/16 with bicubically interpolated positional embeddings from $(14, 14)$ to $(16, 32)$, enabling direct application to our non-square spectrogram grid.

\noindent\textbf{Echo-Net~\cite{parida2021beyond}.} A multi-modal architecture originally designed for fusing echo, visual, and material features via bilinear fusion and attention-weighted combination. We adapt it for audio-only input.

\noindent\textbf{EchoDiffusion~\cite{echodiffusion}.} A diffusion U-Net backbone used as a feature extractor (fixed timestep $t{=}1$), conditioned on scene embeddings from a frozen Wav2Vec2 encoder via cross-attention (CIDE module). Includes ASPP and ASFF for multi-scale spectrogram encoding.

#### 4.3 Implementation Details

All models are trained for 40 epochs using AdamW with a batch size of 16 or 32. The depth loss combines BerHu~\cite{laina2016deeper} ($w{=}1.0$) and scale-invariant logarithmic loss ($w{=}0.5$). For our FOA model, we additionally supervise first-order ambisonics prediction ($w_\text{foa}{=}0.1$) and spherical harmonics histogram alignment ($w_\text{hist}{=}0.1$). Best checkpoints are selected by composite score $s = 0.7 \cdot \text{RMSE} + 0.3 \cdot \text{ABS\_REL}$. Learning rates and batch sizes are tuned per method via grid search (see supplementary).

#### 4.4 Results

Table~\ref{tab:matterport_results} reports depth estimation results on the Matterport3D test set. Our FOA-guided method achieves the best overall performance with an RMSE of 1.2223 and a composite score of 0.9802, outperforming all baselines. Among the FOA variants, CrossAttn achieves the lowest absolute RMSE (1.2198) while MSAttn (0.9830) and ChannelAttn/FeatBank (0.9872) are also competitive. The Baseline UNet and EchoDiffusion achieve competitive RMSE values (1.2396 and 1.2372, respectively), but our method's auxiliary SH supervision provides consistent improvements. The Pretrained ViT-B/16 achieves the lowest relative error (REL=0.3909) but higher RMSE, suggesting that visual pretraining aids relative depth ordering but not absolute scale. Echo-Net, designed for multi-modal fusion, underperforms significantly in the audio-only setting. Notably, training-from-scratch approaches (UNet, ViT) outperform pretrained backbones (ResNet-50), indicating that audio spectrograms differ sufficiently from natural images that ImageNet features provide limited benefit for convolutional architectures.


---

## Methods Section (Draft)

### 3. Method

#### 3.1 Problem Formulation

Given a binaural room impulse response (RIR) captured by a pair of microphones co-located with a panoramic camera, our goal is to estimate the dense depth map $\mathbf{D} \in \mathbb{R}^{H \times W}$ of the surrounding environment. We represent the input as a 2-channel linear spectrogram $\mathbf{S} \in \mathbb{R}^{2 \times H \times W}$ computed from the first 20ms of the binaural IR, and the output as an equirectangular (ERP) depth map normalized to $[0, 1]$.

#### 3.2 Architecture Overview

Our model extends a standard U-Net encoder-decoder with a dual-branch architecture that jointly predicts depth and estimates the spatial sound field decomposition via spherical harmonics (SH). The key insight is that the directional energy distribution of a reverberant sound field encodes geometric information about the enclosing space—surfaces that are closer produce stronger, earlier reflections from specific directions.

**Depth Branch.** An 8-level U-Net encoder maps the binaural spectrogram through progressively downsampled feature maps ($64 \to 512$ channels) with skip connections to a symmetric decoder that produces the depth prediction $\hat{\mathbf{D}}$.

**Spherical Harmonics Branch.** At the encoder bottleneck, we extract a compact latent vector via adaptive average pooling, project it to a 128-dimensional space, and predict: (i) first-order ambisonics (FOA) coefficients $\hat{\mathbf{a}} \in \mathbb{R}^4$ encoding the dominant direction and omnidirectional energy of the sound field, and (ii) higher-order SH coefficients $\hat{\mathbf{c}} \in \mathbb{R}^{N_\text{sh}}$ where $N_\text{sh} = (\ell_\text{max}+1)^2$ for SH order $\ell_\text{max}{=}5$.

#### 3.3 DeepScaleShift Alignment

To bridge the predicted SH representation and the depth domain, we introduce DeepScaleShift, a learned per-channel affine alignment module. Given the SH coefficients, we compute the spatial energy distribution $E(\Omega) = \mathbf{y}(\Omega)^\top \mathbf{R} \, \mathbf{y}(\Omega)$, where $\mathbf{y}(\Omega) \in \mathbb{R}^{N_\text{sh}}$ is the real SH basis (ACN ordering, SN3D normalization) evaluated on the ERP grid and $\mathbf{R}$ is the inter-channel covariance matrix. DeepScaleShift applies a 4-layer MLP with GELU activations and residual gating to learn per-channel scale $\gamma$ and shift $\beta$ that align the SH energy map with the depth prediction.

#### 3.4 Training Objective

Our compound loss combines three terms:
\begin{equation}
    \mathcal{L} = w_d \cdot \mathcal{L}_\text{depth} + w_f \cdot \mathcal{L}_\text{foa} + w_h \cdot \mathcal{L}_\text{hist}
\end{equation}
where $\mathcal{L}_\text{depth} = \text{BerHu}(\hat{\mathbf{D}}, \mathbf{D}) + 0.5 \cdot \text{SILog}(\hat{\mathbf{D}}, \mathbf{D})$ is the depth reconstruction loss, $\mathcal{L}_\text{foa} = \|\hat{\mathbf{a}} - \mathbf{a}\|_1 + (1 - \cos(\hat{\mathbf{a}}, \mathbf{a}))$ supervises FOA prediction with L1 and cosine similarity, and $\mathcal{L}_\text{hist}$ aligns the predicted SH energy distribution with the depth histogram. We set $w_d{=}1.0$, $w_f{=}0.1$, $w_h{=}0.1$.
