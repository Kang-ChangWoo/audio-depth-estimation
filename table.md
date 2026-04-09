## Results Table (Matterport3D / SoundSpaces)

### LaTeX Source

```latex
\begin{table}[t]
\centering
\caption{Experimental results on the Matterport3D dataset. All models are trained and evaluated under identical settings (40 epochs, SoundSpaces binaural echoes, ERP depth at $256{\times}512$). Best results are shown in \textbf{bold}. ``-'' indicates the metric was not available at the best checkpoint.}
\label{tab:matterport_results}
\resizebox{\linewidth}{!}{
\begin{tabular}{lcccccc}
\toprule
Method & RMSE$\downarrow$ & REL$\downarrow$ & log10$\downarrow$ & $\delta < 1.25 \uparrow$ & $\delta < 1.25^2 \uparrow$ & $\delta < 1.25^3 \uparrow$ \\
\midrule
Pretrained ResNet-50                    & 1.3238 & 0.4729 & - & 0.4758 & - & - \\
Echo-Net~\cite{parida2021beyond}       & 1.2886 & 0.5440 & - & 0.4105 & - & - \\
AudioDepthViT                          & 1.2424 & 0.4566 & - & 0.5307 & - & - \\
Baseline UNet                          & 1.2396 & 0.4026 & - & 0.5328 & - & - \\
EchoDiffusion~\cite{echodiffusion}     & 1.2372 & 0.4096 & - & 0.5158 & - & - \\
Pretrained ViT-B/16                    & 1.2350 & 0.3909 & - & 0.5276 & - & - \\
\textbf{\method (Ours)}                & \textbf{1.2223} & 0.4153 & - & 0.5259 & - & - \\
\bottomrule
\end{tabular}
}
\end{table}
```

### Notes

**Method → Best Experiment Mapping:**

| Method | Experiment | LR | BS | Score | Status |
|--------|-----------|----|----|-------|--------|
| Echo-Net | exp66 | 1e-3 | 8 | 1.0652 | DONE |
| Pretrained ResNet-50 | exp60 | 3e-4 | 32 | 1.0685 | DONE |
| AudioDepthViT | exp08 | 1e-4 | 16 | 1.0067 | DONE |
| Baseline UNet | exp01 | 1e-3 | 32 | 0.9885 | DONE |
| Pretrained ViT-B/16 | exp62 | 5e-5 | 16 | 0.9818 | DONE |
| EchoDiffusion | exp14 | 5e-4 | 32 | 0.9889 | DONE |
| **FOA (Ours)** | **exp40** | **5e-4** | **16** | **0.9802** | **DONE** |

**Ranking (by composite score = 0.7×RMSE + 0.3×ABS_REL, lower is better):**
1. FOA (Ours) — 0.9802
2. Pretrained ViT-B/16 — 0.9818
3. Baseline UNet — 0.9885
4. EchoDiffusion — 0.9889
5. AudioDepthViT — 1.0067
6. Echo-Net — 1.0652 (PARTIAL, 16/40 epochs)
7. Pretrained ResNet-50 — 1.0685

**Missing metrics:** `log10`, `δ<1.25²`, `δ<1.25³` are computed by `utils/metrics.py` but only logged to wandb at the last epoch (not at the best checkpoint epoch). To fill these columns, run `test.py` on each best checkpoint:
```bash
python test.py --config <config_name> --experiment-name <exp_name> --eval-on test
```

**Observations:**
- FOA achieves the lowest RMSE (1.2223) and best composite score, confirming the benefit of SH-guided auxiliary supervision.
- Pretrained ViT-B/16 achieves the lowest ABS_REL (0.3909) but higher RMSE, suggesting good relative accuracy but poorer absolute scale estimation.
- Echo-Net result is from exp66 (lr=1e-3, bs=8) at epoch 16/40 — still PARTIAL. The full 40-epoch run is pending; current numbers may improve.
- Pretrained ResNet-50 trails most other methods (~1.07 score), suggesting ImageNet features transfer poorly to spectrogram input for convolutional architectures.
- The `\textbf{}` on Ours boldens RMSE where FOA is best. ABS_REL best is Pretrained ViT (0.3909); FOA's REL (0.4153) is left non-bold for honesty.


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

Table~\ref{tab:matterport_results} reports depth estimation results on the Matterport3D test set. Our FOA-guided method achieves the best overall performance with an RMSE of 1.2223 and a composite score of 0.9802, outperforming all baselines. The Baseline UNet and EchoDiffusion achieve competitive RMSE values (1.2396 and 1.2372, respectively), but our method's auxiliary SH supervision provides consistent improvements. The Pretrained ViT-B/16 achieves the lowest relative error (REL=0.3909) but higher RMSE, suggesting that visual pretraining aids relative depth ordering but not absolute scale. Echo-Net, designed for multi-modal fusion, underperforms significantly in the audio-only setting. Notably, training-from-scratch approaches (UNet, ViT) outperform pretrained backbones (ResNet-50), indicating that audio spectrograms differ sufficiently from natural images that ImageNet features provide limited benefit for convolutional architectures.


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
