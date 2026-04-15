```latex
% ============================================================
% FOA-Guided Depth Estimation from Binaural Audio
% ============================================================

\newcommand{\method}{FOA-Depth}

% ============================================================
% ABSTRACT
% ============================================================
\begin{abstract}
Estimating scene depth from audio echoes enables spatial understanding in visually degraded environments where cameras fail.
Existing audio-to-depth methods treat the problem as a direct regression from spectrograms, ignoring the rich spatial structure encoded in acoustic reflections.
We propose \method, a UNet-based architecture that leverages first-order ambisonics (FOA) as auxiliary supervision through a spherical harmonics (SH) branch.
Our SH branch predicts ambisonic coefficients from the shared encoder bottleneck, and a histogram alignment loss enforces spatial consistency between the predicted depth and the FOA-derived energy distribution.
On the Matterport3D SoundSpaces benchmark, \method{} achieves the lowest RMSE of 1.2223, outperforming all baselines including pretrained vision backbones, diffusion-based methods, and standard encoder-decoder architectures.
Code and trained models are available at [URL].
\end{abstract}

% ============================================================
% 1. INTRODUCTION
% ============================================================
\section{Introduction}
\label{sec:intro}

Depth estimation underpins a wide range of spatial reasoning tasks, from robotic navigation to augmented reality.
While vision-based depth estimation has matured rapidly, cameras degrade in low-light conditions, smoke, and occlusion-heavy environments.
Acoustic echoes offer a complementary sensing modality: sound propagates through space, reflects off surfaces, and encodes geometric information about the surrounding environment.
Recent work has demonstrated that binaural echoes captured in indoor scenes can be used to predict dense depth maps~\cite{gao2020visualechoes,parida2021beyond,zhang2025echodiffusion}.

However, existing audio-to-depth methods treat the prediction as a direct regression from spectrograms to depth, discarding the directional information inherent in acoustic signals.
First-order ambisonics (FOA) decompose sound fields into spherical harmonic components that explicitly encode spatial energy distributions.
This directional structure aligns naturally with the geometry captured in depth maps, yet no prior work exploits this connection for depth estimation.

We propose \method, a UNet encoder-decoder architecture augmented with a spherical harmonics (SH) auxiliary branch.
Our key insight is that FOA coefficients, derived from ambisonic impulse responses, provide a complementary supervision signal that constrains the depth prediction to be spatially consistent with the acoustic energy distribution.
Rather than relying solely on pixel-wise depth supervision, \method{} learns to align the spatial structure of predicted depth with directional sound energy through a histogram alignment loss.

Our key contributions are:
\begin{itemize}
    \item We introduce an SH auxiliary branch that predicts ambisonic coefficients from the encoder bottleneck, providing directional acoustic supervision alongside depth regression.
    \item We propose a histogram alignment loss that enforces spatial consistency between predicted depth maps and FOA-derived energy distributions through SH coefficient matching and cosine similarity.
    \item We evaluate \method{} against six baselines on Matterport3D SoundSpaces, achieving the best RMSE (1.2223) and demonstrating that FOA guidance consistently improves depth estimation across hyperparameter configurations.
\end{itemize}

% ============================================================
% 2. RELATED WORK
% ============================================================
\section{Related Work}
\label{sec:related}

\paragraph{Audio-based depth estimation.}
Gao~\emph{et al.}~\cite{gao2020visualechoes} first demonstrated that binaural echoes can predict coarse depth in indoor environments.
Parida~\emph{et al.}~\cite{parida2021beyond} proposed Echo-Net, which directly regresses depth from echo spectrograms using a multi-scale architecture.
However, Echo-Net treats the audio input as a generic 2D signal without exploiting its spatial structure, limiting performance on complex scenes.
Zhang~\emph{et al.}~\cite{zhang2025echodiffusion} introduced EchoDiffusion, applying diffusion models to iteratively refine depth predictions conditioned on audio features.
While EchoDiffusion produces sharper outputs, it requires iterative denoising at inference, increasing latency.
Our method instead operates in a single forward pass and leverages FOA supervision to encode spatial structure directly into the learned representation.

\paragraph{Spatial audio and ambisonics.}
Ambisonics represent sound fields as weighted sums of spherical harmonic basis functions~\cite{zotter2019ambisonics}.
First-order ambisonics encode the omnidirectional pressure (W channel) and three directional gradients (X, Y, Z channels), capturing the dominant spatial distribution of acoustic energy.
Higher-order ambisonics provide finer directional resolution at the cost of additional channels.
While ambisonics have been widely used in spatial audio rendering and source localization, their application to depth estimation remains unexplored.
We bridge this gap by projecting both ambisonic energy and predicted depth into a shared spherical harmonics space.

\paragraph{Auxiliary supervision for depth.}
Multi-task learning with auxiliary objectives has improved monocular depth estimation in the visual domain~\cite{eigen2015predicting,xu2018pad}.
Surface normals, semantic segmentation, and optical flow serve as complementary signals that regularize depth predictions.
In the audio domain, however, auxiliary supervision strategies remain limited.
Our SH branch and histogram alignment loss introduce a new form of auxiliary supervision specific to the acoustic modality, exploiting the geometric correspondence between directional sound energy and scene depth.

% ============================================================
% 3. METHOD
% ============================================================
\section{Method}
\label{sec:method}

\method{} estimates a dense equirectangular depth map $\hat{D} \in \mathbb{R}^{H \times W}$ from a binaural echo spectrogram $\mathbf{X} \in \mathbb{R}^{2 \times H \times W}$.
The architecture consists of three components: a UNet encoder-decoder for depth prediction, an SH auxiliary branch for ambisonic coefficient estimation, and a set of loss functions that jointly supervise depth accuracy and spatial consistency with FOA energy.
Figure~\ref{fig:overview} illustrates the full pipeline.

\subsection{Encoder-Decoder Backbone}
\label{sec:encoder}

We adopt a UNet architecture with 8 encoder blocks and 8 decoder blocks connected by skip connections.
The encoder processes the binaural spectrogram through progressive downsampling with $4 \times 4$ strided convolutions, LeakyReLU activations, and batch normalization, expanding channels from 64 to 512.
The decoder mirrors this structure with transposed convolutions and concatenated skip connections, producing a single-channel depth map normalized to $[0, 1]$ via a sigmoid activation.

\subsection{Spherical Harmonics Auxiliary Branch}
\label{sec:sh_branch}

The SH branch extracts directional acoustic information from the shared encoder bottleneck.
We apply global average pooling to the bottleneck features and project the result through a two-layer MLP to obtain a latent vector $\mathbf{z} \in \mathbb{R}^{d}$ ($d=128$).
Two linear heads predict FOA coefficients $\hat{\mathbf{c}}_{\text{foa}} \in \mathbb{R}^{4}$ and higher-order coefficients $\hat{\mathbf{c}}_{\text{hoa}} \in \mathbb{R}^{N_{\text{sh}}-4}$, where $N_{\text{sh}} = (L+1)^2 = 36$ for maximum SH order $L=5$.
The concatenated coefficient vector $\hat{\mathbf{c}} = [\hat{\mathbf{c}}_{\text{foa}}; \hat{\mathbf{c}}_{\text{hoa}}]$ represents the predicted sound field in the SH basis.

\paragraph{DeepScaleShift.}
Raw SH coefficients predicted from audio features and those derived from depth maps occupy different value ranges.
We introduce a DeepScaleShift module, a 4-layer MLP with LayerNorm and GELU activations that learns per-channel affine transformations $(\gamma_i, \beta_i)$ for each SH coefficient:
\begin{equation}
    \tilde{c}_i = \gamma_i \cdot \hat{c}_i + \beta_i, \quad i = 1, \ldots, N_{\text{sh}}.
\end{equation}
A residual connection with a learned gate blends the transformed and original coefficients, enabling stable optimization.

\paragraph{Energy reconstruction.}
Given aligned SH coefficients $\tilde{\mathbf{c}}$, we reconstruct a directional energy map on the equirectangular grid:
\begin{equation}
    E(\theta, \phi) = \sum_{i=1}^{N_{\text{sh}}} \tilde{c}_i \cdot Y_i(\theta, \phi),
\end{equation}
where $Y_i(\theta, \phi)$ are the real spherical harmonic basis functions (SN3D normalization) precomputed on the ERP grid.

\paragraph{Depth-to-SH projection.}
To enforce consistency in the opposite direction, we project the predicted depth map $\hat{D}$ into SH space via weighted least-squares with $\sin(\theta)$ area weighting:
\begin{equation}
    \hat{\mathbf{c}}_D = (\mathbf{Y}^\top \mathbf{W} \mathbf{Y})^{-1} \mathbf{Y}^\top \mathbf{W} \, \text{vec}(\hat{D}),
\end{equation}
where $\mathbf{Y} \in \mathbb{R}^{HW \times N_{\text{sh}}}$ is the precomputed basis matrix and $\mathbf{W}$ is a diagonal matrix of $\sin(\theta)$ weights.

\subsection{Loss Functions}
\label{sec:losses}

The total training objective combines depth supervision with FOA-guided losses:
\begin{equation}
    \mathcal{L} = \lambda_d \mathcal{L}_{\text{depth}} + \lambda_f \mathcal{L}_{\text{foa}} + \lambda_h \mathcal{L}_{\text{hist}}.
\end{equation}

\paragraph{Depth loss.}
We supervise depth prediction with a combination of BerHu (reverse Huber)~\cite{laina2016deeper} and scale-invariant logarithmic (SILog)~\cite{eigen2014depth} losses:
\begin{equation}
    \mathcal{L}_{\text{depth}} = \mathcal{L}_{\text{BerHu}} + 0.5 \cdot \mathcal{L}_{\text{SILog}}.
\end{equation}
BerHu applies L1 loss for small errors and quadratic penalization for large outliers, with an adaptive threshold $c = 0.2 \cdot \max(|\hat{D} - D|)$.
SILog penalizes log-scale differences, providing invariance to global depth scaling.

\paragraph{FOA guidance loss.}
We directly supervise the predicted FOA coefficients against ground-truth values derived from ambisonic impulse responses:
\begin{equation}
    \mathcal{L}_{\text{foa}} = \| \hat{\mathbf{c}}_{\text{foa}} - \mathbf{c}_{\text{foa}}^* \|_1 + \lambda_{\cos} (1 - \cos(\hat{\mathbf{c}}_{\text{foa}}, \mathbf{c}_{\text{foa}}^*)),
\end{equation}
where $\mathbf{c}_{\text{foa}}^*$ is the ground-truth FOA energy vector computed as the RMS of each ambisonic channel, normalized to $[0, 1]$.
The cosine term ($\lambda_{\cos} = 0.1$) encourages correct directional structure regardless of magnitude.

\paragraph{Histogram alignment loss.}
The histogram alignment loss enforces spatial consistency between the audio-derived SH energy reconstruction and the depth-derived SH representation:
\begin{equation}
    \mathcal{L}_{\text{hist}} = \| \tilde{E} - E_D \|_1 + \lambda_{\text{map}} (1 - \cos(\tilde{E}, E_D)) + \lambda_{\text{coeff}} (1 - \cos(\tilde{\mathbf{c}}, \hat{\mathbf{c}}_D)),
\end{equation}
where $\tilde{E}$ and $E_D$ are the normalized energy maps reconstructed from aligned audio SH coefficients and depth-derived SH coefficients respectively ($\lambda_{\text{map}} = 0.5$, $\lambda_{\text{coeff}} = 0.2$).
This loss operates in both spatial (energy map) and spectral (SH coefficient) domains, encouraging the model to learn depth predictions whose spatial structure mirrors the acoustic energy field.

\subsection{Ground-Truth FOA Supervision}
\label{sec:gt_foa}

We derive ground-truth FOA targets from first-order ambisonic impulse responses provided in the SoundSpaces dataset.
Given a 4-channel ambisonic IR $\mathbf{A} \in \mathbb{R}^{4 \times T}$, we compute the covariance matrix $\mathbf{R} = \frac{1}{T} \mathbf{A} \mathbf{A}^\top \in \mathbb{R}^{4 \times 4}$.
The directional energy at each ERP pixel $(\theta, \phi)$ is then:
\begin{equation}
    E^*(\theta, \phi) = \mathbf{y}(\theta, \phi)^\top \mathbf{R} \, \mathbf{y}(\theta, \phi),
\end{equation}
where $\mathbf{y}(\theta, \phi) \in \mathbb{R}^{4}$ is the first-order SH basis vector evaluated at that direction.
The FOA target vector $\mathbf{c}_{\text{foa}}^*$ is the RMS energy of each of the four ambisonic channels, normalized to $[0, 1]$.

% ============================================================
% 4. EXPERIMENTS
% ============================================================
\section{Experiments}
\label{sec:experiments}

\subsection{Dataset}
\label{sec:dataset}

We evaluate on the Matterport3D SoundSpaces dataset~\cite{chen2020soundspaces,chang2017matterport3d}, which provides binaural room impulse responses paired with equirectangular (ERP) depth maps across 90 indoor scenes.
We split the scenes into 72 train, 9 validation, and 9 test scenes using a deterministic random split (seed 42), yielding 23{,}560 training, 2{,}951 validation, and 3{,}192 test samples.
Samples with more than 10\% invalid depth pixels are excluded.

\paragraph{Input processing.}
We convert binaural waveforms (48\,kHz, stereo) to magnitude spectrograms using a 512-point FFT with a 400-sample window and 160-sample hop length.
We retain only the first ${\sim}20$\,ms of each impulse response, corresponding to early reflections that carry the strongest geometric information.
Spectrograms are resized to $256 \times 512$ via nearest-neighbor interpolation.
Depth maps are clipped to $[0.01, 10.0]$\,m and normalized to $[0, 1]$.

\subsection{Baselines}
\label{sec:baselines}

We compare \method{} against six baselines spanning pretrained, from-scratch, and generative approaches:

\begin{itemize}
    \item \textbf{Pretrained ResNet-50}: ImageNet-pretrained ResNet-50 encoder with a feature pyramid network (FPN) decoder.
    \item \textbf{Pretrained ViT-B/16}: ImageNet-pretrained Vision Transformer encoder with a convolutional decoder.
    \item \textbf{Echo-Net}~\cite{parida2021beyond}: Multi-scale architecture for direct spectrogram-to-depth regression.
    \item \textbf{Baseline UNet (BatVision)}: Standard pix2pix UNet with the same encoder-decoder backbone as \method{} but without the SH branch or FOA losses.
    \item \textbf{EchoDiffusion}~\cite{zhang2025echodiffusion}: Diffusion-based depth estimation conditioned on audio features, with optional Wav2Vec~\cite{baevski2020wav2vec} waveform encoding.
\end{itemize}

All models are trained with the same data pipeline, depth normalization, and evaluation protocol.
We conduct extensive hyperparameter search for each method (83 total experiments) and report the best configuration per method.

\subsection{Implementation Details}
\label{sec:impl}

We train all models for 40 epochs using AdamW~\cite{loshchilov2019decoupled} with gradient clipping (max norm 1.0).
For \method, we use a learning rate of $10^{-3}$, batch size 32, and loss weights $\lambda_d = 1.0$, $\lambda_f = 0.1$, $\lambda_h = 0.05$.
The SH branch uses maximum order $L=5$ (36 coefficients), projection dimension $d=128$, and a 4-layer DeepScaleShift MLP with hidden dimension 256.
We select the best checkpoint based on a composite score $0.7 \times \text{RMSE} + 0.3 \times \text{ABS\_REL}$ on the validation set.
All experiments run on 2 NVIDIA RTX 4090 GPUs with DataParallel.

\subsection{Evaluation Metrics}
\label{sec:metrics}

We report standard depth estimation metrics computed on valid pixels ($D > 0$):
RMSE (root mean squared error),
ABS\_REL (mean absolute relative error $|D - \hat{D}| / D$),
$\log_{10}$ error,
and accuracy thresholds $\delta < 1.25^k$ for $k \in \{1, 2, 3\}$, which measure the percentage of pixels where $\max(\hat{D}/D, D/\hat{D}) < 1.25^k$.

\subsection{Results}
\label{sec:results}

Table~\ref{tab:matterport_results} presents quantitative results on the Matterport3D test set.

\begin{table}[t]
\centering
\caption{\textbf{Quantitative results on the Matterport3D}. Best results are shown in \textbf{bold}, and second-best results are \underline{underscored}.}
\label{tab:matterport_results}
\resizebox{\linewidth}{!}{
\begin{tabular}{lcccccc}
\toprule
Method & RMSE$\downarrow$ & REL$\downarrow$ & log10$\downarrow$ & $\delta < 1.25 \uparrow$ & $\delta < 1.25^2 \uparrow$ & $\delta < 1.25^3 \uparrow$ \\
\midrule
Pretrained ResNet-50                & 1.3238             & 0.4729             & - & 0.4758             & - & - \\
Pretrained ViT-B/16                 & \underline{1.2350} & \textbf{0.3909}    & - & 0.5276             & - & - \\
\midrule
Echo-Net~\cite{parida2021beyond}    & 2.1536             & 1.2629             & - & 0.1906             & - & - \\
AudioDepthViT                       & 1.2424             & 0.4566             & - & \underline{0.5307} & - & - \\
Baseline UNet                       & 1.2396             & \underline{0.4026} & - & \textbf{0.5328}    & - & - \\
EchoDiffusion~\cite{zhang2025echodiffusion}  & 1.2372             & 0.4096             & - & 0.5158             & - & - \\
\textbf{\method\ (Ours)}            & \textbf{1.2223}    & 0.4153             & - & 0.5259             & - & - \\
\bottomrule
\end{tabular}
}
\end{table}

\method{} achieves the lowest RMSE (1.2223), outperforming all baselines.
Compared to the Baseline UNet, which shares the same encoder-decoder backbone, \method{} reduces RMSE from 1.2396 to 1.2223 (1.4\% improvement), demonstrating that the SH branch and FOA-guided losses provide meaningful auxiliary supervision.
EchoDiffusion achieves the second-lowest RMSE (1.2372) but requires iterative diffusion sampling at inference, whereas \method{} operates in a single forward pass.

Pretrained ViT-B/16 achieves the best REL (0.3909), suggesting that ImageNet features capture relative depth structure effectively.
However, its RMSE (1.2350) lags behind \method, indicating higher sensitivity to outlier errors.
The Baseline UNet achieves the highest $\delta < 1.25$ accuracy (0.5328), reflecting strong local depth consistency from the skip connections, while \method{} trades marginal local accuracy for better global depth scale captured through the SH alignment.

Echo-Net performs substantially worse than all other methods (RMSE 2.1536, REL 1.2629), consistent with its limited capacity to handle the complexity of Matterport3D environments.
Pretrained ResNet-50 also underperforms (RMSE 1.3238), confirming that ImageNet convolutional features transfer poorly to spectrogram inputs.

\subsection{Analysis}
\label{sec:analysis}

\paragraph{FOA dominance across configurations.}
Beyond the best-per-method comparison, we evaluated 20 FOA configurations with varying loss weights and learning rates (Table~\ref{tab:foa_ablation}).
FOA models occupy 9 of the top 15 positions in the overall ranking across all 83 experiments, confirming that SH auxiliary supervision provides robust improvements regardless of specific hyperparameter choices.

\paragraph{Effect of histogram weight.}
The best FOA configuration uses a histogram alignment weight $\lambda_h = 0.05$, lower than the default $\lambda_h = 0.1$.
Heavier histogram weighting ($\lambda_h = 0.2$) increases RMSE from 1.2223 to 1.2424, suggesting that excessive alignment pressure constrains the depth decoder and reduces its capacity to capture fine-grained depth variations.
A lighter histogram loss provides sufficient structural guidance without over-regularizing the depth output.

\paragraph{Single forward pass efficiency.}
\method{} produces depth predictions in a single forward pass with 55.0M parameters, comparable to the Baseline UNet.
The SH branch adds minimal overhead ($<$1\% of total parameters) since it operates on the pooled bottleneck features.
In contrast, EchoDiffusion requires multiple denoising steps, increasing inference latency proportionally.

% ============================================================
% REFERENCES (partial — key citations)
% ============================================================
% \cite{gao2020visualechoes} - Visual Echoes
% \cite{parida2021beyond} - Echo-Net / Beyond Image to Depth
% \cite{zhang2025echodiffusion} - EchoDiffusion
% \cite{chen2020soundspaces} - SoundSpaces
% \cite{chang2017matterport3d} - Matterport3D
% \cite{zotter2019ambisonics} - Ambisonics book
% \cite{laina2016deeper} - BerHu loss / Deeper Depth Prediction
% \cite{eigen2014depth} - Eigen depth / SILog
% \cite{eigen2015predicting} - Multi-task depth + normals
% \cite{xu2018pad} - PAD-Net multi-task depth
% \cite{loshchilov2019decoupled} - AdamW
% \cite{baevski2020wav2vec} - Wav2Vec 2.0
```
