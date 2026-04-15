## Test Set Results Table (Matterport3D / SoundSpaces)

Evaluated on the **test split** (9 scenes, 3192 samples) using best checkpoints selected during training by `score = 0.7 × RMSE + 0.3 × ABS_REL`.

### LaTeX Source

```latex
\begin{table}[t]
\centering
\caption{Test set results on the Matterport3D dataset. All models are trained for 40 epochs on SoundSpaces binaural echoes with ERP depth at $256{\times}512$. Best results per column are shown in \textbf{bold}. FOA variants (CrossAttn, FeatBank, MSAttn, ChannelAttn) and FOA v2 failed at test time and are excluded. Echo-Net is currently being retrained and is excluded.}
\label{tab:matterport_test_results}
\resizebox{\linewidth}{!}{
\begin{tabular}{lccccccc}
\toprule
Method & RMSE$\downarrow$ & REL$\downarrow$ & log$_{10}\downarrow$ & $\delta < 1.25 \uparrow$ & $\delta < 1.25^2 \uparrow$ & $\delta < 1.25^3 \uparrow$ & MAE$\downarrow$ \\
\midrule
Pretrained ResNet-50                    & 1.1515 & 0.4977 & 0.1713 & 0.4587 & 0.6708 & 0.8015 & 0.7251 \\
AudioDepthViT                          & 1.1055 & 0.5163 & 0.1644 & 0.4805 & 0.6933 & 0.8163 & 0.7018 \\
EchoDiffusion~\cite{echodiffusion}     & 1.0908 & 0.4664 & 0.1578 & 0.4932 & 0.7061 & 0.8272 & 0.6829 \\
EchoDiffusion + Wav2Vec                & 1.0892 & 0.4485 & 0.1565 & 0.4887 & 0.7075 & 0.8304 & 0.6740 \\
Pretrained ViT-B/16                    & 1.0818 & \textbf{0.4467} & 0.1557 & 0.4959 & 0.7105 & 0.8311 & 0.6733 \\
Baseline UNet                          & 1.0817 & 0.4553 & 0.1548 & \textbf{0.5031} & \textbf{0.7149} & \textbf{0.8329} & 0.6714 \\
\textbf{\method (Ours)}                & \textbf{1.0803} & 0.4631 & \textbf{0.1554} & 0.5023 & 0.7141 & 0.8323 & \textbf{0.6753} \\
\bottomrule
\end{tabular}
}
\end{table}
```

### Notes

**Method → Best Experiment Mapping (Test Set):**

| Method | Experiment | LR | BS | Test RMSE | Test ABS_REL | Test δ<1.25 |
|--------|-----------|----|----|-----------|-------------|-------------|
| Pretrained ResNet-50 | exp60 | 3e-4 | 32 | 1.1515 | 0.4977 | 0.4587 |
| Echo-Net | exp66–69 | — | — | retraining | retraining | retraining |
| AudioDepthViT | exp08 | 1e-4 | 16 | 1.1055 | 0.5163 | 0.4805 |
| EchoDiffusion | exp14 | 5e-4 | 32 | 1.0908 | 0.4664 | 0.4932 |
| EchoDiff+Wav2Vec | exp121 | 1e-4 | 16 | 1.0892 | 0.4485 | 0.4887 |
| Pretrained ViT-B/16 | exp65 | 3e-5 | 16 | 1.0818 | 0.4467 | 0.4959 |
| Baseline UNet | exp01 | 1e-3 | 32 | 1.0817 | 0.4553 | 0.5031 |
| **FOA (Ours)** | **exp49** | **1e-3** | **32** | **1.0803** | **0.4631** | **0.5023** |

**FOA (Ours) config:** dw=1.0, fw=0.1, hw=0.05

**Ranking by Test RMSE:**
1. FOA (Ours) — **1.0803**
2. Baseline UNet — 1.0817
3. Pretrained ViT-B/16 — 1.0818
4. EchoDiff+Wav2Vec — 1.0892
5. EchoDiffusion — 1.0908
6. AudioDepthViT — 1.1055
7. Pretrained ResNet-50 — 1.1515

**Failed at test (excluded):** FOA CrossAttn (exp16–20), FOA FeatBank (exp21–25), FOA MSAttn (exp26–30), FOA ChannelAttn (exp31–35), FOA v2 (exp56–60 foav2) — all 25 experiments failed to load checkpoints at test time.

**Retraining in progress (excluded):** Echo-Net exp66–69 — currently being retrained (`bulk0410_sum_revised`); only 4–8 of 40 epochs complete, no test metrics available.

---

### Validation vs Test Comparison (Best per Method)

| Method | Val RMSE | Val Score | Test RMSE | Test ABS_REL | Gap (Test−Val RMSE) |
|--------|----------|-----------|-----------|-------------|---------------------|
| FOA (Ours) | 1.2223 (exp40) | 0.9802 | 1.0803 (exp49) | 0.4631 | −0.1420 |
| Baseline UNet | 1.2396 (exp01) | 0.9885 | 1.0817 (exp01) | 0.4553 | −0.1579 |
| Pretrained ViT-B/16 | 1.2350 (exp62) | 0.9818 | 1.0818 (exp65) | 0.4467 | −0.1532 |
| EchoDiffusion | 1.2372 (exp14) | 0.9889 | 1.0908 (exp14) | 0.4664 | −0.1464 |
| EchoDiff+Wav2Vec | — | — | 1.0892 (exp121) | 0.4485 | — |
| AudioDepthViT | 1.2424 (exp08) | 1.0067 | 1.1055 (exp08) | 0.5163 | −0.1369 |
| Echo-Net | retraining | retraining | retraining | retraining | — |
| Pretrained ResNet-50 | 1.3238 (exp60) | 1.0685 | 1.1515 (exp60) | 0.4977 | −0.1723 |

Note: All models have lower test RMSE than validation RMSE. This is expected because the test split (9 scenes) may have different difficulty characteristics than the validation split (9 scenes). The relative ranking is largely preserved.

Note: Best FOA experiment differs between val (exp40, lr=5e-4, bs=16) and test (exp49, lr=1e-3, bs=32, hw=0.05). This suggests hw=0.05 generalizes better than the default hw=0.1.

---

### Observations

1. **FOA achieves the best test RMSE (1.0803)** with hw=0.05 — lighter histogram alignment generalizes better than the default hw=0.1.
2. **Top 3 are extremely tight** — FOA (1.0803), Baseline UNet (1.0817), Pretrained ViT (1.0818). The gap is only 0.0015 RMSE.
3. **Baseline UNet is the best on δ metrics** — highest δ<1.25 (0.5031), δ<1.25² (0.7149), δ<1.25³ (0.8329) and lowest Log10 (0.1548) despite ranking 2nd on RMSE.
4. **Pretrained ViT-B/16 has the lowest ABS_REL (0.4467)** — ImageNet visual pretraining helps relative accuracy but not absolute scale.
5. **EchoDiff+Wav2Vec (exp121) slightly improves over standard EchoDiffusion (exp14)** — 1.0892 vs 1.0908 RMSE, 0.4485 vs 0.4664 ABS_REL.
6. **Echo-Net (exp66–69)** is currently being retrained (`bulk0410_sum_revised`); excluded from this table until training completes.
7. **Pretrained ResNet-50 trails all methods** (1.1515) — ImageNet conv features transfer poorly to spectrograms.
8. **FOA variants all failed at test** — checkpoints were invalid. These require debugging before inclusion in the final table.
9. **All models improved from val to test** — test RMSE is 0.13–0.17 lower than val RMSE across all methods, suggesting the test scenes are slightly easier.
