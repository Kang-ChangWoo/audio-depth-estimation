"""Ablation variants of RenewSingleNet for exp303-304.

exp303 — RenewDPTOnlyNet: disable SF ViT + fusion, use spec ViT DPT only.
         Still predicts k_n and pred_energy (SH head + SF generator stay),
         but depth decoding uses only spec features (no fusion/refinement).
         KL loss still active (controlled by config lambda_kl_energy).

exp304 — Same architecture as exp303, but config sets lambda_kl_energy=0.
         (No code difference — just a YAML change.)
"""

import torch

from .renew_single import RenewSingleNet


class RenewDPTOnlyNet(RenewSingleNet):
    """RenewSingleNet with fusion/refinement bypassed.

    The SF ViT is NOT run. Depth decoding uses only the spec ViT
    features through the DPT neck + decoder. The FOA head and SF
    generator still produce pred_sh and pred_energy for auxiliary
    losses, but they do NOT feed back into depth prediction.
    """

    def forward(self, x, gt_energy=None, teacher_ratio=None, **_unused):
        orig_h, orig_w = x.shape[2], x.shape[3]

        # Spec ViT with taps
        spec_taps = self._run_vit_with_taps(self.vit_spec, x)
        final_spec = spec_taps['final']

        # FOA head → k_n → predicted sound field (for auxiliary losses only)
        k_n = self.foa_head(final_spec[:, 0], final_spec[:, 1:])
        pred_sf = self.sf_gen(k_n)

        # DPT-only: use spec features directly, NO SF ViT, NO fusion
        F4  = spec_taps[self.taps[0]][:, 1:]
        F8  = spec_taps[self.taps[1]][:, 1:]
        F12 = spec_taps[self.taps[2]][:, 1:]

        # DPT neck → depth decoder
        m4, m8, m12 = self.neck(F4, F8, F12)
        pred_depth = self.decoder(m4, m8, m12)
        if pred_depth.shape[-2:] != (orig_h, orig_w):
            pred_depth = torch.nn.functional.interpolate(
                pred_depth, size=(orig_h, orig_w),
                mode='bilinear', align_corners=False)

        return {
            'pred_depth':  pred_depth,
            'pred_sh':     k_n,
            'pred_energy': pred_sf,
            'k_n':         k_n,
            'sf_in':       pred_sf,
        }
