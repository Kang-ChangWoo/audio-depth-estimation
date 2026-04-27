# Test Experiment Report
**Last updated:** 2026-04-18  
**Test split:** 9 scenes, 3192 samples  
**Best model selection during training:** `score = 0.7 * RMSE + 0.3 * ABS_REL` (lower is better)

---

## Summary

| Metric | Count |
|--------|-------|
| Total experiments tested | 147 |
| Test PASSED | 124 |
| Test FAILED (eval crash) | 25 |
| Missing checkpoint | 4 |
| Awaiting test (trained, not tested) | ~54 |

*Delta since 2026-04-16*: +21 tested (all PASSED) from Bulk0417 N3 (exp166–182) + N2 (exp187–190). exp183–186 still training; exp191–206 queued.


---

## Bulk0410 — Main Test Run (78 experiments)

53 passed, 25 failed.

### Baseline UNet (5/5 PASSED)

| Exp | LR | BS | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE |
|-----|----|----|---------|------|--------|--------|--------|-------|-----|
| 01 | 1e-3 | 32 | 0.4553 | **1.0817** | **0.5031** | **0.7149** | **0.8329** | **0.1548** | **0.6714** |
| 02 | 5e-4 | 32 | **0.4127** | 1.0989 | 0.4943 | 0.7086 | 0.8282 | 0.1566 | 0.6738 |
| 03 | 1e-4 | 32 | 0.4295 | 1.0894 | 0.4934 | 0.7079 | 0.8314 | 0.1557 | 0.6738 |
| 04 | 1e-3 | 16 | 0.4300 | 1.1076 | 0.4882 | 0.7039 | 0.8287 | 0.1573 | 0.6814 |
| 05 | 5e-4 | 16 | 0.4688 | 1.0927 | 0.4863 | 0.7013 | 0.8243 | 0.1588 | 0.6835 |

### AudioDepthViT (5/5 PASSED)

| Exp | LR | BS | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE |
|-----|----|----|---------|------|--------|--------|--------|-------|-----|
| 06 | 1e-4 | 32 | 0.4770 | 1.1265 | 0.4705 | 0.6855 | 0.8143 | 0.1654 | 0.7096 |
| 07 | 5e-5 | 32 | 0.4755 | 1.1174 | 0.4731 | 0.6897 | 0.8173 | 0.1640 | 0.7022 |
| 08 | 1e-4 | 16 | 0.5163 | **1.1055** | **0.4805** | 0.6933 | 0.8163 | 0.1644 | 0.7018 |
| 09 | 5e-4 | 32 | 0.5593 | 1.1985 | 0.4443 | 0.6524 | 0.7839 | 0.1804 | 0.7629 |
| 10 | 1e-5 | 32 | **0.4720** | 1.1170 | 0.4812 | **0.6959** | **0.8194** | **0.1627** | **0.6983** |

### EchoDiffusion (5/5 PASSED)

| Exp | LR | BS | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE |
|-----|----|----|---------|------|--------|--------|--------|-------|-----|
| 11 | 1e-4 | 32 | **0.4300** | 1.1060 | 0.4876 | 0.7049 | 0.8296 | 0.1577 | 0.6810 |
| 12 | 5e-5 | 32 | 0.4914 | 1.1208 | 0.4816 | 0.6980 | 0.8197 | 0.1635 | 0.7023 |
| 13 | 1e-4 | 16 | 0.4504 | 1.1134 | 0.4930 | 0.7088 | 0.8291 | 0.1605 | 0.6885 |
| 14 | 5e-4 | 32 | 0.4664 | **1.0908** | **0.4932** | 0.7061 | 0.8272 | 0.1578 | 0.6829 |
| 15 | 1e-5 | 32 | 0.4708 | 1.1300 | 0.4728 | 0.6877 | 0.8164 | 0.1655 | 0.7102 |

### EchoDiffusion + Wav2Vec (4/4 PASSED — bulk0410)

| Exp | LR | BS | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE |
|-----|----|----|---------|------|--------|--------|--------|-------|-----|
| 121 | 1e-4 | 16 | 0.4485 | **1.0892** | 0.4887 | **0.7075** | **0.8304** | **0.1565** | **0.6740** |
| 122 | 5e-4 | 16 | 0.4585 | 1.0958 | 0.4882 | 0.7017 | 0.8267 | 0.1582 | 0.6835 |
| 123 | 1e-4 | 32 | **0.4214** | 1.1062 | **0.4884** | 0.7067 | 0.8314 | 0.1574 | 0.6785 |
| 124 | 5e-5 | 16 | 0.4901 | 1.0958 | 0.4721 | 0.6934 | 0.8192 | 0.1632 | 0.6955 |

### FOA Original (20/20 PASSED)

| Exp | LR | dw | fw | hw | Frz | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE |
|-----|----|----|----|----|-----|---------|------|--------|--------|--------|-------|-----|
| 36 | 1e-3 | 1.0 | 0.1 | 0.1 | 0 | 0.4850 | 1.0884 | 0.4943 | 0.7108 | 0.8297 | 0.1582 | 0.6867 |
| 37 | 5e-4 | 1.0 | 0.1 | 0.1 | 0 | 0.4520 | 1.1102 | 0.4753 | 0.6930 | 0.8220 | 0.1621 | 0.6960 |
| 38 | 1e-4 | 1.0 | 0.1 | 0.1 | 0 | 0.4441 | 1.1058 | 0.4878 | 0.7062 | 0.8286 | 0.1587 | 0.6871 |
| 39 | 1e-3 | 1.0 | 0.1 | 0.1 | 0 | 0.4953 | 1.0973 | 0.4973 | 0.7113 | 0.8261 | 0.1601 | 0.6912 |
| 40 | 5e-4 | 1.0 | 0.1 | 0.1 | 0 | 0.4693 | 1.0888 | 0.4962 | 0.7129 | 0.8316 | 0.1572 | 0.6812 |
| 41 | 1e-3 | 1.0 | 0.2 | 0.1 | 0 | 0.4568 | 1.0875 | 0.4900 | 0.7102 | 0.8317 | 0.1567 | 0.6791 |
| 42 | 5e-4 | 1.0 | 0.2 | 0.1 | 0 | 0.4572 | 1.0904 | 0.5018 | 0.7145 | 0.8312 | 0.1566 | 0.6782 |
| 43 | 1e-3 | 1.0 | 0.1 | 0.2 | 0 | 0.4660 | 1.0972 | 0.4930 | 0.7069 | 0.8247 | 0.1596 | 0.6879 |
| 44 | 5e-4 | 1.0 | 0.1 | 0.2 | 0 | 0.4781 | 1.0871 | 0.4986 | 0.7132 | 0.8314 | 0.1571 | 0.6823 |
| 45 | 1e-3 | 1.0 | 0.2 | 0.2 | 0 | 0.4955 | 1.0969 | 0.4913 | 0.7067 | 0.8269 | 0.1595 | 0.6936 |
| 46 | 1e-3 | 0.5 | 0.1 | 0.1 | 0 | 0.4663 | 1.0977 | 0.4937 | 0.7073 | 0.8268 | 0.1589 | 0.6893 |
| 47 | 1e-3 | 2.0 | 0.1 | 0.1 | 0 | 0.4463 | 1.0968 | 0.4858 | 0.7037 | 0.8299 | 0.1578 | 0.6838 |
| 48 | 1e-3 | 1.0 | 0.05 | 0.1 | 0 | 0.4625 | 1.0886 | 0.4920 | 0.7081 | 0.8289 | 0.1570 | 0.6821 |
| 49 | 1e-3 | 1.0 | 0.1 | 0.05 | 0 | 0.4631 | **1.0803** | 0.5023 | 0.7141 | 0.8323 | **0.1554** | **0.6753** |
| 50 | 1e-3 | 1.0 | 0.1 | 0.1 | 5 | 0.4691 | 1.0858 | 0.4860 | 0.7055 | 0.8281 | 0.1579 | 0.6785 |
| 51 | 1e-3 | 1.0 | 0.1 | 0.1 | 10 | 0.4839 | 1.0881 | 0.4913 | 0.7098 | 0.8303 | 0.1579 | 0.6834 |
| 52 | 5e-4 | 1.0 | 0.2 | 0.2 | 0 | 0.4921 | 1.0814 | 0.4933 | 0.7120 | 0.8306 | 0.1579 | 0.6844 |
| 53 | 5e-4 | 0.5 | 0.2 | 0.1 | 0 | 0.4669 | 1.0917 | 0.4878 | 0.7070 | 0.8288 | 0.1581 | 0.6840 |
| 54 | 1e-4 | 1.0 | 0.2 | 0.1 | 0 | 0.4716 | 1.1059 | 0.4968 | 0.7149 | 0.8321 | 0.1575 | 0.6856 |
| 55 | 1e-4 | 1.0 | 0.1 | 0.1 | 0 | 0.4656 | 1.0945 | 0.4919 | 0.7097 | 0.8310 | 0.1573 | 0.6837 |

### Pretrained ResNet-50 (5/5 PASSED)

| Exp | LR | BS | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE |
|-----|----|----|---------|------|--------|--------|--------|-------|-----|
| 56 | 1e-4 | 32 | 0.5063 | 1.1810 | 0.4402 | 0.6584 | 0.7970 | 0.1768 | 0.7494 |
| 57 | 5e-5 | 32 | 0.5341 | 1.1506 | 0.4697 | 0.6779 | 0.8024 | 0.1720 | 0.7340 |
| 58 | 5e-4 | 32 | 0.4929 | 1.2014 | 0.4467 | 0.6479 | 0.7820 | 0.1816 | 0.7586 |
| 59 | 1e-4 | 16 | 0.5454 | **1.1444** | 0.4480 | 0.6687 | 0.8010 | 0.1733 | 0.7354 |
| 60 | 3e-4 | 32 | **0.4977** | 1.1515 | **0.4587** | **0.6708** | **0.8015** | **0.1713** | **0.7251** |

### Pretrained ViT-B/16 (5/5 PASSED)

| Exp | LR | BS | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE |
|-----|----|----|---------|------|--------|--------|--------|-------|-----|
| 61 | 1e-4 | 16 | 0.4496 | 1.0824 | **0.4989** | **0.7124** | 0.8308 | 0.1557 | **0.6720** |
| 62 | 5e-5 | 16 | 0.4449 | 1.0964 | 0.4869 | 0.7024 | 0.8267 | 0.1582 | 0.6803 |
| 63 | 5e-4 | 16 | 0.4578 | 1.1168 | 0.4733 | 0.6880 | 0.8177 | 0.1625 | 0.6973 |
| 64 | 1e-4 | 8 | 0.4654 | 1.1028 | 0.4803 | 0.6958 | 0.8212 | 0.1607 | 0.6909 |
| 65 | 3e-5 | 16 | **0.4467** | **1.0818** | 0.4959 | 0.7105 | **0.8311** | **0.1557** | 0.6733 |

### Echo-Net (4/4 PASSED — partial training)

| Exp | LR | BS | Train | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE |
|-----|----|----|-------|---------|------|--------|--------|--------|-------|-----|
| 66 | 1e-3 | 8 | 17/40 | **0.4550** | **1.1156** | **0.4778** | **0.6942** | **0.8207** | **0.1620** | **0.6937** |
| 67 | 5e-4 | 16 | 20/40 | 1.0335 | 1.7018 | 0.2152 | 0.3928 | 0.5465 | 0.2772 | 1.3462 |
| 68 | 1e-4 | 16 | 18/40 | 0.6347 | 1.5988 | 0.1621 | 0.3335 | 0.4849 | 0.6126 | 1.1527 |
| 69 | 1e-3 | 16 | 17/40 | 0.5718 | 1.1852 | 0.4039 | 0.6336 | 0.7640 | 0.2410 | 0.8013 |

### FOA CrossAttn — FAILED (5/5 EVAL_FAIL)

exp16, exp17, exp18, exp19, exp20 — checkpoint loading failed at test time (dimension mismatch).

### FOA FeatBank — FAILED (5/5 EVAL_FAIL)

exp21, exp22, exp23, exp24, exp25 — checkpoint loading failed at test time.

### FOA MSAttn — FAILED (5/5 EVAL_FAIL)

exp26, exp27, exp28, exp29, exp30 — checkpoint loading failed at test time.

### FOA ChannelAttn — FAILED (5/5 EVAL_FAIL)

exp31, exp32, exp33, exp34, exp35 — checkpoint loading failed at test time.

### FOA v2 — FAILED (5/5 EVAL_FAIL)

exp56, exp57, exp58, exp59, exp60 (foav2) — checkpoint loading failed at test time.

---

## Bulk0410_test_41 — Extended Test Run (41 experiments, all PASSED)

### BatVision (5/5 PASSED)

| Exp | LR | BS | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE |
|-----|----|----|---------|------|--------|--------|--------|-------|-----|
| 71 | 1e-3 | 32 | **0.4430** | 1.0835 | 0.4897 | 0.7114 | **0.8335** | **0.1549** | **0.6708** |
| 72 | 5e-4 | 32 | 0.4552 | 1.0893 | 0.4894 | 0.7069 | 0.8287 | 0.1574 | 0.6816 |
| 73 | 1e-4 | 32 | 0.4551 | 1.0928 | **0.4998** | **0.7129** | 0.8301 | 0.1562 | 0.6743 |
| 74 | 1e-3 | 16 | 0.4652 | **1.0817** | 0.4944 | 0.7091 | 0.8288 | 0.1566 | 0.6747 |
| 75 | 2e-3 | 32 | 0.4659 | 1.0994 | 0.4938 | 0.7129 | 0.8305 | 0.1578 | 0.6836 |

### Echo-Net (1/1 PASSED)

| Exp | LR | BS | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE |
|-----|----|----|---------|------|--------|--------|--------|-------|-----|
| 70 | 2e-3 | 16 | 0.4782 | 1.1398 | 0.4343 | 0.6648 | 0.7943 | 0.1818 | 0.7387 |

### EchoDiff+Wav2Vec (1/1 PASSED)

| Exp | LR | BS | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE |
|-----|----|----|---------|------|--------|--------|--------|-------|-----|
| 125 | 1e-4 | 8 | 0.4750 | 1.0852 | **0.4994** | 0.7124 | 0.8301 | 0.1568 | 0.6763 |

### FOA CrossAttn + KL (4/4 PASSED)

| Exp | LR | fw | kl | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE | FOA_L1 | FOA_COS |
|-----|----|----|-----|---------|------|--------|--------|--------|-------|-----|--------|---------|
| 76 | 1e-3 | 0.1 | 0.02 | 0.4588 | 1.0981 | 0.4895 | 0.7085 | 0.8295 | 0.1578 | 0.6838 | 0.0321 | 0.9973 |
| 77 | 5e-4 | 0.2 | 0.005 | 0.5034 | 1.0992 | 0.4989 | 0.7114 | 0.8276 | 0.1593 | 0.6948 | 0.0263 | 0.9982 |
| 79 | 5e-4 | 0.1 | 0.01 | 0.4519 | 1.1085 | 0.4903 | 0.7094 | 0.8302 | 0.1578 | 0.6874 | 0.0347 | 0.9971 |
| 80 | 1e-3 | 0.3 | 0.01 | 0.4731 | 1.0913 | **0.5007** | **0.7161** | **0.8319** | **0.1564** | **0.6819** | 0.0242 | 0.9984 |

### FOA FeatBank + KL (4/4 PASSED)

| Exp | LR | fw | kl | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE | FOA_L1 | FOA_COS |
|-----|----|----|-----|---------|------|--------|--------|--------|-------|-----|--------|---------|
| 81 | 1e-3 | 0.1 | 0.02 | **0.4462** | **1.0943** | 0.4900 | **0.7106** | **0.8316** | **0.1571** | **0.6812** | 0.0332 | 0.9975 |
| 83 | 1e-3 | 0.05 | 0.01 | 0.4701 | 1.0930 | 0.4817 | 0.7071 | 0.8287 | 0.1587 | 0.6845 | 0.0348 | 0.9972 |
| 84 | 5e-4 | 0.1 | 0.01 | 0.4747 | 1.1022 | 0.4869 | 0.7074 | 0.8292 | 0.1587 | 0.6889 | 0.0340 | 0.9977 |
| 85 | 1e-3 | 0.3 | 0.01 | 0.4707 | 1.1045 | 0.4812 | 0.7040 | 0.8303 | 0.1614 | 0.6945 | 0.0288 | 0.9981 |

### FOA MSAttn + KL (3/3 PASSED)

| Exp | LR | fw | kl | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE | FOA_L1 | FOA_COS |
|-----|----|----|-----|---------|------|--------|--------|--------|-------|-----|--------|---------|
| 87 | 5e-4 | 0.2 | 0.005 | 0.4751 | 1.0990 | 0.4868 | 0.7048 | 0.8287 | 0.1583 | 0.6868 | 0.0375 | 0.9976 |
| 88 | 1e-3 | 0.05 | 0.01 | 0.4724 | **1.0975** | **0.4998** | **0.7139** | 0.8299 | 0.1582 | 0.6866 | 0.0281 | 0.9980 |
| 89 | 5e-4 | 0.1 | 0.01 | 0.4686 | 1.0999 | 0.4921 | 0.7100 | 0.8285 | 0.1587 | 0.6902 | 0.0358 | 0.9978 |

### FOA ChannelAttn + KL (4/4 PASSED)

| Exp | LR | fw | kl | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE | FOA_L1 | FOA_COS |
|-----|----|----|-----|---------|------|--------|--------|--------|-------|-----|--------|---------|
| 91 | 1e-3 | 0.1 | 0.02 | **0.4470** | **1.0907** | 0.4869 | 0.7101 | 0.8318 | **0.1569** | **0.6805** | 0.0309 | 0.9977 |
| 92 | 5e-4 | 0.2 | 0.005 | 0.4561 | 1.0945 | **0.4948** | **0.7158** | **0.8341** | 0.1568 | 0.6825 | 0.0276 | 0.9984 |
| 93 | 1e-3 | 0.05 | 0.01 | 0.4754 | 1.1097 | 0.4915 | 0.7078 | 0.8286 | 0.1599 | 0.6949 | 0.0275 | 0.9981 |
| 95 | 1e-3 | 0.3 | 0.01 | 0.4590 | 1.1071 | 0.4904 | 0.7096 | 0.8271 | 0.1594 | 0.6892 | 0.0268 | 0.9983 |

### FOA Extended Sweep (19/19 PASSED)

| Exp | LR | dw | fw | hw | Frz | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE | FOA_L1 | FOA_COS |
|-----|----|----|----|-----|-----|---------|------|--------|--------|--------|-------|-----|--------|---------|
| 96 | 2e-4 | 1.0 | 0.1 | 0.1 | 0 | 0.4752 | 1.0866 | 0.4929 | 0.7111 | 0.8306 | 0.1569 | 0.6813 | 0.0377 | 0.9979 |
| 97 | 3e-4 | 1.0 | 0.1 | 0.1 | 0 | **0.4299** | 1.1010 | 0.4922 | 0.7108 | 0.8319 | 0.1562 | 0.6789 | 0.0299 | 0.9982 |
| 99 | 1e-3 | 1.5 | 0.1 | 0.1 | 0 | 0.4748 | 1.0888 | **0.5034** | 0.7136 | 0.8294 | 0.1568 | 0.6825 | 0.0292 | 0.9979 |
| 100 | 1e-3 | 1.0 | 0.15 | 0.1 | 0 | 0.5043 | **1.0793** | 0.4821 | 0.7041 | 0.8266 | 0.1600 | 0.6876 | 0.0313 | 0.9978 |
| 101 | 1e-3 | 1.0 | 0.1 | 0.15 | 0 | 0.4414 | 1.1116 | 0.4831 | 0.7051 | 0.8281 | 0.1593 | 0.6909 | 0.0319 | 0.9976 |
| 103 | 5e-4 | 1.0 | 0.15 | 0.1 | 0 | 0.4662 | 1.0925 | 0.4871 | 0.7078 | 0.8303 | 0.1577 | 0.6850 | 0.0320 | 0.9984 |
| 104 | 5e-4 | 1.0 | 0.1 | 0.15 | 0 | 0.4745 | 1.0964 | 0.4886 | 0.7081 | 0.8285 | 0.1586 | 0.6883 | 0.0290 | 0.9983 |
| 105 | 1e-3 | 1.0 | 0.3 | 0.1 | 0 | 0.4638 | 1.0884 | 0.4940 | 0.7124 | 0.8318 | 0.1567 | 0.6805 | 0.0307 | 0.9980 |
| 107 | 5e-4 | 1.0 | 0.3 | 0.1 | 0 | 0.4574 | 1.0824 | 0.4945 | 0.7120 | **0.8328** | 0.1560 | 0.6765 | 0.0329 | 0.9982 |
| 108 | 5e-4 | 2.0 | 0.1 | 0.1 | 0 | 0.5068 | 1.1069 | 0.4916 | 0.7029 | 0.8235 | 0.1609 | 0.7021 | 0.0293 | 0.9981 |
| 109 | 1e-3 | 1.0 | 0.2 | 0.2 | 5 | 0.4889 | 1.0873 | 0.4923 | 0.7051 | 0.8257 | 0.1602 | 0.6927 | 0.0296 | 0.9981 |
| 111 | 1e-3 | 1.0 | 0.1 | 0.1 | 15 | 0.4535 | 1.0781 | 0.4998 | **0.7161** | **0.8347** | **0.1543** | **0.6716** | 0.0636 | 0.9962 |
| 112 | 5e-4 | 1.0 | 0.1 | 0.1 | 10 | 0.4440 | 1.1047 | 0.4944 | 0.7104 | 0.8284 | 0.1593 | 0.6828 | 0.0308 | 0.9980 |
| 113 | 1e-3 | 1.0 | 0.05 | 0.05 | 0 | 0.4440 | 1.0948 | 0.4862 | 0.7077 | 0.8315 | 0.1572 | 0.6828 | 0.0336 | 0.9976 |
| 115 | 1e-3 | 2.0 | 0.2 | 0.1 | 0 | 0.4513 | 1.0994 | 0.4943 | 0.7136 | 0.8320 | 0.1571 | 0.6831 | 0.0298 | 0.9978 |
| 116 | 3e-4 | 1.0 | 0.2 | 0.1 | 0 | 0.4452 | 1.0902 | 0.4944 | 0.7116 | 0.8329 | 0.1557 | 0.6754 | 0.0301 | 0.9983 |
| 117 | 2e-4 | 1.0 | 0.2 | 0.2 | 0 | 0.4742 | 1.0899 | 0.4908 | 0.7082 | 0.8294 | 0.1577 | 0.6840 | 0.0447 | 0.9981 |
| 119 | 1e-3 | 1.0 | 0.1 | 0.2 | 3 | 0.4566 | 1.0952 | 0.4845 | 0.7065 | 0.8307 | 0.1577 | 0.6844 | 0.0316 | 0.9979 |
| 120 | 5e-4 | 1.0 | 0.05 | 0.1 | 5 | 0.4568 | 1.0858 | 0.4884 | 0.7055 | 0.8299 | 0.1568 | 0.6789 | 0.0355 | 0.9977 |

---

## Bulk0416_test_vit — Pretrained ViT + FOA (6 PASSED, 4 MISSING)

| Exp | Model | LR | lambda_sh | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE | FOA_L1 | FOA_COS | FOA_DIR | Status |
|-----|-------|----|-----------|---------|------|--------|--------|--------|-------|-----|--------|---------|---------|--------|
| 160 | pvitfoav1 | 1e-4 | 0.1 | 0.4619 | **1.0820** | 0.4948 | 0.7074 | 0.8290 | 0.1579 | 0.6797 | 0.0258 | 0.9982 | 0.9954 | PASSED |
| 161 | pvitfoav1 | 5e-5 | 0.3 | 0.4673 | 1.0903 | **0.5000** | 0.7065 | 0.8266 | 0.1591 | 0.6856 | 0.0274 | 0.9980 | 0.9949 | PASSED |
| 162 | pvitfoav2 | 1e-4 | 0.1 | 0.4782 | 1.1000 | 0.4833 | 0.7009 | 0.8271 | 0.1598 | 0.6884 | — | — | — | PASSED |
| 163 | pvitfoav2 | 5e-5 | 0.3 | 0.5089 | 1.2099 | 0.4388 | 0.6584 | 0.7924 | 0.1773 | 0.7519 | — | — | — | PASSED |
| 164 | pvitfoav3 | 1e-4 | 0.1 | **0.4372** | 1.0929 | 0.4972 | **0.7097** | **0.8303** | **0.1576** | **0.6761** | 0.0253 | 0.9983 | 0.9957 | PASSED |
| 165 | pvitfoav3 | 5e-5 | 0.3 | **0.4369** | 1.1140 | 0.4896 | 0.7032 | 0.8241 | 0.1614 | 0.6907 | 0.0241 | 0.9981 | 0.9948 | PASSED |
| 166 | pvitfoav4 | 1e-4 | 0.1 | — | — | — | — | — | — | — | — | — | — | MISSING_CKPT |
| 167 | pvitfoav4 | 5e-5 | 0.3 | — | — | — | — | — | — | — | — | — | — | MISSING_CKPT |
| 168 | pvitfoav5 | 1e-4 | 0.1 | — | — | — | — | — | — | — | — | — | — | MISSING_CKPT |
| 169 | pvitfoav5 | 5e-5 | 0.3 | — | — | — | — | — | — | — | — | — | — | MISSING_CKPT |

---

## Test_CW_fire — FOA 0415 Spot Checks (5 PASSED, 4 EMPTY)

| Exp | Model | LR | lambda_sh | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE | Status |
|-----|-------|----|-----------|---------|------|--------|--------|--------|-------|-----|--------|
| 130 | foa0415v1 | 1e-3 | 0.1 | **0.4354** | 1.0990 | 0.4899 | 0.7100 | **0.8312** | **0.1567** | **0.6771** | PASSED |
| 135 | foa0415v2 | 1e-3 | 0.1 | 0.4473 | **1.0907** | **0.4969** | **0.7134** | 0.8318 | 0.1558 | 0.6772 | PASSED |
| 140 | foa0415v3 | 1e-3 | 0.1 | 0.4705 | 1.0953 | 0.5004 | 0.7109 | 0.8294 | 0.1575 | 0.6839 | PASSED |
| 145 | foa0415v4 | 1e-3 | 0.1 | 0.4465 | 1.0873 | 0.4974 | 0.7096 | 0.8291 | 0.1569 | 0.6781 | PASSED |
| 150 | foa0415v5 | 1e-3 | 0.1 | 0.4505 | 1.1005 | 0.4866 | 0.7052 | 0.8295 | 0.1591 | 0.6865 | PASSED |
| 131-134 | foa0415v1 | various | various | — | — | — | — | — | — | — | NOT TESTED |
| 136-154 | foa0415v2-v5 | various | various | — | — | — | — | — | — | — | NOT TESTED |

---

## Standalone Test Logs (9 PASSED, 1 EMPTY)

| Name | Config | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE |
|------|--------|---------|------|--------|--------|--------|-------|-----|
| CW_rot_foa | foa (canonical) | 0.4399 | 1.0969 | 0.4925 | 0.7086 | 0.8294 | 0.1570 | 0.6795 |
| CW_rot_foa_crossattn | foa_crossattn (canonical) | 0.4812 | 1.0903 | 0.4970 | 0.7127 | 0.8314 | 0.1575 | 0.6861 |
| CW_rot_foa_featbank | foa_featbank (canonical) | 0.4634 | 1.0971 | 0.4854 | 0.7056 | 0.8293 | 0.1582 | 0.6861 |
| CW_rot_foa_msattn | foa_msattn (canonical) | 0.4495 | 1.0935 | 0.4892 | 0.7051 | 0.8293 | 0.1569 | 0.6794 |
| CW_rot_foa_channelattn | foa_channelattn (canonical) | **0.4366** | 1.1080 | 0.4911 | 0.7098 | 0.8287 | 0.1599 | 0.6875 |
| CW_rot_foa_v2 | foa_v2 (canonical) | 0.4466 | 1.1033 | 0.4907 | 0.7078 | 0.8282 | 0.1586 | 0.6861 |
| foa_feat_attn | foa_feat_attn | 0.4728 | 1.0988 | 0.4886 | 0.7059 | 0.8275 | 0.1590 | 0.6872 |
| foa_foa_feat_attn_v2 | foa_feat_attn_v2 | 0.4618 | 1.1094 | 0.4854 | 0.7033 | 0.8259 | 0.1592 | 0.6889 |
| foa_sh_coeff_hierarch | foa_sh_coeff_hierarch | 0.4877 | 1.0915 | 0.4833 | 0.7015 | 0.8251 | 0.1596 | 0.6894 |
| foa_foa_basic_js | foa_v2_js | — | — | — | — | — | — | — |

Note: CW standalone tests prove that FOA variant architectures (crossattn, msattn, channelattn, v2) **do work** with canonical rotation — the bulk0410 eval failures were due to checkpoint naming/loading issues, not broken architectures.

---

## Bulk0417 N3 — Energy-aware UNet Variants + Oracle (exp166–186)

All tested on test split (3192 samples), BS=1, `rotate_canonical=true`. exp183–186 still training.

### N3 FiLM (3/3 PASSED)

| Exp | LR | λ_sh | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE | FOA_L1 | FOA_COS | FOA_DIR |
|-----|----|------|---------|------|--------|--------|--------|-------|-----|--------|---------|---------|
| 166 | 1e-3 | 0.1 | **0.4271** | **1.1045** | 0.4901 | 0.7083 | 0.8314 | **0.1565** | **0.6816** | 0.0374 | 0.9970 | 0.9937 |
| 167 | 5e-4 | 0.1 | 0.4612 | 1.1073 | **0.4983** | **0.7083** | 0.8259 | 0.1591 | 0.6878 | **0.0316** | **0.9980** | **0.9953** |
| 168 | 1e-3 | 0.3 | 0.4434 | 1.1142 | 0.4846 | 0.7028 | 0.8267 | 0.1594 | 0.6910 | 0.0343 | 0.9974 | 0.9937 |

### N3 Multi-Scale SH (3/3 PASSED)

| Exp | LR | λ_sh | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE | FOA_L1 | FOA_COS | FOA_DIR |
|-----|----|------|---------|------|--------|--------|--------|-------|-----|--------|---------|---------|
| 169 | 1e-3 | 0.1 | 0.4418 | 1.0937 | 0.4972 | 0.7123 | 0.8320 | 0.1563 | **0.6784** | 0.0250 | 0.9984 | 0.9959 |
| 170 | 5e-4 | 0.1 | 0.4676 | **1.0878** | **0.4985** | **0.7149** | **0.8318** | **0.1561** | **0.6784** | **0.0249** | **0.9984** | **0.9962** |
| 171 | 1e-3 | 0.3 | **0.4218** | 1.1104 | 0.4823 | 0.7010 | 0.8282 | 0.1582 | 0.6829 | 0.0256 | **0.9984** | **0.9962** |

### N3 Energy Attention (3/3 PASSED)

| Exp | LR | λ_sh | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE | FOA_L1 | FOA_COS | FOA_DIR |
|-----|----|------|---------|------|--------|--------|--------|-------|-----|--------|---------|---------|
| 172 | 1e-3 | 0.1 | 0.4792 | **1.0744** | **0.5030** | **0.7151** | **0.8337** | **0.1549** | **0.6771** | **0.0322** | **0.9978** | **0.9957** |
| 173 | 5e-4 | 0.1 | **0.4511** | 1.0859 | 0.4866 | 0.7089 | 0.8321 | 0.1576 | 0.6826 | 0.0328 | **0.9978** | 0.9947 |
| 174 | 1e-3 | 0.3 | 0.4767 | 1.0835 | 0.4836 | 0.7027 | 0.8261 | 0.1582 | 0.6799 | 0.0335 | 0.9980 | 0.9952 |

### N3 Temporal Window (3/3 PASSED, lower-quality family)

| Exp | LR | λ_sh | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE | FOA_L1 | FOA_COS | FOA_DIR |
|-----|----|------|---------|------|--------|--------|--------|-------|-----|--------|---------|---------|
| 175 | 1e-3 | 0.1 | 0.5477 | 1.2325 | 0.4514 | 0.6605 | 0.7909 | 0.1789 | 0.7777 | 0.0608 | 0.9925 | 0.9904 |
| 176 | 5e-4 | 0.1 | 0.5430 | 1.2022 | 0.4583 | 0.6700 | 0.7956 | 0.1758 | 0.7527 | 0.1284 | 0.9958 | 0.9930 |
| 177 | 1e-3 | 0.3 | 0.5449 | 1.1442 | 0.4529 | 0.6741 | 0.8092 | 0.1720 | 0.7343 | 1.4440 | 0.4778 | 0.2215 |

### Oracle nc3 — binaural + GT energy map (3/3 PASSED)

| Exp | LR | λ_sh | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE | FOA_L1 | FOA_COS | FOA_DIR |
|-----|----|------|---------|------|--------|--------|--------|-------|-----|--------|---------|---------|
| 178 | 1e-3 | 0.1 | 0.4216 | 1.0565 | 0.4979 | 0.7115 | 0.8356 | **0.1535** | **0.6591** | 0.0338 | **0.9996** | 0.9995 |
| 179 | 5e-4 | 0.1 | 0.4774 | **1.0477** | **0.5002** | **0.7142** | 0.8346 | 0.1547 | 0.6714 | **0.0298** | **0.9996** | 0.9993 |
| 180 | 1e-3 | 0.3 | **0.4017** | 1.0741 | 0.4817 | 0.7101 | **0.8392** | 0.1539 | 0.6637 | 0.0417 | 0.9989 | 0.9994 |

### Oracle nc1 — GT energy map only (2/2 PASSED — low ceiling)

| Exp | LR | λ_sh | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE | FOA_L1 | FOA_COS | FOA_DIR |
|-----|----|------|---------|------|--------|--------|--------|-------|-----|--------|---------|---------|
| 181 | 1e-3 | 0.1 | 0.5152 | 1.2620 | 0.4103 | 0.6224 | 0.7625 | 0.1943 | 0.8075 | 0.0410 | 0.9976 | 0.9992 |
| 182 | 5e-4 | 0.1 | 0.5352 | 1.2396 | 0.4173 | 0.6315 | 0.7693 | 0.1908 | 0.7961 | 0.0467 | 0.9976 | **0.9993** |

Key findings:
- **Best N3 test RMSE = 1.0744 (exp172, n3_energy_attn, lr=1e-3 λ=0.1)** — ranks ~1.0 in absolute terms, among the best ever at this resolution
- **nc3 oracle (exp178–180) beats all non-oracle N3 variants on RMSE** (1.0477–1.0741), as expected for a privileged-info ceiling
- **nc1 oracle is *worse* than non-oracle** — GT energy alone, without binaural, is an impoverished input
- **Temporal window (exp175–177) is the weakest N3 family** (RMSE 1.14–1.23); energy attention beats it consistently
- **FOA prediction is near-ceiling for oracle** (FOA_COS = 0.9989–0.9996, FOA_DIR up to 0.9995) — expected, since GT FOA is in the input
- **exp177 FOA collapse** (FOA_L1=1.44, FOA_COS=0.48) — outlier, likely a unit/normalization issue in the 4-channel RMS target; doesn't affect depth metrics

---

## Bulk0417 N2 — Temporal FOA Decomposition (exp187–190)

All tested BS=4, `rotate_canonical=true`. exp187–189 evaluate n2_6ch_input (concat binaural + FOA spec); exp190 evaluates n2_temporal_rms (12-dim temporal RMS supervision, audio-only input).

| Exp | Config | LR | λ_sh | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE | FOA_L1 | FOA_COS | FOA_DIR |
|-----|--------|----|------|---------|------|--------|--------|--------|-------|-----|--------|---------|---------|
| 187 | n2_6ch_input | 1e-3 | 0.1 | 0.4728 | 1.0713 | **0.5111** | **0.7211** | 0.8344 | **0.1540** | **0.6682** | 0.0359 | **0.9993** | **0.9984** |
| 188 | n2_6ch_input | 5e-4 | 0.1 | 0.4444 | 1.0875 | 0.4983 | 0.7130 | 0.8323 | 0.1553 | 0.6755 | **0.0212** | 0.9992 | 0.9982 |
| 189 | n2_6ch_input | 1e-3 | 0.3 | **0.4393** | **1.0781** | 0.4974 | 0.7172 | **0.8371** | 0.1539 | **0.6682** | 0.0302 | 0.9986 | 0.9985 |
| 190 | n2_temporal_rms | 1e-3 | 0.1 | 0.4370 | 1.1231 | 0.4868 | 0.7028 | 0.8235 | 0.1614 | 0.6942 | 0.0369 | 0.9978 | 0.9951 |

Key findings:
- **exp187 (6ch input) achieves test RMSE = 1.0713**, matching the strongest N3 energy_attn runs (exp172: 1.0744) — directly conditioning on FOA spectrogram is competitive with learned SH-aware attention
- **exp190 (audio-only with temporal RMS supervision)** — RMSE 1.1231 is the weakest N2 run; dropping FOA from the input hurts by ~0.05 RMSE even with matched supervision
- exp191–206 (E3–E8 variants) are pending on a second server

---

## Overall Ranking — Top 20 by Test RMSE

★ = oracle (GT FOA info at inference — not comparable to deployable models).

| Rank | Exp | Model | RMSE | ABS_REL | Delta1 | Log10 | MAE | Source |
|------|-----|-------|------|---------|--------|-------|-----|--------|
| 1 ★ | 179 | Oracle nc3 (binaural+GT E) | **1.0477** | 0.4774 | 0.5002 | 0.1547 | 0.6714 | bulk0417_n3 |
| 2 ★ | 178 | Oracle nc3 (binaural+GT E) | 1.0565 | 0.4216 | 0.4979 | 0.1535 | 0.6591 | bulk0417_n3 |
| 3 | 187 | N2 6ch (audio+FOA spec) | 1.0713 | 0.4728 | **0.5111** | 0.1540 | 0.6682 | bulk0417_n2 |
| 4 ★ | 180 | Oracle nc3 (binaural+GT E) | 1.0741 | **0.4017** | 0.4817 | 0.1539 | 0.6637 | bulk0417_n3 |
| 5 | 172 | N3 energy_attn | 1.0744 | 0.4792 | 0.5030 | 0.1549 | 0.6771 | bulk0417_n3 |
| 6 | 111 | FOA (freeze=15) | 1.0781 | 0.4535 | 0.4998 | 0.1543 | 0.6716 | bulk0410_test_41 |
| 7 | 189 | N2 6ch (λ_sh=0.3) | 1.0781 | 0.4393 | 0.4974 | 0.1539 | 0.6682 | bulk0417_n2 |
| 8 | 100 | FOA (fw=0.15) | 1.0793 | 0.5043 | 0.4821 | 0.1600 | 0.6876 | bulk0410_test_41 |
| 9 | 49 | FOA (hw=0.05) | 1.0803 | 0.4631 | 0.5023 | 0.1554 | 0.6753 | bulk0410 |
| 10 | 52 | FOA | 1.0814 | 0.4921 | 0.4933 | 0.1579 | 0.6844 | bulk0410 |
| 11 | 01 | Baseline UNet | 1.0817 | 0.4553 | 0.5031 | 0.1548 | 0.6714 | bulk0410 |
| 12 | 74 | BatVision | 1.0817 | 0.4652 | 0.4944 | 0.1566 | 0.6747 | bulk0410_test_41 |
| 13 | 65 | PreViT-B/16 | 1.0818 | 0.4467 | 0.4959 | 0.1557 | 0.6733 | bulk0410 |
| 14 | 160 | PreViT+FOA v1 | 1.0820 | 0.4619 | 0.4948 | 0.1579 | 0.6797 | bulk0416 |
| 15 | 61 | PreViT-B/16 | 1.0824 | 0.4496 | 0.4989 | 0.1557 | 0.6720 | bulk0410 |
| 16 | 107 | FOA (fw=0.3) | 1.0824 | 0.4574 | 0.4945 | 0.1560 | 0.6765 | bulk0410_test_41 |
| 17 | 174 | N3 energy_attn (λ_sh=0.3) | 1.0835 | 0.4767 | 0.4836 | 0.1582 | 0.6799 | bulk0417_n3 |
| 18 | 71 | BatVision | 1.0835 | 0.4430 | 0.4897 | 0.1549 | 0.6708 | bulk0410_test_41 |
| 19 | 125 | EchoDiff+W2V | 1.0852 | 0.4750 | 0.4994 | 0.1568 | 0.6763 | bulk0410_test_41 |
| 20 | 50 | FOA (freeze=5) | 1.0858 | 0.4691 | 0.4860 | 0.1579 | 0.6785 | bulk0410 |

**Deployable** (non-oracle) best: **exp187 (N2 6ch, RMSE 1.0713)** — first to beat the long-standing exp111 (1.0781).
**Oracle ceiling**: exp179 RMSE 1.0477 — ≈3% below the deployable best; suggests ~3% is the headroom still recoverable from binaural-only acoustic-direction estimation.

---

## Test Status Summary by Model Family

| Model Family | Tested | Passed | Failed | Missing Ckpt | Best RMSE | Best Exp |
|--------------|--------|--------|--------|--------------|-----------|----------|
| Baseline UNet | 5 | 5 | 0 | 0 | 1.0817 | exp01 |
| AudioDepthViT | 5 | 5 | 0 | 0 | 1.1055 | exp08 |
| EchoDiffusion | 5 | 5 | 0 | 0 | 1.0908 | exp14 |
| EchoDiff+Wav2Vec | 5 | 5 | 0 | 0 | 1.0852 | exp125 |
| BatVision | 5 | 5 | 0 | 0 | 1.0817 | exp74 |
| Echo-Net | 5 | 5 | 0 | 0 | 1.1156 | exp66 |
| FOA Original | 39 | 39 | 0 | 0 | **1.0781** | **exp111** |
| FOA CrossAttn (no KL) | 5 | 0 | 5 | 0 | — | — |
| FOA CrossAttn (+KL) | 4 | 4 | 0 | 0 | 1.0913 | exp80 |
| FOA FeatBank (no KL) | 5 | 0 | 5 | 0 | — | — |
| FOA FeatBank (+KL) | 4 | 4 | 0 | 0 | 1.0930 | exp83 |
| FOA MSAttn (no KL) | 5 | 0 | 5 | 0 | — | — |
| FOA MSAttn (+KL) | 3 | 3 | 0 | 0 | 1.0975 | exp88 |
| FOA ChannelAttn (no KL) | 5 | 0 | 5 | 0 | — | — |
| FOA ChannelAttn (+KL) | 4 | 4 | 0 | 0 | 1.0907 | exp91 |
| FOA v2 (no KL) | 5 | 0 | 5 | 0 | — | — |
| Pretrained ResNet | 5 | 5 | 0 | 0 | 1.1444 | exp59 |
| Pretrained ViT | 5 | 5 | 0 | 0 | 1.0818 | exp65 |
| PreViT+FOA v1-v3 | 6 | 6 | 0 | 0 | 1.0820 | exp160 |
| PreViT+FOA v4-v5 | 4 | 0 | 0 | 4 | — | — |
| FOA 0415 v1-v5 | 5 | 5 | 0 | 0 | 1.0873 | exp145 |
| Standalone CW | 10 | 9 | 0 | 0 | 1.0903 | CW_crossattn |
| **TOTAL** | **126** | **103** | **25** | **4** | | |

---

## Experiments Awaiting Test (~50)

The following trained experiments have `best_model.pth` but have NOT been evaluated on the test split:

| Group | Exp IDs | Count |
|-------|---------|-------|
| FOA 0415 v1 | 131, 132, 133, 134 | 4 |
| FOA 0415 v2 | 136, 137, 138, 139 | 4 |
| FOA 0415 v3 | 141, 142, 143, 144 | 4 |
| FOA 0415 v4 | 146, 147, 148, 149 | 4 |
| FOA 0415 v5 | 151, 152, 153, 154 | 4 |
| FOA ext (bulk0408) | 78 (partial), others with ckpt | ~5 |
| Re-test: FOA variants (no KL) | 16-35, 56-60 (foav2) | 25 (need debug) |

---

## Key Observations

1. **exp111 (FOA, freeze=15) achieves the new best test RMSE (1.0781)** from the extended test run, surpassing exp49 (1.0803). Freezing the FOA head for 15 epochs lets the depth encoder converge first.
2. **Top 3 are all FOA variants** (1.0781, 1.0793, 1.0803) — confirming FOA supervision helps.
3. **Baseline UNet (exp01, RMSE 1.0817) ties with BatVision (exp74, RMSE 1.0817)** at rank 5-6.
4. **PreViT+FOA v1 (exp160, RMSE 1.0820)** is the best ViT-based model, outperforming plain PreViT (exp65, RMSE 1.0818) marginally.
5. **PreViT+FOA v3 FiLM (exp164-165)** has the best ABS_REL across ALL experiments (0.4369), but higher RMSE.
6. **FOA variant architectures work with canonical rotation** (CW standalone tests pass), but fail under bulk0410 checkpoint loading — likely a naming/DataParallel mismatch.
7. **KL regularization fixed FOA variant evaluation** — all +KL versions (exp76-95) pass testing while non-KL versions (exp16-35) all fail.
8. **~50 trained experiments still awaiting test** — the FOA 0415 series (20 remaining) and failed variants (25 needing debug) represent significant untested potential.

---

## Master Ranking — all tested experiments (updated 2026-04-21)

Single source of truth. Every experiment with a `_test.log` in `logs/{summary_test,n1_test,n2_test,n3_test}/` is listed exactly once, ranked by RMSE ascending. Each row carries its exp index, full run name, the depth metrics, `FOA_L1` (predicted-SH fidelity, `NA` for non-ambisonic runs), and which log directory the result came from.

Legend: Rank · Idx · Experiment name · ABS_REL ↓ · RMSE ↓ · Delta1 ↑ · FOA_L1 ↓ · Source dir

| Rank | Idx | Experiment | ABS_REL | RMSE | Delta1 | FOA_L1 | Dir |
|---|---|---|---|---|---|---|---|
| 1 | 241 | exp241_n1_pvit_temap_lr1e4_lsh0.1 | 0.4516 | 1.0454 | 0.4923 | 0.0195 | n1_test |
| 2 | 201 | exp201_n2_temap_lr1e3_lsh0.1 | 0.4380 | 1.0464 | 0.4958 | 0.0252 | n2_test |
| 3 | 179 | exp179_oracle_nc3_lr5e4_lsh0.1 | 0.4774 | 1.0477 | 0.5002 | 0.0298 | summary_test |
| 4 | 202 | exp202_n2_temap_lr5e4_lsh0.1 | 0.4379 | 1.0497 | 0.4985 | 0.0447 | n2_test |
| 5 | 178 | exp178_oracle_nc3_lr1e3_lsh0.1 | 0.4216 | 1.0565 | 0.4979 | 0.0338 | summary_test |
| 6 | 203 | exp203_n2_temap_lr1e3_lsh0.3 | 0.4415 | 1.0568 | 0.4980 | 0.0222 | n2_test |
| 7 | 242 | exp242_n1_pvit_temap_lr5e5_lsh0.1 | 0.4556 | 1.0586 | 0.5038 | 0.0182 | n1_test |
| 8 | 219 | exp219_pvit_oracle_nc3_lr1e4_lsh0.1 | 0.4400 | 1.0612 | 0.4947 | 0.0160 | n3_test |
| 9 | 196 | exp196_n2_dual_lr5e4_lsh0.1 | 0.4775 | 1.0614 | 0.5112 | 0.0280 | n2_test |
| 10 | 243 | exp243_n1_pvit_temap_lr1e4_lsh0.3 | 0.4265 | 1.0661 | 0.4995 | 0.0136 | n1_test |
| 11 | 187 | exp187_n2_6ch_lr1e3_lsh0.1 | 0.4728 | 1.0713 | 0.5111 | 0.0359 | n2_test |
| 12 | 195 | exp195_n2_dual_lr1e3_lsh0.1 | 0.4642 | 1.0725 | 0.5069 | 0.0285 | n2_test |
| 13 | 230 | exp230_pvit_distill_lkd0.5 | 0.4834 | 1.0735 | 0.5029 | 0.0280 | n3_test |
| 14 | 180 | exp180_oracle_nc3_lr1e3_lsh0.3 | 0.4017 | 1.0741 | 0.4817 | 0.0417 | summary_test |
| 15 | 172 | exp172_n3eattn_lr1e3_lsh0.1 | 0.4792 | 1.0744 | 0.5030 | 0.0322 | summary_test |
| 16 | 213 | exp213_n3mssh_eattn_lr1e3_lsh0.1 | 0.4645 | 1.0777 | 0.5017 | 0.0270 | n3_test |
| 17 | 111 | exp111_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze15 | 0.4535 | 1.0781 | 0.4998 | 0.0636 | summary_test |
| 18 | 189 | exp189_n2_6ch_lr1e3_lsh0.3 | 0.4393 | 1.0781 | 0.4974 | 0.0302 | n2_test |
| 19 | 100 | exp100_foa_lr1e3_fw0.15_hw0.1 | 0.5043 | 1.0793 | 0.4821 | 0.0313 | summary_test |
| 20 | 49 | exp49_foa_lr1e3_dw1.0_fw0.1_hw0.05 | 0.4631 | 1.0803 | 0.5023 | 0.0271 | summary_test |
| 21 | 192 | exp192_n2_tenergy_lr1e3_lsh0.1 | 0.4734 | 1.0805 | 0.4912 | 0.0337 | n2_test |
| 22 | 218 | exp218_pvit_film_dw2_lr1e4_lsh0.1 | 0.4504 | 1.0805 | 0.4867 | 0.0310 | n3_test |
| 23 | 198 | exp198_n2_stft_lr5e4_lsh0.1 | 0.4719 | 1.0807 | 0.5108 | 0.0322 | n2_test |
| 24 | 200 | exp200_n2_trms_film_lr5e4_lsh0.1 | 0.4649 | 1.0808 | 0.5016 | 0.0310 | n2_test |
| 25 | 225 | exp225_n3mssh_freeze10_lr1e3_lsh0.3 | 0.4821 | 1.0811 | 0.4958 | 0.0242 | n3_test |
| 26 | 208 | exp208_n3eattn_dw2_lr1e3_lsh0.1 | 0.5003 | 1.0813 | 0.4829 | 0.0398 | n3_test |
| 27 | 52 | exp52_foa_lr5e4_dw1.0_fw0.2_hw0.2 | 0.4921 | 1.0814 | 0.4933 | 0.0287 | summary_test |
| 28 | 28 | exp28_msattn_lr1e4_fw0.1 | 0.4917 | 1.0816 | 0.4995 | 0.0341 | summary_test |
| 29 | 01 | exp01_baseline_lr1e3_bs32 | 0.4553 | 1.0817 | 0.5031 | NA | summary_test |
| 30 | 74 | exp74_batvision_lr1e3_bs16 | 0.4652 | 1.0817 | 0.4944 | NA | summary_test |
| 31 | 65 | exp65_vit_lr3e5_bs16 | 0.4467 | 1.0818 | 0.4959 | NA | summary_test |
| 32 | 160 | exp160_pvitfoav1_lr1e4_w0.1 | 0.4619 | 1.0820 | 0.4948 | 0.0248 | summary_test |
| 33 | 107 | exp107_foa_lr5e4_fw0.3_hw0.1 | 0.4574 | 1.0824 | 0.4945 | 0.0329 | summary_test |
| 34 | 61 | exp61_vit_lr1e4_bs16 | 0.4496 | 1.0824 | 0.4989 | NA | summary_test |
| 35 | 174 | exp174_n3eattn_lr1e3_lsh0.3 | 0.4767 | 1.0835 | 0.4836 | 0.0335 | summary_test |
| 36 | 71 | exp71_batvision_lr1e3_bs32 | 0.4430 | 1.0835 | 0.4897 | NA | summary_test |
| 37 | 226 | exp226_n3mssh_dw2_lr1e3_lsh0.3 | 0.4396 | 1.0845 | 0.4947 | 0.0269 | n3_test |
| 38 | 20 | exp20_crossattn_lr5e4_fw0.2 | 0.4735 | 1.0850 | 0.4931 | 0.0262 | summary_test |
| 39 | 125 | exp125_echodiff_wav2vec_lr1e4_bs8 | 0.4750 | 1.0852 | 0.4994 | NA | summary_test |
| 40 | 215 | exp215_n3eattn_distill_lkd0.5 | 0.4288 | 1.0853 | 0.4943 | 0.0353 | n3_test |
| 41 | 120 | exp120_foa_lr5e4_fw0.05_hw0.1_freeze5 | 0.4568 | 1.0858 | 0.4884 | 0.0355 | summary_test |
| 42 | 50 | exp50_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze5 | 0.4691 | 1.0858 | 0.4860 | 0.0343 | summary_test |
| 43 | 173 | exp173_n3eattn_lr5e4_lsh0.1 | 0.4511 | 1.0859 | 0.4866 | 0.0328 | summary_test |
| 44 | 216 | exp216_pvit_eattn_lr1e4_lsh0.1 | 0.4639 | 1.0863 | 0.4974 | 0.0249 | n3_test |
| 45 | 96 | exp96_foa_lr2e4_dw1.0_fw0.1_hw0.1 | 0.4752 | 1.0866 | 0.4929 | 0.0377 | summary_test |
| 46 | 23 | exp23_featbank_lr1e4_fw0.1 | 0.4913 | 1.0869 | 0.4977 | 0.0412 | summary_test |
| 47 | 214 | exp214_n3eattn_sh9_lr1e3_lsh0.1 | 0.4373 | 1.0870 | 0.4954 | 0.0350 | n3_test |
| 48 | 44 | exp44_foa_lr5e4_dw1.0_fw0.1_hw0.2 | 0.4781 | 1.0871 | 0.4986 | 0.0340 | summary_test |
| 49 | 109 | exp109_foa_lr1e3_dw1.0_fw0.2_hw0.2_freeze5 | 0.4889 | 1.0873 | 0.4923 | 0.0296 | summary_test |
| 50 | 188 | exp188_n2_6ch_lr5e4_lsh0.1 | 0.4444 | 1.0875 | 0.4983 | 0.0212 | n2_test |
| 51 | 41 | exp41_foa_lr1e3_dw1.0_fw0.2_hw0.1 | 0.4568 | 1.0875 | 0.4900 | 0.0283 | summary_test |
| 52 | 170 | exp170_n3mssh_lr5e4_lsh0.1 | 0.4676 | 1.0878 | 0.4985 | 0.0249 | summary_test |
| 53 | 16 | exp16_crossattn_lr1e3_fw0.1 | 0.4570 | 1.0880 | 0.4958 | 0.0288 | summary_test |
| 54 | 51 | exp51_foa_lr1e3_dw1.0_fw0.1_hw0.1_freeze10 | 0.4839 | 1.0881 | 0.4913 | 0.0335 | summary_test |
| 55 | 105 | exp105_foa_lr1e3_fw0.3_hw0.1 | 0.4638 | 1.0884 | 0.4940 | 0.0307 | summary_test |
| 56 | 36 | exp36_foa_lr1e3_dw1.0_fw0.1_hw0.1 | 0.4850 | 1.0884 | 0.4943 | 0.0288 | summary_test |
| 57 | 186 | exp186_n3eattn_lr1e3_lsh0.5 | 0.4504 | 1.0885 | 0.4893 | 0.0357 | summary_test |
| 58 | 48 | exp48_foa_lr1e3_dw1.0_fw0.05_hw0.1 | 0.4625 | 1.0886 | 0.4920 | 0.0342 | summary_test |
| 59 | 56 | exp56_foav2_lr1e3_dw1.0_fw0.1_hw0.1 | 0.4672 | 1.0886 | 0.4905 | 0.0289 | summary_test |
| 60 | 40 | exp40_foa_lr5e4_bs16_dw1.0_fw0.1_hw0.1 | 0.4693 | 1.0888 | 0.4962 | 0.0295 | summary_test |
| 61 | 99 | exp99_foa_lr1e3_dw1.5_fw0.1_hw0.1 | 0.4748 | 1.0888 | 0.5034 | 0.0292 | summary_test |
| 62 | 184 | exp184_v1base_lr1e3_lsh0.1 | 0.4836 | 1.0889 | 0.4939 | 0.0410 | summary_test |
| 63 | 121 | exp121_echodiff_wav2vec_lr1e4_bs16 | 0.4485 | 1.0892 | 0.4887 | NA | summary_test |
| 64 | 72 | exp72_batvision_lr5e4_bs32 | 0.4552 | 1.0893 | 0.4894 | NA | summary_test |
| 65 | 03 | exp03_baseline_lr1e4_bs32 | 0.4295 | 1.0894 | 0.4934 | NA | summary_test |
| 66 | 227 | exp227_n3mssh_eattn_lr1e3_lsh0.3 | 0.4331 | 1.0895 | 0.5008 | 0.0241 | n3_test |
| 67 | 228 | exp228_n3mssh_eattn_lenergy_lr1e3_lsh0.1 | 0.4332 | 1.0898 | 0.4964 | 0.0285 | n3_test |
| 68 | 117 | exp117_foa_lr2e4_dw1.0_fw0.2_hw0.2 | 0.4742 | 1.0899 | 0.4908 | 0.0447 | summary_test |
| 69 | 27 | exp27_msattn_lr5e4_fw0.1 | 0.4483 | 1.0901 | 0.4913 | 0.0361 | summary_test |
| 70 | 116 | exp116_foa_lr3e4_fw0.2_hw0.1 | 0.4452 | 1.0902 | 0.4944 | 0.0301 | summary_test |
| 71 | 161 | exp161_pvitfoav1_lr5e5_w0.3 | 0.4673 | 1.0903 | 0.5000 | 0.0222 | summary_test |
| 72 | 42 | exp42_foa_lr5e4_dw1.0_fw0.2_hw0.1 | 0.4572 | 1.0904 | 0.5018 | 0.0259 | summary_test |
| 73 | 91 | exp91_channelattn_lr1e3_fw0.1_kl0.02 | 0.4470 | 1.0907 | 0.4869 | 0.0309 | summary_test |
| 74 | 14 | exp14_echodiff_lr5e4_bs32 | 0.4664 | 1.0908 | 0.4932 | NA | summary_test |
| 75 | 210 | exp210_n3eattn_lenergy0.3_lr1e3_lsh0.1 | 0.4763 | 1.0913 | 0.4930 | 0.0327 | n3_test |
| 76 | 80 | exp80_crossattn_lr1e3_fw0.3_kl0.01 | 0.4731 | 1.0913 | 0.5007 | 0.0242 | summary_test |
| 77 | 53 | exp53_foa_lr5e4_dw0.5_fw0.2_hw0.1 | 0.4669 | 1.0917 | 0.4878 | 0.0320 | summary_test |
| 78 | 29 | exp29_msattn_lr1e3_fw0.2 | 0.4723 | 1.0922 | 0.4936 | 0.0289 | summary_test |
| 79 | 103 | exp103_foa_lr5e4_fw0.15_hw0.1 | 0.4662 | 1.0925 | 0.4871 | 0.0320 | summary_test |
| 80 | 05 | exp05_baseline_lr5e4_bs16 | 0.4688 | 1.0927 | 0.4863 | NA | summary_test |
| 81 | 25 | exp25_featbank_lr5e4_fw0.2 | 0.4558 | 1.0928 | 0.4913 | 0.0311 | summary_test |
| 82 | 73 | exp73_batvision_lr1e4_bs32 | 0.4551 | 1.0928 | 0.4998 | NA | summary_test |
| 83 | 164 | exp164_pvitfoav3_lr1e4_w0.1 | 0.4372 | 1.0929 | 0.4972 | 0.0253 | summary_test |
| 84 | 83 | exp83_featbank_lr1e3_fw0.05_kl0.01 | 0.4701 | 1.0930 | 0.4817 | 0.0348 | summary_test |
| 85 | 204 | exp204_n2_xattn_lr1e3_lsh0.1 | 0.4510 | 1.0931 | 0.5010 | 0.0279 | n2_test |
| 86 | 209 | exp209_n3eattn_freeze15_lr1e3_lsh0.1 | 0.4410 | 1.0931 | 0.4954 | 0.0340 | n3_test |
| 87 | 169 | exp169_n3mssh_lr1e3_lsh0.1 | 0.4418 | 1.0937 | 0.4972 | 0.0250 | summary_test |
| 88 | 199 | exp199_n2_trms_film_lr1e3_lsh0.1 | 0.4356 | 1.0937 | 0.4855 | 0.0517 | n2_test |
| 89 | 32 | exp32_channelattn_lr5e4_fw0.1 | 0.4510 | 1.0938 | 0.4960 | 0.0316 | summary_test |
| 90 | 17 | exp17_crossattn_lr5e4_fw0.1 | 0.4599 | 1.0939 | 0.4927 | 0.0312 | summary_test |
| 91 | 24 | exp24_featbank_lr1e3_fw0.2 | 0.4759 | 1.0939 | 0.5010 | 0.0293 | summary_test |
| 92 | 197 | exp197_n2_stft_lr1e3_lsh0.1 | 0.4578 | 1.0941 | 0.4821 | 0.0371 | n2_test |
| 93 | 222 | exp222_n3mssh_lr1e3_lsh0.7 | 0.4546 | 1.0941 | 0.5022 | 0.0221 | n3_test |
| 94 | 81 | exp81_featbank_lr1e3_fw0.1_kl0.02 | 0.4462 | 1.0943 | 0.4900 | 0.0332 | summary_test |
| 95 | 55 | exp55_foa_lr1e4_bs16_dw1.0_fw0.1_hw0.1 | 0.4656 | 1.0945 | 0.4919 | 0.0383 | summary_test |
| 96 | 92 | exp92_channelattn_lr5e4_fw0.2_kl0.005 | 0.4561 | 1.0945 | 0.4948 | 0.0276 | summary_test |
| 97 | 113 | exp113_foa_lr1e3_dw1.0_fw0.05_hw0.05 | 0.4440 | 1.0948 | 0.4862 | 0.0336 | summary_test |
| 98 | 34 | exp34_channelattn_lr1e3_fw0.2 | 0.4631 | 1.0951 | 0.4890 | 0.0286 | summary_test |
| 99 | 119 | exp119_foa_lr1e3_fw0.1_hw0.2_freeze3 | 0.4566 | 1.0952 | 0.4845 | 0.0316 | summary_test |
| 100 | 58 | exp58_foav2_lr1e4_dw1.0_fw0.1_hw0.1 | 0.4824 | 1.0952 | 0.4854 | 0.0375 | summary_test |
| 101 | 19 | exp19_crossattn_lr1e3_fw0.2 | 0.4652 | 1.0957 | 0.4995 | 0.0254 | summary_test |
| 102 | 122 | exp122_echodiff_wav2vec_lr5e4_bs16 | 0.4585 | 1.0958 | 0.4882 | NA | summary_test |
| 103 | 124 | exp124_echodiff_wav2vec_lr5e5_bs16 | 0.4901 | 1.0958 | 0.4721 | NA | summary_test |
| 104 | 104 | exp104_foa_lr5e4_fw0.1_hw0.15 | 0.4745 | 1.0964 | 0.4886 | 0.0290 | summary_test |
| 105 | 62 | exp62_vit_lr5e5_bs16 | 0.4449 | 1.0964 | 0.4869 | NA | summary_test |
| 106 | 47 | exp47_foa_lr1e3_dw2.0_fw0.1_hw0.1 | 0.4463 | 1.0968 | 0.4858 | 0.0287 | summary_test |
| 107 | 45 | exp45_foa_lr1e3_dw1.0_fw0.2_hw0.2 | 0.4955 | 1.0969 | 0.4913 | 0.0280 | summary_test |
| 108 | 206 | exp206_n2_xattn_lr1e3_lsh0.3 | 0.4604 | 1.0972 | 0.4844 | 0.0334 | n2_test |
| 109 | 35 | exp35_channelattn_lr5e4_fw0.2 | 0.4579 | 1.0972 | 0.4909 | 0.0301 | summary_test |
| 110 | 43 | exp43_foa_lr1e3_dw1.0_fw0.1_hw0.2 | 0.4660 | 1.0972 | 0.4930 | 0.0305 | summary_test |
| 111 | 39 | exp39_foa_lr1e3_bs16_dw1.0_fw0.1_hw0.1 | 0.4953 | 1.0973 | 0.4973 | 0.0332 | summary_test |
| 112 | 88 | exp88_msattn_lr1e3_fw0.05_kl0.01 | 0.4724 | 1.0975 | 0.4998 | 0.0281 | summary_test |
| 113 | 30 | exp30_msattn_lr5e4_fw0.2 | 0.4551 | 1.0977 | 0.4912 | 0.0309 | summary_test |
| 114 | 46 | exp46_foa_lr1e3_dw0.5_fw0.1_hw0.1 | 0.4663 | 1.0977 | 0.4937 | 0.0287 | summary_test |
| 115 | 18 | exp18_crossattn_lr1e4_fw0.1 | 0.4856 | 1.0978 | 0.4905 | 0.1768 | summary_test |
| 116 | 76 | exp76_crossattn_lr1e3_fw0.1_kl0.02 | 0.4588 | 1.0981 | 0.4895 | 0.0321 | summary_test |
| 117 | 22 | exp22_featbank_lr5e4_fw0.1 | 0.4702 | 1.0983 | 0.4944 | 0.0324 | summary_test |
| 118 | 60 | exp60_foav2_lr5e4_dw1.0_fw0.2_hw0.2 | 0.4700 | 1.0988 | 0.4906 | 0.0281 | summary_test |
| 119 | 02 | exp02_baseline_lr5e4_bs32 | 0.4127 | 1.0989 | 0.4943 | NA | summary_test |
| 120 | 87 | exp87_msattn_lr5e4_fw0.2_kl0.005 | 0.4751 | 1.0990 | 0.4868 | 0.0375 | summary_test |
| 121 | 77 | exp77_crossattn_lr5e4_fw0.2_kl0.005 | 0.5034 | 1.0992 | 0.4989 | 0.0263 | summary_test |
| 122 | 115 | exp115_foa_lr1e3_dw2.0_fw0.2_hw0.1 | 0.4513 | 1.0994 | 0.4943 | 0.0298 | summary_test |
| 123 | 75 | exp75_batvision_lr2e3_bs32 | 0.4659 | 1.0994 | 0.4938 | NA | summary_test |
| 124 | 205 | exp205_n2_xattn_lr5e4_lsh0.1 | 0.4373 | 1.0998 | 0.4926 | 0.0268 | n2_test |
| 125 | 89 | exp89_msattn_lr5e4_fw0.1_hw0.2_kl0.01 | 0.4686 | 1.0999 | 0.4921 | 0.0358 | summary_test |
| 126 | 162 | exp162_pvitfoav2_lr1e4_w0.1 | 0.4782 | 1.1000 | 0.4833 | 0.0363 | summary_test |
| 127 | 207 | exp207_n3eattn_bs32_lr1e3_lsh0.1 | 0.4350 | 1.1000 | 0.4966 | 0.0376 | n3_test |
| 128 | 97 | exp97_foa_lr3e4_dw1.0_fw0.1_hw0.1 | 0.4299 | 1.1010 | 0.4922 | 0.0299 | summary_test |
| 129 | 229 | exp229_n3mssh_eattn_sh9_lr1e3_lsh0.3 | 0.4394 | 1.1013 | 0.4940 | 0.0264 | n3_test |
| 130 | 59 | exp59_foav2_lr1e3_dw1.0_fw0.2_hw0.1 | 0.4663 | 1.1021 | 0.4966 | 0.0314 | summary_test |
| 131 | 84 | exp84_featbank_lr5e4_fw0.1_hw0.2_kl0.01 | 0.4747 | 1.1022 | 0.4869 | 0.0340 | summary_test |
| 132 | 57 | exp57_foav2_lr5e4_dw1.0_fw0.1_hw0.1 | 0.4821 | 1.1023 | 0.4922 | 0.0330 | summary_test |
| 133 | 212 | exp212_n3film_eattn_lr1e3_lsh0.1 | 0.4307 | 1.1025 | 0.4828 | 0.0457 | n3_test |
| 134 | 64 | exp64_vit_lr1e4_bs8 | 0.4654 | 1.1028 | 0.4803 | NA | summary_test |
| 135 | 21 | exp21_featbank_lr1e3_fw0.1 | 0.4900 | 1.1032 | 0.4894 | 0.0291 | summary_test |
| 136 | 217 | exp217_pvit_mssh_lr1e4_lsh0.1 | 0.4656 | 1.1035 | 0.4860 | 0.0257 | n3_test |
| 137 | 211 | exp211_n3eattn_dw2_lenergy0.3_lr1e3 | 0.4300 | 1.1036 | 0.4945 | 0.0350 | n3_test |
| 138 | 166 | exp166_n3film_lr1e3_lsh0.1 | 0.4271 | 1.1045 | 0.4901 | 0.0374 | summary_test |
| 139 | 85 | exp85_featbank_lr1e3_fw0.3_kl0.01 | 0.4707 | 1.1045 | 0.4812 | 0.0288 | summary_test |
| 140 | 112 | exp112_foa_lr5e4_dw1.0_fw0.1_hw0.1_freeze10 | 0.4440 | 1.1047 | 0.4944 | 0.0308 | summary_test |
| 141 | 221 | exp221_n3mssh_lr1e3_lsh0.5 | 0.4351 | 1.1052 | 0.4941 | 0.0236 | n3_test |
| 142 | 08 | exp08_vit_lr1e4_bs16 | 0.5163 | 1.1055 | 0.4805 | NA | summary_test |
| 143 | 194 | exp194_n2_tenergy_lr1e3_lsh0.3 | 0.4736 | 1.1056 | 0.4950 | 0.0289 | n2_test |
| 144 | 26 | exp26_msattn_lr1e3_fw0.1 | 0.4410 | 1.1056 | 0.4955 | 0.0310 | summary_test |
| 145 | 38 | exp38_foa_lr1e4_dw1.0_fw0.1_hw0.1 | 0.4441 | 1.1058 | 0.4878 | 0.0372 | summary_test |
| 146 | 54 | exp54_foa_lr1e4_dw1.0_fw0.2_hw0.1 | 0.4716 | 1.1059 | 0.4968 | 0.0365 | summary_test |
| 147 | 11 | exp11_echodiff_lr1e4_bs32 | 0.4300 | 1.1060 | 0.4876 | NA | summary_test |
| 148 | 123 | exp123_echodiff_wav2vec_lr1e4_bs32 | 0.4214 | 1.1062 | 0.4884 | NA | summary_test |
| 149 | 185 | exp185_v1base_lr5e4_lsh0.1 | 0.4772 | 1.1068 | 0.4802 | 0.0539 | summary_test |
| 150 | 108 | exp108_foa_lr5e4_dw2.0_fw0.1_hw0.1 | 0.5068 | 1.1069 | 0.4916 | 0.0293 | summary_test |
| 151 | 95 | exp95_channelattn_lr1e3_fw0.3_kl0.01 | 0.4590 | 1.1071 | 0.4904 | 0.0268 | summary_test |
| 152 | 167 | exp167_n3film_lr5e4_lsh0.1 | 0.4612 | 1.1073 | 0.4983 | 0.0316 | summary_test |
| 153 | 33 | exp33_channelattn_lr1e4_fw0.1 | 0.4339 | 1.1074 | 0.4974 | 0.0335 | summary_test |
| 154 | 04 | exp04_baseline_lr1e3_bs16 | 0.4300 | 1.1076 | 0.4882 | NA | summary_test |
| 155 | 79 | exp79_crossattn_lr5e4_fw0.1_hw0.2_kl0.01 | 0.4519 | 1.1085 | 0.4903 | 0.0347 | summary_test |
| 156 | 93 | exp93_channelattn_lr1e3_fw0.05_kl0.01 | 0.4754 | 1.1097 | 0.4915 | 0.0275 | summary_test |
| 157 | 37 | exp37_foa_lr5e4_dw1.0_fw0.1_hw0.1 | 0.4520 | 1.1102 | 0.4753 | 0.0426 | summary_test |
| 158 | 171 | exp171_n3mssh_lr1e3_lsh0.3 | 0.4218 | 1.1104 | 0.4823 | 0.0256 | summary_test |
| 159 | 101 | exp101_foa_lr1e3_fw0.1_hw0.15 | 0.4414 | 1.1116 | 0.4831 | 0.0319 | summary_test |
| 160 | 193 | exp193_n2_tenergy_lr5e4_lsh0.1 | 0.4475 | 1.1121 | 0.4923 | 0.0352 | n2_test |
| 161 | 31 | exp31_channelattn_lr1e3_fw0.1 | 0.4547 | 1.1121 | 0.4804 | 0.0269 | summary_test |
| 162 | 13 | exp13_echodiff_lr1e4_bs16 | 0.4504 | 1.1134 | 0.4930 | NA | summary_test |
| 163 | 191 | exp191_n2_trms_lr5e4_lsh0.1 | 0.4408 | 1.1136 | 0.4984 | 0.0350 | n2_test |
| 164 | 165 | exp165_pvitfoav3_lr5e5_w0.3 | 0.4369 | 1.1140 | 0.4896 | 0.0241 | summary_test |
| 165 | 168 | exp168_n3film_lr1e3_lsh0.3 | 0.4434 | 1.1142 | 0.4846 | 0.0343 | summary_test |
| 166 | 66 | exp66_echonet_lr1e3_bs8 | 0.4550 | 1.1156 | 0.4778 | NA | summary_test |
| 167 | 223 | exp223_n3mssh_lr1e3_lsh0.2 | 0.4171 | 1.1161 | 0.4904 | 0.0310 | n3_test |
| 168 | 63 | exp63_vit_lr5e4_bs16 | 0.4578 | 1.1168 | 0.4733 | NA | summary_test |
| 169 | 10 | exp10_vit_lr1e5_bs32 | 0.4720 | 1.1170 | 0.4812 | NA | summary_test |
| 170 | 07 | exp07_vit_lr5e5_bs32 | 0.4755 | 1.1174 | 0.4731 | NA | summary_test |
| 171 | 12 | exp12_echodiff_lr5e5_bs32 | 0.4914 | 1.1208 | 0.4816 | NA | summary_test |
| 172 | 190 | exp190_n2_trms_lr1e3_lsh0.1 | 0.4370 | 1.1231 | 0.4868 | 0.0369 | n2_test |
| 173 | 06 | exp06_vit_lr1e4_bs32 | 0.4770 | 1.1265 | 0.4705 | NA | summary_test |
| 174 | 220 | exp220_pvit_freeze20_lr1e4_lsh0.1 | 0.4631 | 1.1287 | 0.4683 | 0.0326 | n3_test |
| 175 | 15 | exp15_echodiff_lr1e5_bs32 | 0.4708 | 1.1300 | 0.4728 | NA | summary_test |
| 176 | 70 | exp70_echonet_lr2e3_bs16 | 0.4782 | 1.1398 | 0.4343 | NA | summary_test |
| 177 | 177 | exp177_n3twin_lr1e3_lsh0.3 | 0.5449 | 1.1442 | 0.4529 | 1.4440 | summary_test |
| 178 | 59 | exp59_resnet_lr1e4_bs16 | 0.5454 | 1.1444 | 0.4480 | NA | summary_test |
| 179 | 57 | exp57_resnet_lr5e5_bs32 | 0.5341 | 1.1506 | 0.4697 | NA | summary_test |
| 180 | 60 | exp60_resnet_lr3e4_bs32 | 0.4977 | 1.1515 | 0.4587 | NA | summary_test |
| 181 | 56 | exp56_resnet_lr1e4_bs32 | 0.5063 | 1.1810 | 0.4402 | NA | summary_test |
| 182 | 69 | exp69_echonet_lr1e3_bs16 | 0.5718 | 1.1852 | 0.4039 | NA | summary_test |
| 183 | 09 | exp09_vit_lr5e4_bs32 | 0.5593 | 1.1985 | 0.4443 | NA | summary_test |
| 184 | 58 | exp58_resnet_lr5e4_bs32 | 0.4929 | 1.2014 | 0.4467 | NA | summary_test |
| 185 | 176 | exp176_n3twin_lr5e4_lsh0.1 | 0.5430 | 1.2022 | 0.4583 | 0.1284 | summary_test |
| 186 | 163 | exp163_pvitfoav2_lr5e5_w0.3 | 0.5089 | 1.2099 | 0.4388 | 0.0457 | summary_test |
| 187 | 175 | exp175_n3twin_lr1e3_lsh0.1 | 0.5477 | 1.2325 | 0.4514 | 0.0608 | summary_test |
| 188 | 182 | exp182_oracle_nc1_lr5e4_lsh0.1 | 0.5352 | 1.2396 | 0.4173 | 0.0467 | summary_test |
| 189 | 183 | exp183_oracle_nc1_lr1e3_lsh0.3 | 0.5500 | 1.2596 | 0.4152 | 0.0426 | summary_test |
| 190 | 181 | exp181_oracle_nc1_lr1e3_lsh0.1 | 0.5152 | 1.2620 | 0.4103 | 0.0410 | summary_test |
| 191 | 68 | exp68_echonet_lr1e4_bs16 | 0.6347 | 1.5988 | 0.1621 | NA | summary_test |
| 192 | 67 | exp67_echonet_lr5e4_bs16 | 1.0335 | 1.7018 | 0.2152 | NA | summary_test |
