# comparison_methods/

External comparison baselines, kept physically and logically separate from
`models/` (which holds our proposed/shared code). These are *not* re-exported
from `models` — import them directly, e.g.
`from comparison_methods.echonet import EchoNet`.

## Methods

| Subpackage      | Main class(es)                                              | Config name(s)                                                  |
|-----------------|-------------------------------------------------------------|-----------------------------------------------------------------|
| `batvision/`    | `BatVisionUNet`                                             | `batvision`                                                     |
| `echonet/`      | `EchoNet`                                                   | `echonet`                                                       |
| `echodiffusion/`| `EchoDiffusion`, `EchoDiffusionAmbi`, `EchoDiffusionAmbiSH`, `EchoDiffusionSHSidePlus` | `echodiffusion`, `echodiffusion_ambi`, `echodiffusion_ambi_sh`, `echodiff_sh_side_plus` |

Historical training runs for these baselines live in `archive/runs/`.
