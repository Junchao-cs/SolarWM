# Latent-WDS releases

Each preencoded latent generation is distributed in a separate dataset
repository. After downloading a generation, preserve its directory name and
place it at:

```text
SolarWM-Data/releases-v1/latent-wds/<generation>/
```

For preencoded training, download the main SolarWM-Data repository and the
matching latent generation. H3 Stage1 validation also uses raw-WDS for the full
camera trajectory. Other workflows need raw-WDS only when they read or encode
source videos.

The following list mirrors the latent generation directories defined by the
SolarWM-Data release.

| Generation | Repository |
|---|---|
| `wan22-ti2v5b-81f-480p-v1` | Coming soon |
| `wan22-ti2v5b-81f-720p-v1` | Coming soon |
| `wan22-ti2v5b-153f-480p-v1` | [ModelScope International](https://modelscope.ai/datasets/Junchao-cs/SolarWM-Data_Latent-WDS_wan22-ti2v5b-153f-480p-v1) · [ModelScope China](https://modelscope.cn/datasets/junchao2003/SolarWM-Data_Latent-WDS_wan22-ti2v5b-153f-480p-v1) |
| `wan22-ti2v5b-153f-720p-v1` | Coming soon |
| `wan22-ti2v5b-957f-480p-v1` | Coming soon |
| `wan22-ti2v5b-957f-720p-v1` | Coming soon |
| `wan22-i2v-a14b-81f-480p-v1` | Coming soon |
| `wan22-i2v-a14b-81f-720p-v1` | Coming soon |
| `wan22-i2v-a14b-153f-480p-v1` | Coming soon |
| `wan22-i2v-a14b-153f-720p-v1` | Coming soon |
| `wan22-i2v-a14b-957f-480p-v1` | Coming soon |
| `wan22-i2v-a14b-957f-720p-v1` | Coming soon |
| `ltx-153f-h512-w768` | Coming soon |
| `ltx-953f-h512-w768` | Coming soon |
| `minimax-h3-158f-768p-nomind-v1` | [ModelScope International](https://modelscope.ai/datasets/Junchao-cs/SolarWM-Data_Latent-WDS_minimax-h3-158f-768p-nomind-v1) · [ModelScope China](https://modelscope.cn/datasets/junchao2003/SolarWM-Data_Latent-WDS_minimax-h3-158f-768p-nomind-v1) |

Download only the generation required by the selected training example. The
main SolarWM-Data repository contains the matching recipe indexes, but it does
not contain these latent payloads.
