# SolarWM-Data

The public repository is available on
[Hugging Face](https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data) and
[ModelScope](https://modelscope.cn/datasets/junchao2003/SolarWM-Data).
After download, the local release root is:

```text
SolarWM-Data/releases-v1/
```

## Data preparation and access

Data preparation pipelines are grouped under [data-engine/](../../data-engine/README.md),
with separate `solarwm-data` and `solar_renderer-data` submodules.

The public [SolarWM-Data release](https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data)
contains release controls, licenses, recipe and test indexes, small format
examples, and the `SolarWM-Data-Annotation/` package. It does **not** include the
full `releases-v1/raw-wds/` or `releases-v1/latent-wds/` payloads.

For a released recipe that uses preencoded training data, download the main
release controls and its matching latent generation. Those training inputs do
not require raw-WDS. Raw-WDS is needed when you want the full processed video
corpus, online encoding, your own latent generation, or another workflow whose
index points to raw data. Validation requirements are separate: H3 Stage1
validation also reads raw test cameras for the full trajectory; see the
[H3 guide](backends/minimax-h3.md) and [dataset access guide](data-access.md).

1. **Use preencoded latents.** Each latent generation is published in a separate
   repository. See the [latent-WDS release list](latent-wds.md) for available
   downloads and generations that are still being uploaded.
2. **Rebuild raw-WDS from annotations.**
   [`SolarWM-Data-Annotation/`](https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data/tree/main/SolarWM-Data-Annotation)
   contains camera trajectories, captions, metadata, source identities, and
   reconstruction tools. It also includes the processed videos for the three
   clean sources: `miradata-clean`, `sekai_walking-clean`, and `spatialvid-clean`.
   The other 11 sources require original videos from their publishers. Follow
   the package README to restore the sources and build the `raw-wds/` layout.
3. **Request prepared raw-WDS.** Submit the
   [Dataset Access Form](https://docs.google.com/forms/d/e/1FAIpQLSfS-SLOiSRVDWwZ2kPl9ywN27aB6QplN0jpdKaBu-gG8aNsvQ/viewform).
   Approved applicants receive download instructions by email.
4. **Download the standalone test set.** The
   [SolarWM standalone test set v1](https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data_test-set-v1)
   contains 1,300 examples from 13 sources in 60 WebDataset tar files
   (approximately 77.4 GB), without the training payload.

See the [dataset access guide](data-access.md) for download commands and the
payload required by each training, validation, and inference example.

## Contents

The complete `raw-wds/` corpus includes samples marked
high, xhigh, and rejected, together with the measurements and intermediate
fields needed to build a different mixture. Expensive video, camera, motion,
quality, and scene processing is therefore independent of the policy that
selects samples or assigns source weights.

The release contains 14 datasets. DL3DV has separate 10s and 60s views, while
MiraData, Sekai Walking, and SpatialVid each include an additional clean variant.

| Source | Total | High | Xhigh | Rejected |
| --- | ---: | ---: | ---: | ---: |
| `abot` | 30,966 | 127 | 30,715 | 124 |
| `dl3dv-10s` | 120,924 | 54,528 | 60,396 | 6,000 |
| `dl3dv-60s` | 10,077 | 3,578 | 6,065 | 434 |
| `mind` | 533 | 117 | 402 | 14 |
| `miradata` | 140,877 | 3,683 | 17,806 | 119,388 |
| `miradata-clean` | 135,224 | 12,865 | 5,740 | 116,619 |
| `multicamvideo` | 123,117 | 89,587 | 5,369 | 28,161 |
| `omniworld` | 19,632 | 3,773 | 13,552 | 2,307 |
| `realcam_vid` | 45,697 | 16,154 | 16,074 | 13,469 |
| `sekai_game` | 2,550 | 537 | 1,410 | 603 |
| `sekai_walking` | 22,990 | 6,054 | 12,976 | 3,960 |
| `sekai_walking-clean` | 109,248 | 46,831 | 33,660 | 28,757 |
| `spatialvid` | 365,345 | 100,815 | 127,180 | 137,350 |
| `spatialvid-clean` | 298,514 | 133,149 | 73,450 | 91,915 |
| **Total** | **1,425,694** | **471,798** | **404,795** | **549,101** |

Every row in this table is part of the frozen annotated corpus. The rejected
tier is published because selection is metadata over an already-annotated
corpus, so a different threshold, metric or source weighting is an index pass
rather than another GPU run. The public reconstruction package includes the
processed videos for the three clean sources. For the other sources, it provides
annotations and tools to rebuild raw-WDS from media acquired by the user under
the applicable upstream terms.

Each source directory contains tiered WebDataset shards and a full
`meta.jsonl`; portable recipe indexes select the training and test rows.
The metadata includes source identity, captions, dimensions, frame rate,
camera summaries, selection state, rejection reasons, VMAF, UniMatch, DOVER,
saturation, scene-cut, and VLM measurements where available.

## Latent generation inventory

| Generation | Samples |
| --- | ---: |
| `wan22-ti2v5b-153f-480p-v1` | 688,424 |
| `wan22-ti2v5b-153f-720p-v1` | 688,418 |
| `wan22-ti2v5b-957f-480p-v1` | 82,321 |
| `wan22-ti2v5b-957f-720p-v1` | 82,321 |
| `wan22-i2v-a14b-81f-480p-v1` | 1,564,464 |
| `wan22-i2v-a14b-81f-720p-v1` | 1,564,464 |
| `wan22-i2v-a14b-153f-480p-v1` | 688,424 |
| `wan22-i2v-a14b-153f-720p-v1` | 688,418 |
| `wan22-i2v-a14b-957f-480p-v1` | 82,321 |
| `wan22-i2v-a14b-957f-720p-v1` | 82,321 |
| `minimax-h3-158f-768p-nomind-v1` | 686,841 |
| `ltx-153f-h512-w768` | 688,418 |
| `ltx-953f-h512-w768` | 82,321 |

The tar member manifests use backend-specific `solarwm_*` schemas. Tensor
payloads keep their native dtype and shape. Each generation will be published
as a separate dataset repository; current repository links and planned
generations are maintained in the [latent-WDS release list](latent-wds.md).

## Camera convention

Camera trajectories and intrinsics are included in each record. The backend
readers expose the same first-frame-relative trajectory to the model. Optional
model-side transforms such as `logd4` belong to the training and checkpoint
contract, not to the dataset.

## Recipes and test data

The public recipes are under `recipes/clean-81f/`, `recipes/clean-153f/`,
`recipes/clean-158f-h3/`, and `recipes/clean-957f/`. Each route provides a
train index, test index, recipe contract, statistics, and an exclusion index
when individual samples are excluded.

Training validation and standalone inference read the selected recipe's
`test-index.jsonl.gz`. Set `validation.sample_count` and
`validation.selection_seed` to choose the same unique rows at every validation
step; the backend's generation seed controls noise independently. The
standalone `test-set/` is a logical view of canonical primary raw shards, so
test semantics are preserved without duplicating video bytes.

`example/` contains small records derived from the release itself: one raw
sample from each annotation tier and one representative of every published
latent reader schema. Its combined index is a portable format catalog for
inspection and reader checks, not a training recipe: it deliberately mixes
raw and backend-specific latent schemas and includes one rejected raw sample.
Example shard paths are relative to `example/`; source archive paths are
relative to `releases-v1/` and carry portable object identities for direct
traceability.
Training uses the homogeneous `train-index.jsonl.gz` under the selected recipe.

## Reading a local release

For a downloaded or mounted release, point both the control root and payload
root at the same `releases-v1/` directory:

```yaml
data:
  index_root: /path/to/SolarWM-Data/releases-v1
  transport:
    kind: local
    root: /path/to/SolarWM-Data/releases-v1
  train_index: recipes/clean-153f/latent-wds/wan22-ti2v5b-153f-480p-v1/train-index.jsonl.gz
```

Every index stores a release-relative object key, so the same downloaded tree
can live at any local absolute path. The full storage and integrity rules are
in the [data contract](data-contract.md).

`release.json` describes the complete logical release across its distribution
repositories. Read the license and citation registry in `licenses/` before
using or redistributing source media or derived tensors.
