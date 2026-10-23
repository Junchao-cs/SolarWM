# SolarWM-Data

The public repository is available on
[Hugging Face](https://huggingface.co/datasets/junchaoh-cs/SolarWM-Data) and
[ModelScope](https://modelscope.cn/datasets/junchao2003/SolarWM-Data).
After download, the local release root is:

```text
SolarWM-Data/releases-v1/
```

The repository contains portable controls, licenses, small format examples,
and the `SolarWM-Data-Annotation/` reconstruction package. The full
`raw-wds/` and `latent-wds/` payloads will be distributed separately. See
[Dataset access](data-access.md) before launching a full training or inference
example.

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
rather than another GPU run. The public annotation package contains no videos;
it provides the released annotations and tools needed to reconstruct raw-WDS
from source media acquired by the user under the applicable upstream terms.

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
