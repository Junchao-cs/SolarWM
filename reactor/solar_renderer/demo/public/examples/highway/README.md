# Synthwave Highway example

The demo imports `structure.mp4`, `first.png` and `prompt.txt` from this checkout's
[included example](../../../../../../examples/solar_renderer/sem-highway_synthwave).
If those local files are missing, it downloads and caches the three official files
from upstream commit a2371eaecff9a8ab378b5ec36199c5863d138074.
Vite serves these files during development and includes them in production builds.
Source: https://github.com/Enderfga/SolarRenderer/tree/a2371eaecff9a8ab378b5ec36199c5863d138074/examples/sem-highway_synthwave

`structure.png` is the first structure frame, used as a thumbnail.
It is the only media asset stored in this directory.
The structure-only UI option uses the same video without an appearance image
and removes Video 2 references from the prompt; it is an adapter-provided input
variant, not a second upstream scene. Example selection never starts inference.

The upstream Apache-2.0 license is included as LICENSE.
