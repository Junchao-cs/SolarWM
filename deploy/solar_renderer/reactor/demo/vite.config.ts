import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const demoRoot = fileURLToPath(new URL(".", import.meta.url));
const revision = "a2371eaecff9a8ab378b5ec36199c5863d138074";
const examplePath = "examples/sem-highway_synthwave";
const samples = {
  "structure.mp4": "8a4c9cbae6538dc2132a2d0353ac9b2f4e124fe6ae301bfe5017371370fea9aa",
  "first.png": "94d8a101299ce65b7c8747dc34118090163e8202ceef06ea4aaefe270a1d1082",
  "prompt.txt": "3530bebe667fd889e116604197701bb86b0cc3bdc25d83e626c399e6f2e983b9",
};

async function resolveExamples() {
  const localRoot = resolve(demoRoot, "../../../../examples/solar_renderer/sem-highway_synthwave");
  if (Object.keys(samples).every(name => existsSync(resolve(localRoot, name)))) return localRoot;

  const cache = resolve(demoRoot, "node_modules/.cache/solarrenderer", revision, examplePath);
  await mkdir(cache, { recursive: true });
  for (const [name, sha256] of Object.entries(samples)) {
    const path = resolve(cache, name);
    const cached = existsSync(path) ? await readFile(path) : null;
    if (cached && createHash("sha256").update(cached).digest("hex") === sha256) continue;
    const response = await fetch(
      `https://raw.githubusercontent.com/Enderfga/SolarRenderer/${revision}/${examplePath}/${name}`,
      { signal: AbortSignal.timeout(30_000) },
    );
    if (!response.ok) throw new Error(`Could not download upstream sample ${name}: HTTP ${response.status}`);
    const data = Buffer.from(await response.arrayBuffer());
    if (createHash("sha256").update(data).digest("hex") !== sha256) {
      throw new Error(`Upstream sample checksum mismatch: ${name}`);
    }
    await writeFile(path, data);
  }
  return cache;
}

export default defineConfig(async () => {
  const exampleRoot = await resolveExamples();
  return {
    resolve: { alias: { "@solarrenderer-example": exampleRoot } },
    server: { fs: { allow: [demoRoot, exampleRoot] } },
  };
});
