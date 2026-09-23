# MiniMax-H3 audio inference

SolarWM can ask MiniMax-H3 to sample its native audio rows together with the
video rows. This is an inference switch; the historical video-only behavior is
unchanged by default.

In a fixed 158-frame Stage0.5 config, set:

```yaml
validation:
  generate_audio: true
```

Then run the normal inference command. The output partition contains:

```text
generated.mp4              # video-only reference
generated.wav              # 32 kHz, stereo PCM16
generated-with-audio.mp4   # generated.mp4 muxed with AAC audio
generated.safetensors      # video and audio latents
```

The sampler uses the native H3 schedules (video shift 12, audio shift 3) and
decodes audio with the checkpoint's `audio_vae` normalization parameters. The
audio switch is currently limited to fixed-length Stage0.5 inference; source-
length Stage2 inference remains video-only.

This exposes the model's audio capability, but it is not a claim of trained
audio quality. The released Stage0.5 LoRA was trained with an audio loss weight
of zero, so the generated sound can be weak, noisy, or poorly aligned. A later
audio-target training run is expected to improve it.
