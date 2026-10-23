# Clown Anchoring for MiniMax H3 (ComfyUI)

Pin the **end** of a MiniMax H3 generation to the **start of the next shot**, using the next shot's real saved latents. The anchored frames come out as the originals, so the join lands exactly and the overlap doesn't flash.

Built for in-between shots: generate shots 1 and 3 first, then generate shot 2 so it starts from shot 1 (for example with Motion Context) and **ends exactly on shot 3**.

## Nodes

### Clown Anchoring
Puts the next shot's first `tail_frames` frames into the end of this shot's starting latent and masks them (noise mask 0), so the sampler holds them and generates everything before them to lead into them.

| Input | What it does |
|---|---|
| `latent` | The starting AV latent that goes into the sampler. |
| `tail_frames` | Overlap in frames. Match your audio overlap; 39 is the usual value. The whole overlap is held. |
| `drop_frames` | **17 by default.** Frames at the end of this shot to generate freely, in steps of 17 (0, 17, 34 …). The held frames still start on the next shot's frame 0. With 39: 17 holds the next shot's frames **0–21** in this shot's 204–225; 34 holds 0–4 in 204–208; 0 holds the whole overlap (204–242 = next shot's 0–38). |
| `next_shot` (optional) | The next shot's saved latent, e.g. from H3 Motion Context **Load Latent**. If it's empty, the latent passes through unchanged. |

With 243-frame clips and `tail_frames` = 39, `drop_frames` 0 holds this shot's frames **204–242** as the next shot's frames **0–38**. With 17 (the default), frames **204–225** are held as the next shot's frames **0–21**. Any existing noise mask, such as an audio mask, is kept. The console prints what was held:

```
[Clown Anchoring] held latents 60..71 = frames 204..242 <- next shot's frames 0..38
```

### H3 Repack AV Latent (list → packed)
The Motion Context **Load Latent** node returns a latent as a plain `[video, audio]` list. VAE Decode, VAE Decode Audio and LTXV Separate AV Latent need the packed form the sampler produces. This node converts it.

## How it works

H3 lays latent frames out as `FRAME_PER_TOKEN = (1, 4, 4, 4, 4)`, counted from frame 0 (`comfy/ldm/minimax/model.py`). Every 5 latents make a 17-frame block, so blocks start at frames 0, 17, 34, and so on. A clip of 5 + 17k frames ends with a short 2-latent piece.

- **A latent only reads correctly at the same position in its block,** so Clown Anchoring only copies latents where the source and target positions line up.
- **With clip lengths and overlaps of the form 5 + 17k** (for example 243 and 39), the overlap always starts on a block boundary, so the whole overlap can be held.
- **Other overlaps are rounded down** to the nearest value that starts on a boundary, and the console says so.

### Why there's no flash
Guide-based anchoring, such as native MiniMax H3 Add Guide or Motion Context's pinned frames, shows the model the old frames as conditioning, and **the model redraws its own copy of them**. In our tests that redraw came out brighter, with a pulse on the 17-frame rhythm, which is why those pinned frames are normally trimmed off. Clown Anchoring puts the real latents into the clip itself and holds them with the mask, so those frames decode as the originals.

## Install

1. Clone or download this folder into `ComfyUI/custom_nodes/`.
2. Restart ComfyUI.

The nodes appear under **latent/H3**.

Requires a ComfyUI version with MiniMax H3 support (`comfy.nested_tensor`).

## Notes and limits

- **Both clips must share** the same model, resolution and latent layout.
- **The sampler must honour noise masks** on the video latent. This works with SamplerCustomAdvanced in our tests.
- **The overlap frames are copies of the next shot.** When joining clips, keep the overlap once: either trim the end of this shot or the start of the next.
- **This fixes the join, not long-chain drift.** Chaining many continuations still slowly darkens and softens the picture, just as it does with Motion Context.
