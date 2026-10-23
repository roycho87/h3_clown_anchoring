"""
Clown Anchoring nodes for MiniMax H3 (one file)
-----------------------------------------------
(formerly h3_repack_av_latent.py - delete that file if it is still in custom_nodes)

H3 latent layout (from ComfyUI's comfy/ldm/minimax/model.py):
  FRAME_PER_TOKEN = (1, 4, 4, 4, 4), counted from frame 0. Every 5 latent frames
  make a 17-frame block, and blocks start at frames 0, 17, 34, ...; a clip of
  5 + 17k frames ends with a short 2-latent piece (1 + 4 frames). A latent only
  reads correctly at the same position in its block, so latents are copied only
  where the source and target positions line up. With clip lengths and overlaps
  of the form 5 + 17k, the overlap always starts on a block boundary, so the
  WHOLE overlap can be held.

1. H3 Repack AV Latent (list -> packed)
   Motion Context "Load Latent" returns samples as a plain [video, audio] list.
   VAE Decode / VAE Decode Audio / LTXV Separate AV need the packed NestedTensor.

2. Clown Anchoring (class H3BlockTailAnchor)
   Holds the NEXT shot's first `tail_frames` frames at the END of this shot:
   this shot's latents for its last tail_frames frames <- next shot's first latents,
   noise mask 0 there so the sampler keeps them. 39 -> this shot's frames 204-242
   = next shot's frames 0-38 (12 latents). drop_17_frames (on by default) skips
   the last block of the overlap: 39 -> this shot's frames 204-225 = next shot's frames 0-21,
   and this shot's last 17 frames (226-242) are generated freely.

Clown Anchoring keeps any existing noise mask (e.g. the audio mask) and passes
the latent through unchanged when there is no next-shot latent. If an overlap would
not start on a block boundary it is rounded down to the nearest one that does,
and the console says so.

Install: drop this file into ComfyUI/custom_nodes/ and restart ComfyUI.
"""

import torch

try:
    from comfy.nested_tensor import NestedTensor
except Exception:  # very old ComfyUI
    NestedTensor = None


def _need_nested(who):
    if NestedTensor is None:
        raise RuntimeError(f"{who}: this ComfyUI has no comfy.nested_tensor; update ComfyUI.")


def _as_tensor(part):
    if isinstance(part, dict) and "samples" in part:
        return part["samples"]
    return part


def _split(value):
    """NestedTensor / list / tensor -> list of tensors, video (5-D) first."""
    if NestedTensor is not None and isinstance(value, NestedTensor):
        parts = list(value.unbind())
    elif isinstance(value, (list, tuple)):
        parts = [_as_tensor(p) for p in value]
    else:
        parts = [value]
    return parts


def _has_samples(lat):
    if lat is None or not isinstance(lat, dict):
        return False
    s = lat.get("samples", None)
    if s is None:
        return False
    if isinstance(s, (list, tuple)) and (len(s) == 0 or any(p is None for p in s)):
        return False
    return True


FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
BLOCK = 17


def _frames_of(n_latents):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(n_latents))


def _latents_in(frames):
    """Latent count for `frames` frames starting at a block boundary (frames must end on a latent edge)."""
    full, r = divmod(frames, BLOCK)
    return 5 * full + (0 if r == 0 else 1 + (r - 1) // 4)


def _aligned_overlap(overlap, clip_frames):
    """Largest overlap <= requested whose start (clip_frames - overlap) is a block boundary."""
    f = int(overlap) - ((int(overlap) - clip_frames) % BLOCK)
    return max(0, min(f, clip_frames))


def _video_first(parts):
    return sorted(parts, key=lambda t: t.ndim, reverse=True)


class H3RepackAVLatent:
    CATEGORY = "latent/H3"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "repack"
    DESCRIPTION = (
        "Packs a [video, audio] list latent (from H3 Motion Context Load Latent) "
        "into the standard AV latent so VAE Decode / VAE Decode Audio can read it."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latent": ("LATENT",)}}

    def repack(self, latent):
        _need_nested("H3 Repack AV Latent")
        samples = latent["samples"]
        if isinstance(samples, NestedTensor):
            return (latent,)
        if not isinstance(samples, (list, tuple)):
            raise ValueError(
                f"H3 Repack AV Latent: expected a [video, audio] list, got {type(samples).__name__}."
            )
        parts = _video_first([_as_tensor(p) for p in samples])
        if len(parts) != 2:
            raise ValueError(f"H3 Repack AV Latent: expected 2 parts (video, audio), got {len(parts)}.")
        out = {k: v for k, v in latent.items() if k not in ("samples", "noise_mask")}
        out["samples"] = NestedTensor(tuple(parts))
        return (out,)


class H3BlockTailAnchor:
    CATEGORY = "latent/H3"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "anchor"
    DESCRIPTION = (
        'Holds the next shot' + chr(39) + 's first frames at the end of this shot, in matching block positions, masked so the sampler keeps them.'
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "Starting AV latent that goes into the sampler."}),
                "tail_frames": ("INT", {"default": 39, "min": 0, "max": 4096,
                                        "tooltip": "Overlap in frames (use the Audio Splitter overlap, e.g. 39). The whole overlap is held."}),
                "drop_17_frames": ("BOOLEAN", {"default": True,
                                               "tooltip": "Skip this shot's last 17 frames (one block): 39 -> holds the next shot's frames 0-21 in this shot's frames 204-225; frames 226-242 are generated freely."}),
            },
            "optional": {
                "next_shot": ("LATENT", {"tooltip": "Next shot's saved latent (list or packed). If empty, the latent passes through."}),
            },
        }

    def anchor(self, latent, tail_frames, drop_17_frames=True, next_shot=None):
        who = "Clown Anchoring"
        if not _has_samples(next_shot):
            print(f"[{who}] no next shot latent - latent passed through unchanged")
            return (latent,)
        samples = latent["samples"]
        packed = NestedTensor is not None and isinstance(samples, NestedTensor)
        parts = _split(samples)
        vi = max(range(len(parts)), key=lambda i: parts[i].ndim)
        video = parts[vi]
        if video.ndim != 5:
            raise ValueError(f"{who}: expected a 5-D video latent [B, C, T, H, W], got {tuple(video.shape)}.")
        src = _video_first(_split(next_shot["samples"]))[0]
        if src.ndim != 5 or src.shape[1] != video.shape[1] or src.shape[3:] != video.shape[3:]:
            raise ValueError(
                f"{who}: source latent {tuple(src.shape)} doesn't match this shot {tuple(video.shape)} "
                "(same model and resolution needed)."
            )
        T, Ts = video.shape[2], src.shape[2]
        Lt, Ls = _frames_of(T), _frames_of(Ts)
        F_all = min(_aligned_overlap(tail_frames, Lt), Lt, Ls)
        if F_all != int(tail_frames):
            print(f"[{who}] overlap {tail_frames} doesn't start on a block boundary - using {F_all}")
        # drop_17_frames skips the LAST block (17 frames) of the overlap (this shot's last 17
        # frames); the held frames still start on the next shot's frame 0. Skipping a whole
        # block keeps positions aligned.
        skip = BLOCK if drop_17_frames else 0
        F = F_all - skip
        n = _latents_in(F_all) - (5 if drop_17_frames else 0)
        if F <= 0 or n <= 0:
            print(f"[{who}] nothing to hold (overlap {F_all}, drop_17_frames {drop_17_frames}) - latent passed through")
            return (latent,)
        if drop_17_frames:
            print(f"[{who}] drop_17_frames on: holding the first {F} overlap frames, this shot's last 17 are generated freely")
        d0 = T - _latents_in(F_all); s0 = 0
        video = video.clone()
        video[:, :, d0:d0 + n] = src[:, :, s0:s0 + n].to(device=video.device, dtype=video.dtype)
        parts[vi] = video

        # noise mask: 0 = keep, 1 = generate. Keep any existing mask (audio, other clown node).
        old = latent.get("noise_mask", None)
        masks = _split(old) if old is not None else None
        if masks is not None and len(masks) != len(parts):
            masks = None
        if masks is None:
            masks = [torch.ones_like(p) for p in parts]
            if old is not None and not packed:
                masks[vi] = old
        vmask = masks[vi]
        vmask = torch.ones_like(video) if vmask.shape != video.shape else vmask.clone()
        vmask[:, :, d0:d0 + n] = 0.0
        masks[vi] = vmask

        out = {kk: v for kk, v in latent.items() if kk not in ("samples", "noise_mask")}
        if packed or len(parts) > 1:
            _need_nested(who)
            out["samples"] = NestedTensor(tuple(parts))
            out["noise_mask"] = NestedTensor(tuple(masks))
        else:
            out["samples"] = parts[0]
            out["noise_mask"] = masks[0]

        df = _frames_of(d0); sf = _frames_of(s0)
        print(f"[{who}] held latents {d0}..{d0 + n - 1} = frames {df}..{df + F - 1} "
              f"<- next shot's frames {sf}..{sf + F - 1}")
        return (out,)


NODE_CLASS_MAPPINGS = {
    "H3RepackAVLatent": H3RepackAVLatent,
    "H3BlockTailAnchor": H3BlockTailAnchor,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3RepackAVLatent": "H3 Repack AV Latent (list -> packed)",
    "H3BlockTailAnchor": "Clown Anchoring",
}
