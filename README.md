# 🍰🎞️ Tiny AutoEncoder for Stable Diffusion Videos

## What is TAESDV?

TAESDV is a Tiny AutoEncoder for Stable Diffusion Videos. TAESDV can decode sequences of Stable Diffusion latents into videos with smoother results than single-frame [TAESD](https://github.com/madebyollin/taesd) (but within the same tiny runtime budget).

| Original Video | Encoded with TAESD, Decoded with TAESD | Encoded with TAESDV, Decoded with TAESDV |
| -------------- | -------------------------------------- | ---------------------------------------- |
| TODO           | TODO                                   | TODO                                     |

## What can I use TAESDV for?

TAESDV is compatible with SD1-based video generation models like Stable Video Diffusion, so you can use TAESDV for previewing SVD-generated videos. **TODO**: example notebook, for now read `taesdv.py`.

You can use TAESDV for real-time video-to-video generation with e.g. StreamDiffusion. **TODO**: example notebook, for now read `taesdv.py`.

## How does TAESDV work?

TAESDV was created by giving TAESD's decoder additional cross-frame-memory and finetuning it on video data.

## What are the limitations of TAESDV?

TAESDV is tiny and trying to work very quickly, so it tends to fudge fine details. If you want maximal quality, you should use the SVD VAE.
