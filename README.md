# 🍰🎞️ Tiny AutoEncoder for Stable Diffusion Videos

## What is TAESDV?

TAESDV is a Tiny AutoEncoder for Stable Diffusion Videos. TAESDV can decode sequences of Stable Diffusion latents into continuous videos with much smoother results than single-frame [TAESD](https://github.com/madebyollin/taesd) (but within the same tiny runtime budget).

Since TAESDV efficiently supports both parallel and sequential frame decoding, TAESDV should be useful for:
1. Fast batched previewing for video-generation systems like [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid)
2. Fast realtime decoding for interactive v2v systems like [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)

| Original Video | TAESD Encode, TAESD Decode | TAESD Encode, TAESDV Decode |
| -------------- | -------------------------------------- | ---------------------------------------- |
| ![](https://github.com/user-attachments/assets/9eb1bba3-3bb5-4430-b0f8-76e176b91e4b)  | ![](https://github.com/user-attachments/assets/1330d79b-49ad-494d-b48e-ac54e4363fa2) | ![](https://github.com/user-attachments/assets/f4ef0531-e2c4-48c9-a7a0-eda6f62b99b8) |

> [!NOTE]
> Lots of TODOs still:
> 1. Add SVD or similar example notebook
> 2. Change repo example videos from roundtrip to gen decode (the TAESD encoder hasn't changed lol)
> 3. Add performance metrics (it's like the same)
> 4. Add StreamDiffusion or other v2v example
> 5. Get a less smudgy checkpoint :)
> 6. Add to Diffusers

## How does TAESDV work?

TAESDV was created by giving TAESD's decoder additional cross-frame-memory and finetuning it on video data.

## What are the limitations of TAESDV?

TAESDV is tiny and trying to work very quickly, so it tends to fudge fine details. If you want maximal quality, you should use the SVD VAE.
