# 🍰🎞️ Tiny AutoEncoder for Stable Diffusion Videos

## What is TAESDV?

TAESDV is a Tiny AutoEncoder for Stable Diffusion Videos. TAESDV can decode sequences of Stable Diffusion latents into continuous videos with much smoother results than single-frame [TAESD](https://github.com/madebyollin/taesd) (but within the same tiny runtime budget).

Since TAESDV efficiently supports both parallel and sequential frame decoding, TAESDV should be useful for:
1. Fast batched previewing for video-generation systems like [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid) or [AnimateLCM](https://animatelcm.github.io).
2. Fast realtime decoding for interactive v2v systems like [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion).

| Original Video | TAESD Encode, TAESD Decode | TAESD Encode, TAESDV Decode |
| -------------- | -------------------------------------- | ---------------------------------------- |
| ![test_video mp4](./images/test_video.mp4.gif) | ![test_video mp4 reconstructed_taesd mp4](./images/test_video.mp4.reconstructed_taesd.mp4.gif) | ![test_video mp4 reconstructed mp4](images/test_video.mp4.reconstructed.mp4.gif) |

> [!NOTE]
> Lots of TODOs still:
>
> 1. Add StreamDiffusion or other v2v example
> 2. Add performance metrics (it's like the same as TAESD)
> 3. Better / more example videos
> 4. Add to Diffusers somehow?
> 5. Even better checkpoint?

## How can I use TAESDV for previewing generated videos?

See the [AnimateLCM previewing example](./examples/TAESDV_Previewing_During_Generation.ipynb), which visualizes a TAESDV preview after each generation step.

<video src="https://github.com/user-attachments/assets/2fa29dd7-1f85-493b-9c9f-dd5337ab3337"/>

## How does TAESDV work?

TAESDV was created by giving TAESD's decoder additional cross-frame-memory and finetuning it on video data.

## What are the limitations of TAESDV?

TAESDV is tiny and trying to work very quickly, so it tends to fudge fine details. If you want maximal quality, you should use the SVD VAE.
