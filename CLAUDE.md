# Project context for Claude Code

Since 2026-10-06 this repo is **Bizzle.ai's production GPU worker**: a
RunPod Serverless worker running ComfyUI + MiniMax H3, driven by
`minimax-h3-website` (server.js + web/, see its CLAUDE.md "ComfyUI engine").
It started as the ComfyLabV2 test worker; `MERGE_NOTES.md` records what was
ported, rejected and deferred, and `PROMPT_FRAMING_RULES.md` is the source
for the site's Prompt Assistant.

Branches: `main` and `claude/hello-4vzjin` are kept identical (push both).
The RunPod endpoint builds from `claude/hello-4vzjin`.

## How it runs

One RunPod job = one GPU session (`handler()` -> `run_session(session_id)`).
Anything without a valid session id is refused before ComfyUI starts. The
session must exist in `gpu_sessions` (not ended); its `user_id` is the owner.

1. Startup (heartbeat running): `ensure_comfyui_engine.sh` (SageAttention
   wheel from the volume cache, base model files), model symlinks, ComfyUI,
   a 1-step warmup on the base model. Billing starts at the very beginning
   of the job (`execution_started_at`, stamped first thing): only RunPod
   queue time is free; startup, model load and warmup are billed, matching
   RunPod's execution time. `worker_started_at` then marks "ready" (the UI
   lets the user generate). `warmup_seconds` in the summary is informational.
2. Loop: heartbeat (`active_gpu_sessions.last_activity_at`, 20s; also kept
   up through startup, every job and a crash restart: since 2026-10-09 the
   website charges GPU time only up to a minute past the last heartbeat, so
   a step without one would go unbilled), session
   still active? (`gpu_sessions.ended_at` null AND claim row present), claim
   the oldest queued `gpu_session_jobs` row, run it, write `progress` /
   `status` / `output`.
3. Ends on: Stop / out of credit / reaper (session no longer active — a
   running job is force-killed within ~2s), 45 minutes idle counted from
   ready and reset after each job (`end_reason: timeout`), 23h safety cutoff
   (`timeout`), or a failed restart (`error`, refunded by the website).
   Two safety stops (2026-10-09, `is_session_active`): if the session can't
   be read for 3 minutes in a row (a database outage) the worker stops; and
   every 30s it checks the billing backstop: when the time since the
   website's meter last charged (`gpu_sessions.metered_until`, else
   `billing_started_at`) is more than the owner's balance + 90s, the meter
   has stalled, so it ends the session as `out_of_credit` itself.

## Jobs

`gpu_session_jobs.input` is written by server.js only (validated there), and
re-validated here by `validate_job` (whitelist; `v: 2`; unknown fields
rejected). Shapes:

- `{"v":2,"mode":"generate","prompt","aspect":"16:9|9:16|1:1","duration":1-15,
  "speed":"standard|fast","seed":null|0..2^32-1,"upscale":bool,
  "ref_images":[<=9 keys],"start_frame":key|null,"end_frame":key|null,
  "ref_video":key|null}`
- `{"v":2,"mode":"face_refine","source_video_key":"<uuid>.mp4","seed"}`

Input keys must be `inputs/<owner_id>/<uuid>.(jpg|png|webp|mp4|mov)`.
Pictures are fully decoded (Pillow) and re-saved as PNG; start/end frames are
centre-cropped to the output canvas. Reference videos must be 24fps and
<=15.5s (probed with PyAV). Face-refine sources must be the owner's own Wasabi
video of kind generate/refine (`refine_source_allowed`).

**Free trial**: `gpu_sessions.is_trial` (read once at session start, never
from the job) makes `validate_job` refuse face fix, motion swap, start/end
frames and more than 2 pictures, force upscale off and clamp the length to
3s; the job's own `is_trial` stamp (added by the database trigger) must
agree. Every trial video gets "Made on Bizzle.Studio" burned in as a
repeating line in a faint band across the middle of the frame, scrolling
left to right without a break (`apply_watermark`, PyAV + Pillow, font
`assets/watermark-font.ttf` = Bricolage Grotesque Bold, OFL; audio
copied), so it can't be cropped or blurred out of one corner; if that
fails the job fails rather than upload an unmarked video.

What the worker decides (never the user):

| Job | Model | LoRA | Steps | Size | Upscale |
|---|---|---|---|---|---|
| generate, standard | base FL2VA | none | 20 | preset | RTX VSR 2x if `upscale` |
| generate, fast | base | fl2v 8-step turbo | 8 | preset | same |
| with pictures | base, MiniMaxH3ReferenceToVideo | as above | | | |
| pictures + frames | base, ReferenceToVideo + MiniMaxH3AddGuide | | | | |
| motion swap, standard | Ref2VA | ref2v 8-step 768p (+SigmaShift 6/3) | 8 | nearest of 16:9/9:16/1:1/4:3/3:4 to the video | off |
| motion swap, fast | Ref2VA | ref2v 4-step | 4 | same | off |
| face_refine | Wan 2.2 low-noise + lightx2v | | 4 | source | back to 2x if the source was 2x |

Sage attention is always on. Presets: 1344x768, 768x1344, 992x992 (2x:
2688x1536 / 1536x2688 / 1984x1984 — inside 4K, H.264 level 5.x, plays on
iPhone/Mac). 4x is not offered. Only one H3 model is resident: a motion swap
loads Ref2VA (~23-28s, "Loading the motion model" stage; ComfyUI's RAM-pressure
cache drops base), the next base job swaps back. Motion swaps keep the
reference video's soundtrack, and their length is the video's (17n+5 frames,
up to 362).

## Configuration

RunPod endpoint (the former test endpoint `qnrreh1dtupfsf`, its network
volume mounted at `/runpod-volume`) env vars:

| Var | Purpose |
|---|---|
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | production tables (service role) |
| `S3_ACCESS_KEY`, `S3_SECRET_KEY` | Wasabi |
| `S3_ENDPOINT` (default `https://s3.eu-west-1.wasabisys.com`), `S3_BUCKET` (default `comfylab-outputs`), `S3_REGION` (default `eu-west-1`) | Wasabi location |
| `HF_TOKEN` (optional) | HuggingFace downloads |
| `COMFY_JOB_TIMEOUT_SECONDS` (default 3600) | per-graph ceiling |
| `NVIDIA_VFX_VERSION` (default 0.2.0.0) | RTX VSR library |

Endpoint settings: RTX 5090, CUDA 13.0 floor, execution timeout above 23h
(the worker returns itself at 23h), container image built from this repo's
Dockerfile.

Volume contents (`/runpod-volume/models/...`): base FL2VA int8, text encoder
qwen3vl_32b nvfp4, video VAE fp16, audio VAE fp32, fl2v 8-step turbo
(engine script, every session start checks them); Ref2VA int8 + ref2v 4-step
and 8-step turbos (downloaded by the first motion swap if missing); Wan 2.2
low-noise 14B fp8, its lightx2v LoRA, umt5 text encoder, Wan VAE (first face
fix); face detector `ultralytics/bbox/face_yolov8m.pt` (engine script); the
SageAttention wheel cache `sageattention_wheel_cu13.0/` + marker.

## Storage (Wasabi bucket)

- `outputs/<uuid>.mp4` — results. Served only through the website's
  `GET /api/video/:key` after an ownership check.
- `inputs/<user_id>/<uuid>.<ext>` — uploads, written through presigned PUT
  URLs the website issues per user; swept after 48h by `cleanup.js`.
The bucket is private (anonymous list/get are refused).

## Not in production

The test page (`test/index.html`) and the comfylab_* tables are retired: the
worker no longer reads them. Removed from the worker: the standalone upscale
job and the upscale-only session job, model/LoRA/attention/refine-engine
choices, the H3 refine engine, DaSiWa, 4x upscale, the base 4-step LoRA.
