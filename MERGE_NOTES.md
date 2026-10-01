# Merge notes: ComfyLabV2 -> production

Decisions and to-dos to carry over when this worker replaces production's
koboldcpp worker (`minimax-h3-worker`) behind `minimax-h3-website`.

## Decided

- **Upscaler: NVIDIA RTX VSR 2x only.** Folded into the same ComfyUI submission
  as the generation (no `/free`), so MiniMax H3 stays resident between jobs.
  Confirmed clean on a real run 2026-10-01. ESRGAN 2x and FlashVSR were tested
  and removed.
- **LoRA: only "Turbo" (8-step)**,
  `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`, verified on the
  test page. At merge, remove every other LoRA from production:
  - production worker `LORA_CHOICES` (`quality` ->
    `minimax_h3_turbo_ema_ckpt500`, `fast` -> `minimax_h3_lightx2v_turbo`)
    and their cold-start downloads
  - the frontend presets (`web/src/components/generate/constants.ts`,
    `GeneratorPanel.tsx`, `settings/GenerationTab.tsx`)
  - this repo's `fast` (4-step 768p) preset and its download in
    `ensure_comfyui_engine.sh`

## Must do at merge

- **Heartbeat -> production reaper.** `touch_session_heartbeat()` and
  `mark_session_worker_started()` write to `comfylab_active_gpu_sessions`
  (filtered by `slot = 'default'`). Repoint them at `active_gpu_sessions`
  (filtered by `session_id`) so `server.js`'s `reapDeadWorkers` and the billing
  meter see them. Both writes fail silently, so on the first merged test,
  verify that `last_activity_at` actually advances.
