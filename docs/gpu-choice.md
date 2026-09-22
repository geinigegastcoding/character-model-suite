# Modal GPU choice

Price snapshot verified 2026-09-22 against the [Modal pricing page](https://modal.com/pricing). Prices are per GPU-hour and exclude CPU, RAM, storage, and any free-credit plan.

| GPU | VRAM | Modal price | Use here |
|---|---:|---:|---|
| L40S | 48 GiB | about $1.95/h | Best value for text/audio LoRA, inference, and cheap experiments. |
| A100 40GB | 40 GiB | about $2.10/h | Not the first choice for Qwen Image training. |
| A100 80GB | 80 GiB | about $2.50/h | Cheapest sensible default for the 20B Qwen Image LoRA run. |
| H100 | 80 GiB | about $3.95/h | Faster, but only cheaper in total if it finishes materially sooner. |

## Decision

Use `A100-80GB` for the first image trainer. The official Modal DreamBooth example labels fine-tuning VRAM-heavy and uses an A100-80GB; Qwen Image is a 20B model, so the 48 GiB L40S is a cost-saving gamble for this first run even though it is roughly 22% cheaper per hour.

Use `L40S` for the planned text, coding, and audio experiments where QLoRA/LoRA and 48 GiB are a much better fit. Use H100 only when a benchmark proves that the saved runtime offsets the approximately 58% higher hourly cost compared with A100-80GB.

The first benchmark should be a fixed 50-step run with the same data, resolution, rank, and seed. Record wall-clock time and total GPU cost. Do not optimize hourly price while ignoring failed/OOM runs: a failed 48 GiB job is more expensive than a completed 80 GiB job.

