# dialogue-sanitize

A pragmatic, local Python utility designed to automatically detect and surgically remove spoken filler words (`"um"`, `"uh"`, `"uhm"`) from audio files. 

The tool optimizes local CPU performance by chunking long audio files at silence points, leveraging an execution-optimized instance of `faster-whisper` for timestamped disfluency tracking, and rebuilding the edit track using zero-crossing alignment and crossfades to guarantee artifact-free transitions.

## Features

- **Silence-Aware Chunking:** Splits long audio files into digestible ~30-second blocks anchored to natural silent intervals ($\leq -40.0$ dBFS) to prevent splitting mid-word before transcription.
- **Optimized Local Inference:** Runs `faster-whisper` locally using the `base.en` model quantized to `int8` for fast, lightweight CPU processing.
- **Context Priming:** Uses an explicit initial prompt style to discourage Whisper from naturally "auto-correcting" or skipping vocal disfluencies.
- **Production-Grade Editing:** - Matches edits to the nearest waveform zero-crossings to prevent phase pops and clicks.
  - Applies a 5ms equal-power crossfade across cut boundaries for transparent dialogue stitching.
  - Merges overlapping or closely timed exclusion zones ($\leq 50$ms apart) to avoid rapid, unnatural micro-cuts.

## Architecture Overview

- `main.py`: CLI entry point and high-level execution orchestrator.
- `splitter.py`: Handles signal analysis (RMS calculation) and segments the input track.
- `transcriber.py`: Runs the ML token prediction pipeline and extracts word-level timestamps.
- `editor.py`: Executes the non-destructive audio array splicing, crossfades, and zero-crossing lookaheads.
- `utils.py`: Core digital signal processing (DSP) helpers (mono conversion, RMS dB calculations, zero-crossing finders).
- `models.py`: Structural data abstractions (`Chunk`, `ExclusionZone`).
- `system_prompt.txt` Core system prompt configuration file. It contains the localized developer steering rules, constraints, and instructions based on the Andrej Karpathy configuration pattern to instantly align LLM context windows upon folder initialization.

## Installation - Tested only on Geoff's local -

Ensure your local terminal environment is running natively in 64-bit mode (`x86_64`). 

1. Create and activate a stable virtual environment:
   ```bash
   /usr/bin/arch -x86_64 python3.13 -m venv .venv
   source .venv/bin/activate
2. Quick start - assuming test file "um-test2.wav" is in "test-files"   
   python main.py test-files/um-test2.wav output.wav

# dialogue-sanitize Roadmap & Backlog

## TODO 1: Implement Adaptive Zero-Crossing Gate & Verbose Skipping

### Architectural Goal
Objective: Refine the current slicing boundaries inside editor.py. The cut infrastructure works, but boundary positioning can be significantly optimized.

### SUGGESTED Steps - we HAVE to review this
1. **Remove DSP code from transcription tasks:** Ensure `transcriber.py` remains focused solely on converting model tokens to absolute time coordinates.
2. **Implement Max Deviation Constraint in `editor.py`:**
   - Update `rebuild_audio()` to verify the sample offset returned by `find_nearest_zero_crossing()`.
   - If the nearest zero-crossing index deviates from the target timestamp boundary by more than a safe threshold (e.g., `MAX_DEVIATION_MS = 15`), flag the cut as high-risk.
3. **Add High-Verbosity Warning Logs:**
   - Instead of silently slicing through a complex waveform or dropping the execution, print an explicit warning to stdout:
     `[SKIP ALERT] Filler word "um" at 00:04:23.150 skipped. No stable zero-crossing found within 15ms radius (Local RMS: -12.4 dBFS). Boundary context preserved.`
4. **Unit Test Coverage:**
   - Create a synthetic test WAV file with a continuous high-amplitude sine wave to simulate a "no zero-crossing failure" scenario and confirm that the engine skips it gracefully.

## TODO 2: Stress-Test with Long-Form Content
- **Objective:** Run a raw, unedited multihour podcast track (>60 mins) to benchmark heap usage.
- **Verification Metrics:** Ensure memory allocation for local array stitching doesn't scale linearly with audio length, verifying that `tmp_chunks` allocation blocks are correctly freed post-inference.

## TODO 3: Cloud Infrastructure Splicing
- **Objective:** Shift local CPU bottlenecks to an on-demand container environment.
- **Target Stack:** Containerize the application using a minimal `Dockerfile` optimized for CPU matrix mathematics (using shared `OpenBLAS` headers) and deploy via an ephemeral AWS Fargate task or Google Cloud Run architecture triggered by cloud bucket uploads.

## TODO 4: Merchant Layer Integration (PayPal linking)
- **Objective:** Build an automated webhook receiver to gateway processing access behind payment verification.
- **Target Stack:** Implement a lightweight web engine layer (FastAPI) that exposes an endpoint for PayPal Instant Payment Notifications (IPN). Upon receipt of a valid `payment_status==Completed` capture flag, release the user's processing session token to run the sanitation pipeline on their payload.

## TODO 5: Implement Multi-Tiered "Um-Removal Intensity" Modes (The Dial/Knob Concept)
Objective: Introduce an edit aggressiveness configurations dial (Low, Medium, High) via CLI flags (e.g., --intensity medium).

Tuning Targets:

Low: Only cuts clear, isolated disfluencies with highly confident ML tokens. Preserves natural conversational pauses.

Medium: Standard default pipeline execution profile.

High: Aggressive sanitation. Drops confidence thresholds, reduces padding parameters to execute lightning-fast micro-cuts, and cleans conversational stutters tightly.

## TODO 6: Implement Global Verbose Logging Framework
Objective: Replace standard script print blocks with structured, repository-wide verbose diagnostics across all runtime modules.

Execution: Introduce a --verbose flag tracking granular pipeline statuses, including local chunk RMS values, model token confidence scores, and specific sample offset corrections computed by the zero-crossing mechanics.
