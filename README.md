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

## Installation

Ensure your local terminal environment is running natively in 64-bit mode (`x86_64`). 

1. Create and activate a stable virtual environment:
   ```bash
   /usr/bin/arch -x86_64 python3.13 -m venv .venv
   source .venv/bin/activate
