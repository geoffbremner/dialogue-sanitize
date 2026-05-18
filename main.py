import argparse

from editor import rebuild_audio
from splitter import split_audio
from transcriber import detect_fillers


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_path",
        type=str,
    )

    parser.add_argument(
        "output_path",
        type=str,
    )

    parser.add_argument(
        "--chunk-dir",
        type=str,
        default="tmp_chunks",
    )

    args = parser.parse_args()

    print("Splitting audio...")
    chunks = split_audio(
        args.input_path,
        args.chunk_dir,
    )

    print(f"Generated {len(chunks)} chunks")

    print("Running Whisper inference...")
    exclusions = detect_fillers(chunks)

    print(f"Detected {len(exclusions)} filler words")

    for exclusion in exclusions:
        print(
            f"{exclusion.word}: "
            f"{exclusion.absolute_start:.3f} -> "
            f"{exclusion.absolute_end:.3f}"
        )

    print("Rebuilding audio...")
    rebuild_audio(
        args.input_path,
        exclusions,
        args.output_path,
    )

    print("Done")


if __name__ == "__main__":
    main()