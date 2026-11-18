"""Download a full YouTube or Twitch video as an MKV file.

Requires yt-dlp and FFmpeg. Optional start/end timestamps are included in the
suggested command for processing the downloaded file; they do not limit the
download.
"""

import argparse
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path


FORMAT = "bestvideo[height<=360]+bestaudio/best[height<=360]/bestvideo+bestaudio/best"


def parse_timestamp(value: str) -> Decimal:
    parts = value.split(":")
    if len(parts) > 3:
        raise argparse.ArgumentTypeError("use seconds, MM:SS, or HH:MM:SS")

    try:
        numbers = [Decimal(part) for part in parts]
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError("timestamp must be numeric") from error

    if any(not number.is_finite() or number < 0 for number in numbers):
        raise argparse.ArgumentTypeError("timestamp cannot be negative")
    if len(numbers) > 1 and numbers[-1] >= 60:
        raise argparse.ArgumentTypeError("seconds in a timestamp must be under 60")
    if len(numbers) == 3 and numbers[-2] >= 60:
        raise argparse.ArgumentTypeError(
            "minutes in an HH:MM:SS timestamp must be under 60"
        )

    total = numbers[-1]
    if len(numbers) > 1:
        total += numbers[-2] * 60
    if len(numbers) > 2:
        total += numbers[-3] * 3600
    return total


def format_timestamp(seconds: Decimal) -> str:
    hours = int(seconds // 3600)
    remainder = seconds - hours * 3600
    minutes = int(remainder // 60)
    seconds_in_minute = remainder - minutes * 60
    whole_seconds = int(seconds_in_minute)
    fractional_seconds = seconds_in_minute - whole_seconds
    formatted_seconds = f"{whole_seconds:02}"
    if fractional_seconds:
        formatted_seconds += format(fractional_seconds, "f")[1:]
    return f"{hours:02}:{minutes:02}:{formatted_seconds}"


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def download_video(
    url: str,
    output: Path,
    concurrent_fragments: int,
    start: Decimal | None,
    end: Decimal | None,
) -> None:
    output = output.resolve()
    if output.suffix.lower() != ".mkv":
        raise ValueError("output file must have an .mkv extension")
    if output.exists():
        raise FileExistsError(f"output file already exists: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.stem}-",
        dir=output.parent,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        download_template = temporary_path / "source.%(ext)s"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--no-playlist",
                "--concurrent-fragments",
                str(concurrent_fragments),
                "--format",
                FORMAT,
                "--merge-output-format",
                "mkv",
                "--remux-video",
                "mkv",
                "--output",
                str(download_template),
                url,
            ],
            check=True,
        )

        downloaded_video = temporary_path / "source.mkv"
        if not downloaded_video.is_file():
            raise RuntimeError("yt-dlp did not produce the expected source.mkv file")
        downloaded_video.rename(output)

    print(f"Saved full video to {output}")
    if start is not None and end is not None:
        print(
            "Process the requested interval with: "
            f'python extract_matching_frames.py "{output}" '
            f"--start {format_timestamp(start)} --end {format_timestamp(end)} "
            "--bars bars.json"
        )
    else:
        print(
            "The extractor still requires --start and --end when processing "
            "this downloaded video."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="YouTube or Twitch video/VOD URL")
    parser.add_argument(
        "--start",
        type=parse_timestamp,
        help="optional processing start time (used only in the suggested command)",
    )
    parser.add_argument(
        "--end",
        type=parse_timestamp,
        help="optional processing end time (used only in the suggested command)",
    )
    parser.add_argument(
        "--concurrent-fragments",
        type=positive_int,
        default=8,
        help="number of HLS/DASH fragments to download concurrently (default: 8)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("video.mkv"),
        help="output MKV path (default: video.mkv; existing files are not overwritten)",
    )
    args = parser.parse_args()

    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must either both be provided or both omitted")
    if args.start is not None and args.end is not None and args.start >= args.end:
        parser.error("--start must be less than --end")

    try:
        download_video(
            args.url,
            args.output,
            args.concurrent_fragments,
            args.start,
            args.end,
        )
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
