"""Detect the appearance and completion of horizontal loading bars.

Requires ffmpeg and ffprobe on PATH. Timestamps accept seconds, MM:SS, or
HH:MM:SS[.fraction]. Coordinates are zero-based from the top-left of the
unrotated video frame.

Each bar is sampled at every x coordinate, inclusive from left_x through
right_x. A partial bar must have completed_color pixels on the left and
loading_color pixels on the right, with at most max_transition_gap unmatched
pixels between them. The default gap is two pixels. An all-loading row is
empty; an all-completed row is full. A load is counted only if it starts empty
and then makes progress. Progress may pause, but a decrease, reset to empty,
disappearance, or full bar ends the load. At the end, the script reports each
load's duration from its first empty frame to its first end-condition frame,
and totals per bar and across bars. With --output-dir, confirmed start and
end frames are saved as PNGs.
"""

import argparse
import json
import struct
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path


@dataclass(frozen=True)
class LoadingBar:
    left_x: int
    right_x: int
    y: int
    loading_color: tuple[int, int, int]
    completed_color: tuple[int, int, int]
    tolerance: int
    max_transition_gap: int


class BarObservation(Enum):
    ABSENT = "absent"
    NON_FULL = "non_full"
    FULL = "full"


class BarPhase(Enum):
    NO_BAR = "no_bar"
    HAS_BAR = "has_bar"
    COMPLETED = "completed"


@dataclass
class BarTracker:
    phase: BarPhase = BarPhase.NO_BAR
    start_timestamp: Decimal | None = None
    start_frame_index: int = 0
    start_frame: bytes | None = None
    last_completed_count: int = 0
    made_progress: bool = False


class PixelClassification(Enum):
    LOADING = "L"
    COMPLETED = "C"
    TRANSITION = "T"
    OTHER = "."


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
        raise argparse.ArgumentTypeError("minutes in an HH:MM:SS timestamp must be under 60")

    total = numbers[-1]
    if len(numbers) > 1:
        total += numbers[-2] * 60
    if len(numbers) > 2:
        total += numbers[-3] * 3600
    return total


def parse_rgb_color(value: object) -> tuple[int, int, int]:
    if not isinstance(value, str):
        raise ValueError("color must be six hexadecimal digits, e.g. #ff8800")
    color = value.removeprefix("#")
    if len(color) != 6:
        raise ValueError("color must be six hexadecimal digits, e.g. #ff8800")
    try:
        return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError as error:
        raise ValueError("color must be six hexadecimal digits, e.g. #ff8800") from error


def load_bars(path: Path) -> tuple[LoadingBar, ...]:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{path}: invalid JSON at line {error.lineno}, column {error.colno}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(f"{path}: the JSON root must be an object")
    raw_bars = data.get("bars")
    if not isinstance(raw_bars, list) or not raw_bars:
        raise ValueError(f"{path}: 'bars' must be a non-empty array")

    bars = []
    for bar_index, raw_bar in enumerate(raw_bars, start=1):
        location = f"bars[{bar_index - 1}]"
        if not isinstance(raw_bar, dict):
            raise ValueError(f"{path}: {location} must be an object")

        left_x = raw_bar.get("left_x")
        right_x = raw_bar.get("right_x")
        y = raw_bar.get("y")
        for field_name, coordinate in (
            ("left_x", left_x),
            ("right_x", right_x),
            ("y", y),
        ):
            if type(coordinate) is not int or coordinate < 0:
                raise ValueError(
                    f"{path}: {location}.{field_name} must be a non-negative integer"
                )
        if left_x > right_x:
            raise ValueError(f"{path}: {location}.left_x must be <= right_x")

        colors = []
        for field_name in ("loading_color", "completed_color"):
            try:
                colors.append(parse_rgb_color(raw_bar.get(field_name)))
            except ValueError as error:
                raise ValueError(f"{path}: {location}.{field_name}: {error}") from error

        tolerance = raw_bar.get("tolerance", 0)
        if type(tolerance) is not int or not 0 <= tolerance <= 255:
            raise ValueError(
                f"{path}: {location}.tolerance must be an integer from 0 to 255"
            )
        max_transition_gap = raw_bar.get("max_transition_gap", 2)
        if type(max_transition_gap) is not int or max_transition_gap < 0:
            raise ValueError(
                f"{path}: {location}.max_transition_gap must be a non-negative integer"
            )
        if all(
            abs(loading - completed) <= 2 * tolerance
            for loading, completed in zip(colors[0], colors[1])
        ):
            raise ValueError(
                f"{path}: {location}.loading_color and completed_color overlap "
                f"at tolerance {tolerance}; lower tolerance or choose more distinct colors"
            )

        bars.append(
            LoadingBar(
                left_x,
                right_x,
                y,
                colors[0],
                colors[1],
                tolerance,
                max_transition_gap,
            )
        )
    return tuple(bars)


def read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def format_timestamp(seconds: Decimal) -> str:
    milliseconds = int(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{whole_seconds:02}.{milliseconds:03}"


def png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


def save_frame_png(path: Path, frame: bytes, width: int, height: int) -> None:
    row_size = width * 3
    compressor = zlib.compressobj()

    with path.open("xb") as image:
        image.write(b"\x89PNG\r\n\x1a\n")
        image.write(png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))

        for y in range(height):
            row = frame[y * row_size : (y + 1) * row_size]
            compressed = compressor.compress(b"\x00" + row)
            if compressed:
                image.write(png_chunk(b"IDAT", compressed))

        compressed = compressor.flush()
        if compressed:
            image.write(png_chunk(b"IDAT", compressed))
        image.write(png_chunk(b"IEND", b""))


def save_bar_event_frame(
    output_dir: Path,
    bar_index: int,
    event: str,
    timestamp: Decimal,
    frame_index: int,
    frame: bytes,
    width: int,
    height: int,
) -> None:
    timestamp_label = format_timestamp(timestamp).replace(":", "-")
    output_path = output_dir / (
        f"{frame_index:07d}_{timestamp_label}_bar_{bar_index + 1:02d}_{event}.png"
    )
    try:
        save_frame_png(output_path, frame, width, height)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing frame image: {output_path}"
        ) from error


def save_debug_frame(
    output_dir: Path,
    frame_index: int,
    timestamp: Decimal,
    frame: bytes,
    width: int,
    height: int,
    bars: tuple[LoadingBar, ...],
    classifications: list[tuple[PixelClassification, ...]],
) -> None:
    highlight_colors = {
        PixelClassification.LOADING: (255, 170, 0),
        PixelClassification.COMPLETED: (0, 255, 0),
        PixelClassification.TRANSITION: (0, 255, 255),
        PixelClassification.OTHER: (255, 0, 255),
    }
    overlay = bytearray(frame)
    for bar, pixels in zip(bars, classifications):
        for x, classification in zip(range(bar.left_x, bar.right_x + 1), pixels):
            offset = (bar.y * width + x) * 3
            overlay[offset : offset + 3] = bytes(highlight_colors[classification])

    timestamp_label = format_timestamp(timestamp).replace(":", "-")
    output_path = output_dir / f"frame_{frame_index:07d}_{timestamp_label}.png"
    try:
        save_frame_png(output_path, bytes(overlay), width, height)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite debug frame: {output_path}"
        ) from error


def read_error(stream) -> str:
    stream.seek(0)
    return stream.read().decode("utf-8", errors="replace").strip()


def read_decoded_frame(decoder, stderr, frame_size: int, frame_index: int) -> bytes:
    frame = read_exact(decoder.stdout, frame_size)
    if len(frame) != frame_size:
        decoder_returncode = decoder.wait()
        if decoder_returncode:
            detail = read_error(stderr)
            raise RuntimeError(
                f"ffmpeg failed with exit code {decoder_returncode}"
                + (f": {detail}" if detail else "")
            )
        raise RuntimeError(
            f"ffmpeg decoded fewer frames than ffprobe reported "
            f"(stopped at frame {frame_index})"
        )
    return frame


def get_video_size(video: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError("the input has no video stream")
    return int(streams[0]["width"]), int(streams[0]["height"])


def pixel_matches_color(
    frame: bytes,
    pixel_offset: int,
    color: tuple[int, int, int],
    tolerance: int,
) -> bool:
    return all(
        abs(frame[pixel_offset + channel] - color[channel]) <= tolerance
        for channel in range(3)
    )


def inspect_loading_bar(
    frame: bytes,
    width: int,
    bar: LoadingBar,
) -> tuple[BarObservation, tuple[PixelClassification, ...], int]:
    classifications = []
    for x in range(bar.left_x, bar.right_x + 1):
        pixel_offset = (bar.y * width + x) * 3
        matches_completed = pixel_matches_color(
            frame, pixel_offset, bar.completed_color, bar.tolerance
        )
        if matches_completed:
            classifications.append(PixelClassification.COMPLETED)
        elif pixel_matches_color(
            frame, pixel_offset, bar.loading_color, bar.tolerance
        ):
            classifications.append(PixelClassification.LOADING)
        else:
            classifications.append(PixelClassification.OTHER)

    if all(pixel is PixelClassification.COMPLETED for pixel in classifications):
        return BarObservation.FULL, tuple(classifications), len(classifications)
    if all(pixel is PixelClassification.LOADING for pixel in classifications):
        return BarObservation.NON_FULL, tuple(classifications), 0

    index = 0
    while index < len(classifications) and classifications[index] is PixelClassification.COMPLETED:
        index += 1
    completed_count = index

    gap_start = index
    while index < len(classifications) and classifications[index] is PixelClassification.OTHER:
        index += 1
    gap_length = index - gap_start

    loading_start = index
    while index < len(classifications) and classifications[index] is PixelClassification.LOADING:
        index += 1
    loading_count = index - loading_start

    valid_transition = (
        index == len(classifications)
        and completed_count > 0
        and loading_count > 0
        and gap_length <= bar.max_transition_gap
    )
    if not valid_transition:
        return BarObservation.ABSENT, tuple(classifications), completed_count

    for gap_index in range(gap_start, gap_start + gap_length):
        classifications[gap_index] = PixelClassification.TRANSITION
    return BarObservation.NON_FULL, tuple(classifications), completed_count


def scan_loading_bars(
    video: Path,
    start: Decimal,
    end: Decimal,
    bars: tuple[LoadingBar, ...],
    output_dir: Path | None,
    debug_dir: Path | None,
) -> None:
    width, height = get_video_size(video)
    for bar in bars:
        if bar.right_x >= width or bar.y >= height:
            raise ValueError(
                f"Bar spans ({bar.left_x}, {bar.y}) to "
                f"({bar.right_x}, {bar.y}), outside the {width}x{height} video"
            )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(
            "[debug] pixel legend: L=loading/orange, C=completed/green, "
            "T=allowed transition/cyan, .=invalid/magenta; right_x is inclusive",
            flush=True,
        )

    probe_stderr = tempfile.TemporaryFile()
    decode_stderr = tempfile.TemporaryFile()
    probe = decoder = None
    try:
        probe = subprocess.Popen(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_frames",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video),
            ],
            stdout=subprocess.PIPE,
            stderr=probe_stderr,
            text=True,
            encoding="utf-8",
        )
        decoder = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-ss",
                str(start),
                "-noautorotate",
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=decode_stderr,
        )

        frame_size = width * height * 3
        trackers = [BarTracker() for _ in bars]
        loading_periods: list[tuple[Decimal, Decimal]] = []
        frame_index = 0
        event_count = 0
        last_progress = time.monotonic()
        last_timestamp = None
        reached_end = False

        def report_event(
            bar_index: int,
            bar: LoadingBar,
            event: str,
            timestamp: Decimal,
            frame: bytes | None,
            event_frame_index: int,
        ) -> None:
            nonlocal event_count
            if output_dir is not None:
                if frame is None:
                    raise RuntimeError(f"missing saved frame {event} event")
                save_bar_event_frame(
                    output_dir,
                    bar_index,
                    event,
                    timestamp,
                    event_frame_index,
                    frame,
                    width,
                    height,
                )
            event_count += 1

        def begin_load(
            tracker: BarTracker,
            timestamp: Decimal,
            frame: bytes,
            current_frame_index: int,
        ) -> None:
            tracker.phase = BarPhase.HAS_BAR
            tracker.start_timestamp = timestamp
            tracker.start_frame_index = current_frame_index
            tracker.start_frame = frame if output_dir is not None else None
            tracker.last_completed_count = 0
            tracker.made_progress = False

        def discard_load(tracker: BarTracker) -> None:
            tracker.phase = BarPhase.NO_BAR
            tracker.start_timestamp = None
            tracker.start_frame = None
            tracker.last_completed_count = 0
            tracker.made_progress = False

        def finish_load(
            bar_index: int,
            bar: LoadingBar,
            tracker: BarTracker,
            timestamp: Decimal,
            frame: bytes,
            current_frame_index: int,
            next_phase: BarPhase,
        ) -> None:
            load_start_timestamp = tracker.start_timestamp
            if load_start_timestamp is None:
                raise RuntimeError(f"missing start time")
            if timestamp < load_start_timestamp:
                raise RuntimeError(f"timestamps went backwards")
            loading_periods.append((load_start_timestamp, timestamp))
            report_event(
                bar_index,
                bar,
                "started",
                load_start_timestamp,
                tracker.start_frame,
                tracker.start_frame_index,
            )
            report_event(
                bar_index,
                bar,
                "ended",
                timestamp,
                frame,
                current_frame_index,
            )
            discard_load(tracker)
            tracker.phase = next_phase

        print("Starting load detection...", flush=True)

        while True:
            timestamp_line = probe.stdout.readline()
            if not timestamp_line:
                break

            try:
                timestamp = Decimal(timestamp_line.strip())
            except InvalidOperation as error:
                raise ValueError(
                    f"ffprobe returned an invalid timestamp for frame {frame_index}"
                ) from error

            if timestamp < start:
                continue
            if timestamp > end:
                reached_end = True
                break

            frame = read_decoded_frame(decoder, decode_stderr, frame_size, frame_index)
            frame_index += 1

            classifications_for_frame = []
            for bar_index, (bar, tracker) in enumerate(zip(bars, trackers)):
                observation, pixel_classifications, completed_count = inspect_loading_bar(
                    frame, width, bar
                )
                classifications_for_frame.append(pixel_classifications)
                previous_phase = tracker.phase
                if tracker.phase is BarPhase.NO_BAR:
                    if observation is BarObservation.NON_FULL and completed_count == 0:
                        begin_load(tracker, timestamp, frame, frame_index)
                elif tracker.phase is BarPhase.HAS_BAR:
                    if observation is BarObservation.FULL:
                        finish_load(
                            bar_index,
                            bar,
                            tracker,
                            timestamp,
                            frame,
                            frame_index,
                            BarPhase.COMPLETED,
                        )
                    elif observation is BarObservation.ABSENT:
                        if tracker.made_progress:
                            finish_load(
                                bar_index,
                                bar,
                                tracker,
                                timestamp,
                                frame,
                                frame_index,
                                BarPhase.NO_BAR,
                            )
                        else:
                            discard_load(tracker)
                    elif completed_count == 0:
                        if tracker.made_progress:
                            finish_load(
                                bar_index,
                                bar,
                                tracker,
                                timestamp,
                                frame,
                                frame_index,
                                BarPhase.NO_BAR,
                            )
                            begin_load(tracker, timestamp, frame, frame_index)
                    elif completed_count > tracker.last_completed_count:
                        tracker.last_completed_count = completed_count
                        tracker.made_progress = True
                    elif completed_count < tracker.last_completed_count:
                        finish_load(
                            bar_index,
                            bar,
                            tracker,
                            timestamp,
                            frame,
                            frame_index,
                            BarPhase.NO_BAR,
                        )
                elif tracker.phase is BarPhase.COMPLETED:
                    if observation is BarObservation.ABSENT:
                        tracker.phase = BarPhase.NO_BAR
                    elif observation is BarObservation.NON_FULL and completed_count == 0:
                        begin_load(tracker, timestamp, frame, frame_index)

                if debug_dir is not None:
                    pixel_status = "".join(
                        classification.value for classification in pixel_classifications
                    )
                    print(
                        f"[debug] frame={frame_index:07d} "
                        f"time={format_timestamp(timestamp)} "
                        f"range=({bar.left_x},{bar.y})-({bar.right_x},{bar.y}) "
                        f"observation={observation.value} "
                        f"progress={completed_count}/{bar.right_x - bar.left_x + 1} "
                        f"state={previous_phase.value}->{tracker.phase.value} "
                        f"pixels={pixel_status}",
                        file=sys.stderr,
                        flush=True,
                    )

            if debug_dir is not None:
                save_debug_frame(
                    debug_dir,
                    frame_index,
                    timestamp,
                    frame,
                    width,
                    height,
                    bars,
                    classifications_for_frame,
                )

            if debug_dir is None:
                last_timestamp = timestamp
                now = time.monotonic()
                if now - last_progress >= 1:
                    print(
                        f"\rProgress: video time {format_timestamp(timestamp)}",
                        file=sys.stderr,
                        end="",
                        flush=True,
                    )
                    last_progress = now

        if reached_end:
            for process in (probe, decoder):
                if process.poll() is None:
                    process.terminate()
                process.wait()
        else:
            extra_data = decoder.stdout.read(1)
            if extra_data:
                raise RuntimeError("ffmpeg decoded more frames than ffprobe reported")

            probe_returncode = probe.wait()
            decoder_returncode = decoder.wait()
            if probe_returncode:
                raise RuntimeError(f"ffprobe failed: {read_error(probe_stderr)}")
            if decoder_returncode:
                raise RuntimeError(f"ffmpeg failed: {read_error(decode_stderr)}")

        if debug_dir is None and last_timestamp is not None:
            print(file=sys.stderr, flush=True)
        print(
            "Detected load durations (first empty frame to first end condition):",
            flush=True,
        )
        total_loading_duration = Decimal(0)
        for load_start, load_end in loading_periods:
            duration = load_end - load_start
            total_loading_duration += duration
            print(
                f"Started at {format_timestamp(load_start)}, "
                f"ended at {format_timestamp(load_end)}, "
                f"duration {format_timestamp(duration)}",
                flush=True,
            )
        print(
            "Combined total detected loading time: "
            f"{format_timestamp(total_loading_duration)}",
            flush=True,
        )

        if last_timestamp is not None:
            saved_message = f"; event frames saved to {output_dir}" if output_dir is not None else ""
            print(
                f"Progress: finished at video time {format_timestamp(last_timestamp)}; "
                f"{event_count} event(s){saved_message}",
                flush=True,
            )
        else:
            print("Progress: no frames in the requested time range", file=sys.stderr, flush=True)
    finally:
        for process in (probe, decoder):
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                process.wait()
                if process.stdout is not None:
                    process.stdout.close()
        probe_stderr.close()
        decode_stderr.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="input video file")
    parser.add_argument("--start", required=True, type=parse_timestamp, help="start timestamp (inclusive)")
    parser.add_argument("--end", required=True, type=parse_timestamp, help="end timestamp (inclusive)")
    parser.add_argument(
        "--bars",
        required=True,
        type=Path,
        help="JSON file describing loading bars",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="save a PNG for every confirmed load start and end event",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        help="log every frame and save an overlay PNG for every frame in the time range",
    )
    args = parser.parse_args()

    if args.start > args.end:
        parser.error("--start must be less than or equal to --end")

    try:
        bars = load_bars(args.bars)
        scan_loading_bars(
            args.video,
            args.start,
            args.end,
            bars,
            args.output_dir,
            args.debug_dir,
        )
    except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
