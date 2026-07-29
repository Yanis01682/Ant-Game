#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
GAME_DIR = REPO_ROOT / "game"
DEFAULT_GAME_BIN = GAME_DIR / "output" / ("main.exe" if os.name == "nt" else "main")
ALT_GAME_BIN = GAME_DIR / "output" / ("main" if os.name == "nt" else "main.exe")
TIMEOUT_SECONDS = 60.0


class PipeReader:
    def __init__(self, stream) -> None:
        self.stream = stream
        self.buffer = bytearray()
        self.chunks: queue.Queue[bytes | BaseException | None] = queue.Queue()
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()

    def _pump(self) -> None:
        try:
            fd = self.stream.fileno()
            while True:
                chunk = os.read(fd, 4096)
                if not chunk:
                    self.chunks.put(None)
                    return
                self.chunks.put(chunk)
        except BaseException as exc:
            self.chunks.put(exc)

    def read_exact(self, size: int, timeout: float) -> bytes:
        deadline = time.monotonic() + timeout
        while len(self.buffer) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out while reading pipe")
            try:
                item = self.chunks.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError("timed out while reading pipe") from None
            if item is None:
                raise EOFError("unexpected EOF while reading pipe")
            if isinstance(item, BaseException):
                raise item
            self.buffer.extend(item)
        data = bytes(self.buffer[:size])
        del self.buffer[:size]
        return data


_PIPE_READERS: dict[int, PipeReader] = {}


def _reader_for(stream) -> PipeReader:
    key = id(stream)
    reader = _PIPE_READERS.get(key)
    if reader is None:
        reader = PipeReader(stream)
        _PIPE_READERS[key] = reader
    return reader


def resolve_game_bin(game_bin: Path) -> Path:
    if game_bin.exists():
        return game_bin
    if game_bin == DEFAULT_GAME_BIN and ALT_GAME_BIN.exists():
        return ALT_GAME_BIN
    if game_bin == ALT_GAME_BIN and DEFAULT_GAME_BIN.exists():
        return DEFAULT_GAME_BIN
    return game_bin


def make_game_if_needed(game_bin: Path) -> Path:
    resolved = resolve_game_bin(game_bin)
    if resolved.exists():
        return resolved
    if game_bin in (DEFAULT_GAME_BIN, ALT_GAME_BIN):
        subprocess.run(["make"], cwd=GAME_DIR, check=True)
        return resolve_game_bin(game_bin)
    return resolved


def packet(payload: object) -> bytes:
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(body)) + body


def read_exact(stream, size: int, proc: subprocess.Popen[bytes], label: str, timeout: float = TIMEOUT_SECONDS) -> bytes:
    try:
        return _reader_for(stream).read_exact(size, timeout)
    except TimeoutError:
        raise TimeoutError(f"timed out while reading {label}") from None
    except EOFError:
        code = proc.poll()
        if code is None:
            raise EOFError(f"unexpected EOF while reading {label}") from None
        raise EOFError(f"{label} closed with exit code {code}") from None


def read_game_packet(game: subprocess.Popen[bytes]) -> tuple[int, bytes]:
    size = struct.unpack(">I", read_exact(game.stdout, 4, game, "game packet length"))[0]
    obj = struct.unpack(">i", read_exact(game.stdout, 4, game, "game packet object"))[0]
    payload = read_exact(game.stdout, size, game, "game packet payload")
    return obj, payload


def read_ai_packet(ai: subprocess.Popen[bytes], name: str) -> bytes:
    size = struct.unpack(">I", read_exact(ai.stdout, 4, ai, f"{name} packet length"))[0]
    payload = read_exact(ai.stdout, size, ai, f"{name} packet payload")
    return struct.pack(">I", size) + payload


def write_all(stream, payload: bytes) -> None:
    stream.write(payload)
    stream.flush()


def launch_ai(ai_dir: Path, stderr_path: Path) -> subprocess.Popen[bytes]:
    stderr_handle = stderr_path.open("wb")
    return subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=ai_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_handle,
    )


def terminate(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def close_stdin(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.stdin is None:
        return
    try:
        proc.stdin.close()
    except OSError:
        pass


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def stage_zip(zip_path: Path, parent: Path, label: str) -> Path:
    output_dir = parent / label
    output_dir.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(output_dir)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local match between any two packaged AI zip files.")
    parser.add_argument("--zip0", type=Path, required=True)
    parser.add_argument("--zip1", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--game-bin", type=Path, default=DEFAULT_GAME_BIN)
    parser.add_argument("--keep-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    game_bin = make_game_if_needed(args.game_bin.resolve())
    workdir_obj = args.keep_dir
    tempdir_obj = None
    if workdir_obj is None:
        tempdir_obj = tempfile.TemporaryDirectory(prefix="agent-packaged-match-")
        workdir_obj = Path(tempdir_obj.name)
    workdir = workdir_obj.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    replay_path = workdir / "replay.json"
    game_stderr_path = workdir / "game.stderr.log"
    ai0_stderr_path = workdir / "ai0.stderr.log"
    ai1_stderr_path = workdir / "ai1.stderr.log"
    stage_root = workdir / "ais"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)

    ai0_dir = stage_zip(args.zip0.resolve(), stage_root, "ai0")
    ai1_dir = stage_zip(args.zip1.resolve(), stage_root, "ai1")

    game_stderr_handle = game_stderr_path.open("wb")
    game = None
    ai0 = None
    ai1 = None
    result: dict[str, object] = {
        "zip0": str(args.zip0.resolve()),
        "zip1": str(args.zip1.resolve()),
        "seed": args.seed,
        "workdir": str(workdir),
        "replay": str(replay_path),
    }
    events: list[dict[str, object]] = []

    def record_event(kind: str, **payload: object) -> None:
        entry = {"kind": kind, **payload}
        events.append(entry)
        if len(events) > 80:
            del events[0]
        if args.verbose:
            print(json.dumps(entry, ensure_ascii=False), flush=True)

    try:
        ai0 = launch_ai(ai0_dir, ai0_stderr_path)
        ai1 = launch_ai(ai1_dir, ai1_stderr_path)
        ais = {0: ai0, 1: ai1}

        game = subprocess.Popen(
            [str(game_bin)],
            cwd=GAME_DIR,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=game_stderr_handle,
        )

        init = {
            "player_list": [1, 1],
            "player_num": 2,
            "config": {"random_seed": args.seed},
            "replay": str(replay_path),
        }
        write_all(game.stdin, packet(init))
        record_event("send_init")

        while True:
            obj, payload = read_game_packet(game)
            record_event("game_packet", object=obj, size=len(payload))
            if obj in (0, 1):
                if payload and not payload.endswith(b"\n"):
                    payload += b"\n"
                write_all(ais[obj].stdin, payload)
                record_event("forward_to_ai", player=obj, preview=payload[:120].decode("utf-8", errors="replace"))
                continue

            message = json.loads(payload.decode("utf-8"))
            record_event("game_message", message=message)
            if isinstance(message, dict) and "player" in message and "content" in message:
                for player, content in zip(message["player"], message["content"]):
                    write_all(ais[int(player)].stdin, content.encode("utf-8"))
                    record_event("broadcast_to_ai", player=int(player), preview=content[:120])
            if isinstance(message, dict) and message.get("listen"):
                for player in message["listen"]:
                    ai_packet = read_ai_packet(ais[int(player)], f"ai{player}")
                    record_event("ai_reply", player=int(player), preview=ai_packet[:120].decode("latin1", errors="replace"))
                    reply = {
                        "player": int(player),
                        "content": ai_packet.decode("latin1"),
                        "time": 0,
                    }
                    write_all(game.stdin, packet(reply))
                    record_event("send_to_game", player=int(player), preview=reply["content"][:120])
            if isinstance(message, dict) and "end_state" in message:
                result["end_state"] = message["end_state"]
                result["end_info"] = message.get("end_info")
                break

        game.wait(timeout=3)
        close_stdin(ai0)
        close_stdin(ai1)
        for ai in (ai0, ai1):
            try:
                ai.wait(timeout=3)
            except subprocess.TimeoutExpired:
                terminate(ai)
        result["game_returncode"] = game.returncode
        result["ai0_returncode"] = ai0.returncode
        result["ai1_returncode"] = ai1.returncode
        result["game_stderr"] = read_text(game_stderr_path)
        result["ai0_stderr"] = read_text(ai0_stderr_path)
        result["ai1_stderr"] = read_text(ai1_stderr_path)
        if replay_path.exists():
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            result["rounds_recorded"] = len(replay)
            if replay:
                last_round = replay[-1].get("round_state", {})
                result["last_winner"] = last_round.get("winner")
                result["last_error"] = last_round.get("error")
        result["events"] = events
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result["events"] = events
        if game is not None:
            result["game_returncode"] = game.poll()
        if ai0 is not None:
            result["ai0_returncode"] = ai0.poll()
        if ai1 is not None:
            result["ai1_returncode"] = ai1.poll()
        result["game_stderr"] = read_text(game_stderr_path)
        result["ai0_stderr"] = read_text(ai0_stderr_path)
        result["ai1_stderr"] = read_text(ai1_stderr_path)
        result["replay_exists"] = replay_path.exists()
        if replay_path.exists():
            result["replay_size"] = replay_path.stat().st_size
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    finally:
        terminate(ai0)
        terminate(ai1)
        terminate(game)
        game_stderr_handle.close()
        if tempdir_obj is not None:
            tempdir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
