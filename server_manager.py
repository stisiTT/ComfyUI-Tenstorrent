# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
tt-metal inference server lifecycle manager for the ComfyUI TT nodes.

The TT_CheckpointLoader node uses this to stand up the tt-metal server on demand
based on the selected model, instead of requiring the user to launch it manually.

Design notes:
- We spawn ``tt-metal/launch_server.sh`` as a subprocess. That script activates
  tt-metal's own ``python_env``, so the server runs with the correct interpreter
  and there is NO cross-venv / NumPy ABI contamination with ComfyUI.
- The server is single-model. Selecting a different model stops the running
  server and starts a new one (multi-minute warmup each switch).
- The child is started in its own session (setsid) so we can signal the whole
  process group on cleanup. A PID/PGID lock file is written as a backstop in case
  ComfyUI is hard-killed before atexit/signal handlers run.
"""

import atexit
import json
import logging
import os
import signal
import subprocess
import threading
import time
from typing import Optional

import requests

logger = logging.getLogger("tenstorrent_nodes.server_manager")

# Default board per model (overridable via env).
MODEL_BOARDS = {
    "sdxl": os.getenv("TT_SDXL_BOARD", "p150"),
    "wan22": os.getenv("TT_WAN22_BOARD", "p300x2"),
}

# Substring expected in the /health "model" label for each model key.
_MODEL_LABEL_HINTS = {
    "sdxl": "sdxl",
    "wan22": "wan",
    "sd35": "sd3",
}

TT_METAL_DIR = os.getenv("TT_METAL_DIR", "/home/stisi/tt-metal")
DEFAULT_HOST = os.getenv("TT_SERVER_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("TT_SERVER_PORT", "8000"))
PID_FILE = os.getenv("TT_SERVER_PID_FILE", "/tmp/tt_comfy_server.pid")

# Warmup can include first-run trace capture (SD35/Wan up to ~25 min).
READY_TIMEOUT_SECONDS = float(os.getenv("TT_SERVER_READY_TIMEOUT", "1800"))


class _ServerManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._model: Optional[str] = None
        self._board: Optional[str] = None
        self._host = DEFAULT_HOST
        self._port = DEFAULT_PORT
        self._cleanup_registered = False

    # -- helpers -----------------------------------------------------------

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def _health(self, timeout: float = 5.0) -> Optional[dict]:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            return None
        return None

    def _health_matches(self, model: str, health: dict) -> bool:
        if not health or health.get("status") != "healthy":
            return False
        label = str(health.get("model", "")).lower()
        hint = _MODEL_LABEL_HINTS.get(model, model)
        return hint in label

    def _log_path(self, model: str) -> str:
        return os.path.join(TT_METAL_DIR, f"{model}_server_comfy.log")

    def _write_pid_file(self):
        try:
            if self._proc is not None:
                pgid = os.getpgid(self._proc.pid)
                with open(PID_FILE, "w") as f:
                    json.dump({"pid": self._proc.pid, "pgid": pgid, "model": self._model}, f)
        except Exception as e:
            logger.warning(f"Could not write PID file {PID_FILE}: {e}")

    def _remove_pid_file(self):
        try:
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
        except Exception:
            pass

    # -- lifecycle ---------------------------------------------------------

    def _register_cleanup(self):
        if self._cleanup_registered:
            return
        atexit.register(self.stop)

        def _handler(signum, frame):
            logger.info(f"server_manager: received signal {signum}, stopping tt-metal server")
            self.stop()
            # Re-raise default behavior for the signal.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handler)
            except (ValueError, RuntimeError):
                # Not in main thread; atexit still covers normal exit.
                pass
        self._cleanup_registered = True

    def stop(self):
        with self._lock:
            proc = self._proc
            if proc is None:
                return
            if proc.poll() is None:
                logger.info(f"Stopping tt-metal server (pid={proc.pid}, model={self._model})")
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGTERM)
                except Exception:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                # Wait up to 15s for graceful shutdown (device release).
                deadline = time.time() + 15
                while proc.poll() is None and time.time() < deadline:
                    time.sleep(0.5)
                if proc.poll() is None:
                    logger.warning("tt-metal server did not stop gracefully, forcing kill")
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    proc.wait(timeout=10)
            self._proc = None
            self._model = None
            self._board = None
            self._remove_pid_file()
            # Settle so devices are fully released before any restart.
            time.sleep(3)

    def _pid_looks_like_server(self, pid: int) -> bool:
        """Best-effort guard against killing a recycled PID.

        Confirm the process command line still references the tt-metal launch
        script / server before we signal it.
        """
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, ProcessLookupError):
            return False
        except Exception:
            # If we cannot read cmdline, do not risk signalling.
            return False
        cmdline = cmdline.lower()
        return ("launch_server.sh" in cmdline) or ("tt-metal" in cmdline) or ("tt_metal" in cmdline)

    def stop_any(self):
        """Stop our managed server, then reap any orphaned/external server.

        Covers the case where a previous ComfyUI process crashed and left a
        tt-metal server running: we recover the recorded PID/PGID from the lock
        file and signal that process group (after a cmdline sanity check to
        avoid killing a recycled PID).
        """
        # 1) Stop the server this process is managing (clears the PID file).
        self.stop()

        # 2) Fallback: reap an orphaned server recorded in the lock file.
        with self._lock:
            try:
                with open(PID_FILE) as f:
                    info = json.load(f)
            except FileNotFoundError:
                return
            except Exception as e:
                logger.warning(f"Could not read PID file {PID_FILE}: {e}")
                return

            pid = info.get("pid")
            pgid = info.get("pgid")
            model = info.get("model")
            if not pgid:
                self._remove_pid_file()
                return

            if pid is not None and not self._pid_looks_like_server(pid):
                logger.info(
                    f"PID-file fallback: pid={pid} no longer looks like a tt-metal "
                    f"server (likely already gone or recycled); clearing stale {PID_FILE}"
                )
                self._remove_pid_file()
                return

            logger.info(f"PID-file fallback: stopping orphaned tt-metal server (pgid={pgid}, model={model})")
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                self._remove_pid_file()
                return
            except Exception as e:
                logger.warning(f"Could not SIGTERM orphaned server group {pgid}: {e}")
                self._remove_pid_file()
                return

            # Wait up to 15s for graceful shutdown (device release).
            deadline = time.time() + 15
            while time.time() < deadline:
                try:
                    os.killpg(pgid, 0)  # probe: alive?
                except ProcessLookupError:
                    break
                except Exception:
                    break
                time.sleep(0.5)
            else:
                logger.warning("Orphaned tt-metal server did not stop gracefully, forcing kill")
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except Exception:
                    pass

            self._remove_pid_file()
            # Settle so devices are fully released before any reset/restart.
            time.sleep(3)

    def _start(self, model: str, board: str):
        script = os.path.join(TT_METAL_DIR, "launch_server.sh")
        if not os.path.isfile(script):
            raise RuntimeError(f"launch_server.sh not found at {script} (set TT_METAL_DIR)")

        cmd = [
            "./launch_server.sh",
            "--model", model,
            "--board", board,
            "--host", self._host,
            "--port", str(self._port),
        ]
        log_path = self._log_path(model)
        logger.info(f"Starting tt-metal server: {' '.join(cmd)} (cwd={TT_METAL_DIR}, log={log_path})")
        log_file = open(log_path, "w")
        self._proc = subprocess.Popen(
            cmd,
            cwd=TT_METAL_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group for clean killpg
        )
        self._model = model
        self._board = board
        self._write_pid_file()
        self._register_cleanup()
        self._wait_for_health(model, log_path)

    def _wait_for_health(self, model: str, log_path: str):
        logger.info(f"Waiting for tt-metal server /health (timeout {READY_TIMEOUT_SECONDS:.0f}s)...")
        deadline = time.time() + READY_TIMEOUT_SECONDS
        last_log = 0.0
        while time.time() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                tail = self._tail(log_path)
                raise RuntimeError(
                    f"tt-metal server exited during startup (model={model}). "
                    f"Last log lines:\n{tail}"
                )
            health = self._health()
            if self._health_matches(model, health):
                logger.info(f"tt-metal server is healthy at {self.base_url} (model={model})")
                return
            now = time.time()
            if now - last_log >= 30:
                logger.info(f"Still warming up (model={model})... first run includes trace capture")
                last_log = now
            time.sleep(5)
        tail = self._tail(log_path)
        self.stop()
        raise RuntimeError(
            f"tt-metal server for '{model}' did not become healthy within "
            f"{READY_TIMEOUT_SECONDS:.0f}s. Last log lines:\n{tail}"
        )

    @staticmethod
    def _tail(path: str, n: int = 40) -> str:
        try:
            with open(path, "r") as f:
                return "".join(f.readlines()[-n:])
        except Exception:
            return "(no log available)"

    # -- public API --------------------------------------------------------

    def ensure_server(self, model: str, board: Optional[str] = None,
                       host: Optional[str] = None, port: Optional[int] = None) -> str:
        """Ensure a healthy tt-metal server for ``model`` is running; return base URL.

        Reuses an existing server if it already serves ``model``; otherwise stops
        any running server and starts the requested one (blocking until healthy).
        """
        if model not in MODEL_BOARDS:
            raise ValueError(f"Unknown model '{model}'. Known: {sorted(MODEL_BOARDS)}")
        board = board or MODEL_BOARDS[model]

        with self._lock:
            if host:
                self._host = host
            if port:
                self._port = int(port)

            # 1) Reuse our own managed server if it already serves this model.
            if self._proc is not None and self._proc.poll() is None and self._model == model:
                if self._health_matches(model, self._health()):
                    logger.info(f"Reusing running tt-metal server (model={model}) at {self.base_url}")
                    return self.base_url
                logger.warning("Managed server unhealthy; restarting")
                self.stop()

            # 2) Reuse an externally-started server that already serves this model.
            if self._proc is None:
                if self._health_matches(model, self._health()):
                    logger.info(f"Found external tt-metal server serving '{model}' at {self.base_url}")
                    return self.base_url

            # 3) Switch models: stop whatever is running, then start the new one.
            if self._proc is not None:
                logger.info(f"Switching model {self._model} -> {model}; restarting server")
                self.stop()

            self._start(model, board)
            return self.base_url


# Process-wide singleton.
_manager = _ServerManager()


def get_manager() -> "_ServerManager":
    return _manager


def ensure_server(model: str, board: Optional[str] = None,
                  host: Optional[str] = None, port: Optional[int] = None) -> str:
    return _manager.ensure_server(model, board=board, host=host, port=port)


# Path to the tt-smi console script (its own venv keeps native deps isolated).
TT_SMI_BIN = os.getenv("TT_SMI_BIN", "/home/stisi/tt-smi/venv/bin/tt-smi")


def _reset_all_boards_subprocess():
    """Fallback: reset all boards by shelling out to the tt-smi console script.

    Isolates tt-smi's native deps and its internal ``sys.exit`` calls in a
    separate process so they cannot tear down ComfyUI.
    """
    logger.info(f"reset_all_boards: running subprocess '{TT_SMI_BIN} -r'")
    try:
        subprocess.run([TT_SMI_BIN, "-r"], check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"tt-smi binary not found at '{TT_SMI_BIN}'. Set TT_SMI_BIN to the "
            f"tt-smi console script (e.g. <tt-smi-venv>/bin/tt-smi)."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"tt-smi reset failed (exit {e.returncode})") from e
    logger.info("reset_all_boards: subprocess reset completed")


def reset_all_boards():
    """Reset all detected Tenstorrent PCI devices.

    Mirrors the ``ResetType.ALL`` path in tt-smi (``pci_scan()`` then
    ``pci_board_reset(...)``). Tries an in-process call first; on import or
    runtime failure it falls back to the tt-smi subprocess.

    IMPORTANT: this is a board-level reset that affects ALL boards on the host
    and must only be run after the tt-metal server has released the device.

    tt-smi's reset code calls ``sys.exit()`` on both success and error paths, so
    the in-process call traps ``SystemExit`` to avoid tearing down ComfyUI.
    """
    try:
        from pyluwen import pci_scan
        from tt_smi.tt_smi_reset import pci_board_reset
    except ImportError as e:
        logger.warning(
            f"reset_all_boards: in-process tt-smi import failed ({e}); "
            f"falling back to subprocess"
        )
        _reset_all_boards_subprocess()
        return

    logger.info("reset_all_boards: in-process pci_scan + pci_board_reset (all devices)")
    try:
        indices = pci_scan()
        if not indices:
            raise RuntimeError("reset_all_boards: pci_scan() found no devices to reset")
        pci_board_reset(
            indices,
            reinit=True,
            print_status=False,
            use_umd=True,
            wait_for_eth=True,
        )
    except SystemExit as e:
        # tt-smi exits 0 on success; non-zero indicates a failure.
        code = e.code
        if code not in (0, None):
            raise RuntimeError(f"tt-smi in-process reset failed (exit {code})") from e
        logger.info("reset_all_boards: in-process reset completed (SystemExit 0 trapped)")
        return
    except Exception as e:
        logger.warning(
            f"reset_all_boards: in-process reset raised ({e}); falling back to subprocess"
        )
        _reset_all_boards_subprocess()
        return
    logger.info("reset_all_boards: in-process reset completed")
