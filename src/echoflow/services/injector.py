"""Text injection via clipboard paste (wl-copy + ydotool/wtype Ctrl+V)."""

from __future__ import annotations

import logging
import subprocess
import time

log = logging.getLogger(__name__)

# Terminal emulators that use Ctrl+Shift+V instead of Ctrl+V
TERMINAL_APP_IDS = frozenset({
    "foot",
    "footclient",
    "ghostty",
    "Alacritty",
    "kitty",
    "org.wezfurlong.wezterm",
    "com.mitchellh.ghostty",
    "org.gnome.Terminal",
    "org.kde.konsole",
    "xterm",
    "urxvt",
})

# Give the target app time to read the clipboard after a paste keystroke
# before we overwrite it with the next chunk (or restore the original).
_PASTE_SETTLE_S = 0.06


class Injector:
    """Injects text into the focused Wayland window via clipboard paste."""

    def inject(self, text: str, app_id: str = "") -> bool:
        """Inject text into focused window in one shot. Returns True on success."""
        if not text:
            return False

        old_clip = self._save_clipboard()
        try:
            self._copy(text)
            self._paste(app_id)
            log.info("Injected %d chars (app: %s)", len(text), app_id)
        except FileNotFoundError as e:
            log.error("Missing tool: %s", e)
            return False
        except subprocess.SubprocessError as e:
            log.error("Clipboard injection failed: %s", e)
            return False
        finally:
            # Apps read the clipboard asynchronously after Ctrl+V lands.
            time.sleep(0.1)
            self._restore_clipboard(old_clip)
        return True

    def begin_session(self, app_id: str = "") -> InjectionSession:
        """Start a streaming injection session for progressive, chunk-by-chunk paste."""
        return InjectionSession(self, app_id)

    # -- low-level clipboard/paste primitives --------------------------------

    def _save_clipboard(self) -> str | None:
        try:
            result = subprocess.run(
                ["wl-paste", "--no-newline"], capture_output=True, timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.decode("utf-8")
        except (subprocess.SubprocessError, UnicodeDecodeError):
            pass
        return None

    def _copy(self, text: str) -> None:
        subprocess.run(["wl-copy", "--"], input=text, text=True, check=True, timeout=5)

    def _paste(self, app_id: str) -> None:
        # Terminals use wtype Ctrl+Shift+V; everything else uses ydotool Ctrl+V
        # (wtype's virtual-keyboard protocol doesn't work with Chrome/Electron).
        if app_id in TERMINAL_APP_IDS:
            subprocess.run(
                ["wtype", "-M", "ctrl", "-M", "shift", "v", "-m", "shift", "-m", "ctrl"],
                check=True, timeout=10,
            )
        else:
            # ydotool key codes: 29=Left Ctrl, 47=V
            subprocess.run(
                ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
                check=True, timeout=10,
            )

    def _restore_clipboard(self, old_clip: str | None) -> None:
        if old_clip is not None:
            try:
                subprocess.run(
                    ["wl-copy", "--"], input=old_clip, text=True, timeout=2,
                )
            except subprocess.SubprocessError as e:
                log.warning("Failed to restore clipboard: %s", e)


class InjectionSession:
    """Pastes text incrementally as chunks arrive, saving/restoring the
    clipboard only once around the whole stream."""

    def __init__(self, injector: Injector, app_id: str) -> None:
        self._inj = injector
        self._app_id = app_id
        self._old_clip = injector._save_clipboard()
        self._chars = 0
        self._ok = True

    def feed(self, text: str) -> None:
        """Paste one chunk. Chunks after the first get a leading space."""
        if not text:
            return
        payload = (" " + text) if self._chars > 0 else text
        try:
            self._inj._copy(payload)
            self._inj._paste(self._app_id)
            self._chars += len(payload)
            # Let the app consume the clipboard before the next chunk overwrites it.
            time.sleep(_PASTE_SETTLE_S)
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            log.error("Streaming injection failed: %s", e)
            self._ok = False

    def end(self) -> bool:
        """Restore the original clipboard. Returns True if anything was injected."""
        time.sleep(0.1)
        self._inj._restore_clipboard(self._old_clip)
        if self._chars:
            log.info("Injected %d chars via streaming (app: %s)", self._chars, self._app_id)
        return self._ok and self._chars > 0
