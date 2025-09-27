"""Helpers for generating movie commentary scripts from the Streamlit UI."""

from __future__ import annotations

import asyncio
import time
from typing import Callable

import streamlit as st
from loguru import logger

from app.config import config
from app.services.movie_commentary import MovieCommentaryService


def _create_progress_updater() -> tuple[Callable[[float, str], None], Callable[[], None]]:
    """Create a Streamlit-friendly progress updater for async callbacks."""

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update(progress: float, message: str = "") -> None:
        value = int(max(0, min(100, round(progress))))
        progress_bar.progress(value)
        if message:
            status_text.text(f"🎬 {message}")
        else:
            status_text.text(f"📊 {value}%")

    def finalize() -> None:
        progress_bar.empty()
        status_text.empty()

    return update, finalize


def generate_movie_commentary_script(tr, params) -> None:  # pragma: no cover - UI helper
    """Generate a movie commentary script via :class:`MovieCommentaryService`."""

    if not params.video_origin_path:
        st.error(tr("Please select a video file first"))
        return

    update_progress, finalize_progress = _create_progress_updater()

    theme = st.session_state.get("video_theme", "")
    custom_prompt = st.session_state.get("custom_prompt", "")
    frame_interval = int(st.session_state.get("frame_interval_input", config.frames.get("frame_interval_input", 3)))
    skip_seconds = int(st.session_state.get("movie_commentary_skip_seconds", 0))
    threshold = int(st.session_state.get("movie_commentary_threshold", 30))
    vision_batch_size = int(st.session_state.get("vision_batch_size", config.frames.get("vision_batch_size", 10)))
    vision_provider = (st.session_state.get("vision_llm_providers") or "gemini").lower()

    service = MovieCommentaryService()

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        segments, script_path = loop.run_until_complete(
            service.generate_commentary_script(
                params.video_origin_path,
                auto_generate=True,
                video_theme=theme,
                custom_prompt=custom_prompt,
                frame_interval=frame_interval,
                skip_seconds=skip_seconds,
                threshold=threshold,
                vision_batch_size=vision_batch_size,
                vision_provider=vision_provider,
                progress_callback=update_progress,
            )
        )
    except Exception as exc:  # pragma: no cover - UI feedback path
        logger.exception("Failed to generate movie commentary script")
        st.error(f"{tr('Failed to generate movie commentary script')}: {exc}")
        return
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:  # pragma: no cover - defensive cleanup
            pass
        finally:
            asyncio.set_event_loop(None)
            loop.close()
            finalize_progress()

    st.session_state["video_clip_json"] = segments
    st.session_state["video_clip_json_path"] = script_path
    params.video_clip_json = segments
    params.video_clip_json_path = script_path
    config.app["video_clip_json_path"] = script_path

    st.success(tr("Movie Commentary Script Generated Successfully"))
    st.info(f"{tr('Movie Commentary Script Saved To')}: {script_path}")

    # Give users a moment to read the success message before the next interaction.
    time.sleep(1)
