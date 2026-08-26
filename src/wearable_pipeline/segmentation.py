"""Stage 2 (event-window segmentation): cuts time windows around labeled onset/resolution
events.

This dataset's tags.csv holds physical button-press timestamps -- present for some
subject/sessions (mostly "Midterm 1") and empty for most others, "Final" sessions especially.
Where a session has tags, a window is centered on each one ("tagged"). Where it doesn't
(most sessions), the exam's own start and end stand in as the onset/resolution boundary the
architecture doc describes cutting windows around: the beginning of a multi-hour exam is where
stress-response physiology should be rising, its end where it should be resolving.
"""

from src.wearable_pipeline.models import SessionSignals

WINDOW_WIDTH_SECONDS = 300.0  # 5 minutes


def _session_end(session: SessionSignals) -> float:
    return session.hr_start + len(session.hr) / session.hr_rate


def build_event_windows(
    session: SessionSignals, window_width: float = WINDOW_WIDTH_SECONDS
) -> list[tuple[float, float, str]]:
    session_start = session.hr_start
    session_end = _session_end(session)

    if session.tags:
        windows = []
        for tag in session.tags:
            start = max(session_start, tag - window_width / 2)
            end = min(session_end, tag + window_width / 2)
            if end - start >= window_width * 0.5:
                windows.append((start, end, "tagged"))
        if windows:
            return windows

    onset_end = min(session_end, session_start + window_width)
    resolution_start = max(session_start, session_end - window_width)

    windows = [(session_start, onset_end, "onset")]
    if resolution_start > onset_end:
        windows.append((resolution_start, session_end, "resolution"))
    return windows
