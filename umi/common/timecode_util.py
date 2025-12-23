from typing import Union
from fractions import Fraction
import datetime
import av


def timecode_to_seconds(
        timecode: str, frame_rate: Union[int, float, Fraction]
        ) -> Union[float, Fraction]:
    """
    Convert non-skip frame timecode into seconds since midnight
    """
    # calculate whole frame rate
    # 29.97 -> 30, 59.94 -> 60
    int_frame_rate = round(frame_rate)

    # parse timecode string - replace any semicolons with colons for proper parsing
    timecode = timecode.replace(';', ':')
    h, m, s, f = [int(x) for x in timecode.split(':')]

    # calculate frames assuming whole frame rate (i.e. non-drop frame)
    frames = (3600 * h + 60 * m + s) * int_frame_rate + f

    # convert to seconds
    seconds = frames / frame_rate
    return seconds


def stream_get_start_datetime(stream: av.stream.Stream) -> datetime.datetime:
    """
    Combines creation time and timecode to get high-precision
    time for the first frame of a video.
    """
    # read metadata
    frame_rate = stream.average_rate
    tc = stream.metadata.get('timecode')
    creation_time_str = stream.metadata.get('creation_time')

    # parse creation_time (store as timezone-aware UTC)
    creation_dt = None
    if creation_time_str is not None:
        try:
            creation_dt = datetime.datetime.fromisoformat(
                creation_time_str.replace('Z', '+00:00')
            )
        except ValueError:
            creation_dt = None

    candidate_from_tc = None
    if tc is not None:
        seconds_since_midnight = float(timecode_to_seconds(timecode=tc, frame_rate=frame_rate))
        if creation_dt is not None:
            # Use the same date as creation_time and wrap by 24h so it stays
            # close to the actual creation timestamp (avoid 24h offsets).
            candidate_from_tc = creation_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            candidate_from_tc = candidate_from_tc + datetime.timedelta(seconds=seconds_since_midnight)
            # shift by whole days to be within 12h of creation_dt
            while (candidate_from_tc - creation_dt).total_seconds() > 12 * 3600:
                candidate_from_tc -= datetime.timedelta(days=1)
            while (candidate_from_tc - creation_dt).total_seconds() < -12 * 3600:
                candidate_from_tc += datetime.timedelta(days=1)
        else:
            # Fallback: timecode only, assume UTC day reference
            candidate_from_tc = datetime.datetime.fromtimestamp(0, tz=datetime.timezone.utc) + \
                datetime.timedelta(seconds=seconds_since_midnight)

    if creation_dt is None and candidate_from_tc is None:
        raise RuntimeError("Video stream missing creation_time/timecode metadata")

    if creation_dt is None:
        return candidate_from_tc
    if candidate_from_tc is None:
        return creation_dt

    # Prefer timecode-based timestamp only when it's already close to creation_time
    # (to keep sub-second/frame accuracy without jumping a full day).
    if abs((candidate_from_tc - creation_dt).total_seconds()) <= 2 * 3600:
        return candidate_from_tc
    return creation_dt


def mp4_get_start_datetime(mp4_path: str) -> datetime.datetime:
    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        return stream_get_start_datetime(stream=stream)
