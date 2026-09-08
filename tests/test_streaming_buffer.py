import threading
import time

import pytest

from crowd_monitoring.streaming import LatestFrameBuffer


def test_maxsize_must_be_positive():
    with pytest.raises(ValueError):
        LatestFrameBuffer(maxsize=0)


def test_put_then_get_round_trips_single_item():
    buf = LatestFrameBuffer(maxsize=1)
    buf.put("a")
    assert buf.get(timeout=1.0) == "a"
    assert buf.stats.frames_put == 1
    assert buf.stats.frames_consumed == 1
    assert buf.stats.frames_dropped == 0


def test_buffer_never_exceeds_maxsize():
    buf = LatestFrameBuffer(maxsize=2)
    for i in range(10):
        buf.put(i)
        assert buf.qsize() <= 2


def test_put_drops_oldest_and_keeps_latest_when_full():
    buf = LatestFrameBuffer(maxsize=1)
    buf.put("stale")
    buf.put("fresh")  # buffer was full -> "stale" must be dropped

    assert buf.qsize() == 1
    assert buf.stats.frames_dropped == 1
    assert buf.get(timeout=1.0) == "fresh"


def test_latest_frame_wins_under_burst_of_puts():
    buf = LatestFrameBuffer(maxsize=1)
    for i in range(50):
        buf.put(i)
    # Only the very last put should survive -- everything else was dropped.
    assert buf.get(timeout=1.0) == 49
    assert buf.stats.frames_dropped == 49
    assert buf.stats.frames_put == 50


def test_get_returns_none_on_timeout_when_not_closed():
    buf = LatestFrameBuffer(maxsize=1)
    start = time.monotonic()
    result = buf.get(timeout=0.2)
    elapsed = time.monotonic() - start
    assert result is None
    assert not buf.closed
    assert elapsed >= 0.15  # actually waited, didn't return immediately


def test_close_unblocks_a_pending_get():
    buf = LatestFrameBuffer(maxsize=1)
    result_holder = {}

    def consumer():
        result_holder["value"] = buf.get(timeout=None)

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.1)  # let the consumer start blocking on an empty buffer
    buf.close()
    t.join(timeout=2.0)

    assert not t.is_alive()
    assert result_holder["value"] is None


def test_get_drains_remaining_items_after_close_before_returning_none():
    buf = LatestFrameBuffer(maxsize=1)
    buf.put("last item")
    buf.close()

    assert buf.get(timeout=1.0) == "last item"
    assert buf.get(timeout=1.0) is None


def test_put_after_close_is_a_noop():
    buf = LatestFrameBuffer(maxsize=1)
    buf.close()
    buf.put("ignored")
    assert buf.get(timeout=0.2) is None


def test_producer_consumer_shutdown_is_clean():
    """A producer pushing frames faster than the consumer drains them must
    never grow the queue past maxsize, and both sides must be able to stop
    cleanly once the producer is done."""
    buf = LatestFrameBuffer(maxsize=1)
    consumed = []
    max_seen_qsize = 0

    def producer():
        nonlocal max_seen_qsize
        for i in range(200):
            buf.put(i)
            max_seen_qsize = max(max_seen_qsize, buf.qsize())
        buf.close()

    def consumer():
        while True:
            item = buf.get(timeout=0.5)
            if item is None and buf.closed:
                break
            if item is not None:
                consumed.append(item)
                time.sleep(0.001)  # simulate slower-than-producer inference

    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join(timeout=5.0)
    consumer_thread.join(timeout=5.0)

    assert not producer_thread.is_alive()
    assert not consumer_thread.is_alive()
    assert max_seen_qsize <= 1
    assert buf.stats.frames_dropped > 0  # consumer really was slower than the producer
    assert consumed == sorted(consumed)  # frames consumed in increasing (never reordered) order
    assert consumed[-1] == 199  # the very last produced frame was not lost
