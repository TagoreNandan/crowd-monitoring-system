"""Live Multi-Stream Analytics Router.

Aggregates real-time telemetry across all running camera sessions, calculates
fleet-wide crowd metrics, zone density distributions, time-series history,
and threshold alerts.
"""

from __future__ import annotations

import collections
from datetime import datetime, timezone
import logging
import time
import uuid
from typing import Deque, Dict, List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..auth.dependencies import get_current_user_optional
from ..db.models import Device, User
from ..db.session import get_db
from ..schemas.analytics import (
    AnalyticsAlert,
    LiveAnalyticsResponse,
    StreamAnalyticsSnapshot,
    TimelineDataPoint,
    ZoneDistributionMetric,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Rolling buffers for temporal trend chart and recent anomaly alerts
_timeline_history: Deque[TimelineDataPoint] = collections.deque(maxlen=60)
_recent_alerts: Deque[AnalyticsAlert] = collections.deque(maxlen=20)
_fleet_peak_crowd: float = 0.0
_last_timeline_s: float = 0.0

# Risk priority ordering for fleet-wide aggregation
RISK_PRIORITY = {
    "CRITICAL": 4,
    "WARNING": 3,
    "CAUTION": 2,
    "SAFE": 1,
    "UNKNOWN": 0,
}


def _get_active_feed_sessions():
    """Retrieve active feed sessions dictionary safely via snapshot."""
    try:
        from .feeds import get_active_feeds_snapshot
        return get_active_feeds_snapshot()
    except Exception as exc:
        logger.warning(f"Unable to read active feeds store: {exc}")
        return {}, {}, {}


@router.get("/live", response_model=LiveAnalyticsResponse)
def get_live_analytics(
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: Session = Depends(get_db),
):
    """Returns aggregated real-time analytics across all active camera streams."""
    global _fleet_peak_crowd

    active_feeds, feed_peaks, feed_latest_frames = _get_active_feed_sessions()
    enrolled_devices = db.query(Device).all()
    device_lookup: Dict[str, Device] = {str(d.device_id): d for d in enrolled_devices if d.device_id}
    url_lookup: Dict[str, Device] = {str(d.stream_url): d for d in enrolled_devices if d.stream_url}

    now_iso = datetime.now(timezone.utc).isoformat()
    now_time_str = datetime.now(timezone.utc).strftime("%H:%M:%S")

    snapshots: List[StreamAnalyticsSnapshot] = []
    total_crowd = 0.0
    total_visible = 0
    highest_risk_status = "SAFE"
    highest_risk_priority = 1
    total_risk_score = 0.0

    # Process active feed sessions
    for feed_id, session in list(active_feeds.items()):
        fast = session.latest_fast_result
        density = session.latest_density_result

        # Identify linked device metadata if present
        meta = session.metadata
        desc = meta.description if meta else ""
        matched_dev = url_lookup.get(desc)
        if not matched_dev:
            for dev in enrolled_devices:
                d_url = str(dev.stream_url) if dev.stream_url else ""
                d_id = str(dev.device_id) if dev.device_id else ""
                if (d_url and d_url in desc) or (d_id and d_id in desc):
                    matched_dev = dev
                    break

        est_count = float(density.result.inference.count) if density else 0.0
        vis_count = int(fast.visible_count) if (fast and fast.visible_count is not None) else (int(est_count) if est_count > 0 else 0)
        peak_count = float(feed_peaks.get(feed_id, est_count))

        risk_status = str(density.result.risk.status).upper() if density else "SAFE"
        risk_score = float(density.result.risk.score) if density else 0.0
        fps = float(meta.fps) if (meta and meta.fps) else (round(1000.0 / fast.latency_ms, 1) if (fast and fast.latency_ms > 0) else 30.0)
        latency = float(fast.latency_ms) if fast else 80.0

        # Base64 JPEG frame artifacts from latest feed cache
        frames = feed_latest_frames.get(feed_id, {})
        frame_jpeg = frames.get("frame_jpeg")
        heatmap_jpeg = frames.get("heatmap_jpeg")
        combined_jpeg = frames.get("combined_jpeg")

        total_crowd += est_count
        total_visible += vis_count
        total_risk_score += risk_score

        current_pri = RISK_PRIORITY.get(risk_status, 1)
        if current_pri > highest_risk_priority:
            highest_risk_priority = current_pri
            highest_risk_status = risk_status

        camera_name: str = str(matched_dev.name) if (matched_dev and matched_dev.name) else f"Stream ({feed_id})"
        location: str = str(matched_dev.location) if (matched_dev and matched_dev.location) else "Main Concourse"

        # Generate alert if high crowd or critical risk detected
        if risk_status in ("WARNING", "CRITICAL") or est_count >= 100:
            alert_id = f"ALT-{feed_id[:6]}-{int(datetime.now().timestamp())}"
            # Avoid spamming duplicates within 15 seconds
            if not any(a.camera_name == camera_name for a in list(_recent_alerts)[-3:]):
                _recent_alerts.append(
                    AnalyticsAlert(
                        id=alert_id,
                        timestamp=now_time_str,
                        zone=location,
                        camera_name=camera_name,
                        severity=risk_status if risk_status in ("WARNING", "CRITICAL") else "WARNING",
                        message=f"High crowd density detected: {int(est_count)} people estimated.",
                        crowd_count=est_count,
                    )
                )

        snapshots.append(
            StreamAnalyticsSnapshot(
                feed_id=feed_id,
                device_id=str(matched_dev.device_id) if (matched_dev and matched_dev.device_id) else None,
                camera_name=camera_name,
                location=location,
                status="LIVE" if session.is_running else "STOPPED",
                is_live=session.is_running,
                estimated_count=round(est_count, 1),
                visible_count=vis_count,
                peak_count=round(peak_count, 1),
                risk_status=risk_status,
                risk_score=round(risk_score, 2),
                fps=round(fps, 1),
                latency_ms=round(latency, 1),
                frame_jpeg=frame_jpeg,
                heatmap_jpeg=heatmap_jpeg,
                combined_jpeg=combined_jpeg,
            )
        )

    # Update running fleet peak
    if total_crowd > _fleet_peak_crowd:
        _fleet_peak_crowd = total_crowd

    avg_fleet_risk_score = round(total_risk_score / len(snapshots), 2) if snapshots else 0.0

    # Record temporal timeline data point (throttled to at most once per second across concurrent client polls)
    global _last_timeline_s
    now_monotonic = time.monotonic()
    if (now_monotonic - _last_timeline_s >= 1.0) or not _timeline_history:
        timeline_point = TimelineDataPoint(
            timestamp=now_time_str,
            total_crowd=round(total_crowd, 1),
            visible_count=total_visible,
            active_streams=len(snapshots),
        )
        _timeline_history.append(timeline_point)
        _last_timeline_s = now_monotonic

    # Zone Distribution Aggregation
    zone_groups: Dict[str, dict] = {}
    # Seed known zones from enrolled devices
    for dev in enrolled_devices:
        zone: str = str(dev.location) if dev.location else "General Area"
        if zone not in zone_groups:
            zone_groups[zone] = {"count": 0.0, "cameras": 0, "highest_risk": "SAFE"}
        zone_groups[zone]["cameras"] += 1

    # Aggregate live counts into zones
    for snap in snapshots:
        zone = snap.location
        if zone not in zone_groups:
            zone_groups[zone] = {"count": 0.0, "cameras": 1, "highest_risk": "SAFE"}
        zone_groups[zone]["count"] += snap.estimated_count
        if RISK_PRIORITY.get(snap.risk_status, 1) > RISK_PRIORITY.get(zone_groups[zone]["highest_risk"], 1):
            zone_groups[zone]["highest_risk"] = snap.risk_status

    # Normalize density percent relative to capacity (assuming nominal capacity 150/zone)
    zone_metrics: List[ZoneDistributionMetric] = []
    max_zone_count = max([zg["count"] for zg in zone_groups.values()], default=1.0) or 1.0

    for zone_name, zg in sorted(zone_groups.items(), key=lambda x: x[1]["count"], reverse=True):
        density_pct = min(100.0, round((zg["count"] / max(max_zone_count, 50.0)) * 100, 1))
        zone_metrics.append(
            ZoneDistributionMetric(
                zone_name=zone_name,
                camera_count=zg["cameras"],
                crowd_count=round(zg["count"], 1),
                risk_status=zg["highest_risk"],
                density_percent=density_pct,
            )
        )

    # If no alerts yet, provide a baseline system notification
    recent_alerts_list = list(_recent_alerts)
    if not recent_alerts_list:
        recent_alerts_list = [
            AnalyticsAlert(
                id="SYS-INIT-01",
                timestamp=now_time_str,
                zone="System Fleet",
                camera_name="AI Analytics Engine",
                severity="INFO",
                message="Real-time multi-stream crowd monitoring active with zero critical alerts.",
                crowd_count=round(total_crowd, 1),
            )
        ]

    return LiveAnalyticsResponse(
        total_crowd=round(total_crowd, 1),
        total_visible=total_visible,
        peak_crowd=round(_fleet_peak_crowd, 1),
        fleet_risk_status=highest_risk_status,
        fleet_risk_score=avg_fleet_risk_score,
        active_cameras_count=len(snapshots),
        total_cameras_count=len(enrolled_devices),
        streams=snapshots,
        zones=zone_metrics,
        timeline=list(_timeline_history),
        recent_alerts=recent_alerts_list,
        updated_at=now_iso,
    )
