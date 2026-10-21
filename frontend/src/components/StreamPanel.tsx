import React, { useState, useEffect, useRef, useCallback, memo } from 'react';
import {
  Maximize2,
  Minimize2,
  Square,
  AlertCircle,
  WifiOff,
  RefreshCw,
  Download,
  Users,
  CheckCircle2,
  Archive,
  Radio,
} from 'lucide-react';
import type {
  FeedResponse,
  FeedSourceMode,
  VideoFrameMetrics,
  WSEnvelope,
  PartialFeedMetrics,
  RecordingSummary,
  SourceStatus,
} from '../types/api';
import {
  connectWebSocket,
  setFeedSourceMode,
  startFeedRecording,
  stopFeedRecording,
  getFeedRecordingStatus,
  getStoredToken,
} from '../lib/api';
import { Button } from './ui/Button';
import { useToast } from '../context/ToastContext';

type ConnectionState = 'CONNECTING' | 'LIVE' | 'RECONNECTING' | 'STOPPED' | 'ERROR';
type ViewMode = 'bbox' | 'heatmap' | 'combined';

interface StreamPanelProps {
  feed: FeedResponse;
  token?: string;
  isFocused?: boolean;
  onFocus?: (feed: FeedResponse) => void;
  onStop?: (feedId: string) => void;
  onMetricsUpdate?: (metrics: VideoFrameMetrics) => void;
  className?: string;
}

export const StreamPanel: React.FC<StreamPanelProps> = memo(({
  feed,
  token,
  isFocused = false,
  onFocus,
  onStop,
  onMetricsUpdate,
  className = '',
}) => {
  const { notifyStorage } = useToast();
  const [connectionState, setConnectionState] = useState<ConnectionState>('CONNECTING');
  const [metrics, setMetrics] = useState<VideoFrameMetrics | null>(feed.latest_metrics || null);
  const [liveFrame, setLiveFrame] = useState<string | null>(null);
  const [liveHeatmap, setLiveHeatmap] = useState<string | null>(null);
  const [liveCombined, setLiveCombined] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('bbox');
  const [error, setError] = useState<string | null>(null);

  // Drone failover: the backend switches to the backup archive on its own
  // when the live link drops; sourceMode is the operator's manual override.
  const [sourceStatus, setSourceStatus] = useState<SourceStatus>(feed.source_status ?? 'LIVE');
  const [sourceMode, setSourceMode] = useState<FeedSourceMode>(feed.source_mode ?? 'auto');
  const [sourceModeLoading, setSourceModeLoading] = useState<boolean>(false);

  // Recording state
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordingDuration, setRecordingDuration] = useState<number>(0);
  const [recordLoading, setRecordLoading] = useState<boolean>(false);
  const [lastRecording, setLastRecording] = useState<RecordingSummary | null>(null);
  const recordingTimerRef = useRef<any>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const intentionalStopRef = useRef<boolean>(false);
  const reconnectAttemptsRef = useRef<number>(0);
  const reconnectTimerRef = useRef<any>(null);
  const connectedFeedIdRef = useRef<string | null>(null);

  // WS messages are buffered here and applied at most once per animation
  // frame: a burst of frames (or a backgrounded tab catching up) causes one
  // render with the newest images instead of one render + decode per message.
  const pendingMetricsRef = useRef<PartialFeedMetrics | null>(null);
  const pendingImagesRef = useRef<{ frame?: string; heatmap?: string; combined?: string }>({});
  const flushHandleRef = useRef<number | null>(null);
  const onMetricsUpdateRef = useRef(onMetricsUpdate);
  onMetricsUpdateRef.current = onMetricsUpdate;

  const flushPending = useCallback(() => {
    flushHandleRef.current = null;
    const partial = pendingMetricsRef.current;
    const images = pendingImagesRef.current;
    pendingMetricsRef.current = null;
    pendingImagesRef.current = {};
    if (partial) {
      setMetrics((prev) => {
        const next = { ...(prev ?? ({} as VideoFrameMetrics)), ...partial };
        if (onMetricsUpdateRef.current) onMetricsUpdateRef.current(next);
        return next;
      });
    }
    if (images.frame) setLiveFrame(images.frame);
    if (images.heatmap) setLiveHeatmap(images.heatmap);
    if (images.combined) setLiveCombined(images.combined);
  }, []);

  const scheduleFlush = useCallback(() => {
    if (flushHandleRef.current !== null) return;
    flushHandleRef.current = window.requestAnimationFrame(flushPending);
  }, [flushPending]);

  const startRecordingTimer = useCallback((fromSeconds: number) => {
    if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
    setRecordingDuration(fromSeconds);
    recordingTimerRef.current = setInterval(() => {
      setRecordingDuration((prev) => prev + 1);
    }, 1000);
  }, []);

  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const attachSocket = useCallback((feedId: string) => {
    if (connectedFeedIdRef.current === feedId && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return;
    }
    connectedFeedIdRef.current = feedId;
    intentionalStopRef.current = false;
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      wsRef.current = connectWebSocket(
        `/ws/feeds/${feedId}`,
        (envelope: WSEnvelope) => {
          if (envelope.type === 'feed.progress' && envelope.data) {
            const { frame_jpeg, heatmap_jpeg, combined_jpeg, ...partial } = envelope.data as PartialFeedMetrics;
            if (partial.source_status) setSourceStatus(partial.source_status);
            pendingMetricsRef.current = { ...(pendingMetricsRef.current ?? {}), ...partial };
            // Only the newest image per view is kept; superseded data URLs are
            // dropped immediately so they can be garbage-collected.
            if (frame_jpeg) pendingImagesRef.current.frame = frame_jpeg;
            if (heatmap_jpeg) pendingImagesRef.current.heatmap = heatmap_jpeg;
            if (combined_jpeg) pendingImagesRef.current.combined = combined_jpeg;
            scheduleFlush();
            reconnectAttemptsRef.current = 0;
            setConnectionState('LIVE');
          } else if (envelope.type === 'feed.status') {
            if (envelope.data?.status === 'stopped') {
              intentionalStopRef.current = true;
              connectedFeedIdRef.current = null;
              setConnectionState('STOPPED');
              setIsRecording(false);
              if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
            } else if (envelope.data?.event === 'recording_started') {
              setIsRecording(true);
            } else if (envelope.data?.event === 'recording_stopped') {
              setIsRecording(false);
              if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
              if (envelope.data?.summary) {
                const sum = envelope.data.summary;
                setLastRecording(sum);
                const isS3 = !!sum.s3_key;
                notifyStorage({
                  title: `Recording Saved (${displayName})`,
                  message: `Captured ${sum.frame_count} frames (${sum.duration_s}s) of live feed telemetry.`,
                  storageType: isS3 ? 's3' : 'local',
                  storagePath: isS3 ? `s3://.../${sum.s3_key}` : `data/live_recordings/${sum.session_uid}_detection.mp4`,
                  action: sum.download_url ? {
                    label: 'Download MP4',
                    onClick: handleDownload,
                  } : undefined,
                });
              }
            }
          } else if (envelope.type === 'feed.error') {
            setError(envelope.data?.message || 'Feed stream error');
            setConnectionState('ERROR');
          }
        },
        (connected: boolean) => {
          if (connected) {
            reconnectAttemptsRef.current = 0;
            setConnectionState((prev) => (prev === 'LIVE' ? prev : 'CONNECTING'));
            return;
          }
          if (intentionalStopRef.current) {
            connectedFeedIdRef.current = null;
            setConnectionState('STOPPED');
            return;
          }
          if (reconnectAttemptsRef.current >= 3) {
            connectedFeedIdRef.current = null;
            setConnectionState('ERROR');
            setError('Stream connection dropped');
            return;
          }
          setConnectionState('RECONNECTING');
          reconnectAttemptsRef.current += 1;
          reconnectTimerRef.current = setTimeout(() => {
            if (!intentionalStopRef.current) {
              attachSocket(feedId);
            }
          }, 2000);
        }
      );
    } catch (e: any) {
      console.error('WebSocket connection error:', e);
      setConnectionState('ERROR');
      setError('Failed to establish WebSocket stream');
    }
  }, []);

  useEffect(() => {
    attachSocket(feed.feed_id);

    // Check if feed is actively recording
    getFeedRecordingStatus(feed.feed_id, token)
      .then((status) => {
        if (status.recording) {
          setIsRecording(true);
          startRecordingTimer(Math.floor(status.duration_s));
        }
      })
      .catch(() => {});

    return () => {
      intentionalStopRef.current = true;
      connectedFeedIdRef.current = null;
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      if (flushHandleRef.current !== null) window.cancelAnimationFrame(flushHandleRef.current);
      flushHandleRef.current = null;
      pendingMetricsRef.current = null;
      pendingImagesRef.current = {};
    };
  }, [feed.feed_id]);

  const handleStartRecording = async () => {
    setRecordLoading(true);
    try {
      await startFeedRecording(feed.feed_id, token);
      setIsRecording(true);
      setLastRecording(null);
      startRecordingTimer(0);
    } catch (err: any) {
      setError(err.message || 'Failed to start recording');
    } finally {
      setRecordLoading(false);
    }
  };

  const handleStopRecording = async () => {
    setRecordLoading(true);
    try {
      const res = await stopFeedRecording(feed.feed_id, token);
      setIsRecording(false);
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
      if (res.summary) {
        setLastRecording(res.summary);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to stop recording');
    } finally {
      setRecordLoading(false);
    }
  };

  const handleToggleSourceMode = async () => {
    const next: FeedSourceMode = sourceMode === 'backup' ? 'auto' : 'backup';
    setSourceModeLoading(true);
    try {
      const res = await setFeedSourceMode(feed.feed_id, next, token);
      setSourceMode(res.source_mode ?? next);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to switch feed source');
    } finally {
      setSourceModeLoading(false);
    }
  };

  const handleDownload = () => {
    if (!lastRecording) return;
    const authToken = token || getStoredToken();
    let url = lastRecording.download_url || `/api/feeds/recordings/${lastRecording.session_uid}/video`;
    if (authToken && !url.includes('token=')) {
      const sep = url.includes('?') ? '&' : '?';
      url = `${url}${sep}token=${encodeURIComponent(authToken)}`;
    }
    const a = document.createElement('a');
    a.href = url;
    a.download = `recording_${lastRecording.session_uid}.mp4`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  // Select video frame based on mode
  const currentVideoSrc =
    viewMode === 'heatmap' && liveHeatmap
      ? liveHeatmap
      : viewMode === 'combined' && liveCombined
      ? liveCombined
      : liveFrame;

  const getRiskColor = (status?: string) => {
    switch (status?.toUpperCase()) {
      case 'CRITICAL':
        return 'text-red-400 border-red-800/80 bg-red-950/60';
      case 'HIGH':
      case 'ELEVATED':
        return 'text-amber-400 border-amber-800/80 bg-amber-950/60';
      case 'MODERATE':
      case 'LOW':
        return 'text-sky-400 border-sky-800/80 bg-sky-950/60';
      default:
        return 'text-emerald-400 border-emerald-800/80 bg-emerald-950/60';
    }
  };

  const countDisplay = metrics?.estimated_count != null
    ? Math.round(metrics.estimated_count)
    : metrics?.visible_count != null
    ? metrics.visible_count
    : '--';

  const riskStatus = metrics?.risk_status || 'SAFE';
  const fpsDisplay = metrics?.fps ? Math.round(metrics.fps) : '--';
  const latencyDisplay = metrics?.latency_ms ? Math.round(metrics.latency_ms) : '--';
  const displayName = feed.device_name || feed.description || feed.feed_id;

  return (
    <div
      className={`group relative flex flex-col h-full min-h-0 bg-zinc-950 rounded-lg border overflow-hidden transition-all duration-200 ${
        isFocused ? 'border-zinc-500 shadow-2xl ring-1 ring-zinc-500' : 'border-zinc-800 hover:border-zinc-700'
      } ${className}`}
    >
      {/* Top Header Bar */}
      <div className="h-9 px-3 bg-zinc-900/90 border-b border-zinc-800/80 flex items-center justify-between z-10 shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          {/* Status Indicator */}
          {connectionState === 'LIVE' && sourceStatus === 'BACKUP_ACTIVE' ? (
            <span className="flex items-center gap-1.5 text-[10px] font-mono font-medium text-amber-400">
              <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
              BACKUP
            </span>
          ) : connectionState === 'LIVE' && sourceStatus === 'SIGNAL_LOST' ? (
            <span className="flex items-center gap-1.5 text-[10px] font-mono font-medium text-red-400">
              <WifiOff className="w-2.5 h-2.5" />
              NO SIGNAL
            </span>
          ) : connectionState === 'LIVE' ? (
            <span className="flex items-center gap-1.5 text-[10px] font-mono font-medium text-emerald-400">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              LIVE
            </span>
          ) : connectionState === 'CONNECTING' || connectionState === 'RECONNECTING' ? (
            <span className="flex items-center gap-1.5 text-[10px] font-mono text-amber-400">
              <RefreshCw className="w-2.5 h-2.5 animate-spin" />
              {connectionState}
            </span>
          ) : (
            <span className="flex items-center gap-1.5 text-[10px] font-mono text-zinc-500">
              <span className="w-2 h-2 rounded-full bg-zinc-600" />
              OFFLINE
            </span>
          )}

          {/* Feed Title */}
          <span className="text-xs font-semibold text-zinc-200 truncate max-w-[150px] md:max-w-[200px]" title={displayName}>
            {displayName}
          </span>

          {feed.device_id && (
            <span className="hidden sm:inline-block font-mono text-[9px] text-zinc-400 bg-zinc-800 px-1.5 py-0.5 rounded border border-zinc-700/60">
              {feed.device_id}
            </span>
          )}
        </div>

        {/* Top Control Actions */}
        <div className="flex items-center gap-1 shrink-0">
          {/* Manual failover override (only for feeds with a backup source) */}
          {feed.failover_enabled && (
            <button
              onClick={handleToggleSourceMode}
              disabled={sourceModeLoading}
              className={`px-2 py-1 rounded text-[11px] font-mono flex items-center gap-1 transition-colors cursor-pointer border ${
                sourceMode === 'backup'
                  ? 'bg-amber-950/80 text-amber-300 border-amber-700/80'
                  : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 border-zinc-800'
              }`}
              title={
                sourceMode === 'backup'
                  ? 'Backup forced. Click to resume automatic failover to the live drone feed'
                  : 'Automatic failover. Click to force the backup archive'
              }
            >
              {sourceMode === 'backup' ? <Archive className="w-3 h-3" /> : <Radio className="w-3 h-3" />}
              <span className="hidden md:inline">{sourceMode === 'backup' ? 'Backup' : 'Auto'}</span>
            </button>
          )}

          {/* Record Button */}
          <button
            onClick={isRecording ? handleStopRecording : handleStartRecording}
            disabled={recordLoading || connectionState !== 'LIVE'}
            className={`px-2 py-1 rounded text-[11px] font-mono flex items-center gap-1 transition-colors cursor-pointer ${
              isRecording
                ? 'bg-red-950/80 text-red-300 border border-red-700/80 animate-pulse'
                : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 border border-zinc-800'
            }`}
            title={isRecording ? 'Stop Recording' : 'Start Clip Recording'}
          >
            {isRecording ? (
              <>
                <Square className="w-2.5 h-2.5 fill-current text-red-400" />
                <span>{formatDuration(recordingDuration)}</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
                <span className="hidden md:inline">Rec</span>
              </>
            )}
          </button>

          {/* Focus / Maximize Button */}
          {onFocus && (
            <button
              onClick={() => onFocus(feed)}
              className="p-1 rounded text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800 transition-colors cursor-pointer"
              title={isFocused ? 'Exit Focus View' : 'Focus Single Feed'}
            >
              {isFocused ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
            </button>
          )}

          {/* Stop / Close Stream Button */}
          {onStop && (
            <button
              onClick={() => onStop(feed.feed_id)}
              className="p-1 rounded text-zinc-500 hover:text-red-400 hover:bg-zinc-800/80 transition-colors cursor-pointer"
              title="Stop Camera Stream"
            >
              <Square className="w-3 h-3 fill-current" />
            </button>
          )}
        </div>
      </div>

      {/* Video Viewport Container */}
      <div className="relative flex-1 min-h-0 bg-zinc-950 flex items-center justify-center overflow-hidden select-none">
        {/* Floating Glassmorphic View Mode Switcher */}
        <div className="absolute top-2 left-2 z-20 flex items-center bg-zinc-950/75 backdrop-blur-md rounded-md p-0.5 border border-zinc-700/60 shadow-lg text-[10px] font-mono select-none">
          <button
            onClick={() => setViewMode('bbox')}
            className={`px-2 py-0.5 rounded transition-all cursor-pointer ${
              viewMode === 'bbox' ? 'bg-zinc-800 text-zinc-100 font-semibold shadow-sm' : 'text-zinc-400 hover:text-zinc-200'
            }`}
            title="Detection Boxes"
          >
            Box
          </button>
          <button
            onClick={() => setViewMode('heatmap')}
            className={`px-2 py-0.5 rounded transition-all cursor-pointer ${
              viewMode === 'heatmap' ? 'bg-zinc-800 text-zinc-100 font-semibold shadow-sm' : 'text-zinc-400 hover:text-zinc-200'
            }`}
            title="Density Heatmap"
          >
            Heat
          </button>
          <button
            onClick={() => setViewMode('combined')}
            className={`px-2 py-0.5 rounded transition-all cursor-pointer ${
              viewMode === 'combined' ? 'bg-zinc-800 text-zinc-100 font-semibold shadow-sm' : 'text-zinc-400 hover:text-zinc-200'
            }`}
            title="Combined Overlay"
          >
            Blend
          </button>
        </div>
        {/* Drone failover HUD badge */}
        {sourceStatus === 'BACKUP_ACTIVE' && (
          <div
            role="status"
            className="absolute bottom-2 left-1/2 -translate-x-1/2 z-20 max-w-[95%] flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-amber-950/90 border border-amber-500/80 text-amber-300 shadow-lg text-[10px] md:text-[11px] font-mono font-semibold tracking-wide backdrop-blur-sm animate-pulse"
          >
            <AlertCircle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
            <span className="truncate">[DRONE DISCONNECTED — PLAYING BACKUP ARCHIVE]</span>
          </div>
        )}
        {sourceStatus === 'SIGNAL_LOST' && (
          <div
            role="status"
            className="absolute bottom-2 left-1/2 -translate-x-1/2 z-20 max-w-[95%] flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-red-950/90 border border-red-600/80 text-red-300 shadow-lg text-[10px] md:text-[11px] font-mono font-semibold tracking-wide backdrop-blur-sm"
          >
            <WifiOff className="w-3.5 h-3.5 text-red-400 shrink-0" />
            <span className="truncate">[DRONE SIGNAL LOST — BACKUP UNAVAILABLE, RECONNECTING]</span>
          </div>
        )}
        {currentVideoSrc ? (
          <img
            src={currentVideoSrc}
            alt={displayName}
            className="w-full h-full object-contain max-h-full transition-opacity duration-150"
          />
        ) : connectionState === 'STOPPED' ? (
          <div className="flex flex-col items-center justify-center gap-2 p-6 text-center text-zinc-500 font-mono">
            <WifiOff className="w-8 h-8 text-zinc-600 mb-1" />
            <div className="text-xs font-semibold text-zinc-300">Camera Feed Inactive</div>
            <p className="text-[11px] text-zinc-600 max-w-xs">The video stream for this camera is currently stopped.</p>
            <Button
              variant="outline"
              size="sm"
              onClick={() => attachSocket(feed.feed_id)}
              className="mt-2 text-xs"
            >
              <RefreshCw className="w-3 h-3 mr-1" /> Reconnect
            </Button>
          </div>
        ) : connectionState === 'ERROR' ? (
          <div className="flex flex-col items-center justify-center gap-2 p-6 text-center text-zinc-500 font-mono">
            <AlertCircle className="w-8 h-8 text-red-500/80 mb-1" />
            <div className="text-xs font-semibold text-red-400">Stream Connection Failed</div>
            <p className="text-[11px] text-zinc-500 max-w-xs">{error || 'Unable to receive frames'}</p>
            <Button
              variant="outline"
              size="sm"
              onClick={() => attachSocket(feed.feed_id)}
              className="mt-2 text-xs"
            >
              <RefreshCw className="w-3 h-3 mr-1" /> Retry Stream
            </Button>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center gap-2 p-6 text-center text-zinc-500 font-mono">
            <RefreshCw className="w-7 h-7 text-zinc-600 animate-spin mb-1" />
            <div className="text-xs text-zinc-400">Awaiting stream signal...</div>
            <span className="text-[10px] text-zinc-600">{feed.feed_id}</span>
          </div>
        )}

        {/* Inline Recording Finished Toast/Banner */}
        {lastRecording && (
          <div className="absolute top-2 right-2 z-20 flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-emerald-950/95 border border-emerald-700/80 text-emerald-300 shadow-lg text-xs font-mono backdrop-blur-sm animate-in fade-in">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
            <span>Clip ({lastRecording.duration_s}s) saved</span>
            <Button
              variant="default"
              size="sm"
              onClick={handleDownload}
              className="h-6 px-2 text-[10px] bg-emerald-600 hover:bg-emerald-500 text-white font-semibold cursor-pointer"
            >
              <Download className="w-3 h-3 mr-1" /> Download
            </Button>
          </div>
        )}
      </div>

      {/* Bottom Telemetry HUD */}
      <div className="h-9 px-3 bg-zinc-900/90 border-t border-zinc-800/80 flex items-center justify-between font-mono text-xs z-10 shrink-0">
        <div className="flex items-center gap-3">
          {/* Real-Time People Count */}
          <div className="flex items-center gap-1.5 text-zinc-200">
            <Users className="w-3.5 h-3.5 text-zinc-400" />
            <span className="text-zinc-500 text-[10px]">Count:</span>
            <span className="font-bold text-zinc-100">{countDisplay}</span>
          </div>

          {/* Risk Level Badge */}
          <span
            className={`px-1.5 py-0.5 rounded text-[10px] font-semibold border uppercase ${getRiskColor(
              riskStatus
            )}`}
          >
            {riskStatus}
          </span>
        </div>

        {/* Telemetry (FPS & Latency) */}
        <div className="flex items-center gap-3 text-[10px] text-zinc-500">
          <span title="Inference Latency">{latencyDisplay}ms</span>
          <span title="Delivery Framerate" className="font-medium text-zinc-400">
            {fpsDisplay} fps
          </span>
        </div>
      </div>
    </div>
  );
});

StreamPanel.displayName = 'StreamPanel';
