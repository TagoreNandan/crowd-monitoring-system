import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Radio,
  Play,
  Square,
  RefreshCw,
  AlertCircle,
  Users,
  Eye,
  Camera,
  Circle,
  Download,
  CheckCircle2,
  X,
  LayoutGrid,
  Maximize2,
  Plus,
  SlidersHorizontal,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import {
  createFeed,
  stopFeed,
  connectWebSocket,
  uploadFeedVideo,
  getDevices,
  listFeeds,
  startFeedRecording,
  stopFeedRecording,
  getFeedRecordingStatus,
  getStoredToken,
} from '../lib/api';
import type { Device, FeedResponse, PartialFeedMetrics, VideoFrameMetrics, WSEnvelope, RecordingSummary } from '../types/api';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { StreamPanel } from '../components/StreamPanel';

type ConnectionState = 'CONNECTING' | 'LIVE' | 'RECONNECTING' | 'STOPPED' | 'ERROR';
type ViewMode = 'bbox' | 'heatmap' | 'combined';

const MAX_RECONNECT_ATTEMPTS = 3;

interface FeedViewProps {
  initialDevice?: Device | null;
}

export const FeedView: React.FC<FeedViewProps> = ({ initialDevice }) => {
  const [enrolledDevices, setEnrolledDevices] = useState<Device[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [sourceType, setSourceType] = useState<string>('sample');
  const [sourceUri, setSourceUri] = useState<string>('examples/sample_crowd.mp4');
  const [feedUploadFile, setFeedUploadFile] = useState<File | null>(null);

  // Multi-feed management
  const [activeFeeds, setActiveFeeds] = useState<FeedResponse[]>([]);
  const [focusedFeedId, setFocusedFeedId] = useState<string | null>(null);
  const [viewModePreference, setViewModePreference] = useState<'auto' | 'grid' | 'single'>('auto');
  const [isIngestionPanelOpen, setIsIngestionPanelOpen] = useState<boolean>(false);
  const [deviceActionLoading, setDeviceActionLoading] = useState<string | null>(null);
  const [batchActionLoading, setBatchActionLoading] = useState<boolean>(false);

  // Single-feed detailed view state (for the focused feed or primary feed)
  const [activeFeed, setActiveFeed] = useState<FeedResponse | null>(null);
  const [starting, setStarting] = useState<boolean>(false);
  const [stopping, setStopping] = useState<boolean>(false);
  const [connectionState, setConnectionState] = useState<ConnectionState>('STOPPED');
  const [metrics, setMetrics] = useState<VideoFrameMetrics | null>(null);
  const [liveFrame, setLiveFrame] = useState<string | null>(null);
  const [liveHeatmap, setLiveHeatmap] = useState<string | null>(null);
  const [liveCombined, setLiveCombined] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<ViewMode>('bbox');
  const [error, setError] = useState<string | null>(null);

  // Manual on-demand recording state for single focus
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [recordingDuration, setRecordingDuration] = useState<number>(0);
  const [recordLoading, setRecordLoading] = useState<boolean>(false);
  const [lastRecording, setLastRecording] = useState<RecordingSummary | null>(null);
  const recordingTimerRef = useRef<any>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const intentionalStopRef = useRef<boolean>(false);
  const reconnectAttemptsRef = useRef<number>(0);
  const feedIdRef = useRef<string | null>(null);
  const reconnectTimerRef = useRef<any>(null);

  const startRecordingTimer = useCallback((fromSeconds: number) => {
    if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
    setRecordingDuration(fromSeconds);
    recordingTimerRef.current = setInterval(() => {
      setRecordingDuration((prev) => prev + 1);
    }, 1000);
  }, []);

  const applyDeviceSource = (dev: Device) => {
    if (dev.stream_type.toUpperCase() === 'WEBCAM') {
      setSourceType('camera');
      setSourceUri(dev.stream_url);
    } else if (dev.stream_type.toUpperCase() === 'VIDEO_FILE') {
      setSourceType('sample');
      setSourceUri(dev.stream_url);
    } else {
      setSourceType('rtsp');
      setSourceUri(dev.stream_url);
    }
  };

  const connectedFeedIdRef = useRef<string | null>(null);

  // Poll list of active feeds from backend with reference stability
  const refreshActiveFeeds = useCallback(async () => {
    try {
      const feeds = await listFeeds();
      const running = feeds.filter((f) => f.status === 'running' || f.status === 'starting');
      setActiveFeeds((prev) => {
        if (
          prev.length === running.length &&
          prev.every(
            (p, i) =>
              p.feed_id === running[i].feed_id &&
              p.status === running[i].status &&
              p.is_recording === running[i].is_recording
          )
        ) {
          return prev;
        }
        return running;
      });

      // If activeFeed stopped running, update single view
      if (feedIdRef.current && !running.some((f) => f.feed_id === feedIdRef.current)) {
        setActiveFeed(null);
        feedIdRef.current = null;
        connectedFeedIdRef.current = null;
        setConnectionState('STOPPED');
        if (wsRef.current) wsRef.current.close();
      }
    } catch (err) {
      console.error('Failed to query active feeds:', err);
    }
  }, []);

  // Initial devices and running feeds load
  useEffect(() => {
    getDevices()
      .then((devs) => {
        setEnrolledDevices(devs);
        if (initialDevice) {
          setSelectedDeviceId(initialDevice.device_id);
          applyDeviceSource(initialDevice);
        }
      })
      .catch((err) => console.error('Failed to load enrolled camera fleet:', err));

    refreshActiveFeeds();
    const interval = setInterval(refreshActiveFeeds, 3000);
    return () => clearInterval(interval);
  }, [initialDevice, refreshActiveFeeds]);

  // Connect WebSocket for single detailed view (guards against redundant reconnects)
  const attachSocket = useCallback((feedId: string) => {
    if (connectedFeedIdRef.current === feedId && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      return;
    }
    connectedFeedIdRef.current = feedId;
    intentionalStopRef.current = false;
    if (wsRef.current) wsRef.current.close();

    wsRef.current = connectWebSocket(
      `/ws/feeds/${feedId}`,
      (envelope: WSEnvelope) => {
        if (envelope.type === 'feed.progress' && envelope.data) {
          const partial = envelope.data as PartialFeedMetrics;
          setMetrics((prev) => ({ ...(prev ?? ({} as VideoFrameMetrics)), ...partial }));
          if (envelope.data.frame_jpeg) setLiveFrame(envelope.data.frame_jpeg);
          if (envelope.data.heatmap_jpeg) setLiveHeatmap(envelope.data.heatmap_jpeg);
          if (envelope.data.combined_jpeg) setLiveCombined(envelope.data.combined_jpeg);
          reconnectAttemptsRef.current = 0;
          setConnectionState('LIVE');
        } else if (envelope.type === 'feed.status') {
          if (envelope.data?.status === 'stopped') {
            intentionalStopRef.current = true;
            connectedFeedIdRef.current = null;
            setConnectionState('STOPPED');
            setMetrics(null);
            setLiveFrame(null);
            setLiveHeatmap(null);
            setLiveCombined(null);
            setIsRecording(false);
            if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
          } else if (envelope.data?.event === 'recording_started') {
            setIsRecording(true);
          } else if (envelope.data?.event === 'recording_stopped') {
            setIsRecording(false);
            if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
            if (envelope.data?.summary) {
              setLastRecording(envelope.data.summary);
            }
          }
        } else if (envelope.type === 'feed.error') {
          setError(envelope.data?.message || 'Feed processing error');
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
        if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
          connectedFeedIdRef.current = null;
          setConnectionState('ERROR');
          setError('Lost connection to the real-time feed and could not reconnect.');
          return;
        }
        setConnectionState('RECONNECTING');
        reconnectAttemptsRef.current += 1;
        reconnectTimerRef.current = setTimeout(() => {
          if (!intentionalStopRef.current && feedIdRef.current) {
            attachSocket(feedIdRef.current);
          }
        }, 2000);
      }
    );
  }, []);

  // When focused feed changes, update active single feed (only connects in Single view mode!)
  useEffect(() => {
    // In grid mode, the individual StreamPanels handle their own WebSockets; close single socket
    if (isShowingGrid) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      connectedFeedIdRef.current = null;
      return;
    }

    const targetFeedId = focusedFeedId || (activeFeeds.length > 0 ? activeFeeds[0].feed_id : null);
    if (!targetFeedId) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      connectedFeedIdRef.current = null;
      setActiveFeed(null);
      setConnectionState('STOPPED');
      return;
    }

    const feed = activeFeeds.find((f) => f.feed_id === targetFeedId);
    if (feed) {
      setActiveFeed(feed);
      feedIdRef.current = feed.feed_id;
      if (connectedFeedIdRef.current !== feed.feed_id) {
        setConnectionState('CONNECTING');
        attachSocket(feed.feed_id);

        getFeedRecordingStatus(feed.feed_id)
          .then((status) => {
            if (status.recording) {
              setIsRecording(true);
              startRecordingTimer(Math.floor(status.duration_s));
            }
          })
          .catch(() => {});
      }
    }
  }, [isShowingGrid, focusedFeedId, activeFeeds, attachSocket, startRecordingTimer]);

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (recordingTimerRef.current) clearInterval(recordingTimerRef.current);
    };
  }, []);

  // Toggle stream for a fleet device (Start or Stop)
  const handleToggleDeviceStream = async (dev: Device) => {
    const running = activeFeeds.find((f) => f.device_id === dev.device_id);
    setDeviceActionLoading(dev.device_id);
    try {
      if (running) {
        await stopFeed(running.feed_id);
        if (focusedFeedId === running.feed_id) {
          setFocusedFeedId(null);
        }
      } else {
        let st = 'rtsp';
        if (dev.stream_type.toUpperCase() === 'WEBCAM') st = 'camera';
        else if (dev.stream_type.toUpperCase() === 'VIDEO_FILE') st = 'sample';
        await createFeed(st, dev.stream_url, true, undefined, dev.device_id, dev.name);
      }
      await refreshActiveFeeds();
    } catch (err: any) {
      console.error('Error toggling device stream:', err);
      setError(err.message || 'Failed to toggle stream');
    } finally {
      setDeviceActionLoading(null);
    }
  };

  // Launch all online cameras at once
  const handleStartAllFleet = async () => {
    setBatchActionLoading(true);
    try {
      const onlineDevices = enrolledDevices.filter((d) => d.status.toUpperCase() === 'ONLINE');
      for (const dev of onlineDevices.slice(0, 6)) {
        const isRunning = activeFeeds.some((f) => f.device_id === dev.device_id);
        if (!isRunning) {
          let st = 'rtsp';
          if (dev.stream_type.toUpperCase() === 'WEBCAM') st = 'camera';
          else if (dev.stream_type.toUpperCase() === 'VIDEO_FILE') st = 'sample';
          await createFeed(st, dev.stream_url, true, undefined, dev.device_id, dev.name).catch(() => {});
        }
      }
      await refreshActiveFeeds();
    } finally {
      setBatchActionLoading(false);
    }
  };

  // Stop all running feeds
  const handleStopAllFeeds = async () => {
    setBatchActionLoading(true);
    try {
      await Promise.allSettled(activeFeeds.map((f) => stopFeed(f.feed_id)));
      setActiveFeeds([]);
      setActiveFeed(null);
      setFocusedFeedId(null);
      feedIdRef.current = null;
      setConnectionState('STOPPED');
    } finally {
      setBatchActionLoading(false);
    }
  };

  // Start custom feed (from upload or input)
  const handleStartCustomFeed = async () => {
    setStarting(true);
    setError(null);
    try {
      let feedRes: FeedResponse;
      if (sourceType === 'upload') {
        if (!feedUploadFile) {
          throw new Error('Please select a video file to upload.');
        }
        feedRes = await uploadFeedVideo(feedUploadFile, true);
      } else {
        feedRes = await createFeed(sourceType, sourceUri, true);
      }
      await refreshActiveFeeds();
      setFocusedFeedId(feedRes.feed_id);
      setIsIngestionPanelOpen(false);
    } catch (err: any) {
      setError(err.message || 'Failed to initialize real-time feed');
    } finally {
      setStarting(false);
    }
  };

  // Stop single feed from tile
  const handleStopSingleFeed = async (feedId: string) => {
    try {
      await stopFeed(feedId);
      if (focusedFeedId === feedId) {
        setFocusedFeedId(null);
      }
      await refreshActiveFeeds();
    } catch (err: any) {
      console.error('Failed to stop feed:', err);
    }
  };

  // Recording timer formatting
  const formatDuration = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleStartRecording = async () => {
    if (!activeFeed) return;
    setRecordLoading(true);
    setError(null);
    try {
      await startFeedRecording(activeFeed.feed_id);
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
    if (!activeFeed) return;
    setRecordLoading(true);
    try {
      const res = await stopFeedRecording(activeFeed.feed_id);
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

  const getRiskVariant = (st?: string) => {
    switch (st?.toUpperCase()) {
      case 'CRITICAL': return 'critical';
      case 'WARNING': return 'warning';
      case 'CAUTION': return 'caution';
      default: return 'safe';
    }
  };

  // Grid column CSS based on active stream count
  const gridLayoutClass = useMemo(() => {
    const count = activeFeeds.length;
    switch (count) {
      case 1:
        return 'grid-cols-1';
      case 2:
        return 'grid-cols-1 md:grid-cols-2';
      case 3:
        return 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3';
      case 4:
        return 'grid-cols-1 md:grid-cols-2';
      case 5:
      case 6:
      default:
        return 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3';
    }
  }, [activeFeeds.length]);

  // Aggregate stats across all active feeds
  const aggregateMetrics = useMemo(() => {
    let totalCount = 0;
    let maxRisk = 'SAFE';
    activeFeeds.forEach((f) => {
      const c = f.latest_metrics?.estimated_count ?? f.latest_metrics?.visible_count ?? 0;
      totalCount += c;
      const r = f.latest_metrics?.risk_status;
      if (r === 'CRITICAL') maxRisk = 'CRITICAL';
      else if (r === 'ELEVATED' && maxRisk !== 'CRITICAL') maxRisk = 'ELEVATED';
      else if (r === 'LOW' && maxRisk === 'SAFE') maxRisk = 'LOW';
    });
    return { totalCount: Math.round(totalCount), maxRisk };
  }, [activeFeeds]);

  // Determine whether to display the Multi-Stream Grid or Single Focus View
  const isShowingGrid = useMemo(() => {
    if (viewModePreference === 'grid') return true;
    if (viewModePreference === 'single') return false;
    // 'auto' mode: if focusedFeedId is set, show single. If > 1 stream, show grid.
    if (focusedFeedId) return false;
    return activeFeeds.length > 1;
  }, [viewModePreference, focusedFeedId, activeFeeds.length]);

  const getDisplayedFrame = () => {
    if (viewMode === 'heatmap' && liveHeatmap) return liveHeatmap;
    if (viewMode === 'combined' && liveCombined) return liveCombined;
    return liveFrame;
  };

  const currentDisplayedFrame = getDisplayedFrame();

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto space-y-6">
      {/* Top Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-zinc-100 flex items-center gap-2">
              Real-Time Feed
            </h1>

            {/* Stream Count Badge */}
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono border ${
                activeFeeds.length > 0
                  ? 'text-emerald-400 bg-emerald-950/60 border-emerald-800/60'
                  : 'text-zinc-500 bg-zinc-900 border-zinc-800'
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  activeFeeds.length > 0 ? 'bg-emerald-400 animate-pulse' : 'bg-zinc-600'
                }`}
              />
              {activeFeeds.length} {activeFeeds.length === 1 ? 'Stream Active' : 'Streams Active'}
            </span>
          </div>
          <p className="text-xs text-zinc-400 mt-1">
            Simultaneous multi-camera CCTV monitoring &amp; real-time crowd analytics
          </p>
        </div>

        {/* Global Action Controls */}
        <div className="flex items-center gap-2">
          {/* Mode Switcher (Grid vs Single) */}
          {activeFeeds.length > 0 && (
            <div className="flex items-center bg-zinc-900 p-1 rounded-md border border-zinc-800 text-xs font-mono">
              <button
                onClick={() => {
                  setViewModePreference('grid');
                  setFocusedFeedId(null);
                }}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded transition-colors cursor-pointer ${
                  isShowingGrid
                    ? 'bg-zinc-800 text-zinc-100 font-semibold'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
                title="Surveillance Wall Grid View"
              >
                <LayoutGrid className="w-3.5 h-3.5" />
                <span>Grid ({activeFeeds.length})</span>
              </button>

              <button
                onClick={() => {
                  setViewModePreference('single');
                  if (!focusedFeedId && activeFeeds.length > 0) {
                    setFocusedFeedId(activeFeeds[0].feed_id);
                  }
                }}
                className={`flex items-center gap-1.5 px-2.5 py-1 rounded transition-colors cursor-pointer ${
                  !isShowingGrid
                    ? 'bg-zinc-800 text-zinc-100 font-semibold'
                    : 'text-zinc-400 hover:text-zinc-200'
                }`}
                title="Single Camera Detailed Focus"
              >
                <Maximize2 className="w-3.5 h-3.5" />
                <span>Focus</span>
              </button>
            </div>
          )}

          {/* Quick Ingest Toggle Button */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsIngestionPanelOpen((prev) => !prev)}
            className="text-xs border-zinc-800 text-zinc-300 hover:bg-zinc-900"
          >
            <SlidersHorizontal className="w-3.5 h-3.5 mr-1" />
            <span>Custom Ingest</span>
            {isIngestionPanelOpen ? <ChevronUp className="w-3 h-3 ml-1" /> : <ChevronDown className="w-3 h-3 ml-1" />}
          </Button>

          {/* Stop All Feeds */}
          {activeFeeds.length > 0 && (
            <Button
              variant="destructive"
              size="sm"
              onClick={handleStopAllFeeds}
              disabled={batchActionLoading}
              className="text-xs"
            >
              <Square className="w-3 h-3 mr-1 fill-current" />
              <span>Stop All</span>
            </Button>
          )}
        </div>
      </div>

      {/* Fleet Device Selector Strip (Synced directly with Device Management) */}
      <div className="p-3.5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-2.5">
        <div className="flex items-center justify-between">
          <span className="font-mono text-xs font-semibold text-zinc-300 flex items-center gap-1.5">
            <Camera className="w-3.5 h-3.5 text-zinc-400" />
            Fleet Camera Stream Toggles
          </span>
          <span className="text-[11px] text-zinc-500 font-mono">
            Check or uncheck cameras to stream concurrently
          </span>
        </div>

        {enrolledDevices.length === 0 ? (
          <p className="text-xs text-zinc-500 font-mono py-1">
            No enrolled devices in fleet. Onboard RTSP or Webcam devices in the Device Fleet tab to stream directly.
          </p>
        ) : (
          <div className="flex flex-wrap gap-2 pt-1">
            {enrolledDevices.map((dev) => {
              const active = activeFeeds.find((f) => f.device_id === dev.device_id);
              const isRunning = !!active;
              const isLoading = deviceActionLoading === dev.device_id;

              return (
                <button
                  key={dev.device_id}
                  onClick={() => handleToggleDeviceStream(dev)}
                  disabled={isLoading}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-mono transition-all border cursor-pointer ${
                    isRunning
                      ? 'bg-emerald-950/70 border-emerald-700/80 text-emerald-300 shadow-sm'
                      : 'bg-zinc-950/80 border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
                  }`}
                  title={isRunning ? `Click to stop stream for ${dev.name}` : `Click to stream ${dev.name}`}
                >
                  {isLoading ? (
                    <RefreshCw className="w-3 h-3 animate-spin text-amber-400" />
                  ) : (
                    <span
                      className={`w-2 h-2 rounded-full ${
                        isRunning ? 'bg-emerald-400 animate-pulse' : 'bg-zinc-600'
                      }`}
                    />
                  )}
                  <span className="font-medium text-zinc-200">{dev.name}</span>
                  <span className="text-[10px] text-zinc-500">({dev.stream_type})</span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Collapsible Custom Ingestion Settings Panel */}
      {isIngestionPanelOpen && (
        <div className="p-4 rounded-lg bg-zinc-900/60 border border-zinc-800 space-y-3 text-xs animate-in fade-in">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-2">
            <span className="font-mono font-medium text-zinc-300 uppercase tracking-wider text-[11px]">
              Ingest Custom Source (Sample Video / Webcam / Custom URL)
            </span>
            <button
              onClick={() => setIsIngestionPanelOpen(false)}
              className="text-zinc-500 hover:text-zinc-300"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
            <div>
              <label className="text-zinc-400 block mb-1">Source Type</label>
              <select
                value={sourceType}
                onChange={(e) => {
                  setSourceType(e.target.value);
                  setFeedUploadFile(null);
                }}
                className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-200 focus:outline-none focus:border-zinc-700"
              >
                <option value="sample">Sample Video File</option>
                <option value="upload">Upload Video File</option>
                <option value="camera">Local USB Camera (index 0, 1...)</option>
                <option value="rtsp">RTSP / RTMP / YouTube Stream</option>
              </select>
            </div>

            <div className="sm:col-span-2">
              {sourceType === 'upload' ? (
                <div>
                  <label className="text-zinc-400 block mb-1">Upload Video File</label>
                  <input
                    type="file"
                    accept="video/*"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) {
                        setFeedUploadFile(e.target.files[0]);
                      }
                    }}
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2 py-1 text-zinc-200 font-mono text-xs focus:outline-none focus:border-zinc-700 file:mr-3 file:py-1 file:px-2.5 file:rounded file:border-0 file:text-xs file:font-semibold file:bg-zinc-800 file:text-zinc-200 hover:file:bg-zinc-700"
                  />
                </div>
              ) : (
                <div>
                  <label className="text-zinc-400 block mb-1">
                    {sourceType === 'camera'
                      ? 'Camera Index (e.g. 0 or 1)'
                      : sourceType === 'sample'
                      ? 'Local Video File Path'
                      : 'Stream URL (RTSP, RTMP, or YouTube URL)'}
                  </label>
                  <input
                    type="text"
                    value={sourceUri}
                    onChange={(e) => setSourceUri(e.target.value)}
                    placeholder={
                      sourceType === 'camera'
                        ? '0'
                        : sourceType === 'sample'
                        ? 'examples/sample_crowd.mp4'
                        : 'rtsp://... or https://www.youtube.com/watch?v=...'
                    }
                    className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-200 font-mono text-xs focus:outline-none focus:border-zinc-700"
                  />
                </div>
              )}
            </div>
          </div>

          <div className="flex justify-end pt-2">
            <Button onClick={handleStartCustomFeed} isLoading={starting} size="sm">
              <Play className="w-3 h-3 mr-1 fill-current" />
              Launch Stream
            </Button>
          </div>
        </div>
      )}

      {error && (
        <div className="p-3.5 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-start gap-2">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">{error}</div>
          <button onClick={() => setError(null)} className="text-rose-400 hover:text-rose-200">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* =========================================================================
          VIEW MODE 1: MULTI-STREAM SURVEILLANCE WALL (GRID VIEW)
          ========================================================================= */}
      {isShowingGrid && activeFeeds.length > 0 ? (
        <div className="space-y-4">
          {/* Top Aggregate Fleet KPI Strip */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-xs">
            <div className="p-3 rounded-lg bg-zinc-900/50 border border-zinc-800 flex items-center justify-between">
              <span className="text-zinc-400">Total Active Feeds:</span>
              <span className="text-base font-bold text-emerald-400">{activeFeeds.length} / 6</span>
            </div>

            <div className="p-3 rounded-lg bg-zinc-900/50 border border-zinc-800 flex items-center justify-between">
              <span className="text-zinc-400">Total Monitored People:</span>
              <span className="text-base font-bold text-zinc-100">{aggregateMetrics.totalCount}</span>
            </div>

            <div className="p-3 rounded-lg bg-zinc-900/50 border border-zinc-800 flex items-center justify-between">
              <span className="text-zinc-400">Highest Risk Zone:</span>
              <Badge variant={getRiskVariant(aggregateMetrics.maxRisk)} className="font-bold text-[10px]">
                {aggregateMetrics.maxRisk}
              </Badge>
            </div>

            <div className="p-3 rounded-lg bg-zinc-900/50 border border-zinc-800 flex items-center justify-between">
              <span className="text-zinc-400">Layout Matrix:</span>
              <span className="text-zinc-300 font-semibold uppercase">{activeFeeds.length} Streams Auto</span>
            </div>
          </div>

          {/* Dynamic Adaptive CCTV Grid */}
          <div className={`grid gap-4 ${gridLayoutClass}`}>
            {activeFeeds.map((feed) => (
              <StreamPanel
                key={feed.feed_id}
                feed={feed}
                onFocus={(f) => {
                  setFocusedFeedId(f.feed_id);
                  setViewModePreference('single');
                }}
                onStop={handleStopSingleFeed}
              />
            ))}
          </div>
        </div>
      ) : activeFeeds.length === 0 ? (
        /* =========================================================================
            EMPTY STATE: ZERO CAMERAS RUNNING
            ========================================================================= */
        <div className="p-12 text-center rounded-xl bg-zinc-900/20 border border-zinc-800/80 space-y-4 my-4">
          <div className="w-12 h-12 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center mx-auto text-zinc-500 shadow-inner">
            <Radio className="w-6 h-6" />
          </div>

          <div className="space-y-1">
            <h3 className="text-base font-semibold text-zinc-200">No Active Camera Streams</h3>
            <p className="text-xs text-zinc-500 max-w-md mx-auto">
              Select cameras from the fleet panel above to start concurrent live monitoring, or launch a quick sample demonstration.
            </p>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2.5 pt-2">
            {enrolledDevices.length > 0 && (
              <Button
                variant="default"
                size="sm"
                onClick={handleStartAllFleet}
                isLoading={batchActionLoading}
                className="bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs"
              >
                <Play className="w-3 h-3 mr-1.5 fill-current" />
                Launch All Fleet Cameras
              </Button>
            )}

            <Button
              variant="outline"
              size="sm"
              onClick={handleStartCustomFeed}
              isLoading={starting}
              className="text-xs text-zinc-300 border-zinc-700"
            >
              <Play className="w-3 h-3 mr-1.5" />
              Launch Sample Crowd Feed
            </Button>
          </div>
        </div>
      ) : (
        /* =========================================================================
            VIEW MODE 2: SINGLE CAMERA DETAILED INSPECTION VIEW
            ========================================================================= */
        <div className="space-y-4">
          {/* Quick Return to Grid Banner if multiple streams are running */}
          {activeFeeds.length > 1 && (
            <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-900/80 border border-zinc-800 text-xs font-mono">
              <div className="flex items-center gap-2 text-zinc-300">
                <Maximize2 className="w-4 h-4 text-cyan-400" />
                <span>
                  Currently inspecting:{' '}
                  <strong className="text-white">
                    {activeFeed?.device_name || activeFeed?.description || activeFeed?.feed_id}
                  </strong>
                </span>
                <span className="text-zinc-500">({activeFeeds.length} streams running in background)</span>
              </div>

              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setFocusedFeedId(null);
                  setViewModePreference('grid');
                }}
                className="text-xs border-zinc-700 text-zinc-200 hover:bg-zinc-800 cursor-pointer"
              >
                <LayoutGrid className="w-3.5 h-3.5 mr-1" />
                Back to Multi-Stream Wall
              </Button>
            </div>
          )}

          {/* Viewport & Side Telemetry Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-8 space-y-4">
              <div className="relative rounded-lg border border-zinc-800 bg-black min-h-[440px] flex flex-col justify-between overflow-hidden">
                {connectionState === 'LIVE' && (
                  <div className="flex items-center justify-between px-3 py-2 bg-zinc-950/90 border-b border-zinc-800 z-10">
                    <div className="flex items-center gap-3 font-mono text-xs text-emerald-400">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                        <span>STREAM: {activeFeed?.device_name || activeFeed?.feed_id}</span>
                      </div>
                      {isRecording && (
                        <span className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-rose-950/90 border border-rose-800/80 text-rose-400 text-[11px] font-semibold animate-pulse">
                          <span className="w-1.5 h-1.5 rounded-full bg-rose-500 animate-ping" />
                          REC {formatDuration(recordingDuration)}
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-1 bg-zinc-900 p-1 rounded border border-zinc-800">
                      <button
                        onClick={() => setViewMode('bbox')}
                        className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${
                          viewMode === 'bbox' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'
                        }`}
                      >
                        Live Video
                      </button>
                      <button
                        onClick={() => setViewMode('heatmap')}
                        className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${
                          viewMode === 'heatmap' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'
                        }`}
                      >
                        Live Heatmap
                      </button>
                      <button
                        onClick={() => setViewMode('combined')}
                        className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${
                          viewMode === 'combined' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'
                        }`}
                      >
                        Combined Overlay
                      </button>
                    </div>
                  </div>
                )}

                <div className="w-full flex-1 flex items-center justify-center relative min-h-[380px]">
                  {currentDisplayedFrame && (connectionState === 'LIVE' || connectionState === 'RECONNECTING') ? (
                    <img src={currentDisplayedFrame} alt="live feed" className="w-full max-h-[500px] object-contain" />
                  ) : connectionState === 'CONNECTING' || connectionState === 'RECONNECTING' ? (
                    <div className="p-16 text-center space-y-3">
                      <RefreshCw className="w-6 h-6 text-zinc-400 animate-spin mx-auto" />
                      <p className="text-xs text-zinc-500">
                        {connectionState === 'CONNECTING' ? 'Connecting to feed…' : 'Reconnecting…'}
                      </p>
                    </div>
                  ) : (
                    <div className="p-16 text-center space-y-3">
                      <div className="w-10 h-10 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center mx-auto text-zinc-500">
                        <Radio className="w-5 h-5" />
                      </div>
                      <div>
                        <h3 className="text-sm font-semibold text-zinc-300">Feed Offline</h3>
                        <p className="text-xs text-zinc-500 max-w-xs mx-auto mt-1">
                          Toggle a camera in the fleet bar above to begin streaming.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Single View Recording Download Banner */}
              {lastRecording && (
                <div className="flex items-center justify-between p-3.5 rounded-lg bg-zinc-900/90 border border-emerald-900/60 text-xs">
                  <div className="flex items-center gap-2.5">
                    <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                    <div>
                      <span className="text-zinc-200 font-medium">Live recording saved successfully</span>
                      <span className="text-zinc-400 ml-2 font-mono">
                        {lastRecording.duration_s}s • {lastRecording.frame_count} frames{' '}
                        {lastRecording.s3_key ? '• S3 Cloud' : '• Local Storage'}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    {lastRecording.download_url && (
                      <a
                        href={(() => {
                          const url = lastRecording.download_url;
                          const token = getStoredToken();
                          if (!token || url.startsWith('http://') || url.startsWith('https://')) return url;
                          const sep = url.includes('?') ? '&' : '?';
                          return `${url}${sep}token=${encodeURIComponent(token)}`;
                        })()}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1.5 px-3 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-100 rounded text-xs font-medium transition-colors border border-zinc-700"
                      >
                        <Download className="w-3.5 h-3.5" />
                        Download Video
                      </a>
                    )}
                    <button
                      onClick={() => setLastRecording(null)}
                      className="text-zinc-500 hover:text-zinc-300 p-1"
                      title="Dismiss"
                    >
                      <X className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              )}
            </div>

            {/* Right Telemetry Column (Deep Analytics) */}
            <div className="lg:col-span-4 space-y-4">
              <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
                <span className="text-xs text-zinc-400 font-medium flex items-center gap-1.5">
                  <Users className="w-3.5 h-3.5 text-zinc-400" />
                  Estimated Crowd
                </span>
                <div className="text-4xl font-semibold font-mono text-zinc-100">
                  {connectionState === 'LIVE' && metrics?.estimated_count != null ? metrics.estimated_count : '—'}
                </div>
                <p className="text-xs text-zinc-500">Spatial density estimate</p>
              </div>

              <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
                <span className="text-xs text-zinc-400 font-medium flex items-center gap-1.5">
                  <Eye className="w-3.5 h-3.5 text-zinc-400" />
                  Visible People
                </span>
                <div className="text-3xl font-semibold font-mono text-zinc-200">
                  {connectionState === 'LIVE'
                    ? metrics?.visible_count !== undefined && metrics?.visible_count !== null
                      ? metrics.visible_count
                      : 'N/A'
                    : '—'}
                </div>
                <p className="text-xs text-zinc-500">Optical detection count</p>
              </div>

              <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
                <span className="text-xs text-zinc-400 font-medium">Peak Count</span>
                <div className="text-2xl font-semibold font-mono text-zinc-200">
                  {connectionState === 'LIVE' ? (metrics?.peak_count ?? '—') : '—'}
                </div>
              </div>

              <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-zinc-400 font-medium">Risk Assessment</span>
                  {connectionState === 'LIVE' && metrics?.risk_status && (
                    <Badge variant={getRiskVariant(metrics.risk_status)}>{metrics.risk_status}</Badge>
                  )}
                </div>
                <div className="text-xs font-mono text-zinc-400 space-y-1 pt-1">
                  <div className="flex justify-between">
                    <span>Risk Score:</span>
                    <span className="text-zinc-200 font-medium">
                      {connectionState === 'LIVE' && metrics?.risk_score != null
                        ? `${metrics.risk_score.toFixed(1)} / 100`
                        : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Density Score:</span>
                    <span className="text-zinc-200 font-medium">
                      {connectionState === 'LIVE' && metrics?.density != null ? metrics.density : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Clusters:</span>
                    <span className="text-zinc-200 font-medium">
                      {connectionState === 'LIVE' && metrics?.cluster_count != null
                        ? `${metrics.cluster_count} (largest ${metrics.largest_cluster})`
                        : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Inference FPS:</span>
                    <span className="text-zinc-200 font-medium">
                      {connectionState === 'LIVE' && metrics?.fps ? `${metrics.fps} FPS` : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Detector Latency:</span>
                    <span className="text-zinc-200 font-medium">
                      {connectionState === 'LIVE' && metrics?.latency_ms ? `${metrics.latency_ms}ms` : '—'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Density Latency:</span>
                    <span className="text-zinc-200 font-medium">
                      {connectionState === 'LIVE' && metrics?.density_latency_ms
                        ? `${metrics.density_latency_ms}ms`
                        : '—'}
                    </span>
                  </div>
                </div>
              </div>

              {/* Focus Controls */}
              {activeFeed && (
                <div className="pt-2 flex items-center gap-2">
                  {!isRecording ? (
                    <Button
                      variant="outline"
                      onClick={handleStartRecording}
                      isLoading={recordLoading}
                      className="w-full border-rose-900/60 hover:bg-rose-950/40 text-rose-400 hover:text-rose-300"
                    >
                      <Circle className="w-3.5 h-3.5 mr-1.5 fill-rose-500 text-rose-500" />
                      Start Record
                    </Button>
                  ) : (
                    <Button
                      variant="destructive"
                      onClick={handleStopRecording}
                      isLoading={recordLoading}
                      className="w-full"
                    >
                      <Square className="w-3.5 h-3.5 mr-1.5 fill-current" />
                      Stop Record
                    </Button>
                  )}

                  <Button
                    variant="outline"
                    onClick={() => handleStopSingleFeed(activeFeed.feed_id)}
                    className="border-zinc-800 text-zinc-400 hover:text-red-400"
                  >
                    <Square className="w-3.5 h-3.5 mr-1 text-zinc-400" />
                    Stop Stream
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
