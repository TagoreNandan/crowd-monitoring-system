import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Radio, Play, Square, RefreshCw, AlertCircle, Users, Eye, Camera } from 'lucide-react';
import { createFeed, stopFeed, connectWebSocket, uploadFeedVideo, getDevices } from '../lib/api';
import type { Device, FeedResponse, PartialFeedMetrics, VideoFrameMetrics, WSEnvelope } from '../types/api';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';

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
  const [sourceUri, setSourceUri] = useState<string>('dashboard/static/uploads/077cc45b94_14968235_2160_3840_60fps.mp4');
  const [feedUploadFile, setFeedUploadFile] = useState<File | null>(null);

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

  const wsRef = useRef<WebSocket | null>(null);
  const intentionalStopRef = useRef<boolean>(false);
  const reconnectAttemptsRef = useRef<number>(0);
  const feedIdRef = useRef<string | null>(null);
  const reconnectTimerRef = useRef<any>(null);

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
  }, [initialDevice]);

  useEffect(() => {
    return () => {
      intentionalStopRef.current = true;
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    };
  }, []);

  const attachSocket = useCallback((feedId: string) => {
    if (wsRef.current) wsRef.current.close();

    wsRef.current = connectWebSocket(
      `/ws/feeds/${feedId}`,
      (envelope: WSEnvelope) => {
        if (envelope.type === 'feed.progress' && envelope.data) {
          // Fast (detector) and density (DM-Count) loops each broadcast
          // only the fields they own -- merge, never replace, or one
          // loop's update would wipe out the other's last known values.
          const partial = envelope.data as PartialFeedMetrics;
          setMetrics((prev) => ({ ...(prev ?? ({} as VideoFrameMetrics)), ...partial }));
          if (envelope.data.frame_jpeg) setLiveFrame(envelope.data.frame_jpeg);
          if (envelope.data.heatmap_jpeg) setLiveHeatmap(envelope.data.heatmap_jpeg);
          if (envelope.data.combined_jpeg) setLiveCombined(envelope.data.combined_jpeg);
          reconnectAttemptsRef.current = 0;
          setConnectionState('LIVE');
        } else if (envelope.type === 'feed.status' && envelope.data?.status === 'stopped') {
          intentionalStopRef.current = true;
          setConnectionState('STOPPED');
          setMetrics(null);
          setLiveFrame(null);
          setLiveHeatmap(null);
          setLiveCombined(null);
        } else if (envelope.type === 'feed.error') {
          setError(envelope.data?.message || 'Feed processing error');
          setConnectionState('ERROR');
        }
      },
      (connected: boolean) => {
        if (connected) {
          setConnectionState((prev) => (prev === 'LIVE' ? prev : 'CONNECTING'));
          return;
        }
        // Socket closed/errored.
        if (intentionalStopRef.current) {
          setConnectionState('STOPPED');
          return;
        }
        if (reconnectAttemptsRef.current >= MAX_RECONNECT_ATTEMPTS) {
          setConnectionState('ERROR');
          setError('Lost connection to the real-time feed and could not reconnect.');
          return;
        }
        setConnectionState('RECONNECTING');
        reconnectAttemptsRef.current += 1;
        reconnectTimerRef.current = setTimeout(() => {
          if (!intentionalStopRef.current && feedIdRef.current === feedId) {
            attachSocket(feedId);
          }
        }, 1000 * reconnectAttemptsRef.current);
      }
    );
  }, []);

  const handleStartFeed = async () => {
    setStarting(true);
    setError(null);
    setMetrics(null);
    setLiveFrame(null);
    intentionalStopRef.current = false;
    reconnectAttemptsRef.current = 0;
    setConnectionState('CONNECTING');

    try {
      let feedRes: FeedResponse;
      if (sourceType === 'upload') {
        if (!feedUploadFile) {
          throw new Error('Please select a video file to upload for the real-time feed.');
        }
        feedRes = await uploadFeedVideo(feedUploadFile, true);
      } else {
        feedRes = await createFeed(sourceType, sourceUri, true);
      }
      setActiveFeed(feedRes);
      feedIdRef.current = feedRes.feed_id;
      attachSocket(feedRes.feed_id);
    } catch (err: any) {
      setError(err.message || 'Failed to initialize real-time feed');
      setConnectionState('ERROR');
    } finally {
      setStarting(false);
    }
  };

  const handleStopFeed = async () => {
    if (!activeFeed) return;
    setStopping(true);
    intentionalStopRef.current = true;
    if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);

    try {
      await stopFeed(activeFeed.feed_id);
      if (wsRef.current) wsRef.current.close();
      setActiveFeed(null);
      feedIdRef.current = null;
      setMetrics(null);
      setLiveFrame(null);
      setConnectionState('STOPPED');
    } catch (err: any) {
      setError(err.message || 'Failed to stop feed');
    } finally {
      setStopping(false);
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

  const isLive = connectionState === 'LIVE';

  const statusBadge = () => {
    switch (connectionState) {
      case 'LIVE':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-800/50">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
            ● LIVE
          </span>
        );
      case 'CONNECTING':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono text-amber-400 bg-amber-950/60 border border-amber-800/50">
            <RefreshCw className="w-3 h-3 animate-spin" />
            Connecting
          </span>
        );
      case 'RECONNECTING':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono text-amber-400 bg-amber-950/60 border border-amber-800/50">
            <RefreshCw className="w-3 h-3 animate-spin" />
            Reconnecting
          </span>
        );
      case 'ERROR':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono text-rose-400 bg-rose-950/60 border border-rose-800/50">
            Error
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono text-zinc-500 border border-zinc-800">
            Offline
          </span>
        );
    }
  };

  const getDisplayedFrame = () => {
    if (viewMode === 'heatmap' && liveHeatmap) return liveHeatmap;
    if (viewMode === 'combined' && liveCombined) return liveCombined;
    return liveFrame;
  };

  const currentDisplayedFrame = getDisplayedFrame();

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-5">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight text-zinc-100 flex items-center gap-2">
            Real-Time Feed
          </h1>
          {statusBadge()}
        </div>

        <div>
          {!activeFeed ? (
            <Button onClick={handleStartFeed} isLoading={starting}>
              <Play className="w-3.5 h-3.5 mr-1.5" />
              Start Feed
            </Button>
          ) : (
            <Button variant="destructive" onClick={handleStopFeed} isLoading={stopping}>
              <Square className="w-3.5 h-3.5 mr-1.5" />
              Stop Feed
            </Button>
          )}
        </div>
      </div>

      {/* Viewport & Side Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 space-y-4">
          <div className="relative rounded-lg border border-zinc-800 bg-black min-h-[440px] flex flex-col justify-between overflow-hidden">
            {isLive && (
              <div className="flex items-center justify-between px-3 py-2 bg-zinc-950/90 border-b border-zinc-800 z-10">
                <div className="flex items-center gap-2 font-mono text-xs text-emerald-400">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                  <span>STREAM: {activeFeed?.feed_id}</span>
                </div>
                <div className="flex items-center gap-1 bg-zinc-900 p-1 rounded border border-zinc-800">
                  <button
                    onClick={() => setViewMode('bbox')}
                    className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${viewMode === 'bbox' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'}`}
                  >
                    Live Video
                  </button>
                  <button
                    onClick={() => setViewMode('heatmap')}
                    className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${viewMode === 'heatmap' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'}`}
                  >
                    Live Heatmap
                  </button>
                  <button
                    onClick={() => setViewMode('combined')}
                    className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${viewMode === 'combined' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'}`}
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
                /* Stopped/Error Clean State */
                <div className="p-16 text-center space-y-3">
                  <div className="w-10 h-10 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center mx-auto text-zinc-500">
                    <Radio className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-zinc-300">Feed Offline</h3>
                    <p className="text-xs text-zinc-500 max-w-xs mx-auto mt-1">
                      Select stream source configuration below and press <strong className="text-zinc-300">Start Feed</strong>.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Stream Ingestion Source Panel */}
          <div className="p-4 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-3 text-xs">
            <span className="font-mono font-medium text-zinc-400 uppercase tracking-wider block text-[11px]">
              Ingestion Source Settings
            </span>

            {/* Quick Enrolled Camera Selector */}
            {enrolledDevices.length > 0 && (
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-2.5 rounded bg-zinc-950/80 border border-zinc-800/80">
                <div className="flex items-center gap-2">
                  <Camera className="w-3.5 h-3.5 text-zinc-400" />
                  <span className="text-zinc-300 font-medium">Select Fleet Camera:</span>
                </div>
                <select
                  value={selectedDeviceId}
                  onChange={(e) => {
                    const id = e.target.value;
                    setSelectedDeviceId(id);
                    const found = enrolledDevices.find((d) => d.device_id === id);
                    if (found) applyDeviceSource(found);
                  }}
                  disabled={activeFeed !== null}
                  className="bg-zinc-900 border border-zinc-700 text-zinc-200 text-xs rounded px-2.5 py-1 focus:outline-none font-mono"
                >
                  <option value="">-- Choose from Enrolled Fleet --</option>
                  {enrolledDevices.map((d) => (
                    <option key={d.device_id} value={d.device_id}>
                      [{d.device_id}] {d.name} ({d.location}) &bull; {d.status}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <div>
                <label className="text-zinc-400 block mb-1">Source Type</label>
                <select
                  value={sourceType}
                  onChange={(e) => {
                    setSourceType(e.target.value);
                    setFeedUploadFile(null);
                  }}
                  disabled={activeFeed !== null}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-200 focus:outline-none focus:border-zinc-700"
                >
                  <option value="upload">Upload Video File</option>
                  <option value="sample">Sample Video File</option>
                  <option value="camera">Local USB Camera</option>
                  <option value="rtsp">RTSP / RTMP Stream</option>
                </select>
              </div>

              <div className="sm:col-span-2">
                {sourceType === 'upload' ? (
                  <div>
                    <label className="text-zinc-400 block mb-1">Upload Video File</label>
                    <input
                      type="file"
                      accept="video/*"
                      disabled={activeFeed !== null}
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
                    <label className="text-zinc-400 block mb-1">Source URI / Path</label>
                    <input
                      type="text"
                      value={sourceUri}
                      onChange={(e) => setSourceUri(e.target.value)}
                      disabled={activeFeed !== null}
                      className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-200 font-mono text-xs focus:outline-none focus:border-zinc-700"
                    />
                  </div>
                )}
              </div>
            </div>
          </div>

          {error && (
            <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              <div>{error}</div>
            </div>
          )}
        </div>

        {/* Right Telemetry Column */}
        <div className="lg:col-span-4 space-y-4">
          <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
            <span className="text-xs text-zinc-400 font-medium flex items-center gap-1.5">
              <Users className="w-3.5 h-3.5 text-zinc-400" />
              Estimated Crowd
            </span>
            <div className="text-4xl font-semibold font-mono text-zinc-100">
              {isLive && metrics?.estimated_count != null ? metrics.estimated_count : '—'}
            </div>
            <p className="text-xs text-zinc-500">DM-Count dense estimate</p>
          </div>

          <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
            <span className="text-xs text-zinc-400 font-medium flex items-center gap-1.5">
              <Eye className="w-3.5 h-3.5 text-zinc-400" />
              Visible People
            </span>
            <div className="text-3xl font-semibold font-mono text-zinc-200">
              {isLive ? (metrics?.visible_count !== undefined && metrics?.visible_count !== null ? metrics.visible_count : 'N/A') : '—'}
            </div>
            <p className="text-xs text-zinc-500">YOLOv8 bounding box count</p>
          </div>

          <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
            <span className="text-xs text-zinc-400 font-medium">Peak Count</span>
            <div className="text-2xl font-semibold font-mono text-zinc-200">{isLive ? (metrics?.peak_count ?? '—') : '—'}</div>
          </div>

          <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-zinc-400 font-medium">Risk Assessment</span>
              {isLive && metrics?.risk_status && <Badge variant={getRiskVariant(metrics.risk_status)}>{metrics.risk_status}</Badge>}
            </div>
            <div className="text-xs font-mono text-zinc-400 space-y-1 pt-1">
              <div className="flex justify-between">
                <span>Risk Score:</span>
                <span className="text-zinc-200 font-medium">{isLive && metrics?.risk_score != null ? `${metrics.risk_score.toFixed(1)} / 100` : '—'}</span>
              </div>
              <div className="flex justify-between">
                <span>Density Score:</span>
                <span className="text-zinc-200 font-medium">{isLive && metrics?.density != null ? metrics.density : '—'}</span>
              </div>
              <div className="flex justify-between">
                <span>Clusters:</span>
                <span className="text-zinc-200 font-medium">{isLive && metrics?.cluster_count != null ? `${metrics.cluster_count} (largest ${metrics.largest_cluster})` : '—'}</span>
              </div>
              <div className="flex justify-between">
                <span>Inference FPS:</span>
                <span className="text-zinc-200 font-medium">{isLive && metrics?.fps ? `${metrics.fps} FPS` : '—'}</span>
              </div>
              <div className="flex justify-between">
                <span>Detector Latency:</span>
                <span className="text-zinc-200 font-medium">{isLive && metrics?.latency_ms ? `${metrics.latency_ms}ms` : '—'}</span>
              </div>
              <div className="flex justify-between">
                <span>Density Latency:</span>
                <span className="text-zinc-200 font-medium">{isLive && metrics?.density_latency_ms ? `${metrics.density_latency_ms}ms` : '—'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
