import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Radio,
  Play,
  Square,
  RefreshCw,
  AlertCircle,
  Users,
  Eye,
  Camera,
  X,
  LayoutGrid,
  Maximize2,
  SlidersHorizontal,
  ChevronDown,
  ChevronUp,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import {
  createFeed,
  stopFeed,
  uploadFeedVideo,
  getDevices,
  listFeeds,
} from '../lib/api';
import type { Device, FeedResponse, VideoFrameMetrics } from '../types/api';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { StreamPanel } from '../components/StreamPanel';

interface FeedViewProps {
  initialDevice?: Device | null;
}

export const FeedView: React.FC<FeedViewProps> = ({ initialDevice }) => {
  const [enrolledDevices, setEnrolledDevices] = useState<Device[]>([]);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_selectedDeviceId, setSelectedDeviceId] = useState<string>('');
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
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [liveMetricsMap, setLiveMetricsMap] = useState<Record<string, VideoFrameMetrics>>({});

  const handleMetricsUpdate = useCallback((feedId: string, metrics: VideoFrameMetrics) => {
    setLiveMetricsMap((prev) => {
      if (prev[feedId] === metrics) return prev;
      return { ...prev, [feedId]: metrics };
    });
  }, []);

  // Determine whether to display the Multi-Stream Grid or Single Focus View
  // Must be declared before the useEffect that depends on it.
  const isShowingGrid = useMemo(() => {
    if (viewModePreference === 'grid') return true;
    if (viewModePreference === 'single') return false;
    // 'auto' mode: if focusedFeedId is set, show single. If > 1 stream, show grid.
    if (focusedFeedId) return false;
    return activeFeeds.length > 1;
  }, [viewModePreference, focusedFeedId, activeFeeds.length]);

  // Single-feed detailed view state (for the focused feed or primary feed)
  const [activeFeed, setActiveFeed] = useState<FeedResponse | null>(null);
  const [starting, setStarting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

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

  // Sync focused feed
  useEffect(() => {
    const targetFeedId = focusedFeedId || (activeFeeds.length > 0 ? activeFeeds[0].feed_id : null);
    if (!targetFeedId) {
      setActiveFeed(null);
      return;
    }
    const feed = activeFeeds.find((f) => f.feed_id === targetFeedId) || null;
    setActiveFeed(feed);
  }, [focusedFeedId, activeFeeds]);

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

  // Launch all online cameras concurrently
  const handleStartAllFleet = async () => {
    setBatchActionLoading(true);
    try {
      const onlineDevices = enrolledDevices.filter((d) => d.status.toUpperCase() === 'ONLINE');
      await Promise.allSettled(
        onlineDevices.slice(0, 6).map(async (dev) => {
          const isRunning = activeFeeds.some((f) => f.device_id === dev.device_id);
          if (!isRunning) {
            let st = 'rtsp';
            if (dev.stream_type.toUpperCase() === 'WEBCAM') st = 'camera';
            else if (dev.stream_type.toUpperCase() === 'VIDEO_FILE') st = 'sample';
            await createFeed(st, dev.stream_url, true, undefined, dev.device_id, dev.name).catch(() => {});
          }
        })
      );
      await refreshActiveFeeds();
    } catch (err: any) {
      console.error('Error starting fleet cameras:', err);
    } finally {
      setBatchActionLoading(false);
    }
  };

  // Stop all running feeds
  const handleStopAllFeeds = async () => {
    setBatchActionLoading(true);
    try {
      await Promise.allSettled(activeFeeds.map((f) => stopFeed(f.feed_id).catch(() => {})));
      setActiveFeeds([]);
      setActiveFeed(null);
      setFocusedFeedId(null);
      await refreshActiveFeeds();
    } catch (err: any) {
      console.error('Error stopping all feeds:', err);
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

  const getRiskVariant = (st?: string) => {
    switch (st?.toUpperCase()) {
      case 'CRITICAL': return 'critical';
      case 'WARNING': return 'warning';
      case 'CAUTION': return 'caution';
      default: return 'safe';
    }
  };

  // 6 cameras maximum per page pagination
  const CAMS_PER_PAGE = 6;
  const totalPages = Math.max(1, Math.ceil(activeFeeds.length / CAMS_PER_PAGE));
  const safePage = Math.min(currentPage, totalPages);

  // Feeds to display on current page
  const visibleFeeds = useMemo(() => {
    const startIdx = (safePage - 1) * CAMS_PER_PAGE;
    return activeFeeds.slice(startIdx, startIdx + CAMS_PER_PAGE);
  }, [activeFeeds, safePage]);

  // Viewport-fitted Grid layout CSS based on visible stream count (max 6)
  const gridLayoutClass = useMemo(() => {
    const count = visibleFeeds.length;
    switch (count) {
      case 1:
        return 'grid-cols-1 grid-rows-1';
      case 2:
        return 'grid-cols-2 grid-rows-1';
      case 3:
        return 'grid-cols-3 grid-rows-1';
      case 4:
        return 'grid-cols-2 grid-rows-2';
      case 5:
      case 6:
      default:
        return 'grid-cols-3 grid-rows-2';
    }
  }, [visibleFeeds.length]);

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



  return (
    <div className="h-full min-h-0 flex flex-col p-3 md:p-4 gap-2 overflow-hidden select-none">
      {/* Top Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-zinc-800 pb-2.5 shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold tracking-tight text-zinc-100 flex items-center gap-2">
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

        {/* Global Action Controls */}
        <div className="flex items-center gap-2">
          {/* Pagination Controls when multiple pages (max 6 cams/page) */}
          {totalPages > 1 && (
            <div className="flex items-center gap-1.5 bg-zinc-900 px-2 py-1 rounded-md border border-zinc-800 text-xs font-mono">
              <button
                onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                disabled={safePage === 1}
                className="p-1 rounded text-zinc-400 hover:text-zinc-100 disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
                title="Previous 6 Cameras"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
              <span className="text-zinc-300 font-semibold px-1">
                Page {safePage} / {totalPages}
              </span>
              <button
                onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                disabled={safePage >= totalPages}
                className="p-1 rounded text-zinc-400 hover:text-zinc-100 disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
                title="Next 6 Cameras"
              >
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

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
      <div className="px-3 py-1.5 rounded-lg bg-zinc-900/40 border border-zinc-800 shrink-0">
        <div className="flex items-center justify-between gap-3 overflow-x-auto">
          <div className="flex items-center gap-2 shrink-0">
            <Camera className="w-3.5 h-3.5 text-zinc-400" />
            <span className="font-mono text-xs font-semibold text-zinc-300">Fleet Cameras:</span>
          </div>

          {enrolledDevices.length === 0 ? (
            <p className="text-xs text-zinc-500 font-mono py-0.5">
              No enrolled devices in fleet. Onboard RTSP or Webcam devices in Device Fleet to stream directly.
            </p>
          ) : (
            <div className="flex items-center gap-1.5 overflow-x-auto py-0.5 scrollbar-thin">
              {enrolledDevices.map((dev) => {
                const active = activeFeeds.find((f) => f.device_id === dev.device_id);
                const isRunning = !!active;
                const isLoading = deviceActionLoading === dev.device_id;

                return (
                  <button
                    key={dev.device_id}
                    onClick={() => handleToggleDeviceStream(dev)}
                    disabled={isLoading}
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono transition-all border shrink-0 cursor-pointer ${
                      isRunning
                        ? 'bg-emerald-950/70 border-emerald-700/80 text-emerald-300 shadow-sm'
                        : 'bg-zinc-950/80 border-zinc-800 text-zinc-400 hover:border-zinc-700 hover:text-zinc-200'
                    }`}
                    title={isRunning ? `Click to stop stream for ${dev.name}` : `Click to stream ${dev.name}`}
                  >
                    {isLoading ? (
                      <RefreshCw className="w-2.5 h-2.5 animate-spin text-amber-400" />
                    ) : (
                      <span
                        className={`w-2 h-2 rounded-full ${
                          isRunning ? 'bg-emerald-400 animate-pulse' : 'bg-zinc-600'
                        }`}
                      />
                    )}
                    <span className="font-medium text-zinc-200 text-xs">{dev.name}</span>
                    <span className="text-[10px] text-zinc-500 hidden sm:inline">({dev.stream_type})</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Collapsible Custom Ingestion Settings Panel */}
      {isIngestionPanelOpen && (
        <div className="p-3.5 rounded-lg bg-zinc-900/90 border border-zinc-800 space-y-2.5 text-xs animate-in fade-in shrink-0 z-20">
          <div className="flex items-center justify-between border-b border-zinc-800 pb-1.5">
            <span className="font-mono font-medium text-zinc-300 uppercase tracking-wider text-[11px]">
              Ingest Custom Source (Sample Video / Webcam / Custom URL)
            </span>
            <button
              onClick={() => setIsIngestionPanelOpen(false)}
              className="text-zinc-500 hover:text-zinc-300 cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 pt-0.5">
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

          <div className="flex justify-end pt-1">
            <Button onClick={handleStartCustomFeed} isLoading={starting} size="sm">
              <Play className="w-3 h-3 mr-1 fill-current" />
              Launch Stream
            </Button>
          </div>
        </div>
      )}

      {error && (
        <div className="p-2.5 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-start gap-2 shrink-0">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1">{error}</div>
          <button onClick={() => setError(null)} className="text-rose-400 hover:text-rose-200 cursor-pointer">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* =========================================================================
          DYNAMIC SURVEILLANCE WORKSPACE (PERMANENT STREAM MOUNTING)
          ========================================================================= */}
      {activeFeeds.length === 0 ? (
        /* EMPTY STATE: ZERO CAMERAS RUNNING */
        <div className="flex-1 min-h-0 flex flex-col items-center justify-center p-8 text-center rounded-xl bg-zinc-900/20 border border-zinc-800/80 space-y-4">
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
                className="bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs cursor-pointer"
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
              className="text-xs text-zinc-300 border-zinc-700 cursor-pointer"
            >
              <Play className="w-3 h-3 mr-1.5" />
              Launch Sample Crowd Feed
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex-1 min-h-0 flex flex-col gap-2 overflow-hidden">
          {/* Top Mode Header / Sub-Nav */}
          {isShowingGrid ? (
            /* Top Aggregate Fleet KPI Strip */
            <div className="flex items-center justify-between px-3 py-1.5 rounded-md bg-zinc-900/50 border border-zinc-800 font-mono text-xs shrink-0">
              <div className="flex items-center gap-4">
                <span className="text-zinc-400">
                  Active Feeds: <strong className="text-emerald-400">{activeFeeds.length}</strong>
                </span>
                <span className="text-zinc-400">
                  Total Monitored: <strong className="text-zinc-100">{aggregateMetrics.totalCount}</strong>
                </span>
                <span className="text-zinc-400 flex items-center gap-1.5">
                  Highest Risk:
                  <Badge variant={getRiskVariant(aggregateMetrics.maxRisk)} className="font-bold text-[10px] py-0 px-1.5">
                    {aggregateMetrics.maxRisk}
                  </Badge>
                </span>
                <span className="text-zinc-500 hidden md:inline">
                  Wall Matrix: Showing {visibleFeeds.length} of {activeFeeds.length}
                </span>
              </div>

              {totalPages > 1 && (
                <div className="flex items-center gap-1.5">
                  <button
                    onClick={() => setCurrentPage((p) => Math.max(1, p - 1))}
                    disabled={safePage === 1}
                    className="px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 hover:text-white disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1 text-[11px] cursor-pointer"
                  >
                    <ChevronLeft className="w-3 h-3" /> Prev
                  </button>
                  <span className="text-zinc-300 text-[11px]">
                    Page {safePage}/{totalPages}
                  </span>
                  <button
                    onClick={() => setCurrentPage((p) => Math.min(totalPages, p + 1))}
                    disabled={safePage >= totalPages}
                    className="px-2 py-0.5 rounded bg-zinc-800 text-zinc-300 hover:text-white disabled:opacity-40 disabled:cursor-not-allowed flex items-center gap-1 text-[11px] cursor-pointer"
                  >
                    Next <ChevronRight className="w-3 h-3" />
                  </button>
                </div>
              )}
            </div>
          ) : (
            /* Quick Return to Grid Banner if multiple streams are running */
            activeFeeds.length > 1 && (
              <div className="flex items-center justify-between px-3 py-1.5 rounded-lg bg-zinc-900/80 border border-zinc-800 text-xs font-mono shrink-0">
                <div className="flex items-center gap-2 text-zinc-300">
                  <Maximize2 className="w-4 h-4 text-cyan-400" />
                  <span>
                    Currently inspecting:{' '}
                    <strong className="text-white">
                      {activeFeed?.device_name || activeFeed?.description || activeFeed?.feed_id}
                    </strong>
                  </span>
                  <span className="text-zinc-500">({activeFeeds.length} streams running)</span>
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setFocusedFeedId(null);
                    setViewModePreference('grid');
                  }}
                  className="text-xs border-zinc-700 text-zinc-200 hover:bg-zinc-800 cursor-pointer h-7 px-2.5"
                >
                  <LayoutGrid className="w-3.5 h-3.5 mr-1" />
                  Back to Multi-Stream Wall
                </Button>
              </div>
            )
          )}

          {/* Unified Stream Grid + Side Telemetry (Streams Never Unmount) */}
          <div className="flex-1 min-h-0 flex flex-col lg:flex-row gap-3 overflow-hidden">
            {/* Left Primary Stream Display Area */}
            <div
              className={`flex-1 min-h-0 ${
                isShowingGrid ? `grid gap-2.5 ${gridLayoutClass}` : 'flex flex-col'
              }`}
            >
              {visibleFeeds.map((feed) => {
                const targetId = focusedFeedId || (activeFeeds.length > 0 ? activeFeeds[0].feed_id : null);
                const isTarget = feed.feed_id === targetId;
                const isFocused = !isShowingGrid && isTarget;
                const isHidden = !isShowingGrid && !isTarget;

                return (
                  <div
                    key={feed.feed_id}
                    className={
                      isHidden
                        ? 'hidden'
                        : isFocused
                        ? 'flex-1 h-full min-h-0 flex flex-col'
                        : 'h-full min-h-0 flex flex-col'
                    }
                  >
                    <StreamPanel
                      feed={feed}
                      isFocused={isFocused}
                      onFocus={(f) => {
                        if (isFocused) {
                          setFocusedFeedId(null);
                          setViewModePreference('grid');
                        } else {
                          setFocusedFeedId(f.feed_id);
                          setViewModePreference('single');
                        }
                      }}
                      onStop={handleStopSingleFeed}
                      onMetricsUpdate={(m) => handleMetricsUpdate(feed.feed_id, m)}
                      className="h-full"
                    />
                  </div>
                );
              })}
            </div>

            {/* Right Telemetry Column (Visible in Single Focus View, powered by real-time WebSocket metrics) */}
            {!isShowingGrid && (() => {
              const targetFeed = activeFeed || (activeFeeds.length > 0 ? activeFeeds[0] : null);
              const curMetrics = (targetFeed && liveMetricsMap[targetFeed.feed_id]) || targetFeed?.latest_metrics;

              return (
                <div className="w-full lg:w-80 shrink-0 h-full overflow-y-auto space-y-2.5 pr-1 font-mono">
                  <div className="p-3.5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
                    <span className="text-xs text-zinc-400 font-medium flex items-center gap-1.5">
                      <Users className="w-3.5 h-3.5 text-zinc-400" />
                      Estimated Crowd
                    </span>
                    <div className="text-3xl font-bold font-mono text-zinc-100">
                      {curMetrics?.estimated_count != null ? Math.round(curMetrics.estimated_count) : '—'}
                    </div>
                    <p className="text-[11px] text-zinc-500">Spatial density estimate</p>
                  </div>

                  <div className="p-3.5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
                    <span className="text-xs text-zinc-400 font-medium flex items-center gap-1.5">
                      <Eye className="w-3.5 h-3.5 text-zinc-400" />
                      Visible People
                    </span>
                    <div className="text-2xl font-bold font-mono text-zinc-200">
                      {curMetrics?.visible_count != null ? curMetrics.visible_count : '—'}
                    </div>
                    <p className="text-[11px] text-zinc-500">Optical detection count</p>
                  </div>

                  <div className="p-3 rounded-lg bg-zinc-900/40 border border-zinc-800 flex items-center justify-between">
                    <span className="text-xs text-zinc-400 font-medium">Peak Count</span>
                    <div className="text-lg font-bold font-mono text-zinc-200">
                      {curMetrics?.peak_count != null ? curMetrics.peak_count : '—'}
                    </div>
                  </div>

                  <div className="p-3.5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-zinc-400 font-medium">Risk Assessment</span>
                      {curMetrics?.risk_status && (
                        <Badge variant={getRiskVariant(curMetrics.risk_status)}>{curMetrics.risk_status}</Badge>
                      )}
                    </div>
                    <div className="text-xs font-mono text-zinc-400 space-y-1 pt-1">
                      <div className="flex justify-between">
                        <span>Risk Score:</span>
                        <span className="text-zinc-200 font-medium">
                          {curMetrics?.risk_score != null
                            ? `${curMetrics.risk_score.toFixed(1)} / 100`
                            : '—'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Density Score:</span>
                        <span className="text-zinc-200 font-medium">
                          {curMetrics?.density != null ? curMetrics.density : '—'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Clusters:</span>
                        <span className="text-zinc-200 font-medium">
                          {curMetrics?.cluster_count != null
                            ? `${curMetrics.cluster_count} (max ${curMetrics.largest_cluster})`
                            : '—'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Inference FPS:</span>
                        <span className="text-zinc-200 font-medium">
                          {curMetrics?.fps ? `${curMetrics.fps} FPS` : '—'}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span>Latency:</span>
                        <span className="text-zinc-200 font-medium">
                          {curMetrics?.latency_ms ? `${curMetrics.latency_ms}ms` : '—'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })()}
          </div>
        </div>
      )}
    </div>
  );
};

