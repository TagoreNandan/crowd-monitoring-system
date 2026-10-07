import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Video as VideoIcon, Play, RefreshCw, AlertCircle, Eye, Users, Cloud } from 'lucide-react';
import { submitVideoJob, getJob, connectWebSocket, getJobVideoUrl, getJobVideoUrlInfo, type VideoView as VideoViewMode, type VideoUrlInfo } from '../lib/api';
import type { JobRecordResponse, VideoFrameMetrics, WSEnvelope } from '../types/api';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Progress } from '../components/ui/Progress';
import { useToast } from '../context/ToastContext';

type LocalViewMode = 'original' | VideoViewMode;

const VIEW_TABS: { key: LocalViewMode; label: string }[] = [
  { key: 'original', label: 'Original' },
  { key: 'detection', label: 'Person Detection' },
  { key: 'heatmap', label: 'Heatmap' },
  { key: 'combined', label: 'Combined' },
];

export const VideoView: React.FC = () => {
  const { notifyStorage } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [job, setJob] = useState<JobRecordResponse | null>(null);
  const [metrics, setMetrics] = useState<VideoFrameMetrics | null>(null);
  const [livePreview, setLivePreview] = useState<string | null>(null);
  const [liveHeatmapPreview, setLiveHeatmapPreview] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<LocalViewMode>('detection');
  const [activeStreamInfo, setActiveStreamInfo] = useState<VideoUrlInfo | null>(null);
  const [activeStreamUrl, setActiveStreamUrl] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const pollTimerRef = useRef<any>(null);
  const renewalTimerRef = useRef<any>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const videoObjectUrlRef = useRef<string | null>(null);
  const notifiedJobIdRef = useRef<string | null>(null);

  const handleJobCompletionNotification = useCallback((j: JobRecordResponse) => {
    if (notifiedJobIdRef.current === j.job_id) return;
    notifiedJobIdRef.current = j.job_id;
    const isS3 = j.storage_backend === 's3';
    notifyStorage({
      title: 'Video Analysis Completed',
      message: `Processed ${j.total_frames || j.current_frame || 0} video frames. All analytical overlays generated.`,
      storageType: isS3 ? 's3' : 'local',
      storagePath: isS3 ? `s3://.../jobs/${j.job_id}/` : `data/jobs/${j.job_id}/`,
    });
  }, [notifyStorage]);

  const refreshStream = useCallback(async (jobId: string, mode: VideoViewMode, keepPosition = false) => {
    try {
      const info = await getJobVideoUrlInfo(jobId, mode);
      setActiveStreamInfo(info);
      const currentTime = videoRef.current?.currentTime || 0;
      const wasPlaying = videoRef.current ? !videoRef.current.paused : false;

      setActiveStreamUrl(info.url);

      if (keepPosition && videoRef.current) {
        videoRef.current.currentTime = currentTime;
        if (wasPlaying) videoRef.current.play().catch(() => {});
      }

      // Schedule proactive renewal before token TTL expires (e.g. 45s before 300s TTL)
      if (renewalTimerRef.current) clearTimeout(renewalTimerRef.current);
      if (info.mode === 's3' && info.expires_in) {
        const delayMs = Math.max((info.expires_in - 45) * 1000, 15000);
        renewalTimerRef.current = setTimeout(() => {
          refreshStream(jobId, mode, true);
        }, delayMs);
      }
    } catch (err) {
      console.error('Failed to get video URL info:', err);
      setActiveStreamUrl(getJobVideoUrl(jobId, mode));
    }
  }, []);

  // When job completes or view mode switches, obtain streaming URL
  useEffect(() => {
    if (job?.status === 'completed' && viewMode !== 'original') {
      refreshStream(job.job_id, viewMode as VideoViewMode, false);
    }
    return () => {
      if (renewalTimerRef.current) clearTimeout(renewalTimerRef.current);
    };
  }, [job?.status, job?.job_id, viewMode, refreshStream]);

  // Video error recovery handler (handles token expiration during idle/pause)
  const handleVideoError = () => {
    if (job?.status === 'completed' && viewMode !== 'original') {
      console.warn('Video stream error or token expired. Auto-renewing session...');
      refreshStream(job.job_id, viewMode as VideoViewMode, true);
    }
  };

  useEffect(() => {
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
      if (renewalTimerRef.current) clearTimeout(renewalTimerRef.current);
      if (videoObjectUrlRef.current) {
        URL.revokeObjectURL(videoObjectUrlRef.current);
      }
    };
  }, []);

  const resetSelection = (file: File) => {
    if (videoObjectUrlRef.current) {
      URL.revokeObjectURL(videoObjectUrlRef.current);
    }
    const newUrl = URL.createObjectURL(file);
    videoObjectUrlRef.current = newUrl;
    setSelectedFile(file);
    setVideoUrl(newUrl);
    setJob(null);
    setMetrics(null);
    setLivePreview(null);
    setLiveHeatmapPreview(null);
    setViewMode('detection');
    setActiveStreamInfo(null);
    setActiveStreamUrl(null);
    setError(null);
    notifiedJobIdRef.current = null;
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) resetSelection(e.target.files[0]);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) resetSelection(e.dataTransfer.files[0]);
  };

  const startVideoAnalysis = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setError(null);

    try {
      const initialJob = await submitVideoJob(selectedFile);
      setJob(initialJob);

      const jobId = initialJob.job_id;

      if (wsRef.current) wsRef.current.close();

      wsRef.current = connectWebSocket(
        `/ws/jobs/${jobId}`,
        (envelope: WSEnvelope) => {
          if (envelope.type === 'job.progress' && envelope.data) {
            setMetrics(envelope.data as VideoFrameMetrics);
            if (envelope.data.frame_jpeg) setLivePreview(envelope.data.frame_jpeg);
            if (envelope.data.heatmap_jpeg) setLiveHeatmapPreview(envelope.data.heatmap_jpeg);
            setJob((prev) => prev ? {
              ...prev,
              status: 'running',
              current_frame: envelope.data.frame_number,
              total_frames: envelope.data.total_frames ?? prev.total_frames,
              latest_metrics: envelope.data,
              progress: envelope.data.total_frames ? envelope.data.frame_number / envelope.data.total_frames : prev.progress
            } : prev);
          } else if (envelope.type === 'job.completed') {
            setJob((prev) => prev ? { ...prev, status: 'completed', progress: 1.0 } : prev);
            setLivePreview(null);
            setLiveHeatmapPreview(null);
          } else if (envelope.type === 'job.error') {
            setError(envelope.data?.message || 'Video processing job failed');
            setJob((prev) => prev ? { ...prev, status: 'failed' } : prev);
          }
        }
      );

      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
      pollTimerRef.current = setInterval(async () => {
        try {
          const updatedJob = await getJob(jobId);
          setJob(updatedJob);
          if (updatedJob.latest_metrics) {
            setMetrics(updatedJob.latest_metrics);
          }
          if (updatedJob.status === 'completed') {
            clearInterval(pollTimerRef.current);
            handleJobCompletionNotification(updatedJob);
          } else if (updatedJob.status === 'failed') {
            clearInterval(pollTimerRef.current);
          }
        } catch (e) {
          console.error('Job poll error:', e);
        }
      }, 1000);

    } catch (err: any) {
      setError(err.message || 'Failed to submit video processing job');
    } finally {
      setUploading(false);
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

  const currentStatus = job?.status || 'idle';
  const progressPercent = Math.round((job?.progress || 0) * 100);
  const isCompleted = currentStatus === 'completed';
  const isRunning = currentStatus === 'running';

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-5">
        <div className="flex items-center gap-3">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-zinc-100">Video Analysis</h1>
            <p className="text-sm text-zinc-400 mt-0.5">
              Per-frame person detection/tracking + periodic dense crowd estimation.
            </p>
          </div>
          {job && (
            <Badge variant={isCompleted ? 'safe' : isRunning ? 'secondary' : 'outline'}>
              {currentStatus}
            </Badge>
          )}
        </div>

        {selectedFile && !isRunning && (
          <Button onClick={startVideoAnalysis} isLoading={uploading}>
            <Play className="w-3.5 h-3.5 mr-1.5" />
            Start Video Analysis
          </Button>
        )}
      </div>

      {/* Main Viewport & Side Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Hero Video Viewport */}
        <div className="lg:col-span-8 space-y-4">
          {videoUrl && (
            <div className="flex items-center justify-between border-b border-zinc-800 pb-2 text-xs">
              <div className="flex items-center gap-1">
                {VIEW_TABS.map((tab) => (
                  <button
                    key={tab.key}
                    onClick={() => setViewMode(tab.key)}
                    className={`px-3 py-1 rounded-md transition-colors ${
                      viewMode === tab.key
                        ? 'bg-zinc-900 text-zinc-100 border border-zinc-800 font-medium'
                        : 'text-zinc-400 hover:text-zinc-200'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
              {isCompleted && viewMode !== 'original' && activeStreamInfo && (
                <div className="flex items-center gap-1.5 font-mono text-[11px]">
                  {activeStreamInfo.mode === 's3' ? (
                    <span className="flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-emerald-950/70 border border-emerald-800/80 text-emerald-400 font-medium">
                      <Cloud className="w-3 h-3 text-emerald-400 animate-pulse" />
                      <span>S3 Direct Stream (TTL: {activeStreamInfo.expires_in ? `${Math.round(activeStreamInfo.expires_in / 60)}m` : 'Active'})</span>
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 px-2.5 py-0.5 rounded-full bg-zinc-900 border border-zinc-800 text-zinc-400">
                      <span>Local Server Stream</span>
                    </span>
                  )}
                </div>
              )}
            </div>
          )}
          <div className="relative rounded-lg border border-zinc-800 bg-black min-h-[420px] flex items-center justify-center overflow-hidden">
            {!videoUrl ? (
              /* Dropzone */
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                className="p-16 text-center space-y-3 cursor-pointer group"
              >
                <div className="w-10 h-10 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center mx-auto text-zinc-400 group-hover:scale-105 transition-transform">
                  <VideoIcon className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-zinc-200">Select Video File</h3>
                  <p className="text-xs text-zinc-500 mt-1 max-w-xs mx-auto">
                    Supports MP4, MOV, AVI (e.g. 4K high-density crowd video)
                  </p>
                </div>
                <label className="inline-block mt-2 px-3.5 py-1.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-200 text-xs font-medium rounded-md border border-zinc-800 cursor-pointer transition-colors">
                  Browse Video
                  <input type="file" accept="video/*" onChange={handleFileChange} className="hidden" />
                </label>
              </div>
            ) : isCompleted && viewMode === 'original' ? (
              <div className="w-full h-full flex flex-col items-center justify-center">
                <video src={videoUrl} controls className="w-full max-h-[500px] object-contain" />
              </div>
            ) : isCompleted ? (
              /* Real output video for the selected view: person boxes,
                 density heatmap, or both -- baked in server-side,
                 never re-derived client-side. With automatic S3 token renewal & recovery. */
              <div className="w-full h-full flex flex-col items-center justify-center">
                <video
                  ref={videoRef}
                  key={viewMode}
                  src={activeStreamUrl || getJobVideoUrl(job!.job_id, viewMode as VideoViewMode)}
                  controls
                  autoPlay
                  loop
                  onError={handleVideoError}
                  className="w-full max-h-[500px] object-contain"
                />
              </div>
            ) : isRunning && (viewMode === 'heatmap' ? liveHeatmapPreview : livePreview) ? (
              /* Live annotated-frame preview streamed over the job's WebSocket
                 while processing is still in progress */
              <div className="w-full h-full flex flex-col items-center justify-center relative">
                <img
                  src={(viewMode === 'heatmap' ? liveHeatmapPreview : livePreview) as string}
                  alt="live analysis preview"
                  className="w-full max-h-[500px] object-contain"
                />
                <div className="absolute top-3 left-3 bg-zinc-950/90 border border-zinc-800 px-3 py-1.5 rounded text-xs font-mono text-zinc-300 flex items-center gap-2">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-zinc-400" />
                  <span>Processing: {progressPercent}%</span>
                </div>
              </div>
            ) : (
              <div className="w-full h-full flex flex-col items-center justify-center">
                <video src={videoUrl} controls className="w-full max-h-[500px] object-contain" />
                {isRunning && (
                  <div className="absolute top-3 left-3 bg-zinc-950/90 border border-zinc-800 px-3 py-1.5 rounded text-xs font-mono text-zinc-300 flex items-center gap-2">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin text-zinc-400" />
                    <span>Processing Frame Stream: {progressPercent}%</span>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Progress Indicator */}
          {isRunning && (
            <div className="p-4 rounded-lg bg-zinc-900/30 border border-zinc-800 space-y-2">
              <div className="flex justify-between text-xs font-mono text-zinc-300">
                <span>Frame {job?.current_frame || 0} / {job?.total_frames || '?'}</span>
                <span>{progressPercent}%</span>
              </div>
              <Progress value={progressPercent} />
            </div>
          )}

          {error && (
            <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              <div>{error}</div>
            </div>
          )}
        </div>

        {/* Right Column: Hero Metrics Panel */}
        <div className="lg:col-span-4 space-y-4">
          {/* Estimated Crowd */}
          <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
            <span className="text-xs text-zinc-400 font-medium flex items-center gap-1.5">
              <Users className="w-3.5 h-3.5 text-zinc-400" />
              Estimated Crowd
            </span>
            <div className="text-4xl font-semibold font-mono text-zinc-100">
              {metrics?.estimated_count ?? '—'}
            </div>
            <p className="text-xs text-zinc-500">Density regression (periodic)</p>
          </div>

          {/* Visible People */}
          <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
            <span className="text-xs text-zinc-400 font-medium flex items-center gap-1.5">
              <Eye className="w-3.5 h-3.5 text-zinc-400" />
              Visible People
            </span>
            <div className="text-3xl font-semibold font-mono text-zinc-200">
              {metrics?.visible_count !== undefined && metrics?.visible_count !== null ? metrics.visible_count : 'N/A'}
            </div>
            <p className="text-xs text-zinc-500">Person detections (periodic, held between passes)</p>
          </div>

          {/* Risk Level */}
          <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs text-zinc-400 font-medium">Risk Status</span>
              {metrics && <Badge variant={getRiskVariant(metrics.risk_status)}>{metrics.risk_status}</Badge>}
            </div>
            <div className="text-xs font-mono text-zinc-400 flex justify-between">
              <span>Risk Score</span>
              <span className="text-zinc-200 font-bold">{metrics ? `${metrics.risk_score.toFixed(1)} / 100` : '—'}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Compact Bottom Telemetry Row */}
      <div className="p-4 rounded-lg bg-zinc-900/30 border border-zinc-800 grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-4 text-xs font-mono">
        <div>
          <span className="text-zinc-500 block">Frame</span>
          <span className="text-zinc-200 font-medium">{job?.current_frame || 0} / {job?.total_frames || '—'}</span>
        </div>
        <div>
          <span className="text-zinc-500 block">Time</span>
          <span className="text-zinc-200 font-medium">
            {metrics?.video_time_s != null
              ? `${metrics.video_time_s.toFixed(1)}s${metrics.total_duration_s != null ? ` / ${metrics.total_duration_s.toFixed(1)}s` : ''}`
              : '—'}
          </span>
        </div>
        <div>
          <span className="text-zinc-500 block">Processing FPS</span>
          <span className="text-zinc-200 font-medium">{metrics?.fps ? `${metrics.fps}` : '—'}</span>
        </div>
        <div>
          <span className="text-zinc-500 block">Latency</span>
          <span className="text-zinc-200 font-medium">{metrics?.latency_ms ? `${metrics.latency_ms}ms` : '—'}</span>
        </div>
        <div>
          <span className="text-zinc-500 block">Max Count</span>
          <span className="text-zinc-200 font-medium">{metrics?.peak_count ?? '—'}</span>
        </div>
        <div>
          <span className="text-zinc-500 block">Density Score</span>
          <span className="text-zinc-200 font-medium">{metrics?.density ?? '—'}</span>
        </div>
        <div>
          <span className="text-zinc-500 block">DBSCAN Clusters</span>
          <span className="text-zinc-200 font-medium">{metrics?.cluster_count ?? '—'}</span>
        </div>
      </div>
    </div>
  );
};
