import React, { useEffect, useState, useRef } from 'react';
import {
  ArrowRight,
  Image as ImageIcon,
  Video,
  Radio,
  Maximize2,
  Activity,
} from 'lucide-react';
import type { SystemHealth, SystemStatus, VideoFrameMetrics, StreamAnalyticsSnapshot } from '../types/api';
import { getLiveAnalytics } from '../lib/api';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';

interface OverviewViewProps {
  health: SystemHealth | null;
  status: SystemStatus | null;
  latestMetrics: VideoFrameMetrics | null;
  onNavigate: (view: any) => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({
  health,
  status,
  latestMetrics,
  onNavigate,
}) => {
  const [activeStream, setActiveStream] = useState<StreamAnalyticsSnapshot | null>(null);
  const pollIntervalRef = useRef<any>(null);

  const fetchLiveStatus = async () => {
    try {
      const data = await getLiveAnalytics();
      if (data.streams && data.streams.length > 0) {
        const live = data.streams.find((s) => s.status === 'LIVE') || data.streams[0];
        setActiveStream(live);
      } else {
        setActiveStream(null);
      }
    } catch {
      // Best effort background sync
    }
  };

  useEffect(() => {
    fetchLiveStatus();
    pollIntervalRef.current = setInterval(fetchLiveStatus, 2500);
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  const isHealthy = health?.status === 'ok' && health?.model_loaded;
  const isStreaming = activeStream !== null || latestMetrics !== null;

  // Aggregate current metrics preferring active stream snapshot or WS telemetry
  const estCrowd = activeStream?.estimated_count ?? latestMetrics?.estimated_count ?? 0;
  const visCount = activeStream?.visible_count ?? latestMetrics?.visible_count ?? null;
  const density = activeStream ? (activeStream.risk_score * 0.1).toFixed(2) : (latestMetrics?.density ?? 0);
  const riskStatus = (activeStream?.risk_status || latestMetrics?.risk_status || 'SAFE').toUpperCase();
  const fps = activeStream?.fps ?? latestMetrics?.fps ?? null;
  const previewFrame = activeStream?.frame_jpeg || latestMetrics?.frame_jpeg || null;
  const streamTitle = activeStream?.camera_name || 'Active Camera Stream';
  const streamLocation = activeStream?.location || 'Main Surveillance Zone';

  const getRiskVariant = (st: string): 'critical' | 'warning' | 'caution' | 'safe' => {
    switch (st) {
      case 'CRITICAL':
        return 'critical';
      case 'WARNING':
        return 'warning';
      case 'CAUTION':
        return 'caution';
      default:
        return 'safe';
    }
  };

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-10">
      {/* Header section */}
      <div className="space-y-1">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight text-zinc-100">Overview</h1>
          <Badge variant={isHealthy ? 'safe' : 'secondary'}>
            {isHealthy ? 'System Operational' : 'Initializing'}
          </Badge>
          {isStreaming && (
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-800/60">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              1 Stream Active
            </span>
          )}
        </div>
        <p className="text-sm text-zinc-400">
          Real-time crowd estimation, spatial density analysis, and multi-camera safety monitoring.
        </p>
      </div>

      {/* Current Monitoring State */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-mono font-medium text-zinc-400 uppercase tracking-wider">
            Current Live Stream
          </h2>
          {isStreaming && (
            <button
              onClick={() => onNavigate('feed')}
              className="text-xs text-zinc-400 hover:text-zinc-100 flex items-center gap-1 transition-colors font-mono"
            >
              <span>View Full Screen Feed</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {isStreaming ? (
          <div className="rounded-lg bg-zinc-900/50 border border-zinc-800 overflow-hidden shadow-xl">
            <div className="grid grid-cols-1 lg:grid-cols-12">
              {/* Live Preview Thumbnail / Feed Area */}
              <div
                onClick={() => onNavigate('feed')}
                className="lg:col-span-7 bg-black min-h-[220px] max-h-[320px] relative flex items-center justify-center cursor-pointer group overflow-hidden border-b lg:border-b-0 lg:border-r border-zinc-800"
              >
                {previewFrame ? (
                  <img
                    src={previewFrame}
                    alt="Live surveillance camera"
                    className="w-full h-full object-contain group-hover:scale-[1.02] transition-transform duration-300"
                  />
                ) : (
                  <div className="p-8 text-center space-y-2">
                    <Radio className="w-8 h-8 text-emerald-400 animate-pulse mx-auto" />
                    <p className="text-xs text-zinc-400 font-mono">Streaming in progress...</p>
                  </div>
                )}

                {/* Overlaid Live Tag */}
                <div className="absolute top-3 left-3 flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded bg-zinc-950/80 backdrop-blur-xs border border-zinc-800 text-[11px] font-mono text-emerald-400 flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                    LIVE
                  </span>
                  <span className="px-2 py-0.5 rounded bg-zinc-950/80 backdrop-blur-xs border border-zinc-800 text-[11px] font-mono text-zinc-300 truncate max-w-[180px]">
                    {streamTitle}
                  </span>
                </div>

                {/* Overlaid Hover Hint */}
                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2 text-zinc-100 text-xs font-medium backdrop-blur-2xs">
                  <Maximize2 className="w-4 h-4" />
                  <span>Click to Expand Live Feed</span>
                </div>
              </div>

              {/* Side Telemetry Details */}
              <div className="lg:col-span-5 p-6 flex flex-col justify-between space-y-6">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <h3 className="text-sm font-semibold text-zinc-100 truncate">{streamTitle}</h3>
                      <p className="text-xs text-zinc-400 flex items-center gap-1 mt-0.5">
                        <span className="w-1.5 h-1.5 rounded-full bg-zinc-500 inline-block" />
                        {streamLocation}
                      </p>
                    </div>
                    <Badge variant={getRiskVariant(riskStatus)}>{riskStatus}</Badge>
                  </div>

                  <div className="grid grid-cols-2 gap-4 pt-2">
                    <div className="p-3 rounded bg-zinc-950 border border-zinc-800/80">
                      <span className="text-[11px] text-zinc-500 font-mono block">Estimated Crowd</span>
                      <span className="text-2xl font-bold font-mono text-zinc-100 mt-0.5 block">
                        {estCrowd}
                      </span>
                    </div>

                    <div className="p-3 rounded bg-zinc-950 border border-zinc-800/80">
                      <span className="text-[11px] text-zinc-500 font-mono block">Visible People</span>
                      <span className="text-2xl font-bold font-mono text-zinc-100 mt-0.5 block">
                        {visCount ?? '—'}
                      </span>
                    </div>

                    <div className="p-3 rounded bg-zinc-950 border border-zinc-800/80">
                      <span className="text-[11px] text-zinc-500 font-mono block">Density Score</span>
                      <span className="text-xl font-bold font-mono text-zinc-200 mt-0.5 block">
                        {density}
                      </span>
                    </div>

                    <div className="p-3 rounded bg-zinc-950 border border-zinc-800/80">
                      <span className="text-[11px] text-zinc-500 font-mono block">Processing Speed</span>
                      <span className="text-xl font-bold font-mono text-zinc-200 mt-0.5 block">
                        {fps ? `${fps} fps` : '—'}
                      </span>
                    </div>
                  </div>
                </div>

                <Button size="sm" onClick={() => onNavigate('feed')} className="w-full">
                  <Radio className="w-3.5 h-3.5 mr-1.5" />
                  Jump to Real-Time Feed
                </Button>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-8 rounded-lg bg-zinc-900/20 border border-zinc-800/80 text-center space-y-3">
            <p className="text-sm text-zinc-400">No active real-time feed session currently running.</p>
            <div className="flex justify-center gap-3 pt-1">
              <Button size="sm" onClick={() => onNavigate('feed')}>
                <Radio className="w-3.5 h-3.5 mr-1.5" />
                Launch Live Feed
              </Button>
              <Button size="sm" variant="outline" onClick={() => onNavigate('image')}>
                <ImageIcon className="w-3.5 h-3.5 mr-1.5" />
                Analyze Image
              </Button>
            </div>
          </div>
        )}
      </div>

      {/* Available Analysis Workflows */}
      <div className="space-y-3">
        <h2 className="text-xs font-mono font-medium text-zinc-400 uppercase tracking-wider">
          Analysis Capabilities
        </h2>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div
            onClick={() => onNavigate('image')}
            className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 hover:border-zinc-700 transition-colors cursor-pointer group space-y-2"
          >
            <div className="flex items-center justify-between text-zinc-300">
              <span className="text-sm font-medium flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-zinc-400 group-hover:text-zinc-200" />
                Image Analysis
              </span>
              <ArrowRight className="w-4 h-4 text-zinc-500 group-hover:text-zinc-300 group-hover:translate-x-0.5 transition-all" />
            </div>
            <p className="text-xs text-zinc-500 leading-relaxed">
              High-resolution DM-Count spatial density regression & keypoint localization.
            </p>
          </div>

          <div
            onClick={() => onNavigate('video')}
            className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 hover:border-zinc-700 transition-colors cursor-pointer group space-y-2"
          >
            <div className="flex items-center justify-between text-zinc-300">
              <span className="text-sm font-medium flex items-center gap-2">
                <Video className="w-4 h-4 text-zinc-400 group-hover:text-zinc-200" />
                Video Analysis
              </span>
              <ArrowRight className="w-4 h-4 text-zinc-500 group-hover:text-zinc-300 group-hover:translate-x-0.5 transition-all" />
            </div>
            <p className="text-xs text-zinc-500 leading-relaxed">
              Async frame processing, YOLOv8n bounding boxes & IoU tracking trails.
            </p>
          </div>

          <div
            onClick={() => onNavigate('analytics')}
            className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 hover:border-zinc-700 transition-colors cursor-pointer group space-y-2"
          >
            <div className="flex items-center justify-between text-zinc-300">
              <span className="text-sm font-medium flex items-center gap-2">
                <Activity className="w-4 h-4 text-zinc-400 group-hover:text-zinc-200" />
                Live Multi-Stream Wall
              </span>
              <ArrowRight className="w-4 h-4 text-zinc-500 group-hover:text-zinc-300 group-hover:translate-x-0.5 transition-all" />
            </div>
            <p className="text-xs text-zinc-500 leading-relaxed">
              Fleet-wide real-time analytics wall, temporal trend curves & zone density.
            </p>
          </div>
        </div>
      </div>

      {/* System Quick Details */}
      <div className="space-y-3 pt-2">
        <h2 className="text-xs font-mono font-medium text-zinc-400 uppercase tracking-wider">
          Runtime Specifications
        </h2>

        <div className="p-4 rounded-lg bg-zinc-900/30 border border-zinc-800/80 grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs">
          <div>
            <span className="text-zinc-500 block">Dense Model</span>
            <span className="text-zinc-200 font-medium font-mono">DM-Count (QNRF)</span>
          </div>
          <div>
            <span className="text-zinc-500 block">Person Detector</span>
            <span className="text-zinc-200 font-medium font-mono">YOLOv8n</span>
          </div>
          <div>
            <span className="text-zinc-500 block">Hardware Accelerator</span>
            <span className="text-zinc-200 font-medium font-mono">{status?.device || 'CPU / DirectML'}</span>
          </div>
          <div>
            <span className="text-zinc-500 block">WebSocket Protocol</span>
            <span className="text-zinc-200 font-medium font-mono">Active (/ws/feeds, /ws/system)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
