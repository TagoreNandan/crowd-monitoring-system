import React from 'react';
import { ArrowRight, Image as ImageIcon, Video, Radio } from 'lucide-react';
import type { SystemHealth, SystemStatus, VideoFrameMetrics } from '../types/api';
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
  const isHealthy = health?.status === 'ok' && health?.model_loaded;
  const hasActiveTelemetry = latestMetrics !== null;

  return (
    <div className="p-8 max-w-5xl mx-auto space-y-10">
      {/* Header section */}
      <div className="space-y-1">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight text-zinc-100">Overview</h1>
          <Badge variant={isHealthy ? 'safe' : 'secondary'}>
            {isHealthy ? 'System Operational' : 'Initializing'}
          </Badge>
        </div>
        <p className="text-sm text-zinc-400">
          Real-time crowd estimation, spatial density analysis, and safety monitoring.
        </p>
      </div>

      {/* Current Monitoring State */}
      <div className="space-y-3">
        <h2 className="text-xs font-mono font-medium text-zinc-400 uppercase tracking-wider">
          Current Live Stream
        </h2>

        {hasActiveTelemetry ? (
          <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs text-emerald-400 font-mono flex items-center gap-1.5 font-medium">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                Active Feed Telemetry
              </span>
              <Badge variant="outline">{latestMetrics?.risk_status || 'SAFE'}</Badge>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div>
                <span className="text-xs text-zinc-500 block">Estimated Crowd</span>
                <span className="text-2xl font-semibold font-mono text-zinc-100">{latestMetrics.estimated_count}</span>
              </div>
              <div>
                <span className="text-xs text-zinc-500 block">Visible People</span>
                <span className="text-2xl font-semibold font-mono text-zinc-100">{latestMetrics.visible_count ?? 'N/A'}</span>
              </div>
              <div>
                <span className="text-xs text-zinc-500 block">Density Score</span>
                <span className="text-2xl font-semibold font-mono text-zinc-100">{latestMetrics.density}</span>
              </div>
              <div>
                <span className="text-xs text-zinc-500 block">Processing FPS</span>
                <span className="text-2xl font-semibold font-mono text-zinc-100">{latestMetrics.fps ? `${latestMetrics.fps}` : '—'}</span>
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
            onClick={() => onNavigate('feed')}
            className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 hover:border-zinc-700 transition-colors cursor-pointer group space-y-2"
          >
            <div className="flex items-center justify-between text-zinc-300">
              <span className="text-sm font-medium flex items-center gap-2">
                <Radio className="w-4 h-4 text-zinc-400 group-hover:text-zinc-200" />
                Real-Time Feed
              </span>
              <ArrowRight className="w-4 h-4 text-zinc-500 group-hover:text-zinc-300 group-hover:translate-x-0.5 transition-all" />
            </div>
            <p className="text-xs text-zinc-500 leading-relaxed">
              Live WebSocket telemetry stream for IP cameras, RTSP streams, or sample videos.
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
            <span className="text-zinc-200 font-medium font-mono">{status?.device || 'Apple MPS'}</span>
          </div>
          <div>
            <span className="text-zinc-500 block">WebSocket Protocol</span>
            <span className="text-zinc-200 font-medium font-mono">Active (/ws)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
