import React, { useState, useEffect } from 'react';
import { RefreshCw } from 'lucide-react';
import { getHealth, getStatus, getConfig } from '../lib/api';
import type { SystemConfig, SystemHealth, SystemStatus } from '../types/api';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';

export const SystemView: React.FC = () => {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [config, setConfig] = useState<SystemConfig | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const fetchSystemInfo = async () => {
    setLoading(true);
    try {
      const [h, s, c] = await Promise.all([
        getHealth().catch(() => null),
        getStatus().catch(() => null),
        getConfig().catch(() => null),
      ]);
      setHealth(h);
      setStatus(s);
      setConfig(c);
    } catch (err: any) {
      console.error('Failed to query system status telemetry');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemInfo();
  }, []);

  const isHealthy = health?.status === 'ok' && health?.model_loaded;

  return (
    <div className="p-8 max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-zinc-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-100">System Status</h1>
          <p className="text-sm text-zinc-400 mt-0.5">
            Backend runtime diagnostics and PyTorch configuration.
          </p>
        </div>

        <Button variant="outline" size="sm" onClick={fetchSystemInfo} isLoading={loading}>
          <RefreshCw className="w-3.5 h-3.5 mr-1.5" />
          Refresh
        </Button>
      </div>

      {/* Clean Table / Status List */}
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/30 overflow-hidden divide-y divide-zinc-800/60 font-mono text-xs">
        <div className="p-4 flex items-center justify-between">
          <span className="text-zinc-400">FastAPI Core Backend</span>
          <Badge variant={isHealthy ? 'safe' : 'critical'}>
            {isHealthy ? 'Online' : 'Offline'}
          </Badge>
        </div>

        <div className="p-4 flex items-center justify-between">
          <span className="text-zinc-400">DM-Count Model Checkpoint</span>
          <span className="text-zinc-200">{config?.model_weights || 'weights/dm_count_qnrf.pth'}</span>
        </div>

        <div className="p-4 flex items-center justify-between">
          <span className="text-zinc-400">YOLOv8n Person Detector</span>
          <span className="text-zinc-200">weights/yolov8n.pt</span>
        </div>

        <div className="p-4 flex items-center justify-between">
          <span className="text-zinc-400">PyTorch Acceleration Device</span>
          <span className="text-zinc-200">{status?.device || config?.device || 'Apple MPS'}</span>
        </div>

        <div className="p-4 flex items-center justify-between">
          <span className="text-zinc-400">DBSCAN Spatial Cluster Radius</span>
          <span className="text-zinc-200">{config?.eps_px || 30} px</span>
        </div>

        <div className="p-4 flex items-center justify-between">
          <span className="text-zinc-400">WebSocket Channels</span>
          <span className="text-emerald-400 font-medium">Active (/ws/system, /ws/jobs, /ws/feeds)</span>
        </div>
      </div>
    </div>
  );
};
