import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  Activity,
  Users,
  Eye,
  AlertTriangle,
  Radio,
  RefreshCw,
  Play,
  TrendingUp,
  MapPin,
  Camera,
  ShieldCheck,
  ShieldAlert,
  Maximize2,
} from 'lucide-react';
import type { LiveAnalyticsResponse } from '../types/api';
import { getLiveAnalytics } from '../lib/api';
import { useAuth } from '../context/AuthContext';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';

interface AnalyticsViewProps {
  onFocusFeed?: (feedId?: string, deviceId?: string) => void;
}

type CardViewMode = 'combined' | 'heatmap' | 'frame';

export const AnalyticsView: React.FC<AnalyticsViewProps> = ({ onFocusFeed }) => {
  const { token } = useAuth();
  const [data, setData] = useState<LiveAnalyticsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [cardModes, setCardModes] = useState<Record<string, CardViewMode>>({});
  const intervalRef = useRef<any>(null);

  const fetchAnalytics = async () => {
    try {
      const res = await getLiveAnalytics(token || undefined);
      setData(res);
    } catch (err) {
      console.error('Failed to fetch live analytics', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchAnalytics();
    // Auto poll live telemetry every 2 seconds
    intervalRef.current = setInterval(fetchAnalytics, 2000);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [token]);

  const handleManualRefresh = () => {
    setRefreshing(true);
    fetchAnalytics();
  };

  const setViewModeForStream = (streamId: string, mode: CardViewMode) => {
    setCardModes((prev) => ({ ...prev, [streamId]: mode }));
  };

  const getRiskColor = (status: string) => {
    switch (status.toUpperCase()) {
      case 'CRITICAL':
        return { text: 'text-rose-400', bg: 'bg-rose-950/60', border: 'border-rose-800/80', badge: 'critical' as const };
      case 'WARNING':
        return { text: 'text-amber-400', bg: 'bg-amber-950/60', border: 'border-amber-800/80', badge: 'warning' as const };
      case 'CAUTION':
        return { text: 'text-yellow-400', bg: 'bg-yellow-950/60', border: 'border-yellow-800/80', badge: 'caution' as const };
      default:
        return { text: 'text-emerald-400', bg: 'bg-emerald-950/60', border: 'border-emerald-800/80', badge: 'safe' as const };
    }
  };

  const riskInfo = useMemo(() => {
    return getRiskColor(data?.fleet_risk_status || 'SAFE');
  }, [data?.fleet_risk_status]);

  // Generate SVG points for dynamic temporal line chart
  const chartPoints = useMemo(() => {
    if (!data || !data.timeline || data.timeline.length < 2) return '';
    const points = data.timeline;
    const maxVal = Math.max(...points.map((p) => p.total_crowd), 20);
    const width = 600;
    const height = 140;

    return points
      .map((p, idx) => {
        const x = (idx / (points.length - 1)) * width;
        const y = height - (p.total_crowd / maxVal) * (height - 20) - 10;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }, [data]);

  return (
    <div className="p-8 max-w-7xl mx-auto space-y-8 animate-in fade-in duration-300">
      {/* Top Header Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 border-b border-zinc-800 pb-5">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold tracking-tight text-zinc-100 flex items-center gap-2">
              <Activity className="w-6 h-6 text-emerald-400" />
              Live Multi-Stream Analytics
            </h1>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono text-emerald-400 bg-emerald-950/60 border border-emerald-800/50">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
              LIVE TELEMETRY
            </span>
          </div>
          <p className="text-sm text-zinc-400 mt-1">
            Real-time cross-camera aggregation, spatial heatmaps, situational risk assessment, and surge alerts.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleManualRefresh}
            disabled={refreshing}
            className="flex items-center gap-1.5"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>

          {onFocusFeed && (
            <Button
              variant="default"
              size="sm"
              onClick={() => onFocusFeed()}
              className="flex items-center gap-1.5 bg-zinc-100 text-zinc-900 hover:bg-white"
            >
              <Radio className="w-3.5 h-3.5" />
              Open Live Viewer
            </Button>
          )}
        </div>
      </div>

      {/* Fleet Executive KPI Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        {/* Total Crowd */}
        <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-2 relative overflow-hidden">
          <div className="flex items-center justify-between text-xs text-zinc-400 font-medium">
            <span>Fleet Crowd Estimate</span>
            <Users className="w-4 h-4 text-zinc-400" />
          </div>
          <div className="text-3xl font-bold font-mono text-zinc-100 flex items-baseline gap-1">
            <span>{data ? data.total_crowd : 0}</span>
            <span className="text-xs font-normal text-zinc-500 font-sans">people</span>
          </div>
          <p className="text-[11px] text-zinc-500 flex items-center gap-1">
            <TrendingUp className="w-3 h-3 text-emerald-400" />
            Spatial density regression
          </p>
        </div>

        {/* Visible People */}
        <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-2">
          <div className="flex items-center justify-between text-xs text-zinc-400 font-medium">
            <span>Visible Detections</span>
            <Eye className="w-4 h-4 text-zinc-400" />
          </div>
          <div className="text-3xl font-bold font-mono text-zinc-200 flex items-baseline gap-1">
            <span>{data ? data.total_visible : 0}</span>
            <span className="text-xs font-normal text-zinc-500 font-sans">boxes</span>
          </div>
          <p className="text-[11px] text-zinc-500">Optical detector</p>
        </div>

        {/* Fleet Risk */}
        <div className={`p-5 rounded-lg border ${riskInfo.bg} ${riskInfo.border} space-y-2`}>
          <div className="flex items-center justify-between text-xs font-medium text-zinc-300">
            <span>Fleet Risk Status</span>
            {data?.fleet_risk_status === 'CRITICAL' || data?.fleet_risk_status === 'WARNING' ? (
              <ShieldAlert className={`w-4 h-4 ${riskInfo.text}`} />
            ) : (
              <ShieldCheck className={`w-4 h-4 ${riskInfo.text}`} />
            )}
          </div>
          <div className="text-2xl font-bold font-mono uppercase tracking-wider flex items-center gap-2">
            <span className={riskInfo.text}>{data?.fleet_risk_status || 'SAFE'}</span>
          </div>
          <p className="text-[11px] text-zinc-400">
            Score: {data?.fleet_risk_score ?? '0.00'} &bull; Multi-factor rule engine
          </p>
        </div>

        {/* Active Feeds vs Fleet */}
        <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-2">
          <div className="flex items-center justify-between text-xs text-zinc-400 font-medium">
            <span>Active Streams</span>
            <Radio className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-bold font-mono text-emerald-400 flex items-baseline gap-1.5">
            <span>{data ? data.active_cameras_count : 0}</span>
            <span className="text-xs font-normal text-zinc-500 font-sans">
              / {data?.total_cameras_count || 1} cameras
            </span>
          </div>
          <p className="text-[11px] text-zinc-500">Live AI inference pipelines</p>
        </div>

        {/* Fleet Peak Record */}
        <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-2 col-span-2 lg:col-span-1">
          <div className="flex items-center justify-between text-xs text-zinc-400 font-medium">
            <span>Peak Crowd Surge</span>
            <TrendingUp className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-3xl font-bold font-mono text-amber-400">
            {data ? data.peak_crowd : 0}
          </div>
          <p className="text-[11px] text-zinc-500">Session max concurrency</p>
        </div>
      </div>

      {/* Main Analytics Grid: Temporal Trend & Zone Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Real-time Temporal Trend Chart */}
        <div className="lg:col-span-8 p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-4">
          <div className="flex items-center justify-between">
            <div className="space-y-0.5">
              <h3 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-emerald-400" />
                Real-Time Temporal Crowd Trend
              </h3>
              <p className="text-xs text-zinc-500">Rolling aggregate crowd headcount over recent time intervals</p>
            </div>
            <Badge variant="outline" className="font-mono text-[11px]">
              {data?.timeline?.length || 0} Data Points
            </Badge>
          </div>

          {/* SVG Chart Container */}
          <div className="relative h-44 w-full bg-zinc-950/80 rounded border border-zinc-800/80 p-3 flex flex-col justify-between">
            {chartPoints ? (
              <svg
                viewBox="0 0 600 140"
                className="w-full h-full overflow-hidden"
                preserveAspectRatio="none"
              >
                <defs>
                  <linearGradient id="crowdGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#10b981" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="#10b981" stopOpacity="0.0" />
                  </linearGradient>
                </defs>

                {/* Grid guidelines */}
                <line x1="0" y1="35" x2="600" y2="35" stroke="#27272a" strokeDasharray="3 3" />
                <line x1="0" y1="70" x2="600" y2="70" stroke="#27272a" strokeDasharray="3 3" />
                <line x1="0" y1="105" x2="600" y2="105" stroke="#27272a" strokeDasharray="3 3" />

                {/* Area fill */}
                <polygon
                  points={`0,140 ${chartPoints} 600,140`}
                  fill="url(#crowdGradient)"
                />

                {/* Primary Trend Line */}
                <polyline
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  points={chartPoints}
                />

                {/* Pulsing Live Point on latest reading */}
                {data && data.timeline && data.timeline.length > 0 && (() => {
                  const latestVal = data.timeline[data.timeline.length - 1].total_crowd;
                  const maxVal = Math.max(...data.timeline.map((t) => t.total_crowd), 20);
                  const cy = 140 - (latestVal / maxVal) * 120 - 10;
                  return (
                    <g>
                      <circle cx="595" cy={cy} r="4" fill="#10b981" />
                      <circle cx="595" cy={cy} r="8" fill="#10b981" opacity="0.3">
                        <animate
                          attributeName="r"
                          values="4;9;4"
                          dur="1.8s"
                          repeatCount="indefinite"
                        />
                        <animate
                          attributeName="opacity"
                          values="0.5;0.1;0.5"
                          dur="1.8s"
                          repeatCount="indefinite"
                        />
                      </circle>
                    </g>
                  );
                })()}
              </svg>
            ) : (
              <div className="h-full flex items-center justify-center text-xs font-mono text-zinc-500">
                Awaiting telemetry frames to render surge line...
              </div>
            )}

            {/* X-Axis Labels */}
            <div className="flex items-center justify-between text-[10px] font-mono text-zinc-500 pt-1 border-t border-zinc-800/60">
              <span>-60s</span>
              <span>-45s</span>
              <span>-30s</span>
              <span>-15s</span>
              <span className="text-emerald-400 font-semibold">NOW</span>
            </div>
          </div>
        </div>

        {/* Spatial Zone Density Breakdown */}
        <div className="lg:col-span-4 p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-4">
          <div className="space-y-0.5">
            <h3 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
              <MapPin className="w-4 h-4 text-blue-400" />
              Zone Density Distribution
            </h3>
            <p className="text-xs text-zinc-500">Crowd concentration by location</p>
          </div>

          <div className="space-y-3">
            {data && data.zones && data.zones.length > 0 ? (
              data.zones.map((zone) => (
                <div key={zone.zone_name} className="space-y-1.5">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-zinc-300 font-medium truncate max-w-[140px]">
                      {zone.zone_name}
                    </span>
                    <div className="flex items-center gap-2 font-mono text-[11px]">
                      <span className="text-zinc-100 font-semibold">{zone.crowd_count} ppl</span>
                      <Badge variant={getRiskColor(zone.risk_status).badge} className="text-[9px] py-0 px-1">
                        {zone.risk_status}
                      </Badge>
                    </div>
                  </div>
                  {/* Progress Bar */}
                  <div className="h-1.5 w-full bg-zinc-950 rounded-full overflow-hidden border border-zinc-800/80">
                    <div
                      className={`h-full transition-all duration-500 rounded-full ${
                        zone.density_percent > 80
                          ? 'bg-rose-500'
                          : zone.density_percent > 50
                          ? 'bg-amber-500'
                          : 'bg-emerald-500'
                      }`}
                      style={{ width: `${Math.max(zone.density_percent, 6)}%` }}
                    />
                  </div>
                </div>
              ))
            ) : (
              <div className="p-6 text-center text-xs text-zinc-500 font-mono">
                No active operational zones detected
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Multi-Camera Live Wall / Heatmap Grid */}
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-lg font-bold text-zinc-100 flex items-center gap-2">
              <Camera className="w-5 h-5 text-zinc-300" />
              Multi-Stream Surveillance Wall
            </h2>
            <p className="text-xs text-zinc-400">
              Simultaneous visual telemetry, heatmaps, and spatial density across operational cameras.
            </p>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-xs text-zinc-500 font-mono">
              {data?.streams?.length || 0} active stream(s)
            </span>
          </div>
        </div>

        {loading ? (
          <div className="p-16 text-center text-zinc-500 font-mono text-xs flex items-center justify-center gap-2">
            <RefreshCw className="w-4 h-4 animate-spin" />
            Loading surveillance grid telemetry...
          </div>
        ) : !data || data.streams.length === 0 ? (
          <div className="p-10 text-center rounded-lg bg-zinc-900/20 border border-zinc-800/80 space-y-4">
            <Radio className="w-10 h-10 text-zinc-600 mx-auto" />
            <div className="space-y-1">
              <h4 className="text-sm font-semibold text-zinc-200">No Active Surveillance Feeds Running</h4>
              <p className="text-xs text-zinc-500 max-w-md mx-auto">
                All enrolled fleet cameras are currently on standby. Launch a real-time monitoring session to view live heatmaps and bounding box telemetry.
              </p>
            </div>
            {onFocusFeed && (
              <Button
                variant="default"
                size="sm"
                onClick={() => onFocusFeed()}
                className="bg-zinc-100 text-zinc-900 hover:bg-white inline-flex items-center gap-1.5"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                Start Live Feed Session
              </Button>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {data.streams.map((stream) => {
              const mode = cardModes[stream.feed_id] || 'combined';
              const risk = getRiskColor(stream.risk_status);

              // Determine image source based on selected mode
              const activeImage =
                mode === 'combined' && stream.combined_jpeg
                  ? stream.combined_jpeg
                  : mode === 'heatmap' && stream.heatmap_jpeg
                  ? stream.heatmap_jpeg
                  : stream.frame_jpeg;

              return (
                <div
                  key={stream.feed_id}
                  className="rounded-lg bg-zinc-900/40 border border-zinc-800 hover:border-zinc-700 transition-all overflow-hidden flex flex-col justify-between"
                >
                  {/* Card Header */}
                  <div className="p-3.5 border-b border-zinc-800 flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-1.5">
                        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                        <h4 className="text-sm font-semibold text-zinc-100 truncate max-w-[180px]">
                          {stream.camera_name}
                        </h4>
                      </div>
                      <p className="text-[11px] text-zinc-400 flex items-center gap-1 mt-0.5">
                        <MapPin className="w-3 h-3 text-zinc-500" />
                        {stream.location}
                      </p>
                    </div>

                    <div className="flex items-center gap-1.5">
                      <Badge variant={risk.badge} className="text-[10px] uppercase font-mono">
                        {stream.risk_status}
                      </Badge>
                    </div>
                  </div>

                  {/* Frame / Heatmap Viewport */}
                  <div className="relative bg-black min-h-[220px] max-h-[260px] flex items-center justify-center overflow-hidden group">
                    {activeImage ? (
                      <img
                        src={activeImage}
                        alt={`Live camera feed ${stream.camera_name}`}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="text-center p-6 space-y-2 text-zinc-500 font-mono text-xs">
                        <RefreshCw className="w-5 h-5 animate-spin mx-auto text-zinc-600" />
                        <span>Streaming inference frames...</span>
                      </div>
                    )}

                    {/* Overlay Mode Switcher */}
                    <div className="absolute top-2 right-2 flex items-center gap-1 bg-zinc-950/80 backdrop-blur-xs p-1 rounded border border-zinc-800 opacity-90 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={() => setViewModeForStream(stream.feed_id, 'frame')}
                        className={`px-2 py-0.5 text-[10px] font-mono rounded ${
                          mode === 'frame' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'
                        }`}
                        title="Live Video"
                      >
                        Video
                      </button>
                      <button
                        onClick={() => setViewModeForStream(stream.feed_id, 'heatmap')}
                        className={`px-2 py-0.5 text-[10px] font-mono rounded ${
                          mode === 'heatmap' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'
                        }`}
                        title="Heatmap"
                      >
                        Heatmap
                      </button>
                      <button
                        onClick={() => setViewModeForStream(stream.feed_id, 'combined')}
                        className={`px-2 py-0.5 text-[10px] font-mono rounded ${
                          mode === 'combined' ? 'bg-zinc-800 text-zinc-100' : 'text-zinc-400 hover:text-zinc-200'
                        }`}
                        title="Combined Overlay"
                      >
                        Combined
                      </button>
                    </div>

                    {/* Live FPS / Latency Badge */}
                    <div className="absolute bottom-2 left-2 bg-zinc-950/80 px-2 py-0.5 rounded text-[10px] font-mono text-zinc-300 border border-zinc-800/80">
                      {stream.fps || 30} FPS &bull; {stream.latency_ms || 80}ms
                    </div>
                  </div>

                  {/* Card Telemetry Footer */}
                  <div className="p-3.5 bg-zinc-950/50 border-t border-zinc-800/80 space-y-3">
                    <div className="grid grid-cols-3 gap-2 text-center font-mono">
                      <div className="p-1.5 rounded bg-zinc-900/60 border border-zinc-800/60">
                        <span className="text-[10px] text-zinc-500 block">Current</span>
                        <span className="text-base font-bold text-zinc-100">{stream.estimated_count}</span>
                      </div>
                      <div className="p-1.5 rounded bg-zinc-900/60 border border-zinc-800/60">
                        <span className="text-[10px] text-zinc-500 block">Visible</span>
                        <span className="text-base font-bold text-zinc-200">{stream.visible_count ?? '—'}</span>
                      </div>
                      <div className="p-1.5 rounded bg-zinc-900/60 border border-zinc-800/60">
                        <span className="text-[10px] text-zinc-500 block">Peak</span>
                        <span className="text-base font-bold text-amber-400">{stream.peak_count}</span>
                      </div>
                    </div>

                    {onFocusFeed && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => onFocusFeed(stream.feed_id, stream.device_id || undefined)}
                        className="w-full text-xs flex items-center justify-center gap-1.5 text-zinc-300 hover:text-zinc-100"
                      >
                        <Maximize2 className="w-3.5 h-3.5" />
                        Focus Full Feed
                      </Button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Real-Time Situational Alerts Log */}
      <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-4">
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <h3 className="text-sm font-semibold text-zinc-100 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
              Live Anomaly & Threshold Alerts Log
            </h3>
            <p className="text-xs text-zinc-500">Autonomous surge notifications and critical safety events</p>
          </div>
          <Badge variant="outline" className="font-mono text-xs">
            {data?.recent_alerts?.length || 0} Alerts Logged
          </Badge>
        </div>

        <div className="space-y-2">
          {data && data.recent_alerts && data.recent_alerts.length > 0 ? (
            data.recent_alerts.map((alert) => {
              const isCrit = alert.severity.toUpperCase() === 'CRITICAL';
              const isWarn = alert.severity.toUpperCase() === 'WARNING';

              return (
                <div
                  key={alert.id}
                  className={`p-3 rounded-md border flex items-center justify-between gap-4 text-xs font-mono transition-colors ${
                    isCrit
                      ? 'bg-rose-950/40 border-rose-800/60 text-rose-200'
                      : isWarn
                      ? 'bg-amber-950/40 border-amber-800/60 text-amber-200'
                      : 'bg-zinc-950/60 border-zinc-800 text-zinc-300'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <span
                      className={`w-2 h-2 rounded-full ${
                        isCrit ? 'bg-rose-400 animate-ping' : isWarn ? 'bg-amber-400' : 'bg-emerald-400'
                      }`}
                    />
                    <span className="text-zinc-500">{alert.timestamp}</span>
                    <span className="font-semibold text-zinc-100">[{alert.camera_name}]</span>
                    <span className="text-zinc-400">{alert.message}</span>
                  </div>

                  <div className="flex items-center gap-2 shrink-0">
                    <span className="text-zinc-500">{alert.zone}</span>
                    <Badge
                      variant={isCrit ? 'critical' : isWarn ? 'warning' : 'safe'}
                      className="text-[9px] py-0 px-1.5 uppercase"
                    >
                      {alert.severity}
                    </Badge>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="p-4 text-center text-xs text-zinc-500 font-mono">
              No recent anomalies detected. Fleet is operating within normal density parameters.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
