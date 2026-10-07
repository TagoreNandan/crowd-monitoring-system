import React, { useState, useEffect, useMemo } from 'react';
import {
  Camera,
  Plus,
  RefreshCw,
  Play,
  Square,
  CheckCircle2,
  AlertCircle,
  Trash2,
  Edit3,
  Search,
  Wifi,
  WifiOff,
  Radio,
  X,
} from 'lucide-react';
import type { Device, DeviceCreatePayload, StreamProbeResult, FeedResponse } from '../types/api';
import { getDevices, createDevice, updateDevice, deleteDevice, testDevice, probeStream, listFeeds, createFeed, stopFeed } from '../lib/api';
import { useAuth } from '../context/AuthContext';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';

interface DeviceManagementViewProps {
  onSelectDeviceForFeed?: (device: Device) => void;
}

export const DeviceManagementView: React.FC<DeviceManagementViewProps> = ({ onSelectDeviceForFeed }) => {
  const { isAdmin, token } = useAuth();
  const [devices, setDevices] = useState<Device[]>([]);
  const [activeFeeds, setActiveFeeds] = useState<FeedResponse[]>([]);
  const [streamActionLoading, setStreamActionLoading] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [statusFilter, setStatusFilter] = useState<string>('ALL');
  const [protocolFilter, setProtocolFilter] = useState<string>('ALL');

  // Testing stream state
  const [testingDeviceId, setTestingDeviceId] = useState<string | null>(null);
  const [testResult, setTestResult] = useState<{ deviceId: string; result: StreamProbeResult } | null>(null);

  // Modal State
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [editingDevice, setEditingDevice] = useState<Device | null>(null);
  const [modalForm, setModalForm] = useState<DeviceCreatePayload>({
    device_id: '',
    name: '',
    description: '',
    location: '',
    stream_type: 'RTSP',
    stream_url: '',
    status: 'ONLINE',
    fps: 30,
    resolution: '1920x1080',
  });
  const [probingModalStream, setProbingModalStream] = useState<boolean>(false);
  const [modalProbeResult, setModalProbeResult] = useState<StreamProbeResult | null>(null);
  const [modalError, setModalError] = useState<string | null>(null);
  const [modalSaving, setModalSaving] = useState<boolean>(false);

  // Load devices
  const loadDevices = async () => {
    try {
      const data = await getDevices(
        {
          status: statusFilter,
          stream_type: protocolFilter,
          search: searchQuery,
        },
        token || undefined
      );
      setDevices(data);
    } catch (err) {
      console.error('Failed to load camera fleet:', err);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const loadActiveFeeds = async () => {
    try {
      const feeds = await listFeeds(token || undefined);
      setActiveFeeds(feeds);
    } catch (err) {
      console.error('Failed to load active feeds:', err);
    }
  };

  useEffect(() => {
    loadDevices();
    loadActiveFeeds();
    const interval = setInterval(loadActiveFeeds, 4000);
    return () => clearInterval(interval);
  }, [statusFilter, protocolFilter, searchQuery, token]);

  const handleRefresh = () => {
    setRefreshing(true);
    loadDevices();
    loadActiveFeeds();
  };

  const handleToggleStream = async (device: Device) => {
    const activeFeed = activeFeeds.find((f) => f.device_id === device.device_id);
    setStreamActionLoading(device.device_id);
    try {
      if (activeFeed) {
        await stopFeed(activeFeed.feed_id, token || undefined);
      } else {
        let sourceType = 'rtsp';
        if (device.stream_type.toUpperCase() === 'WEBCAM') sourceType = 'camera';
        else if (device.stream_type.toUpperCase() === 'VIDEO_FILE') sourceType = 'sample';
        await createFeed(sourceType, device.stream_url, true, token || undefined, device.device_id, device.name);
      }
      await loadActiveFeeds();
    } catch (err) {
      console.error('Failed to toggle stream for device:', err);
    } finally {
      setStreamActionLoading(null);
    }
  };

  // Quick connectivity test
  const handleTestDevice = async (deviceId: string) => {
    setTestingDeviceId(deviceId);
    setTestResult(null);
    try {
      const result = await testDevice(deviceId, token || undefined);
      setTestResult({ deviceId, result });
      // Reload devices to update status badge
      loadDevices();
    } catch (err: any) {
      setTestResult({
        deviceId,
        result: {
          is_reachable: false,
          error: err.message || 'Stream test request failed',
        },
      });
    } finally {
      setTestingDeviceId(null);
    }
  };

  // Delete Device
  const handleDeleteDevice = async (deviceId: string, name: string) => {
    if (!window.confirm(`Are you sure you want to decommission camera "${name}" (${deviceId})?`)) {
      return;
    }
    try {
      await deleteDevice(deviceId, token || undefined);
      setDevices((prev) => prev.filter((d) => d.device_id !== deviceId));
    } catch (err: any) {
      alert(`Failed to delete device: ${err.message}`);
    }
  };

  // Open Modal for New Device
  const handleOpenCreateModal = () => {
    setEditingDevice(null);
    setModalForm({
      device_id: `CAM-${Math.floor(100 + Math.random() * 900)}`,
      name: '',
      description: '',
      location: '',
      stream_type: 'RTSP',
      stream_url: '',
      status: 'ONLINE',
      fps: 30,
      resolution: '1920x1080',
    });
    setModalProbeResult(null);
    setModalError(null);
    setIsModalOpen(true);
  };

  // Open Modal for Edit
  const handleOpenEditModal = (dev: Device) => {
    setEditingDevice(dev);
    setModalForm({
      device_id: dev.device_id,
      name: dev.name,
      description: dev.description || '',
      location: dev.location,
      stream_type: dev.stream_type,
      stream_url: dev.stream_url,
      status: dev.status,
      fps: dev.fps || 30,
      resolution: dev.resolution || '',
    });
    setModalProbeResult(null);
    setModalError(null);
    setIsModalOpen(true);
  };

  // Live probe inside modal
  const handleProbeModalStream = async () => {
    if (!modalForm.stream_url) {
      setModalError('Please enter a stream URL or device index first.');
      return;
    }
    setProbingModalStream(true);
    setModalProbeResult(null);
    setModalError(null);
    try {
      const probe = await probeStream(
        {
          stream_url: modalForm.stream_url,
          stream_type: modalForm.stream_type,
        },
        token || undefined
      );
      setModalProbeResult(probe);
      if (probe.is_reachable) {
        if (probe.resolution) {
          setModalForm((prev) => ({ ...prev, resolution: probe.resolution || prev.resolution }));
        }
        if (probe.fps) {
          setModalForm((prev) => ({ ...prev, fps: probe.fps || prev.fps }));
        }
      }
    } catch (err: any) {
      setModalProbeResult({
        is_reachable: false,
        error: err.message || 'Stream unreachable',
      });
    } finally {
      setProbingModalStream(false);
    }
  };

  // Save Modal (Create or Edit)
  const handleSaveModal = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!modalForm.name.trim() || !modalForm.location.trim() || !modalForm.stream_url.trim()) {
      setModalError('Please fill in all required fields (Name, Location, Stream URL).');
      return;
    }

    setModalSaving(true);
    setModalError(null);
    try {
      if (editingDevice) {
        const updated = await updateDevice(
          editingDevice.device_id,
          {
            name: modalForm.name,
            description: modalForm.description,
            location: modalForm.location,
            stream_type: modalForm.stream_type,
            stream_url: modalForm.stream_url,
            status: modalForm.status,
            fps: modalForm.fps,
            resolution: modalForm.resolution,
          },
          token || undefined
        );
        setDevices((prev) => prev.map((d) => (d.device_id === updated.device_id ? updated : d)));
      } else {
        const created = await createDevice(modalForm, token || undefined);
        setDevices((prev) => [created, ...prev]);
      }
      setIsModalOpen(false);
    } catch (err: any) {
      setModalError(err.message || 'Failed to save camera device.');
    } finally {
      setModalSaving(false);
    }
  };

  // Telemetry KPIs
  const totalCount = devices.length;
  const onlineCount = useMemo(() => devices.filter((d) => d.status.toUpperCase() === 'ONLINE').length, [devices]);
  const standbyCount = useMemo(
    () => devices.filter((d) => ['STANDBY', 'OFFLINE', 'MAINTENANCE', 'ERROR'].includes(d.status.toUpperCase())).length,
    [devices]
  );
  const rtspCount = useMemo(() => devices.filter((d) => d.stream_type.toUpperCase() === 'RTSP').length, [devices]);

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-8 animate-in fade-in duration-300">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight text-zinc-100">Device Fleet Management</h1>
            <Badge variant="outline" className="text-xs font-mono">
              {totalCount} Enrolled
            </Badge>
          </div>
          <p className="text-sm text-zinc-400 mt-1">
            Onboard, probe connectivity, and manage multi-camera surveillance streams across operational zones.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={refreshing}
            className="flex items-center gap-1.5"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>

          {isAdmin && (
            <Button
              variant="default"
              size="sm"
              onClick={handleOpenCreateModal}
              className="flex items-center gap-1.5 bg-zinc-100 text-zinc-900 hover:bg-white"
            >
              <Plus className="w-4 h-4" />
              Onboard Device
            </Button>
          )}
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="p-4 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
          <div className="flex items-center justify-between text-xs text-zinc-500">
            <span>Total Cameras</span>
            <Camera className="w-4 h-4 text-zinc-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-zinc-100">{totalCount}</div>
          <div className="text-[11px] text-zinc-500">Enrolled fleet devices</div>
        </div>

        <div className="p-4 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
          <div className="flex items-center justify-between text-xs text-zinc-500">
            <span>Active Streams</span>
            <Wifi className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-emerald-400">{onlineCount}</div>
          <div className="text-[11px] text-emerald-500/80">Online & reachable</div>
        </div>

        <div className="p-4 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
          <div className="flex items-center justify-between text-xs text-zinc-500">
            <span>Standby / Offline</span>
            <WifiOff className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-amber-400">{standbyCount}</div>
          <div className="text-[11px] text-zinc-500">Awaiting stream connection</div>
        </div>

        <div className="p-4 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
          <div className="flex items-center justify-between text-xs text-zinc-500">
            <span>RTSP Protocols</span>
            <Radio className="w-4 h-4 text-blue-400" />
          </div>
          <div className="text-2xl font-bold font-mono text-zinc-100">{rtspCount}</div>
          <div className="text-[11px] text-zinc-500">Network IP feeds</div>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 p-3 rounded-lg bg-zinc-900/40 border border-zinc-800">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-zinc-500 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search by ID, name, or location..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 text-xs bg-zinc-950/60 border border-zinc-800/80 rounded-md text-zinc-200 placeholder-zinc-500 focus:outline-none focus:border-zinc-600 font-mono"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          {/* Status Filter */}
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            aria-label="Filter devices by operational status"
            className="px-2.5 py-1.5 text-xs bg-zinc-950/60 border border-zinc-800/80 rounded-md text-zinc-300 focus:outline-none focus:border-zinc-600 font-mono"
          >
            <option value="ALL">All Statuses</option>
            <option value="ONLINE">Online</option>
            <option value="STANDBY">Standby</option>
            <option value="OFFLINE">Offline</option>
          </select>

          {/* Protocol Filter */}
          <select
            value={protocolFilter}
            onChange={(e) => setProtocolFilter(e.target.value)}
            aria-label="Filter devices by stream protocol"
            className="px-2.5 py-1.5 text-xs bg-zinc-950/60 border border-zinc-800/80 rounded-md text-zinc-300 focus:outline-none focus:border-zinc-600 font-mono"
          >
            <option value="ALL">All Protocols</option>
            <option value="YOUTUBE">YouTube Live / Video</option>
            <option value="RTSP">RTSP</option>
            <option value="WEBCAM">Webcam</option>
            <option value="HTTP_HLS">HTTP / HLS</option>
            <option value="VIDEO_FILE">Video File</option>
          </select>
        </div>
      </div>

      {/* Device Grid */}
      {loading ? (
        <div className="p-12 text-center text-zinc-500 font-mono text-xs flex items-center justify-center gap-2">
          <RefreshCw className="w-4 h-4 animate-spin" />
          Loading camera fleet...
        </div>
      ) : devices.length === 0 ? (
        <div className="p-12 text-center rounded-lg bg-zinc-900/20 border border-zinc-800/60 space-y-3">
          <Camera className="w-8 h-8 text-zinc-600 mx-auto" />
          <div className="text-sm font-medium text-zinc-300">No camera devices found</div>
          <p className="text-xs text-zinc-500 max-w-sm mx-auto">
            {isAdmin
              ? 'Get started by onboarding your first RTSP, Webcam, or video stream device.'
              : 'No cameras matching your criteria are currently active.'}
          </p>
          {isAdmin && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleOpenCreateModal}
              className="mt-2"
            >
              <Plus className="w-3.5 h-3.5 mr-1" />
              Onboard Device
            </Button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {devices.map((device) => {
            const isOnline = device.status.toUpperCase() === 'ONLINE';
            const isTesting = testingDeviceId === device.device_id;
            const currentTest = testResult?.deviceId === device.device_id ? testResult.result : null;
            const activeFeed = activeFeeds.find((f) => f.device_id === device.device_id);
            const isStreaming = !!activeFeed && activeFeed.status !== 'stopped';
            const isActionLoading = streamActionLoading === device.device_id;

            return (
              <div
                key={device.id}
                className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 hover:border-zinc-700 transition-colors flex flex-col justify-between space-y-4"
              >
                {/* Device Header */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-semibold text-zinc-400 bg-zinc-950 px-2 py-0.5 rounded border border-zinc-800">
                      {device.device_id}
                    </span>
                    <div className="flex items-center gap-1.5">
                      {isStreaming && (
                        <span className="flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-800/80 animate-pulse">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block" />
                          LIVE
                        </span>
                      )}
                      <Badge
                        variant={isOnline ? 'safe' : 'secondary'}
                        className="text-[10px] font-mono uppercase"
                      >
                        {device.status}
                      </Badge>
                      <Badge variant="outline" className="text-[10px] font-mono">
                        {device.stream_type}
                      </Badge>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-base font-semibold text-zinc-100 truncate">{device.name}</h3>
                    <p className="text-xs text-zinc-400 flex items-center gap-1 mt-0.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-zinc-500 inline-block" />
                      {device.location}
                    </p>
                  </div>

                  {device.description && (
                    <p className="text-xs text-zinc-500 line-clamp-2">{device.description}</p>
                  )}
                </div>

                {/* Device Details */}
                <div className="space-y-2 pt-2 border-t border-zinc-800/80 font-mono text-[11px] text-zinc-400">
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-500">Resolution</span>
                    <span>{device.resolution || 'Auto / Undetected'}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-500">FPS</span>
                    <span>{device.fps ? `${device.fps} fps` : 'Variable'}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-zinc-500">Endpoint</span>
                    <span className="truncate max-w-[170px] text-zinc-400" title={device.stream_url}>
                      {device.stream_url}
                    </span>
                  </div>

                  {/* Connectivity Test Feedback */}
                  {currentTest && (
                    <div
                      className={`p-2 rounded mt-2 text-[11px] ${
                        currentTest.is_reachable
                          ? 'bg-emerald-950/40 text-emerald-300 border border-emerald-800/60'
                          : 'bg-red-950/40 text-red-300 border border-red-800/60'
                      }`}
                    >
                      {currentTest.is_reachable ? (
                        <div className="flex items-center gap-1.5">
                          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                          <span>
                            Reachable ({currentTest.latency_ms}ms) &bull; {currentTest.resolution}
                          </span>
                        </div>
                      ) : (
                        <div className="flex items-center gap-1.5">
                          <AlertCircle className="w-3.5 h-3.5 text-red-400 shrink-0" />
                          <span className="truncate">{currentTest.error || 'Connection timed out'}</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div className="flex items-center justify-between pt-2 border-t border-zinc-800/80 gap-2">
                  <div className="flex items-center gap-1.5">
                    <Button
                      variant={isStreaming ? 'outline' : 'default'}
                      size="sm"
                      onClick={() => handleToggleStream(device)}
                      disabled={isActionLoading}
                      className={`text-xs flex items-center gap-1 font-medium cursor-pointer ${
                        isStreaming
                          ? 'text-red-400 border-red-800/80 hover:bg-red-950/40 hover:text-red-300'
                          : 'bg-emerald-600 hover:bg-emerald-500 text-white'
                      }`}
                      title={isStreaming ? 'Stop Live Stream' : 'Start Live Stream'}
                    >
                      {isActionLoading ? (
                        <RefreshCw className="w-3 h-3 animate-spin" />
                      ) : isStreaming ? (
                        <Square className="w-3 h-3 fill-current" />
                      ) : (
                        <Play className="w-3 h-3 fill-current" />
                      )}
                      <span>{isStreaming ? 'Stop' : 'Stream'}</span>
                    </Button>

                    {onSelectDeviceForFeed && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onSelectDeviceForFeed(device)}
                        className="text-xs text-zinc-300 hover:text-white hover:bg-zinc-800 flex items-center gap-1 cursor-pointer"
                        title="Open in Real-Time Feed"
                      >
                        <Radio className="w-3 h-3 text-cyan-400" />
                        <span>View</span>
                      </Button>
                    )}

                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleTestDevice(device.device_id)}
                      disabled={isTesting}
                      className="text-xs flex items-center gap-1 text-zinc-300"
                      title="Probe camera stream"
                    >
                      <Wifi className={`w-3 h-3 ${isTesting ? 'animate-pulse text-amber-400' : ''}`} />
                      {isTesting ? 'Probing...' : 'Test'}
                    </Button>
                  </div>

                  {isAdmin && (
                    <div className="flex items-center gap-1.5">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleOpenEditModal(device)}
                        className="text-zinc-300 hover:text-white hover:bg-zinc-800 border-zinc-800 h-8 w-8 p-0 flex items-center justify-center cursor-pointer"
                        title="Edit Camera Details"
                      >
                        <Edit3 className="w-3.5 h-3.5 text-zinc-300 hover:text-cyan-400 transition-colors" />
                      </Button>

                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => handleDeleteDevice(device.device_id, device.name)}
                        className="text-zinc-400 hover:text-rose-300 hover:bg-rose-950/40 border-zinc-800 hover:border-rose-900/60 h-8 w-8 p-0 flex items-center justify-center cursor-pointer"
                        title="Decommission Camera"
                      >
                        <Trash2 className="w-3.5 h-3.5 transition-colors" />
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Onboard / Edit Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-xs p-4 animate-in fade-in duration-200">
          <div className="bg-zinc-950 border border-zinc-800 rounded-lg max-w-lg w-full p-6 shadow-2xl space-y-5">
            <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
              <div className="flex items-center gap-2">
                <Camera className="w-4 h-4 text-zinc-400" />
                <h3 className="text-base font-semibold text-zinc-100">
                  {editingDevice ? 'Edit Camera Configuration' : 'Onboard New Camera Device'}
                </h3>
              </div>
              <button
                onClick={() => setIsModalOpen(false)}
                className="text-zinc-400 hover:text-zinc-200"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {modalError && (
              <div className="p-3 bg-red-950/50 border border-red-800/80 rounded-md text-red-300 text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{modalError}</span>
              </div>
            )}

            <form onSubmit={handleSaveModal} className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs font-mono text-zinc-400 block">Device Identifier *</label>
                  <input
                    type="text"
                    required
                    disabled={editingDevice !== null}
                    value={modalForm.device_id}
                    onChange={(e) => setModalForm({ ...modalForm, device_id: e.target.value.toUpperCase() })}
                    placeholder="CAM-01"
                    className="w-full px-3 py-1.5 text-xs font-mono bg-zinc-900 border border-zinc-800 rounded-md text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-zinc-600 disabled:opacity-50"
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-mono text-zinc-400 block">Camera Name *</label>
                  <input
                    type="text"
                    required
                    value={modalForm.name}
                    onChange={(e) => setModalForm({ ...modalForm, name: e.target.value })}
                    placeholder="North Gate Turnstiles"
                    className="w-full px-3 py-1.5 text-xs bg-zinc-900 border border-zinc-800 rounded-md text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-zinc-600"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs font-mono text-zinc-400 block">Location / Zone *</label>
                  <input
                    type="text"
                    required
                    value={modalForm.location}
                    onChange={(e) => setModalForm({ ...modalForm, location: e.target.value })}
                    placeholder="Concourse Gate 3"
                    className="w-full px-3 py-1.5 text-xs bg-zinc-900 border border-zinc-800 rounded-md text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-zinc-600"
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-mono text-zinc-400 block">Protocol Type *</label>
                  <select
                    value={modalForm.stream_type}
                    onChange={(e) => setModalForm({ ...modalForm, stream_type: e.target.value })}
                    className="w-full px-3 py-1.5 text-xs font-mono bg-zinc-900 border border-zinc-800 rounded-md text-zinc-100 focus:outline-none focus:border-zinc-600"
                  >
                    <option value="RTSP">RTSP (Network IP Camera)</option>
                    <option value="YOUTUBE">YouTube Live / Video Stream</option>
                    <option value="WEBCAM">Webcam / USB Capture</option>
                    <option value="HTTP_HLS">HTTP / HLS Stream</option>
                    <option value="VIDEO_FILE">Local Video File / Sample</option>
                  </select>
                </div>
              </div>

              {/* Stream URL + Probe button */}
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-mono text-zinc-400 block">
                    Stream URL / Device Index *
                  </label>
                  <button
                    type="button"
                    onClick={handleProbeModalStream}
                    disabled={probingModalStream || !modalForm.stream_url}
                    className="text-[11px] text-zinc-400 hover:text-zinc-100 flex items-center gap-1 font-mono disabled:opacity-40"
                  >
                    <Wifi className={`w-3 h-3 ${probingModalStream ? 'animate-pulse text-amber-400' : ''}`} />
                    {probingModalStream ? 'Testing stream...' : 'Test Stream Probe'}
                  </button>
                </div>
                <input
                  type="text"
                  required
                  value={modalForm.stream_url}
                  onChange={(e) => setModalForm({ ...modalForm, stream_url: e.target.value })}
                  placeholder={
                    modalForm.stream_type === 'YOUTUBE'
                      ? 'https://www.youtube.com/watch?v=...'
                      : modalForm.stream_type === 'WEBCAM'
                      ? '0 (Device Index)'
                      : modalForm.stream_type === 'VIDEO_FILE'
                      ? 'examples/sample_crowd.mp4'
                      : 'rtsp://admin:pass@192.168.1.100:554/stream1'
                  }
                  className="w-full px-3 py-1.5 text-xs font-mono bg-zinc-900 border border-zinc-800 rounded-md text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-zinc-600"
                />

                {/* Stream Probe Feedback */}
                {modalProbeResult && (
                  <div
                    className={`p-2 rounded text-xs font-mono flex items-center gap-2 mt-1.5 ${
                      modalProbeResult.is_reachable
                        ? 'bg-emerald-950/40 text-emerald-300 border border-emerald-800/60'
                        : 'bg-red-950/40 text-red-300 border border-red-800/60'
                    }`}
                  >
                    {modalProbeResult.is_reachable ? (
                      <>
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                        <span>
                          Stream Verified! Resolution: {modalProbeResult.resolution || 'Auto'}, FPS: {modalProbeResult.fps || 30} ({modalProbeResult.latency_ms}ms)
                        </span>
                      </>
                    ) : (
                      <>
                        <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
                        <span>{modalProbeResult.error || 'Connection failed'}</span>
                      </>
                    )}
                  </div>
                )}
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1">
                  <label className="text-xs font-mono text-zinc-400 block">Initial Status</label>
                  <select
                    value={modalForm.status}
                    onChange={(e) => setModalForm({ ...modalForm, status: e.target.value })}
                    className="w-full px-3 py-1.5 text-xs font-mono bg-zinc-900 border border-zinc-800 rounded-md text-zinc-100 focus:outline-none focus:border-zinc-600"
                  >
                    <option value="ONLINE">ONLINE (Active)</option>
                    <option value="STANDBY">STANDBY (Awaiting)</option>
                    <option value="OFFLINE">OFFLINE</option>
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-mono text-zinc-400 block">Resolution</label>
                  <input
                    type="text"
                    value={modalForm.resolution}
                    onChange={(e) => setModalForm({ ...modalForm, resolution: e.target.value })}
                    placeholder="1920x1080"
                    className="w-full px-3 py-1.5 text-xs font-mono bg-zinc-900 border border-zinc-800 rounded-md text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-zinc-600"
                  />
                </div>
              </div>

              <div className="space-y-1">
                <label className="text-xs font-mono text-zinc-400 block">Description / Notes</label>
                <textarea
                  rows={2}
                  value={modalForm.description}
                  onChange={(e) => setModalForm({ ...modalForm, description: e.target.value })}
                  placeholder="Optional camera installation details, coverage angle, or zone notes."
                  className="w-full px-3 py-1.5 text-xs bg-zinc-900 border border-zinc-800 rounded-md text-zinc-100 placeholder-zinc-500 focus:outline-none focus:border-zinc-600 resize-none"
                />
              </div>

              <div className="flex items-center justify-end gap-2 pt-3 border-t border-zinc-800">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => setIsModalOpen(false)}
                  disabled={modalSaving}
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  variant="default"
                  size="sm"
                  disabled={modalSaving}
                  className="bg-zinc-100 text-zinc-900 hover:bg-white"
                >
                  {modalSaving ? 'Saving...' : editingDevice ? 'Update Camera' : 'Enroll Camera'}
                </Button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
