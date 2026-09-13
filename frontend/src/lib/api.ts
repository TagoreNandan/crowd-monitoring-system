import type {
  ActivityLog,
  AdminOverviewStats,
  AuthTokenResponse,
  Device,
  DeviceCreatePayload,
  DeviceUpdatePayload,
  FeedResponse,
  JobRecordResponse,
  LiveAnalyticsResponse,
  StreamProbeResult,
  SystemConfig,
  SystemHealth,
  SystemStatus,
  User,
  WSEnvelope,
} from '../types/api';

const API_BASE = '/api';

export async function getHealth(): Promise<SystemHealth> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function getStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/status`);
  if (!res.ok) throw new Error(`Status check failed: ${res.statusText}`);
  return res.json();
}

export async function getConfig(): Promise<SystemConfig> {
  const res = await fetch(`${API_BASE}/config`);
  if (!res.ok) throw new Error(`Config fetch failed: ${res.statusText}`);
  return res.json();
}

export async function analyzeImage(file: File): Promise<JobRecordResponse> {
  const formData = new FormData();
  formData.append('image', file);

  const res = await fetch(`${API_BASE}/jobs/image`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error?.message || errData.message || `Image analysis failed (${res.status})`);
  }

  return res.json();
}

export async function submitVideoJob(file: File): Promise<JobRecordResponse> {
  const formData = new FormData();
  formData.append('video', file);

  const res = await fetch(`${API_BASE}/jobs/video`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error?.message || errData.message || `Video submission failed (${res.status})`);
  }

  return res.json();
}

export async function getJob(jobId: string): Promise<JobRecordResponse> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Failed to fetch job ${jobId}`);
  return res.json();
}

export type VideoView = 'detection' | 'heatmap' | 'combined';

// URL of a real output video view for a completed video job: `detection`
// (person boxes + stats panel), `heatmap` (real DM-Count density overlay,
// no boxes), or `combined` (both together). Only valid once the job has
// completed -- callers should check `job.status === 'completed'` first.
export function getJobVideoUrl(jobId: string, view: VideoView = 'detection'): string {
  return `${API_BASE}/jobs/${jobId}/video/${view}`;
}

export async function createFeed(
  sourceType: string = 'video',
  sourceUri: string = 'dashboard/static/uploads/077cc45b94_14968235_2160_3840_60fps.mp4',
  loop: boolean = true
): Promise<FeedResponse> {
  const res = await fetch(`${API_BASE}/feeds`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      source_type: sourceType,
      source_uri: sourceUri,
      loop: loop,
    }),
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error?.message || errData.message || `Feed creation failed (${res.status})`);
  }

  return res.json();
}

export async function uploadFeedVideo(
  file: File,
  loop: boolean = true
): Promise<FeedResponse> {
  const formData = new FormData();
  formData.append('video', file);
  formData.append('loop', String(loop));

  const res = await fetch(`${API_BASE}/feeds/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.error?.message || errData.message || `Feed video upload failed (${res.status})`);
  }

  return res.json();
}

export async function getFeed(feedId: string): Promise<FeedResponse> {
  const res = await fetch(`${API_BASE}/feeds/${feedId}`);
  if (!res.ok) throw new Error(`Failed to fetch feed ${feedId}`);
  return res.json();
}

export async function stopFeed(feedId: string): Promise<FeedResponse> {
  const res = await fetch(`${API_BASE}/feeds/${feedId}/stop`, {
    method: 'POST',
  });

  if (!res.ok) throw new Error(`Failed to stop feed ${feedId}`);
  return res.json();
}

// WebSocket Helper
export function connectWebSocket(
  path: string,
  onMessage: (msg: WSEnvelope) => void,
  onStatusChange?: (connected: boolean) => void
): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = window.location.host;
  const wsUrl = `${protocol}//${host}${path}`;

  const ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    onStatusChange?.(true);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error('WebSocket parse error:', e);
    }
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    onStatusChange?.(false);
  };

  ws.onclose = () => {
    onStatusChange?.(false);
  };

  return ws;
}

/* =========================================================================
   Authentication & Admin API
   ========================================================================= */

const TOKEN_KEY = 'cms_auth_token';

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setStoredToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function removeStoredToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

export async function loginUser(credentials: { username: string; password: string }): Promise<AuthTokenResponse> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(credentials),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Login failed');
  }

  const data: AuthTokenResponse = await res.json();
  setStoredToken(data.access_token);
  return data;
}

export async function registerUser(payload: {
  username: string;
  email: string;
  password: string;
  full_name?: string;
}): Promise<User> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Registration failed');
  }

  return res.json();
}

export async function getCurrentUser(token?: string): Promise<User> {
  const authToken = token || getStoredToken();
  if (!authToken) throw new Error('No authentication token found');

  const res = await fetch(`${API_BASE}/auth/me`, {
    headers: { Authorization: `Bearer ${authToken}` },
  });

  if (!res.ok) {
    throw new Error('Failed to authenticate session');
  }

  return res.json();
}

export async function getAdminStats(token?: string): Promise<AdminOverviewStats> {
  const authToken = token || getStoredToken();
  const res = await fetch(`${API_BASE}/admin/stats`, {
    headers: { Authorization: `Bearer ${authToken}` },
  });

  if (!res.ok) throw new Error('Failed to load admin stats');
  return res.json();
}

export async function getAdminUsers(token?: string): Promise<User[]> {
  const authToken = token || getStoredToken();
  const res = await fetch(`${API_BASE}/admin/users`, {
    headers: { Authorization: `Bearer ${authToken}` },
  });

  if (!res.ok) throw new Error('Failed to load user directory');
  return res.json();
}

export async function updateAdminUser(
  userId: number,
  updates: { is_active?: boolean; role?: string },
  token?: string
): Promise<User> {
  const authToken = token || getStoredToken();
  const res = await fetch(`${API_BASE}/admin/users/${userId}`, {
    method: 'PATCH',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${authToken}`,
    },
    body: JSON.stringify(updates),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to update user');
  }

  return res.json();
}

export async function getAdminActivities(limit = 50, token?: string): Promise<ActivityLog[]> {
  const authToken = token || getStoredToken();
  const res = await fetch(`${API_BASE}/admin/activities?limit=${limit}`, {
    headers: { Authorization: `Bearer ${authToken}` },
  });

  if (!res.ok) throw new Error('Failed to load system activities');
  return res.json();
}

export async function getDevices(
  filters?: { status?: string; stream_type?: string; search?: string },
  token?: string
): Promise<Device[]> {
  const authToken = token || getStoredToken();
  const params = new URLSearchParams();
  if (filters?.status && filters.status !== 'ALL') params.append('status', filters.status);
  if (filters?.stream_type && filters.stream_type !== 'ALL') params.append('stream_type', filters.stream_type);
  if (filters?.search) params.append('search', filters.search);

  const query = params.toString() ? `?${params.toString()}` : '';
  const res = await fetch(`${API_BASE}/devices${query}`, {
    headers: { Authorization: `Bearer ${authToken}` },
  });

  if (!res.ok) throw new Error('Failed to fetch devices');
  return res.json();
}

export async function getDevice(deviceId: string, token?: string): Promise<Device> {
  const authToken = token || getStoredToken();
  const res = await fetch(`${API_BASE}/devices/${encodeURIComponent(deviceId)}`, {
    headers: { Authorization: `Bearer ${authToken}` },
  });

  if (!res.ok) throw new Error(`Device ${deviceId} not found`);
  return res.json();
}

export async function createDevice(payload: DeviceCreatePayload, token?: string): Promise<Device> {
  const authToken = token || getStoredToken();
  const res = await fetch(`${API_BASE}/devices`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${authToken}`,
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to enroll device');
  }

  return res.json();
}

export async function updateDevice(
  deviceId: string,
  payload: DeviceUpdatePayload,
  token?: string
): Promise<Device> {
  const authToken = token || getStoredToken();
  const res = await fetch(`${API_BASE}/devices/${encodeURIComponent(deviceId)}`, {
    method: 'PUT',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${authToken}`,
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to update device');
  }

  return res.json();
}

export async function deleteDevice(deviceId: string, token?: string): Promise<{ message: string; device_id: string }> {
  const authToken = token || getStoredToken();
  const res = await fetch(`${API_BASE}/devices/${encodeURIComponent(deviceId)}`, {
    method: 'DELETE',
    headers: { Authorization: `Bearer ${authToken}` },
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to delete device');
  }

  return res.json();
}

export async function probeStream(
  payload: { stream_url: string; stream_type: string },
  token?: string
): Promise<StreamProbeResult> {
  const authToken = token || getStoredToken();
  const res = await fetch(`${API_BASE}/devices/probe`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${authToken}`,
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Stream probe failed');
  }

  return res.json();
}

export async function testDevice(deviceId: string, token?: string): Promise<StreamProbeResult> {
  const authToken = token || getStoredToken();
  const res = await fetch(`${API_BASE}/devices/${encodeURIComponent(deviceId)}/test`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${authToken}` },
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Device test failed');
  }

  return res.json();
}

export async function getLiveAnalytics(token?: string): Promise<LiveAnalyticsResponse> {
  const authToken = token || getStoredToken();
  const res = await fetch(`${API_BASE}/analytics/live`, {
    headers: { Authorization: `Bearer ${authToken}` },
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Failed to load live analytics');
  }

  return res.json();
}


