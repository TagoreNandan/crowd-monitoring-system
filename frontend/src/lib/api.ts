import type {
  FeedResponse,
  JobRecordResponse,
  SystemConfig,
  SystemHealth,
  SystemStatus,
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
