export interface ClusterSummary {
  n_clusters: number;
  n_noise?: number;
  largest_cluster_size: number;
  average_cluster_size: number;
}

export interface RiskSummary {
  status: 'SAFE' | 'CAUTION' | 'WARNING' | 'CRITICAL' | string;
  score: number;
  explanation: string;
  components?: Record<string, number | null>;
}

export interface DetectionBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
}

export interface ImageVisualizations {
  detection?: string | null;
  density?: string | null;
  heatmap?: string | null;
  localization?: string | null;
  combined?: string | null;
}

export interface ImageAnalysisResult {
  count: number;
  visible_count?: number | null;
  detections?: DetectionBox[];
  model_name: string;
  model_weights: string;
  device: string;
  latency_ms: number;
  detector_latency_ms?: number | null;
  n_points: number;
  clusters: ClusterSummary;
  risk: RiskSummary;
  images?: ImageVisualizations | null;
  s3_urls?: Record<string, string> | null;
  warnings?: string[];
}

export interface VideoFrameMetrics {
  frame_number: number;
  total_frames?: number | null;
  // Wall-clock processing time since the job started -- progress/ETA only,
  // NOT the video's own timestamp (see video_time_s for that; the two can
  // legitimately diverge a lot on a slow run).
  elapsed_s: number;
  // The video's own playback position for this frame, and its total
  // duration if known -- this is what the burned-in overlay's "Time" line
  // shows, and the ONLY field the UI should use to display "Time" so it
  // never disagrees with the video.
  video_time_s?: number;
  total_duration_s?: number | null;
  visible_count?: number | null;
  estimated_count: number;
  peak_count: number;
  density: number;
  cluster_count: number;
  largest_cluster: number;
  risk_status: string;
  risk_score: number;
  latency_ms: number;
  fps?: number | null;
  density_latency_ms?: number | null;
  frames_dropped?: number;
  frame_jpeg?: string | null;
  heatmap_jpeg?: string | null;
  combined_jpeg?: string | null;
  timestamp_s?: number;
}

// A real-time feed's `feed.progress` messages are partial: the fast
// (detector) loop and slow (DM-Count) loop each broadcast only the fields
// they own (see backend/app/routers/feeds.py), so the frontend must MERGE
// incoming messages into its metrics state rather than replacing it --
// otherwise a fast-path update would wipe out the last known
// estimated_count/heatmap, and vice versa.
export type PartialFeedMetrics = Partial<VideoFrameMetrics>;

export interface VideoJobSummary {
  frames_processed: number;
  total_frames?: number | null;
  elapsed_s: number;
  peak_count: number;
  final_metrics: VideoFrameMetrics;
}

export interface JobRecordResponse {
  job_id: string;
  job_type: 'image' | 'video';
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at?: number;
  started_at?: number | null;
  completed_at?: number | null;
  progress: number;
  source_description?: string;
  message?: string | null;
  result?: ImageAnalysisResult | null;
  video_result?: VideoJobSummary | null;
  current_frame: number;
  total_frames?: number | null;
  peak_count: number;
  latest_metrics?: VideoFrameMetrics | null;
  error?: string | null;
}

export interface FeedResponse {
  feed_id: string;
  source_type: string;
  status: 'starting' | 'running' | 'stopping' | 'stopped' | 'failed';
  frames_produced: number;
  is_recording?: boolean;
  latest_metrics?: VideoFrameMetrics | null;
  error?: string | null;
  description?: string;
}

export interface RecordingSummary {
  feed_id: string;
  session_uid: string;
  frame_count: number;
  duration_s: number;
  s3_key?: string | null;
  download_url?: string | null;
  is_finalized: boolean;
}

export interface RecordingStatusResponse {
  feed_id: string;
  recording: boolean;
  session_uid?: string | null;
  duration_s: number;
  frames: number;
}

export interface SystemHealth {
  status: string;
  model_loaded: boolean;
  uptime_seconds?: number;
}

export interface SystemStatus {
  status?: string;
  model_loaded?: boolean;
  device?: string;
  model_name?: string;
  model_weights?: string;
  fps?: number;
  latency_ms?: number;
  errors?: string[];
}

export interface SystemConfig {
  model_name: string;
  model_weights: string;
  device: string;
  max_side?: number;
  eps_px?: number;
  min_samples?: number;
  risk_config?: Record<string, any>;
}

export interface WSEnvelope {
  type: string;
  job_id?: string;
  channel?: string;
  timestamp?: number;
  data: any;
}

export interface User {
  id: number;
  username: string;
  email: string;
  full_name?: string;
  role: 'admin' | 'user';
  is_active: boolean;
  created_at: string;
  last_login?: string | null;
}

export interface AuthTokenResponse {
  access_token: string;
  token_type: string;
  username: string;
  role: 'admin' | 'user';
  full_name?: string;
}

export interface ActivityLog {
  id: number;
  user_id?: number | null;
  username: string;
  action: string;
  category: string;
  details?: string;
  severity: 'INFO' | 'WARNING' | 'CRITICAL';
  timestamp: string;
}

export interface AdminOverviewStats {
  total_users: number;
  active_users: number;
  admin_users: number;
  total_activities: number;
  database_engine?: string;
  database_status?: string;
  recent_activities: ActivityLog[];
}

export interface Device {
  id: number;
  device_id: string;
  name: string;
  description?: string;
  location: string;
  stream_type: 'RTSP' | 'WEBCAM' | 'HTTP_HLS' | 'VIDEO_FILE' | 'YOUTUBE' | string;
  stream_url: string;
  status: 'ONLINE' | 'STANDBY' | 'OFFLINE' | 'MAINTENANCE' | 'ERROR' | string;
  fps?: number | null;
  resolution?: string;
  created_by?: number | null;
  created_at: string;
  updated_at: string;
}

export interface DeviceCreatePayload {
  device_id: string;
  name: string;
  description?: string;
  location: string;
  stream_type: string;
  stream_url: string;
  status?: string;
  fps?: number | null;
  resolution?: string;
}

export interface DeviceUpdatePayload {
  name?: string;
  description?: string;
  location?: string;
  stream_type?: string;
  stream_url?: string;
  status?: string;
  fps?: number | null;
  resolution?: string;
}

export interface StreamProbeResult {
  is_reachable: boolean;
  resolution?: string | null;
  fps?: number | null;
  latency_ms?: number | null;
  error?: string | null;
}

export interface StreamAnalyticsSnapshot {
  feed_id: string;
  device_id?: string | null;
  camera_name: string;
  location: string;
  status: string;
  is_live: boolean;
  estimated_count: number;
  visible_count?: number | null;
  peak_count: number;
  risk_status: string;
  risk_score: number;
  fps?: number | null;
  latency_ms?: number | null;
  frame_jpeg?: string | null;
  heatmap_jpeg?: string | null;
  combined_jpeg?: string | null;
}

export interface ZoneDistributionMetric {
  zone_name: string;
  camera_count: number;
  crowd_count: number;
  risk_status: string;
  density_percent: number;
}

export interface TimelineDataPoint {
  timestamp: string;
  total_crowd: number;
  visible_count: number;
  active_streams: number;
}

export interface AnalyticsAlert {
  id: string;
  timestamp: string;
  zone: string;
  camera_name: string;
  severity: 'INFO' | 'WARNING' | 'CRITICAL';
  message: string;
  crowd_count: number;
}

export interface LiveAnalyticsResponse {
  total_crowd: number;
  total_visible: number;
  peak_crowd: number;
  fleet_risk_status: string;
  fleet_risk_score: number;
  active_cameras_count: number;
  total_cameras_count: number;
  streams: StreamAnalyticsSnapshot[];
  zones: ZoneDistributionMetric[];
  timeline: TimelineDataPoint[];
  recent_alerts: AnalyticsAlert[];
  updated_at: string;
}


