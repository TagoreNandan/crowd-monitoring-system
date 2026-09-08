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
  latest_metrics?: VideoFrameMetrics | null;
  error?: string | null;
  description?: string;
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
