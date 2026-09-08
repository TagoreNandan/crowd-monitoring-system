import React, { useState } from 'react';
import { Upload, RefreshCw, AlertCircle, Sparkles } from 'lucide-react';
import { analyzeImage } from '../lib/api';
import type { JobRecordResponse } from '../types/api';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';

type Tab = 'original' | 'detection' | 'density' | 'heatmap' | 'localization' | 'combined';

const TABS: { key: Tab; label: string }[] = [
  { key: 'original', label: 'Original' },
  { key: 'detection', label: 'Person Detection' },
  { key: 'density', label: 'Density Map' },
  { key: 'heatmap', label: 'Heatmap Overlay' },
  { key: 'localization', label: 'Localization' },
  { key: 'combined', label: 'Combined' },
];

export const ImageView: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [jobResult, setJobResult] = useState<JobRecordResponse | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>('original');

  const resetSelection = (file: File) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setJobResult(null);
    setError(null);
    setActiveTab('original');
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) resetSelection(e.target.files[0]);
  };

  const handleDragOver = (e: React.DragEvent) => e.preventDefault();

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) resetSelection(e.dataTransfer.files[0]);
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);

    try {
      const response = await analyzeImage(selectedFile);
      if (response.status === 'failed') {
        setError(response.error || 'Image analysis failed');
      } else {
        setJobResult(response);
        setActiveTab('detection');
      }
    } catch (err: any) {
      setError(err.message || 'Image analysis failed');
    } finally {
      setLoading(false);
    }
  };

  const result = jobResult?.result;
  const images = result?.images;
  const riskStatus = result?.risk?.status || 'SAFE';
  const hasDetector = result ? result.visible_count !== undefined && result.visible_count !== null : false;
  const avgConfidence =
    result && result.detections && result.detections.length > 0
      ? result.detections.reduce((acc, d) => acc + d.confidence, 0) / result.detections.length
      : null;

  const getRiskVariant = (st: string) => {
    switch (st?.toUpperCase()) {
      case 'CRITICAL': return 'critical';
      case 'WARNING': return 'warning';
      case 'CAUTION': return 'caution';
      default: return 'safe';
    }
  };

  const tabImageSrc = (): string | null => {
    if (!previewUrl) return null;
    if (activeTab === 'original') return previewUrl;
    if (!images) return null;
    return images[activeTab] ?? null;
  };

  const currentSrc = tabImageSrc();

  return (
    <div className="p-8 max-w-6xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-zinc-100">Image Analysis</h1>
          <p className="text-sm text-zinc-400 mt-0.5">
            Person detection (YOLOv8) + dense crowd estimation (DM-Count) with separate visual outputs.
          </p>
        </div>

        {selectedFile && (
          <Button onClick={runAnalysis} isLoading={loading}>
            <Sparkles className="w-3.5 h-3.5 mr-1.5" />
            Analyze Image
          </Button>
        )}
      </div>

      {!previewUrl ? (
        <div
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          className="border border-dashed border-zinc-800 hover:border-zinc-700 transition-colors rounded-xl p-16 flex flex-col items-center justify-center text-center cursor-pointer bg-zinc-950/40 group"
        >
          <div className="w-10 h-10 rounded-full bg-zinc-900 border border-zinc-800 flex items-center justify-center mb-3 group-hover:scale-105 transition-transform text-zinc-400">
            <Upload className="w-5 h-5" />
          </div>
          <h3 className="text-sm font-semibold text-zinc-200">Upload Image for Crowd Analysis</h3>
          <p className="text-xs text-zinc-500 mt-1 max-w-xs">
            Drag and drop JPEG, PNG, or WEBP (e.g. sample <code className="text-zinc-300">examples/crowd.jpeg</code>)
          </p>
          <label className="mt-4 px-3.5 py-1.5 bg-zinc-900 hover:bg-zinc-800 text-zinc-200 text-xs font-medium rounded-md border border-zinc-800 cursor-pointer transition-colors">
            Browse File
            <input type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
          </label>
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <div className="lg:col-span-8 space-y-4">
            {jobResult && (
              <div className="flex items-center gap-1 border-b border-zinc-800 pb-2 text-xs flex-wrap">
                {TABS.map((tab) => {
                  const unavailable = tab.key !== 'original' && !images?.[tab.key];
                  return (
                    <button
                      key={tab.key}
                      onClick={() => !unavailable && setActiveTab(tab.key)}
                      disabled={unavailable}
                      title={unavailable ? 'Not available for this image' : undefined}
                      className={`px-3 py-1 rounded-md transition-colors ${
                        activeTab === tab.key
                          ? 'bg-zinc-900 text-zinc-100 border border-zinc-800 font-medium'
                          : unavailable
                          ? 'text-zinc-700 cursor-not-allowed'
                          : 'text-zinc-400 hover:text-zinc-200'
                      }`}
                    >
                      {tab.label}
                    </button>
                  );
                })}
              </div>
            )}

            <div className="relative rounded-lg border border-zinc-800 bg-black min-h-[420px] flex items-center justify-center overflow-hidden">
              {currentSrc ? (
                <img src={currentSrc} alt={activeTab} className="max-h-[550px] w-auto object-contain" />
              ) : (
                <div className="text-xs text-zinc-500 font-mono px-6 text-center">
                  {activeTab === 'detection' && !hasDetector
                    ? 'Person detector unavailable for this run — visible_count is N/A.'
                    : 'View not available.'}
                </div>
              )}

              {loading && (
                <div className="absolute inset-0 bg-zinc-950/80 backdrop-blur-xs flex flex-col items-center justify-center gap-2">
                  <RefreshCw className="w-6 h-6 text-zinc-400 animate-spin" />
                  <span className="text-xs text-zinc-400 font-mono">Running YOLOv8 + DM-Count inference...</span>
                </div>
              )}
            </div>
          </div>

          {/* Analytics Panel */}
          <div className="lg:col-span-4 space-y-6">
            <div className="grid grid-cols-2 gap-3">
              <div className="p-4 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
                <span className="text-xs text-zinc-400 font-medium">Visible People</span>
                <div className="text-2xl font-semibold font-mono text-zinc-100">
                  {hasDetector ? result?.visible_count : 'N/A'}
                </div>
              </div>
              <div className="p-4 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-1">
                <span className="text-xs text-zinc-400 font-medium">Estimated Crowd</span>
                <div className="text-2xl font-semibold font-mono text-zinc-100">{result?.count ?? '—'}</div>
              </div>
            </div>
            <p className="text-[11px] text-zinc-500 -mt-4">
              Visible People (YOLOv8 boxes) and Estimated Crowd (DM-Count density) are separate signals — a dense,
              occluded crowd can validly show visible ≪ estimated.
            </p>

            <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-3">
              <span className="text-xs font-mono font-medium text-zinc-400 uppercase tracking-wider block">
                Analysis Breakdown
              </span>

              <div className="space-y-2.5 text-xs font-mono">
                <Row label="Localization Points" value={result ? result.n_points : '—'} />
                <Row label="Density Score" value={result ? result.risk.components?.relative_density_per_100px2 ?? '—' : '—'} />
                <Row label="DBSCAN Clusters" value={result ? result.clusters.n_clusters : '—'} />
                <Row label="Largest Cluster" value={result ? result.clusters.largest_cluster_size : '—'} />
                <Row
                  label="Detection Confidence"
                  value={avgConfidence !== null ? `${(avgConfidence * 100).toFixed(0)}%` : hasDetector ? '—' : 'N/A'}
                />
                <Row label="DM-Count Latency" value={result ? `${result.latency_ms} ms` : '—'} />
                <Row
                  label="Detector Latency"
                  value={result?.detector_latency_ms != null ? `${result.detector_latency_ms.toFixed(0)} ms` : 'N/A'}
                />
                <Row label="Device" value={result?.device ?? '—'} last />
              </div>
            </div>

            <div className="p-5 rounded-lg bg-zinc-900/40 border border-zinc-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs text-zinc-400 font-medium">Risk Status</span>
                {result && <Badge variant={getRiskVariant(riskStatus)}>{riskStatus}</Badge>}
              </div>

              <div className="text-xs font-mono text-zinc-400 space-y-1">
                <div className="flex justify-between">
                  <span>Risk Score:</span>
                  <span className="text-zinc-200 font-bold">{result ? `${result.risk.score.toFixed(1)} / 100` : '—'}</span>
                </div>
                {result?.risk?.explanation && (
                  <p className="text-zinc-500 font-sans text-[11px] mt-1 leading-relaxed">{result.risk.explanation}</p>
                )}
              </div>
            </div>

            {error && (
              <div className="p-4 rounded-lg bg-rose-950/40 border border-rose-800/60 text-xs text-rose-300 flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                <div>{error}</div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

const Row: React.FC<{ label: string; value: React.ReactNode; last?: boolean }> = ({ label, value, last }) => (
  <div className={`flex justify-between py-1 ${last ? '' : 'border-b border-zinc-800/60'}`}>
    <span className="text-zinc-400">{label}</span>
    <span className="text-zinc-200 font-medium">{value}</span>
  </div>
);
