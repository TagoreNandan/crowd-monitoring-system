import React, { useEffect, useState } from 'react';
import { ChevronLeft, ChevronRight, Download, Filter, RefreshCw, Search } from 'lucide-react';
import type { ActivityListItem } from '../types/api';
import { ActivityDownloadError, getActivities, requestActivityDownload } from '../lib/api';
import { useAuth } from '../context/AuthContext';
import { Badge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Card, CardContent } from '../components/ui/Card';

const PAGE_SIZE = 50;
const categories = ['AUTH', 'INFERENCE', 'DOWNLOAD', 'SECURITY', 'SYSTEM', 'DEVICES', 'GENERAL'];
const severities = ['INFO', 'WARNING', 'CRITICAL'];

type DownloadState = { id: number; view: string } | null;

const formatTimestamp = (timestamp: string): string => {
  const iso = timestamp.endsWith('Z') || timestamp.includes('+') ? timestamp : `${timestamp}Z`;
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return timestamp;
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
    hour12: true,
  });
};

const severityBadge = (severity: string) => {
  const variant = severity === 'CRITICAL' ? 'critical' : severity === 'WARNING' ? 'warning' : severity === 'INFO' ? 'safe' : 'outline';
  return <Badge variant={variant}>{severity}</Badge>;
};

export const ActivityLogView: React.FC = () => {
  const { token, isAdmin } = useAuth();
  const [items, setItems] = useState<ActivityListItem[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [category, setCategory] = useState('');
  const [severity, setSeverity] = useState('');
  const [action, setAction] = useState('');
  const [username, setUsername] = useState('');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [downloadState, setDownloadState] = useState<DownloadState>(null);
  const [selectedViews, setSelectedViews] = useState<Record<number, string>>({});
  const [refreshNonce, setRefreshNonce] = useState(0);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);

    getActivities(
      {
        limit: PAGE_SIZE,
        offset,
        category: category || undefined,
        severity: severity || undefined,
        action: action || undefined,
        username: isAdmin ? username || undefined : undefined,
      },
      token || undefined
    )
      .then((response) => {
        if (!active) return;
        setItems(response.items);
        setTotal(response.total);
      })
      .catch((err: Error) => {
        if (active) setError(err.message || 'Failed to load activity log');
      })
      .finally(() => {
        if (active) {
          setLoading(false);
          setRefreshing(false);
        }
      });

    return () => {
      active = false;
    };
  }, [action, category, isAdmin, offset, refreshNonce, severity, token, username]);

  const resetPage = (setter: (value: string) => void, value: string) => {
    setter(value);
    setOffset(0);
  };

  const handleRefresh = () => {
    setRefreshing(true);
    setRefreshNonce((current) => current + 1);
  };

  const handleDownload = async (item: ActivityListItem) => {
    const view = selectedViews[item.id] || item.available_views[0];
    if (!view) return;

    setDownloadError(null);
    setDownloadState({ id: item.id, view });
    try {
      const download = await requestActivityDownload(item.id, view, token || undefined);
      let downloadUrl = download.url;
      const currentToken = token || getStoredToken();
      if (currentToken && (downloadUrl.startsWith('/api') || !downloadUrl.startsWith('http'))) {
        const sep = downloadUrl.includes('?') ? '&' : '?';
        downloadUrl = `${downloadUrl}${sep}token=${encodeURIComponent(currentToken)}`;
      }
      const anchor = document.createElement('a');
      anchor.href = downloadUrl;
      anchor.download = download.filename;
      anchor.target = '_blank';
      anchor.rel = 'noopener noreferrer';
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
    } catch (err) {
      const status = err instanceof ActivityDownloadError ? err.status : 0;
      setDownloadError(
        status === 404
          ? 'Output not found'
          : status === 410
            ? 'This output has expired or was deleted'
            : 'Unable to prepare the download'
      );
    } finally {
      setDownloadState(null);
    }
  };

  return (
    <div className="p-5 sm:p-8 max-w-[1500px] mx-auto space-y-6 animate-in fade-in duration-150">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-zinc-800 pb-5">
        <div>
          <p className="text-[11px] font-mono uppercase tracking-[0.18em] text-zinc-500">Audit telemetry</p>
          <h1 className="mt-1 text-2xl font-semibold tracking-tight text-zinc-100">Activity Log</h1>
          <p className="mt-1 text-sm text-zinc-400">Analysis history and available output files.</p>
        </div>
        <Button variant="outline" size="sm" onClick={handleRefresh} disabled={loading}>
          <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      <Card>
        <CardContent className="p-3">
          <div className="flex flex-wrap items-center gap-2">
            <Filter className="h-3.5 w-3.5 text-zinc-500" />
            <select
              value={category}
              onChange={(event) => resetPage(setCategory, event.target.value)}
              className="rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-300 focus:outline-none focus:ring-1 focus:ring-zinc-400"
              aria-label="Filter by category"
            >
              <option value="">All categories</option>
              {categories.map((value) => <option key={value} value={value}>{value}</option>)}
            </select>
            <select
              value={severity}
              onChange={(event) => resetPage(setSeverity, event.target.value)}
              className="rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-300 focus:outline-none focus:ring-1 focus:ring-zinc-400"
              aria-label="Filter by severity"
            >
              <option value="">All severities</option>
              {severities.map((value) => <option key={value} value={value}>{value}</option>)}
            </select>
            {isAdmin && (
              <input
                value={username}
                onChange={(event) => resetPage(setUsername, event.target.value)}
                placeholder="Filter username"
                className="w-36 rounded-md border border-zinc-800 bg-zinc-950 px-2.5 py-1.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-400"
                aria-label="Filter by username"
              />
            )}
            <div className="relative min-w-[190px] flex-1 sm:max-w-xs">
              <Search className="absolute left-2.5 top-2 h-3.5 w-3.5 text-zinc-600" />
              <input
                value={action}
                onChange={(event) => resetPage(setAction, event.target.value)}
                placeholder="Search action"
                className="w-full rounded-md border border-zinc-800 bg-zinc-950 py-1.5 pl-8 pr-2.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-400"
                aria-label="Search actions"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {error && <div className="rounded-md border border-rose-900/60 bg-rose-950/30 px-4 py-3 text-sm text-rose-300">{error}</div>}
      {downloadError && <div className="rounded-md border border-amber-900/60 bg-amber-950/30 px-4 py-3 text-sm text-amber-300">{downloadError}</div>}

      <Card>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[900px] text-left text-xs text-zinc-300">
            <thead className="border-b border-zinc-800 bg-zinc-900/70 text-[10px] uppercase tracking-wider text-zinc-500">
              <tr>
                <th className="px-4 py-3 font-medium">Time</th>
                {isAdmin && <th className="px-4 py-3 font-medium">User</th>}
                <th className="px-4 py-3 font-medium">Action</th>
                <th className="px-4 py-3 font-medium">Category</th>
                <th className="px-4 py-3 font-medium">Severity</th>
                <th className="px-4 py-3 font-medium">Details</th>
                <th className="px-4 py-3 text-right font-medium">Output</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-800/60">
              {loading ? (
                <tr><td colSpan={isAdmin ? 7 : 6} className="px-4 py-12 text-center text-zinc-500">Loading activity...</td></tr>
              ) : items.length === 0 ? (
                <tr><td colSpan={isAdmin ? 7 : 6} className="px-4 py-12 text-center text-zinc-500">No activity records found.</td></tr>
              ) : items.map((item) => {
                const view = selectedViews[item.id] || item.available_views[0] || '';
                const isDownloading = downloadState?.id === item.id;
                return (
                  <tr key={item.id} className="align-middle hover:bg-zinc-800/20">
                    <td className="whitespace-nowrap px-4 py-3 font-mono text-zinc-500">{formatTimestamp(item.timestamp)}</td>
                    {isAdmin && <td className="px-4 py-3 font-medium text-zinc-200">{item.username}</td>}
                    <td className="px-4 py-3 font-mono text-zinc-100">{item.action}</td>
                    <td className="px-4 py-3 text-zinc-400">{item.category}</td>
                    <td className="whitespace-nowrap px-4 py-3">{severityBadge(item.severity)}</td>
                    <td className="max-w-[360px] truncate px-4 py-3 text-zinc-400" title={item.details || ''}>{item.details || '—'}</td>
                    <td className="px-4 py-3 text-right">
                      {item.has_download && (
                        <div className="inline-flex items-center gap-1.5">
                          {item.available_views.length > 1 && (
                            <select
                              value={view}
                              onChange={(event) => setSelectedViews((current) => ({ ...current, [item.id]: event.target.value }))}
                              className="rounded-md border border-zinc-800 bg-zinc-950 px-2 py-1.5 text-[11px] text-zinc-300 focus:outline-none focus:ring-1 focus:ring-zinc-400"
                              aria-label={`Select output view for activity ${item.id}`}
                            >
                              {item.available_views.map((availableView) => <option key={availableView} value={availableView}>{availableView}</option>)}
                            </select>
                          )}
                          <Button
                            variant="outline"
                            size="sm"
                            isLoading={isDownloading}
                            onClick={() => handleDownload(item)}
                            title={`Download ${view}`}
                            aria-label={`Download ${view}`}
                          >
                            {!isDownloading && <Download className="h-3.5 w-3.5" />}
                            <span className="hidden sm:inline">Download</span>
                          </Button>
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="flex items-center justify-between border-t border-zinc-800 px-4 py-3 text-xs text-zinc-500">
          <span>{total === 0 ? '0 records' : `${offset + 1}-${Math.min(offset + PAGE_SIZE, total)} of ${total}`}</span>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="sm" onClick={() => setOffset((current) => Math.max(0, current - PAGE_SIZE))} disabled={offset === 0 || loading} aria-label="Previous page" title="Previous page">
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setOffset((current) => current + PAGE_SIZE)} disabled={offset + PAGE_SIZE >= total || loading} aria-label="Next page" title="Next page">
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
};
