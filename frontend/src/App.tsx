import { useState, useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { OverviewView } from './views/OverviewView';
import { ImageView } from './views/ImageView';
import { VideoView } from './views/VideoView';
import { FeedView } from './views/FeedView';
import { getHealth, getStatus, connectWebSocket } from './lib/api';
import type { SystemHealth, SystemStatus, VideoFrameMetrics, WSEnvelope } from './types/api';

export type ViewType = 'overview' | 'image' | 'video' | 'feed';

export function App() {
  const [activeView, setActiveView] = useState<ViewType>('overview');
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [latestMetrics, setLatestMetrics] = useState<VideoFrameMetrics | null>(null);

  const wsRef = useRef<WebSocket | null>(null);

  // Poll system health and status periodically
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const [h, s] = await Promise.all([
          getHealth().catch(() => null),
          getStatus().catch(() => null),
        ]);
        setHealth(h);
        setStatus(s);
      } catch (e) {
        console.error('Health check failed', e);
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  // System-level WebSocket connection (/ws/system)
  useEffect(() => {
    try {
      wsRef.current = connectWebSocket(
        '/ws/system',
        (envelope: WSEnvelope) => {
          if (envelope.type === 'system.metrics' && envelope.data) {
            setLatestMetrics(envelope.data);
          }
        }
      );
    } catch (e) {
      console.error('WS System connection error', e);
    }

    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const viewTitles: Record<ViewType, string> = {
    overview: 'Overview',
    image: 'Image Analysis',
    video: 'Video Analysis',
    feed: 'Real-Time Feed',
  };

  const renderView = () => {
    switch (activeView) {
      case 'overview':
        return (
          <OverviewView
            health={health}
            status={status}
            latestMetrics={latestMetrics}
            onNavigate={(view) => setActiveView(view)}
          />
        );
      case 'image':
        return <ImageView />;
      case 'video':
        return <VideoView />;
      case 'feed':
        return <FeedView />;
      default:
        return (
          <OverviewView
            health={health}
            status={status}
            latestMetrics={latestMetrics}
            onNavigate={(view) => setActiveView(view)}
          />
        );
    }
  };

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100 font-sans overflow-hidden antialiased">
      {/* Sidebar Navigation */}
      <Sidebar activeView={activeView} onViewChange={setActiveView} />

      {/* Main Content Workspace */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top Header Bar */}
        <TopBar activeViewTitle={viewTitles[activeView]} />

        {/* Dynamic Viewport Container */}
        <main className="flex-1 overflow-y-auto bg-zinc-950">
          {renderView()}
        </main>
      </div>
    </div>
  );
}

export default App;
