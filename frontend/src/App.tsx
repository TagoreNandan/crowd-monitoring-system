import { useState, useEffect, useRef } from 'react';
import { Sidebar, type ViewType } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { OverviewView } from './views/OverviewView';
import { ImageView } from './views/ImageView';
import { VideoView } from './views/VideoView';
import { FeedView } from './views/FeedView';
import { AdminView } from './views/AdminView';
import { DeviceManagementView } from './views/DeviceManagementView';
import { AnalyticsView } from './views/AnalyticsView';
import { AuthModal } from './components/AuthModal';
import { AuthProvider, useAuth } from './context/AuthContext';
import { getHealth, getStatus, connectWebSocket } from './lib/api';
import type { Device, SystemHealth, SystemStatus, VideoFrameMetrics, WSEnvelope } from './types/api';

function AppContent() {
  const [activeView, setActiveView] = useState<ViewType>('overview');
  const [selectedFeedDevice, setSelectedFeedDevice] = useState<Device | null>(null);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [latestMetrics, setLatestMetrics] = useState<VideoFrameMetrics | null>(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  const { isAdmin } = useAuth();
  const wsRef = useRef<WebSocket | null>(null);

  // If user loses admin privileges while on admin page, switch to overview
  useEffect(() => {
    if (activeView === 'admin' && !isAdmin) {
      setActiveView('overview');
    }
  }, [isAdmin, activeView]);

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
    feed: 'Real-Time Feed',
    analytics: 'Live Multi-Stream Analytics',
    devices: 'Device Fleet Management',
    image: 'Image Analysis',
    video: 'Video Analysis',
    admin: 'Administrator Console',
  };

  const renderView = () => {
    switch (activeView) {
      case 'overview':
        return (
          <OverviewView
            health={health}
            status={status}
            latestMetrics={latestMetrics}
            onNavigate={(view) => setActiveView(view as ViewType)}
          />
        );
      case 'feed':
        return null;
      case 'analytics':
        return (
          <AnalyticsView
            onFocusFeed={() => {
              setActiveView('feed');
            }}
          />
        );
      case 'devices':
        return (
          <DeviceManagementView
            onSelectDeviceForFeed={(device) => {
              setSelectedFeedDevice(device);
              setActiveView('feed');
            }}
          />
        );
      case 'image':
        return <ImageView />;
      case 'video':
        return <VideoView />;
      case 'admin':
        return <AdminView />;
      default:
        return (
          <OverviewView
            health={health}
            status={status}
            latestMetrics={latestMetrics}
            onNavigate={(view) => setActiveView(view as ViewType)}
          />
        );
    }
  };

  return (
    <div className="flex h-screen bg-zinc-950 text-zinc-100 font-sans overflow-hidden antialiased">
      {/* Sidebar Navigation */}
      <Sidebar
        activeView={activeView}
        onViewChange={setActiveView}
        onOpenAuth={() => setIsAuthModalOpen(true)}
      />

      {/* Main Content Workspace */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Top Header Bar */}
        <TopBar
          activeViewTitle={viewTitles[activeView]}
          health={health}
          status={status}
          onOpenAuth={() => setIsAuthModalOpen(true)}
        />

        {/* Dynamic Viewport Container */}
        <main className="flex-1 overflow-y-auto bg-zinc-950">
          {/* Keep FeedView continuously mounted so its live WebSocket and streaming state persist across tab switches */}
          <div className={activeView === 'feed' ? 'block' : 'hidden'}>
            <FeedView initialDevice={selectedFeedDevice} />
          </div>
          {activeView !== 'feed' && renderView()}
        </main>
      </div>

      {/* Auth Modal */}
      <AuthModal
        isOpen={isAuthModalOpen}
        onClose={() => setIsAuthModalOpen(false)}
      />
    </div>
  );
}

export function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
