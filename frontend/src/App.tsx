import { useState, useEffect, useRef } from 'react';
import { Sidebar, type ViewType } from './components/Sidebar';
import { TopBar } from './components/TopBar';
import { ImageView } from './views/ImageView';
import { VideoView } from './views/VideoView';
import { FeedView } from './views/FeedView';
import { AdminView } from './views/AdminView';
import { DeviceManagementView } from './views/DeviceManagementView';
import { AnalyticsView } from './views/AnalyticsView';
import { ActivityLogView } from './views/ActivityLogView';
import { LoginView } from './views/LoginView';
import { AuthModal } from './components/AuthModal';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ToastProvider } from './context/ToastContext';
import { getHealth, getStatus, connectWebSocket } from './lib/api';
import type { Device, SystemHealth, SystemStatus, WSEnvelope } from './types/api';

function AppContent() {
  const { user, isAdmin, isLoading } = useAuth();
  const [activeView, setActiveView] = useState<ViewType>('dashboard');
  const [selectedFeedDevice, setSelectedFeedDevice] = useState<Device | null>(null);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [isAuthModalOpen, setIsAuthModalOpen] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);

  // If user loses admin privileges while on admin page, switch to dashboard
  useEffect(() => {
    if (activeView === 'admin' && !isAdmin) {
      setActiveView('dashboard');
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
        (_envelope: WSEnvelope) => {
          // System channel heartbeat / telemetry
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
    dashboard: 'Live Analytics Dashboard',
    overview: 'Live Analytics Dashboard',
    analytics: 'Live Analytics Dashboard',
    feed: 'Real-Time Feed',
    devices: 'Device Fleet Management',
    image: 'Image Analysis',
    video: 'Video Analysis',
    activity: 'Activity Log',
    admin: 'Administrator Console',
  };

  const renderView = () => {
    switch (activeView) {
      case 'dashboard':
      case 'overview':
      case 'analytics':
        return (
          <AnalyticsView
            onFocusFeed={() => {
              setActiveView('feed');
            }}
          />
        );
      case 'feed':
        return null;
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
      case 'activity':
        return <ActivityLogView />;
      case 'admin':
        return isAdmin ? (
          <AdminView />
        ) : (
          <AnalyticsView
            onFocusFeed={() => {
              setActiveView('feed');
            }}
          />
        );
      default:
        return (
          <AnalyticsView
            onFocusFeed={() => {
              setActiveView('feed');
            }}
          />
        );
    }
  };

  // While validating stored session token
  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center bg-zinc-950 text-zinc-500 font-mono text-xs">
        Authenticating session...
      </div>
    );
  }

  // Without signing in, nothing in the workspace is exposed; redirects straight to LoginView
  if (!user) {
    return <LoginView />;
  }

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
        <main
          className={`flex-1 bg-zinc-950 ${
            activeView === 'feed' ? 'overflow-hidden flex flex-col min-h-0' : 'overflow-y-auto'
          }`}
        >
          {/* Keep FeedView continuously mounted so its live WebSocket and streaming state persist across tab switches */}
          <div className={activeView === 'feed' ? 'flex-1 min-h-0 flex flex-col h-full overflow-hidden' : 'hidden'}>
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
      <ToastProvider>
        <AppContent />
      </ToastProvider>
    </AuthProvider>
  );
}

export default App;
