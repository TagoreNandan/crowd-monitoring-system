import React from 'react';
import { LayoutDashboard, Image as ImageIcon, Video, Radio, Shield, Camera } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export type ViewType = 'dashboard' | 'overview' | 'feed' | 'devices' | 'image' | 'video' | 'analytics' | 'admin';

interface SidebarProps {
  activeView: ViewType;
  onSelectView?: (view: ViewType) => void;
  onViewChange?: (view: ViewType) => void;
  onOpenAuth?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeView,
  onSelectView,
  onViewChange,
  onOpenAuth,
}) => {
  const { isAdmin } = useAuth();

  const handleSelect = (view: ViewType) => {
    if (view === 'admin' && !isAdmin) {
      if (onOpenAuth) onOpenAuth();
      return;
    }
    if (onSelectView) onSelectView(view);
    if (onViewChange) onViewChange(view);
  };

  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'feed', label: 'Real-Time Feed', icon: Radio },
    { id: 'devices', label: 'Device Fleet', icon: Camera },
    { id: 'image', label: 'Image Analysis', icon: ImageIcon },
    { id: 'video', label: 'Video Analysis', icon: Video },
  ];

  return (
    <aside className="w-60 bg-zinc-950 border-r border-zinc-800/80 flex flex-col justify-between h-screen sticky top-0 shrink-0 select-none">
      <div>
        {/* Header Area */}
        <div className="h-14 px-5 border-b border-zinc-800/80 flex items-center justify-between">
          <span className="text-xs font-mono font-medium text-zinc-300 tracking-wider uppercase">
            Crowd Monitor
          </span>
          <span className="text-[10px] font-mono text-zinc-600">v0.1</span>
        </div>

        {/* Minimal Nav List */}
        <nav className="p-2 space-y-0.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive =
              activeView === item.id ||
              (item.id === 'dashboard' && (activeView === 'overview' || activeView === 'analytics'));

            return (
              <button
                key={item.id}
                onClick={() => handleSelect(item.id as ViewType)}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-md text-xs font-medium transition-colors cursor-pointer ${
                  isActive
                    ? 'bg-zinc-900 text-zinc-100 border border-zinc-800'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/50'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-zinc-100' : 'text-zinc-500'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}

          {/* Admin Navigation Item - Strictly RBAC protected: visible ONLY to authenticated Administrators */}
          {isAdmin && (
            <>
              <div className="pt-3 pb-1 px-3">
                <div className="border-t border-zinc-900" />
              </div>

              <button
                onClick={() => handleSelect('admin')}
                className={`w-full flex items-center justify-between px-3 py-2 rounded-md text-xs font-medium transition-colors cursor-pointer ${
                  activeView === 'admin'
                    ? 'bg-zinc-900 text-zinc-100 border border-zinc-800'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/50'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  <Shield className={`w-4 h-4 ${activeView === 'admin' ? 'text-zinc-100' : 'text-zinc-500'}`} />
                  <span>Admin Console</span>
                </div>
              </button>
            </>
          )}
        </nav>
      </div>

      <div className="p-4 border-t border-zinc-900 text-[10px] font-mono text-zinc-600">
        AI Density & Clustering
      </div>
    </aside>
  );
};
