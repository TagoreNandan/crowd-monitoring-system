import React, { useState } from 'react';
import {
  Activity,
  LayoutDashboard,
  Image as ImageIcon,
  Video,
  Radio,
  Shield,
  Camera,
  PanelLeftClose,
  PanelLeftOpen,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export type ViewType = 'dashboard' | 'overview' | 'feed' | 'devices' | 'image' | 'video' | 'analytics' | 'activity' | 'admin';

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
  const [isCollapsed, setIsCollapsed] = useState<boolean>(() => {
    try {
      return localStorage.getItem('cms_sidebar_collapsed') === 'true';
    } catch {
      return false;
    }
  });

  const toggleCollapse = () => {
    setIsCollapsed((prev) => {
      const next = !prev;
      try {
        localStorage.setItem('cms_sidebar_collapsed', String(next));
      } catch {}
      return next;
    });
  };

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
    { id: 'activity', label: 'Activity Log', icon: Activity },
  ];

  return (
    <aside
      className={`${
        isCollapsed ? 'w-16' : 'w-60'
      } bg-zinc-950 border-r border-zinc-800/80 flex flex-col justify-between h-screen sticky top-0 shrink-0 select-none transition-all duration-200 ease-in-out z-20`}
    >
      <div>
        {/* Header Area */}
        <div
          className={`h-14 border-b border-zinc-800/80 flex items-center ${
            isCollapsed ? 'justify-center px-2' : 'justify-between px-4'
          }`}
        >
          {!isCollapsed && (
            <div className="flex items-center gap-2 overflow-hidden">
              <span className="text-xs font-mono font-bold text-zinc-200 tracking-wider uppercase truncate">
                Crowd Monitor
              </span>
              <span className="text-[10px] font-mono text-zinc-500 bg-zinc-900 px-1 py-0.5 rounded border border-zinc-800">
                v0.1
              </span>
            </div>
          )}

          <button
            onClick={toggleCollapse}
            className="p-1.5 rounded-md text-zinc-500 hover:text-zinc-200 hover:bg-zinc-900 border border-transparent hover:border-zinc-800 transition-colors cursor-pointer"
            title={isCollapsed ? 'Expand sidebar' : 'Minimize sidebar'}
          >
            {isCollapsed ? (
              <PanelLeftOpen className="w-4 h-4 text-zinc-400" />
            ) : (
              <PanelLeftClose className="w-4 h-4 text-zinc-400" />
            )}
          </button>
        </div>

        {/* Minimal Nav List */}
        <nav className="p-2 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive =
              activeView === item.id ||
              (item.id === 'dashboard' && (activeView === 'overview' || activeView === 'analytics'));

            return (
              <button
                key={item.id}
                onClick={() => handleSelect(item.id as ViewType)}
                title={isCollapsed ? item.label : undefined}
                className={`w-full flex items-center rounded-md text-xs font-medium transition-colors cursor-pointer ${
                  isCollapsed ? 'justify-center p-2.5' : 'gap-2.5 px-3 py-2'
                } ${
                  isActive
                    ? 'bg-zinc-900 text-zinc-100 border border-zinc-800'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/50'
                }`}
              >
                <Icon className={`w-4 h-4 shrink-0 ${isActive ? 'text-zinc-100' : 'text-zinc-500'}`} />
                {!isCollapsed && <span className="truncate">{item.label}</span>}
              </button>
            );
          })}

          {/* Admin Navigation Item - Strictly RBAC protected: visible ONLY to authenticated Administrators */}
          {isAdmin && (
            <>
              <div className="pt-2 pb-1 px-1">
                <div className="border-t border-zinc-900" />
              </div>

              <button
                onClick={() => handleSelect('admin')}
                title={isCollapsed ? 'Admin Console' : undefined}
                className={`w-full flex items-center rounded-md text-xs font-medium transition-colors cursor-pointer ${
                  isCollapsed ? 'justify-center p-2.5' : 'gap-2.5 px-3 py-2'
                } ${
                  activeView === 'admin'
                    ? 'bg-zinc-900 text-zinc-100 border border-zinc-800'
                    : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/50'
                }`}
              >
                <Shield
                  className={`w-4 h-4 shrink-0 ${activeView === 'admin' ? 'text-zinc-100' : 'text-zinc-500'}`}
                />
                {!isCollapsed && <span className="truncate">Admin Console</span>}
              </button>
            </>
          )}
        </nav>
      </div>

      <div
        className={`p-3 border-t border-zinc-900 text-[10px] font-mono text-zinc-600 ${
          isCollapsed ? 'text-center' : ''
        }`}
      >
        {isCollapsed ? 'AI' : 'AI Density & Clustering'}
      </div>
    </aside>
  );
};
