import React from 'react';
import { LayoutDashboard, Image as ImageIcon, Video, Radio } from 'lucide-react';

export type ViewType = 'overview' | 'image' | 'video' | 'feed';

interface SidebarProps {
  activeView: ViewType;
  onSelectView?: (view: ViewType) => void;
  onViewChange?: (view: ViewType) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeView, onSelectView, onViewChange }) => {
  const handleSelect = (view: ViewType) => {
    if (onSelectView) onSelectView(view);
    if (onViewChange) onViewChange(view);
  };

  const navItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'image', label: 'Image Analysis', icon: ImageIcon },
    { id: 'video', label: 'Video Analysis', icon: Video },
    { id: 'feed', label: 'Real-Time Feed', icon: Radio },
  ];

  return (
    <aside className="w-60 bg-zinc-950 border-r border-zinc-800/80 flex flex-col justify-between h-screen sticky top-0 shrink-0 select-none">
      <div>
        {/* Clean Header Area */}
        <div className="h-14 px-5 border-b border-zinc-800/80 flex items-center" />

        {/* Minimal Nav List */}
        <nav className="p-2 space-y-0.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeView === item.id;

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
        </nav>
      </div>
    </aside>
  );
};
