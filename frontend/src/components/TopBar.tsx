import React from 'react';
import type { SystemHealth, SystemStatus } from '../types/api';

interface TopBarProps {
  health?: SystemHealth | null;
  status?: SystemStatus | null;
  wsConnected?: boolean;
  activeViewTitle?: string;
}

export const TopBar: React.FC<TopBarProps> = ({ activeViewTitle = '' }) => {
  return (
    <header className="h-14 bg-zinc-950/80 backdrop-blur border-b border-zinc-800/80 px-6 flex items-center justify-between sticky top-0 z-20 select-none">
      <div className="flex items-center gap-3">
        <h1 className="text-sm font-semibold text-zinc-200 tracking-tight">{activeViewTitle}</h1>
      </div>
      <div />
    </header>
  );
};
