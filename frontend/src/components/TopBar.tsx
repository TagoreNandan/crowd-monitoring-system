import React from 'react';
import { Shield, User, LogOut, LogIn } from 'lucide-react';
import type { SystemHealth, SystemStatus } from '../types/api';
import { useAuth } from '../context/AuthContext';
import { Badge } from './ui/Badge';
import { Button } from './ui/Button';

interface TopBarProps {
  health?: SystemHealth | null;
  status?: SystemStatus | null;
  wsConnected?: boolean;
  activeViewTitle?: string;
  onOpenAuth?: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({
  activeViewTitle = '',
  status,
  onOpenAuth,
}) => {
  const { user, isAdmin, logout } = useAuth();

  return (
    <header className="h-14 bg-zinc-950/80 backdrop-blur border-b border-zinc-800/80 px-6 flex items-center justify-between sticky top-0 z-20 select-none">
      <div className="flex items-center gap-3">
        <h1 className="text-sm font-semibold text-zinc-200 tracking-tight">{activeViewTitle}</h1>
        {status?.device && (
          <Badge variant="outline" className="hidden sm:inline-flex text-[10px] font-mono">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
            {status.device.toUpperCase()}
          </Badge>
        )}
      </div>

      <div className="flex items-center space-x-3">
        {user ? (
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-md bg-zinc-900 border border-zinc-800 text-xs">
              <div className="text-zinc-400">
                {isAdmin ? <Shield className="w-3.5 h-3.5 text-zinc-200" /> : <User className="w-3.5 h-3.5 text-zinc-400" />}
              </div>
              <div className="flex flex-col text-left">
                <span className="font-medium text-zinc-200 leading-none">{user.username}</span>
                <span className="text-[10px] uppercase font-mono text-zinc-400 font-semibold leading-tight">
                  {user.role === 'admin' ? 'ADMIN' : 'ANALYST'}
                </span>
              </div>
            </div>

            <button
              onClick={logout}
              title="Sign Out"
              className="p-1.5 rounded-md text-zinc-400 hover:text-rose-400 hover:bg-zinc-900 transition-colors cursor-pointer"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <Button
            variant="outline"
            size="sm"
            onClick={onOpenAuth}
            className="flex items-center gap-1.5 text-xs font-medium"
          >
            <LogIn className="w-3.5 h-3.5 text-zinc-400" />
            <span>Sign In</span>
          </Button>
        )}
      </div>
    </header>
  );
};
