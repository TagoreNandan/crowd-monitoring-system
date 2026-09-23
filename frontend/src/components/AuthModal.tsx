import React, { useState } from 'react';
import { X, AlertCircle, Shield, User as UserIcon, Lock } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { Button } from './ui/Button';

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AuthModal: React.FC<AuthModalProps> = ({ isOpen, onClose }) => {
  const { login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      await login(username, password);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Authentication failed. Please verify credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-xs animate-in fade-in duration-150">
      <div className="relative w-full max-w-sm bg-zinc-950 border border-zinc-800 rounded-lg shadow-2xl p-6 text-zinc-100">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1 rounded-md text-zinc-400 hover:text-zinc-100 hover:bg-zinc-900 transition-colors cursor-pointer"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Modal Header */}
        <div className="mb-6 space-y-1">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-md bg-zinc-900 border border-zinc-800 text-zinc-300">
              <Shield className="w-4 h-4" />
            </div>
            <h2 className="text-base font-semibold tracking-tight text-zinc-100">
              Sign In
            </h2>
          </div>
          <p className="text-xs text-zinc-400">
            Enter your credentials to access your session.
          </p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-4 p-2.5 rounded-md bg-rose-950/40 border border-rose-800/50 text-rose-300 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
            <span>{error}</span>
          </div>
        )}

        {/* Auth Form */}
        <form onSubmit={handleSubmit} className="space-y-3.5">
          <div className="space-y-1">
            <label className="block text-xs font-medium text-zinc-300">Username</label>
            <div className="relative">
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Username"
                className="w-full pl-8 pr-3 py-1.5 bg-zinc-900/60 border border-zinc-800 rounded-md text-xs text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-400 transition-colors"
              />
              <UserIcon className="w-3.5 h-3.5 text-zinc-500 absolute left-2.5 top-2" />
            </div>
          </div>

          <div className="space-y-1">
            <label className="block text-xs font-medium text-zinc-300">Password</label>
            <div className="relative">
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Password"
                className="w-full pl-8 pr-3 py-1.5 bg-zinc-900/60 border border-zinc-800 rounded-md text-xs text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:ring-1 focus:ring-zinc-400 transition-colors"
              />
              <Lock className="w-3.5 h-3.5 text-zinc-500 absolute left-2.5 top-2" />
            </div>
          </div>

          <div className="pt-2">
            <Button
              type="submit"
              variant="default"
              size="sm"
              isLoading={loading}
              className="w-full"
            >
              Sign In
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};
