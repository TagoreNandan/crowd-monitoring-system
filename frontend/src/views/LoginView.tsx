import React, { useState } from 'react';
import { Shield, User as UserIcon, Lock, AlertCircle, ArrowRight } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { Button } from '../components/ui/Button';

export const LoginView: React.FC = () => {
  const { login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      await login(username, password);
    } catch (err: any) {
      setError(err.message || 'Authentication failed. Please verify your credentials.');
    } finally {
      setLoading(false);
    }
  };

  const fillCredentials = (u: string, p: string) => {
    setUsername(u);
    setPassword(p);
    setError(null);
  };

  return (
    <div className="min-h-screen w-full bg-zinc-950 text-zinc-100 flex flex-col justify-center items-center px-4 py-12 relative overflow-hidden select-none">
      {/* Background ambient glow */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-emerald-500/5 rounded-full blur-3xl pointer-events-none" />

      {/* Brand Header */}
      <div className="mb-8 text-center space-y-2 z-10">
        <div className="inline-flex items-center justify-center p-3 rounded-xl bg-zinc-900 border border-zinc-800 shadow-xl mb-2">
          <Shield className="w-8 h-8 text-emerald-400" />
        </div>
        <h1 className="text-2xl font-bold tracking-tight text-zinc-100">
          Crowd Monitoring System
        </h1>
        <p className="text-xs text-zinc-400 max-w-sm mx-auto">
          Enterprise real-time crowd density analysis & fleet surveillance.
        </p>
      </div>

      {/* Main Card */}
      <div className="w-full max-w-sm bg-zinc-900/60 border border-zinc-800 rounded-xl shadow-2xl p-7 backdrop-blur-md z-10">
        <div className="mb-6 space-y-1">
          <h2 className="text-lg font-semibold tracking-tight text-zinc-100">
            Sign In
          </h2>
          <p className="text-xs text-zinc-400">
            Enter your credentials to access system telemetry and video feeds.
          </p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-5 p-3 rounded-lg bg-rose-950/40 border border-rose-800/60 text-rose-300 text-xs flex items-start gap-2">
            <AlertCircle className="w-4 h-4 shrink-0 text-rose-400 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-1.5">
            <label className="block text-xs font-medium text-zinc-300">Username</label>
            <div className="relative">
              <input
                type="text"
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Username (e.g. admin or analyst)"
                className="w-full pl-9 pr-3.5 py-2 bg-zinc-950/70 border border-zinc-800 rounded-lg text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-400 transition-colors"
              />
              <UserIcon className="w-4 h-4 text-zinc-500 absolute left-3 top-2.5" />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="block text-xs font-medium text-zinc-300">Password</label>
            <div className="relative">
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full pl-9 pr-3.5 py-2 bg-zinc-950/70 border border-zinc-800 rounded-lg text-xs text-zinc-100 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-zinc-400 transition-colors"
              />
              <Lock className="w-4 h-4 text-zinc-500 absolute left-3 top-2.5" />
            </div>
          </div>

          <Button
            type="submit"
            variant="default"
            size="default"
            isLoading={loading}
            className="w-full mt-2 font-medium"
          >
            <span>Sign In</span>
            <ArrowRight className="w-4 h-4 ml-1.5" />
          </Button>
        </form>

        {/* Quick Credentials Assistance */}
        <div className="mt-6 pt-5 border-t border-zinc-800/80">
          <span className="text-[11px] font-mono text-zinc-500 block mb-2 uppercase tracking-wider text-center">
            Quick Select Role
          </span>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => fillCredentials('admin', 'admin123')}
              className="p-2 rounded-lg bg-zinc-950/60 border border-zinc-800 hover:border-zinc-700 text-left transition-colors cursor-pointer group"
            >
              <div className="text-[11px] font-semibold text-zinc-200 group-hover:text-emerald-400">
                Admin
              </div>
              <div className="text-[10px] font-mono text-zinc-500">Full Access</div>
            </button>
            <button
              type="button"
              onClick={() => fillCredentials('analyst', 'analyst123')}
              className="p-2 rounded-lg bg-zinc-950/60 border border-zinc-800 hover:border-zinc-700 text-left transition-colors cursor-pointer group"
            >
              <div className="text-[11px] font-semibold text-zinc-200 group-hover:text-emerald-400">
                Analyst
              </div>
              <div className="text-[10px] font-mono text-zinc-500">Standard User</div>
            </button>
          </div>
        </div>
      </div>

      <div className="mt-8 text-xs font-mono text-zinc-600 z-10">
        Authentication & Role-Based Access Control
      </div>
    </div>
  );
};
