import React, { createContext, useContext, useState, useCallback } from 'react';
import { CheckCircle2, AlertCircle, Info, HardDrive, Cloud, X } from 'lucide-react';

export type ToastType = 'success' | 'info' | 'warning' | 'error' | 'storage';

export interface ToastItem {
  id: string;
  type: ToastType;
  title: string;
  message: string;
  storageType?: 's3' | 'local';
  storagePath?: string;
  durationMs?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface ToastContextType {
  toasts: ToastItem[];
  showToast: (toast: Omit<ToastItem, 'id'>) => string;
  dismissToast: (id: string) => void;
  notifyStorage: (params: {
    title: string;
    message: string;
    storageType: 's3' | 'local';
    storagePath?: string;
    action?: { label: string; onClick: () => void };
  }) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const showToast = useCallback(
    ({ durationMs = 6000, ...toast }: Omit<ToastItem, 'id'>) => {
      const id = `toast-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      const newToast: ToastItem = { ...toast, id, durationMs };

      setToasts((prev) => [...prev.slice(-4), newToast]); // keep max 5 visible

      if (durationMs > 0) {
        setTimeout(() => {
          dismissToast(id);
        }, durationMs);
      }
      return id;
    },
    [dismissToast]
  );

  const notifyStorage = useCallback(
    ({
      title,
      message,
      storageType,
      storagePath,
      action,
    }: {
      title: string;
      message: string;
      storageType: 's3' | 'local';
      storagePath?: string;
      action?: { label: string; onClick: () => void };
    }) => {
      showToast({
        type: 'storage',
        title,
        message,
        storageType,
        storagePath,
        action,
        durationMs: 7000,
      });
    },
    [showToast]
  );

  return (
    <ToastContext.Provider value={{ toasts, showToast, dismissToast, notifyStorage }}>
      {children}
      {/* Toast Notification Container */}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2.5 max-w-sm w-full pointer-events-none select-none">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`pointer-events-auto flex items-start gap-3 p-3.5 rounded-lg shadow-2xl border backdrop-blur-md transition-all duration-300 animate-in slide-in-from-bottom-3 ${
              toast.type === 'storage'
                ? toast.storageType === 's3'
                  ? 'bg-zinc-950/95 border-cyan-800/80 text-zinc-100 shadow-cyan-950/40'
                  : 'bg-zinc-950/95 border-emerald-800/80 text-zinc-100 shadow-emerald-950/40'
                : toast.type === 'error'
                ? 'bg-zinc-950/95 border-rose-800/80 text-zinc-100 shadow-rose-950/40'
                : toast.type === 'warning'
                ? 'bg-zinc-950/95 border-amber-800/80 text-zinc-100 shadow-amber-950/40'
                : 'bg-zinc-950/95 border-zinc-800 text-zinc-100 shadow-black/60'
            }`}
          >
            {/* Icon */}
            <div className="shrink-0 mt-0.5">
              {toast.type === 'storage' ? (
                toast.storageType === 's3' ? (
                  <Cloud className="w-5 h-5 text-cyan-400" />
                ) : (
                  <HardDrive className="w-5 h-5 text-emerald-400" />
                )
              ) : toast.type === 'success' ? (
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              ) : toast.type === 'error' ? (
                <AlertCircle className="w-5 h-5 text-rose-400" />
              ) : (
                <Info className="w-5 h-5 text-zinc-400" />
              )}
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0 space-y-1">
              <div className="flex items-center justify-between gap-2">
                <h4 className="text-xs font-semibold text-zinc-100 truncate">{toast.title}</h4>
                {toast.storageType && (
                  <span
                    className={`text-[10px] font-mono font-medium px-1.5 py-0.2 rounded uppercase border ${
                      toast.storageType === 's3'
                        ? 'text-cyan-300 bg-cyan-950/70 border-cyan-800/60'
                        : 'text-emerald-300 bg-emerald-950/70 border-emerald-800/60'
                    }`}
                  >
                    {toast.storageType === 's3' ? 'AWS S3 / MinIO' : 'Local Storage'}
                  </span>
                )}
              </div>

              <p className="text-xs text-zinc-300 leading-relaxed font-sans">{toast.message}</p>

              {toast.storagePath && (
                <p className="text-[11px] font-mono text-zinc-400 truncate bg-zinc-900/90 px-2 py-0.5 rounded border border-zinc-800/80" title={toast.storagePath}>
                  {toast.storagePath}
                </p>
              )}

              {toast.action && (
                <div className="pt-1">
                  <button
                    onClick={() => {
                      toast.action?.onClick();
                      dismissToast(toast.id);
                    }}
                    className="text-xs font-medium text-emerald-400 hover:text-emerald-300 underline cursor-pointer"
                  >
                    {toast.action.label}
                  </button>
                </div>
              )}
            </div>

            {/* Close Button */}
            <button
              onClick={() => dismissToast(toast.id)}
              className="text-zinc-500 hover:text-zinc-300 p-0.5 rounded shrink-0 cursor-pointer"
              title="Dismiss"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
};

export const useToast = () => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
};
