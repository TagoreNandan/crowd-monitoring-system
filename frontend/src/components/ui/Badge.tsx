import React from 'react';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'safe' | 'caution' | 'warning' | 'critical' | 'default' | 'outline' | 'secondary';
  children: React.ReactNode;
}

export const Badge: React.FC<BadgeProps> = ({ variant = 'default', className = '', children, ...props }) => {
  const base = 'inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium transition-colors border';

  const variants = {
    safe: 'bg-emerald-950/40 text-emerald-400 border-emerald-800/50',
    caution: 'bg-amber-950/40 text-amber-400 border-amber-800/50',
    warning: 'bg-orange-950/40 text-orange-400 border-orange-800/50',
    critical: 'bg-rose-950/40 text-rose-400 border-rose-800/50',
    secondary: 'bg-zinc-800 text-zinc-300 border-zinc-700',
    default: 'bg-zinc-800/60 text-zinc-200 border-zinc-700/60',
    outline: 'bg-transparent text-zinc-400 border-zinc-800',
  };

  return (
    <span className={`${base} ${variants[variant] || variants.default} ${className}`} {...props}>
      {children}
    </span>
  );
};
