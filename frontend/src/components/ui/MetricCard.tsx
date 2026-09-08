import React from 'react';

interface MetricCardProps {
  title: string;
  value: string | number | null | undefined;
  subtitle?: string;
  icon?: React.ReactNode;
  badge?: React.ReactNode;
  isPrimary?: boolean;
  unit?: string;
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  badge,
  unit,
  isPrimary = false,
  className = '',
}) => {
  const displayValue = value === null || value === undefined || value === '' ? '—' : value;

  return (
    <div
      className={`p-4 rounded-lg border bg-zinc-900/40 border-zinc-800 ${className}`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-medium text-zinc-400 flex items-center gap-1.5">
          {icon && <span className="text-zinc-400">{icon}</span>}
          {title}
        </span>
        {badge}
      </div>

      <div className="mt-2 flex items-baseline gap-1.5">
        <span
          className={`font-mono tracking-tight ${
            isPrimary ? 'text-3xl font-semibold text-zinc-100' : 'text-2xl font-medium text-zinc-200'
          }`}
        >
          {displayValue}
        </span>
        {unit && <span className="text-xs font-mono text-zinc-400">{unit}</span>}
      </div>

      {subtitle && <p className="mt-1 text-xs text-zinc-500">{subtitle}</p>}
    </div>
  );
};
