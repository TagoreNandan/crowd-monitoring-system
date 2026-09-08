import React from 'react';

interface ProgressProps {
  value: number; // 0 to 100
  className?: string;
  barClassName?: string;
}

export const Progress: React.FC<ProgressProps> = ({ value, className = '', barClassName = '' }) => {
  const clamped = Math.min(100, Math.max(0, value));

  return (
    <div className={`w-full bg-zinc-900 border border-zinc-800 rounded-full h-1.5 overflow-hidden ${className}`}>
      <div
        className={`bg-zinc-200 h-full transition-all duration-300 ease-out rounded-full ${barClassName}`}
        style={{ width: `${clamped}%` }}
      />
    </div>
  );
};
