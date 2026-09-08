import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'outline' | 'secondary' | 'destructive' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  isLoading?: boolean;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'default',
  size = 'md',
  isLoading = false,
  className = '',
  children,
  disabled,
  ...props
}) => {
  const base = 'inline-flex items-center justify-center font-medium transition-colors rounded-md focus:outline-none focus:ring-1 focus:ring-zinc-400 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer';

  const variants = {
    default: 'bg-zinc-100 text-zinc-900 hover:bg-zinc-200 font-semibold shadow-xs',
    outline: 'border border-zinc-800 bg-zinc-950 text-zinc-200 hover:bg-zinc-900 hover:text-white',
    secondary: 'bg-zinc-900 text-zinc-200 hover:bg-zinc-800 border border-zinc-800',
    destructive: 'bg-rose-950/80 text-rose-200 border border-rose-800/60 hover:bg-rose-900/90',
    ghost: 'text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100',
  };

  const sizes = {
    sm: 'px-3 py-1.5 text-xs gap-1.5',
    md: 'px-4 py-2 text-sm gap-2',
    lg: 'px-5 py-2.5 text-base gap-2',
  };

  return (
    <button
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled || isLoading}
      {...props}
    >
      {isLoading ? (
        <>
          <svg className="animate-spin h-3.5 w-3.5 text-current" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          {children}
        </>
      ) : (
        children
      )}
    </button>
  );
};
