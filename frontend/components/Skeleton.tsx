import type { JSX } from "react";

/**
 * Skeleton primitive: pulsating placeholder block used while data loads.
 *
 * Uses Tailwind's `animate-pulse` with a low-contrast background so it
 * reads clearly on dark surfaces without screaming for attention.
 */
export interface SkeletonProps {
  className?: string;
  rounded?: "md" | "lg" | "xl" | "2xl" | "full";
}

export function Skeleton({
  className = "",
  rounded = "md",
}: SkeletonProps): JSX.Element {
  return (
    <span
      aria-hidden
      className={`block animate-pulse bg-neutral-800/70 rounded-${rounded} ${className}`}
    />
  );
}

export function SkeletonCard({
  className = "",
}: {
  className?: string;
}): JSX.Element {
  return (
    <div
      aria-hidden
      className={`rounded-2xl border border-neutral-800 bg-neutral-900/50 p-5 ${className}`}
    >
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <Skeleton className="h-5 w-20" />
          <Skeleton className="h-3 w-28" />
        </div>
        <Skeleton className="h-5 w-10 rounded-full" />
      </div>
      <div className="mt-5 space-y-2">
        <Skeleton className="h-7 w-32" />
        <Skeleton className="h-3 w-24" />
      </div>
      <div className="mt-5 flex items-center justify-between rounded-lg border border-neutral-800 bg-neutral-950/60 px-3 py-2">
        <Skeleton className="h-3 w-12" />
        <Skeleton className="h-3 w-20" />
        <Skeleton className="h-3 w-12" />
      </div>
    </div>
  );
}
