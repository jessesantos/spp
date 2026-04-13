"use client";

/**
 * ExplanationModal: dialog showing the narrative explanation for a horizon.
 *
 * Pure Tailwind, no external deps. Closes on backdrop click, ESC key and the
 * explicit close button. Renders a friendly placeholder when the explanation
 * has not been generated yet.
 */
import { useEffect, useId, useMemo, type JSX } from "react";

export interface ExplanationModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  explanation: string | null | undefined;
}

const PLACEHOLDER =
  "Explicacao ainda nao disponivel. Faca uma nova previsao para gerar o texto.";

function countWords(text: string): number {
  const matches = text.match(/\S+/g);
  return matches ? matches.length : 0;
}

export function ExplanationModal({
  open,
  onClose,
  title,
  explanation,
}: ExplanationModalProps): JSX.Element | null {
  const titleId = useId();

  useEffect(() => {
    if (!open) return;
    function onKey(event: KeyboardEvent): void {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const words = useMemo(
    () => (explanation ? countWords(explanation) : 0),
    [explanation],
  );

  if (!open) return null;

  const body = explanation && explanation.trim().length > 0 ? explanation : PLACEHOLDER;
  const hasText = Boolean(explanation && explanation.trim().length > 0);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4"
      onClick={onClose}
    >
      <div
        className="relative w-full max-w-2xl rounded-xl border border-neutral-800 bg-neutral-950 p-6 shadow-2xl"
        onClick={(event) => event.stopPropagation()}
      >
        <header className="flex items-start justify-between gap-4 border-b border-neutral-800 pb-3">
          <div>
            <p className="text-[10px] uppercase tracking-wider text-neutral-500">
              Por que essa tendencia?
            </p>
            <h2
              id={titleId}
              className="mt-1 text-lg font-semibold text-neutral-50"
            >
              {title}
            </h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Fechar"
            className="rounded-md border border-neutral-800 px-2 py-1 text-sm text-neutral-400 hover:bg-neutral-900 hover:text-neutral-100"
          >
            x
          </button>
        </header>
        <div className="mt-4 max-h-[60vh] overflow-y-auto whitespace-pre-wrap text-sm leading-relaxed text-neutral-200">
          {body}
        </div>
        <footer className="mt-4 flex items-center justify-between border-t border-neutral-800 pt-3 text-xs text-neutral-500">
          <span>{hasText ? `${words} palavras` : "Sem texto persistido"}</span>
          <button
            type="button"
            onClick={onClose}
            className="rounded-md border border-neutral-700 px-3 py-1 text-neutral-200 hover:bg-neutral-900"
          >
            Fechar
          </button>
        </footer>
      </div>
    </div>
  );
}
