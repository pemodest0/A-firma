export default function HelpHint({ text }: { text: string }) {
  return (
    <span
      title={text}
      aria-label={text}
      className="inline-flex h-4 w-4 items-center justify-center rounded-full border border-zinc-700 text-[10px] font-medium text-zinc-400"
    >
      ?
    </span>
  );
}
