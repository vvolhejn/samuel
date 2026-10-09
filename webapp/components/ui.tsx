/** The small shared pieces of the page's idiom: the two section wrappers, the
 * link look, and the panel headings. */

/** Faint box around a stage; the active one drops the outline and fills with the
 * accent wash instead, so only one thing at a time reads as raised. The border
 * stays declared but transparent when lit, or the box would shift a pixel.
 * Padding is left to the call site.
 *
 * On a phone it's a full-bleed strip instead — no corners, and nothing at all
 * until it lights up — which buys back the 26px of side padding and border the
 * controls inside need to stay on one line, and reads as a stronger "you are
 * here" than a small rounded box. The border is still declared there, just
 * transparent, so lighting a strip can't change its height. Call sites pair
 * this with STRIP for the bleed itself. */
export function sectionBox(active: boolean): string {
  return `border-y border-transparent transition-[color,background-color,border-color,filter] md:rounded-xl md:border ${
    active ? "bg-highlight-50/60" : "bg-white md:border-neutral-200"
  }`;
}

/** Escapes the column's own padding so a section reaches both screen edges, and
 * re-inserts it inside so the contents stay in line with the prose above. Back
 * to an inset box at `md`, where "full width" would only mean the column's.
 *
 * `self-stretch` rather than `w-full`, and it has to be: the column is
 * `items-start`, so a width of 100% would measure the column's *content* box
 * and the negative margin would only slide that width leftwards, leaving the
 * strip short by its own bleed at the right. Stretching sizes it against the
 * margins instead, which is what reaches both edges. */
export const STRIP = "-mx-4 self-stretch px-4 py-3 md:mx-0 md:p-3";

/** TextLink's `muted` look, for the controls that read as links in the prose
 * without being one. */
export const MUTED_LINK =
  "text-inherit underline decoration-highlight-600 decoration-dotted underline-offset-2 hover:text-highlight-600";

/** A link in the prose. Anything off-origin opens in a new tab — which, as it
 * stands, is every link here.
 *
 * `muted` is for asides nobody needs to follow (the QWOP joke, the credit for
 * the original): the pink underline still says "link", but the text stays the
 * colour of the sentence around it so it doesn't ask for the click. */
export function TextLink({
  href,
  muted,
  children,
}: {
  href: string;
  muted?: boolean;
  children: React.ReactNode;
}) {
  const external = /^(https?:)?\/\//.test(href);
  return (
    <a
      href={href}
      target={external ? "_blank" : undefined}
      rel={external ? "noreferrer" : undefined}
      className={
        muted
          ? MUTED_LINK
          : "text-highlight-600 underline decoration-dotted hover:text-highlight-700"
      }
    >
      {children}
    </a>
  );
}

export function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <span className="font-semibold tracking-wide text-neutral-600 uppercase">
      {children}
    </span>
  );
}
