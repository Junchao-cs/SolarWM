// Native source sampling is 24 FPS; this does not pin Runtime playout FPS.
export class ComparisonClock {
  private previous: number | null = null;
  private frame = -1;
  reset() { this.previous = null; this.frame = -1; }
  observe(presented: number, width: number, height: number): number | null {
    // Exclude Runtime's initial placeholder, including when it arrives late.
    if (!((width === 1344 && height === 768) || (width === 768 && height === 1344))) return null;
    const advance = this.previous === null ? 1 : Math.max(0, presented - this.previous);
    this.previous = presented;
    this.frame = Math.min(157, this.frame + advance);
    return Math.max(0, this.frame) / 24;
  }
}
