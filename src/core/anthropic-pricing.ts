/**
 * Anthropic chat pricing — a bare-keyed VIEW of the canonical pricing table
 * (`src/core/model-pricing.ts`).
 *
 * Kept as a distinct export because some legacy callers look up by bare Claude
 * id (`claude-opus-4-7`). `estimateMaxCostUsd` carries the null-on-miss
 * contract the dream-cycle budget gate depends on. That estimator keeps an
 * explicit consumer allowlist: Anthropic plus the Gemini deployment routes
 * used by this fork. The dollar
 * numbers live in model-pricing.ts — DO NOT hand-edit prices here; this map is
 * derived from the `anthropic:` canonical entries (prefix stripped), so it
 * cannot drift from the other pricing views. (Pre-unification this map and
 * takes-quality-eval/pricing.ts duplicated the numbers and drifted: Opus 4.7
 * read $15/$75 in one and $5/$25 in the other.)
 *
 * Models absent from this consumer's explicit pricing policy bypass the
 * budget gate with a `BUDGET_METER_NO_PRICING` warn once per process. The
 * cycle still runs unbounded for those models.
 */

import { CANONICAL_PRICING, canonicalLookup, type ModelPricing } from './model-pricing.ts';
import { splitProviderModelId } from './model-id.ts';

export type { ModelPricing };

/**
 * Bare-keyed Anthropic view, derived from the canonical table. Both the
 * dateless ids (`claude-haiku-4-5`, used by aliases / TIER_DEFAULTS / most
 * callers) and the dated snapshots (`claude-haiku-4-5-20251001`) are present
 * because canonical carries both.
 */
export const ANTHROPIC_PRICING: Record<string, ModelPricing> = Object.fromEntries(
  Object.entries(CANONICAL_PRICING)
    .filter(([key]) => key.startsWith('anthropic:'))
    .map(([key, pricing]) => [key.slice('anthropic:'.length), pricing]),
);

/**
 * Non-Anthropic chat routes this legacy estimator is explicitly allowed to
 * price. Keep this consumer-owned allowlist narrow: adding an entry to the
 * canonical table must not silently change the dream-cycle budget policy.
 */
const DREAM_BUDGET_CANONICAL_MODELS: ReadonlySet<string> = new Set([
  'google:gemini-3.5-flash',
  'google:gemini-3-flash-preview',
  'google:gemini-3.1-flash-lite',
  'litellm:gemini-3.5-flash',
  'litellm:gemini-3-flash',
  'litellm:gemini-3-flash-preview',
  'litellm:gemini-3.1-flash-lite',
]);

/**
 * Estimate the upper-bound USD cost of a single submit.
 * Uses (estimatedInputTokens × inputRate) + (maxOutputTokens × outputRate).
 * The maxOutputTokens upper-bounds the output cost — actual completions
 * usually return less.
 *
 * Returns null when this consumer's explicit pricing policy rejects the
 * model. Callers warn once and treat it as zero-cost (the cycle runs
 * unbounded for that submit).
 *
 * Accepts bare Claude ids and explicit Anthropic colon/slash forms. The
 * deployment Gemini routes above use their exact provider:model ids. Other
 * canonical providers remain outside this consumer's policy by design.
 */
export function estimateMaxCostUsd(
  modelId: string,
  estimatedInputTokens: number,
  maxOutputTokens: number,
): number | null {
  let p: ModelPricing | undefined = ANTHROPIC_PRICING[modelId];
  if (!p) {
    const { provider, model } = splitProviderModelId(modelId);
    if (provider === 'anthropic') p = ANTHROPIC_PRICING[model];
  }
  if (!p && DREAM_BUDGET_CANONICAL_MODELS.has(modelId)) {
    p = canonicalLookup(modelId) ?? undefined;
  }
  if (!p) return null;
  return (
    (estimatedInputTokens / 1_000_000) * p.input +
    (maxOutputTokens     / 1_000_000) * p.output
  );
}
