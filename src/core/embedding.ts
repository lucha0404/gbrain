/**
 * Embedding Service (Vertex AI Gemini fork)
 *
 * Patched from upstream (which uses OpenAI text-embedding-3-large at 1536d)
 * to Vertex AI gemini-embedding-001 at 1536d (matryoshka). Auth via ADC —
 * no API key required when running on GCE / under a service account that
 * has roles/aiplatform.user.
 *
 * Env:
 *   GBRAIN_VERTEX_PROJECT   GCP project id (or GOOGLE_CLOUD_PROJECT)
 *   GBRAIN_VERTEX_LOCATION  Vertex region (default: us-central1)
 *
 * Retry with exponential backoff (4s base, 120s cap, 5 retries).
 * 8000 character input truncation.
 */

import { GoogleGenAI } from '@google/genai';

const MODEL = 'gemini-embedding-001';
const DIMENSIONS = 1536;
const MAX_CHARS = 8000;
const MAX_RETRIES = 5;
const BASE_DELAY_MS = 4000;
const MAX_DELAY_MS = 120000;
const BATCH_SIZE = 100;
const REQUEST_CONCURRENCY = 10;

function getProject(): string {
  return (
    process.env.GBRAIN_VERTEX_PROJECT ||
    process.env.GOOGLE_CLOUD_PROJECT ||
    ''
  );
}

function getLocation(): string {
  return process.env.GBRAIN_VERTEX_LOCATION || 'us-central1';
}

export function isEmbeddingAvailable(): boolean {
  return Boolean(getProject());
}

let client: GoogleGenAI | null = null;

function getClient(): GoogleGenAI {
  if (!client) {
    const project = getProject();
    if (!project) {
      throw new Error(
        'GBRAIN_VERTEX_PROJECT (or GOOGLE_CLOUD_PROJECT) must be set for Vertex AI embedding',
      );
    }
    client = new GoogleGenAI({
      vertexai: true,
      project,
      location: getLocation(),
    });
  }
  return client;
}

export async function embed(text: string): Promise<Float32Array> {
  const truncated = text.slice(0, MAX_CHARS);
  const result = await embedBatch([truncated]);
  return result[0];
}

export interface EmbedBatchOptions {
  /**
   * Optional callback fired after each 100-item sub-batch completes.
   * CLI wrappers tick a reporter; Minion handlers can call
   * job.updateProgress here instead of hooking the per-page callback.
   */
  onBatchComplete?: (done: number, total: number) => void;
}

export async function embedBatch(
  texts: string[],
  options: EmbedBatchOptions = {},
): Promise<Float32Array[]> {
  const truncated = texts.map(t => t.slice(0, MAX_CHARS));
  const results: Float32Array[] = [];

  for (let i = 0; i < truncated.length; i += BATCH_SIZE) {
    const batch = truncated.slice(i, i + BATCH_SIZE);
    const batchResults = await embedSubBatchWithConcurrency(batch);
    results.push(...batchResults);
    options.onBatchComplete?.(results.length, truncated.length);
  }

  return results;
}

/**
 * Vertex AI Gemini embedContent processes one input per call, so we fan out
 * single-item requests with bounded concurrency to approximate batch throughput
 * while respecting the per-project QPS quota.
 */
async function embedSubBatchWithConcurrency(texts: string[]): Promise<Float32Array[]> {
  const results: Float32Array[] = new Array(texts.length);

  for (let i = 0; i < texts.length; i += REQUEST_CONCURRENCY) {
    const slice = texts.slice(i, i + REQUEST_CONCURRENCY);
    const sliceResults = await Promise.all(
      slice.map((t, j) =>
        embedSingleWithRetry(t).then(emb => ({ idx: i + j, emb })),
      ),
    );
    for (const { idx, emb } of sliceResults) {
      results[idx] = emb;
    }
  }

  return results;
}

async function embedSingleWithRetry(text: string): Promise<Float32Array> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const response = await getClient().models.embedContent({
        model: MODEL,
        contents: text,
        config: {
          outputDimensionality: DIMENSIONS,
          taskType: 'RETRIEVAL_DOCUMENT',
        },
      });

      const values = response.embeddings?.[0]?.values;
      if (!values || values.length === 0) {
        throw new Error('Vertex embedding response missing values');
      }
      if (values.length !== DIMENSIONS) {
        throw new Error(
          `Expected ${DIMENSIONS}-dim embedding, got ${values.length}`,
        );
      }
      return new Float32Array(values);
    } catch (e: unknown) {
      if (attempt === MAX_RETRIES - 1) throw e;

      const status = errorStatus(e);
      if (status && status >= 400 && status < 500 && status !== 429) {
        throw e;
      }

      await sleep(exponentialDelay(attempt));
    }
  }

  throw new Error('Embedding failed after all retries');
}

function errorStatus(e: unknown): number | undefined {
  if (e && typeof e === 'object' && 'status' in e) {
    const s = (e as { status?: unknown }).status;
    if (typeof s === 'number') return s;
  }
  return undefined;
}

function exponentialDelay(attempt: number): number {
  const delay = BASE_DELAY_MS * Math.pow(2, attempt);
  return Math.min(delay, MAX_DELAY_MS);
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export { MODEL as EMBEDDING_MODEL, DIMENSIONS as EMBEDDING_DIMENSIONS };

/**
 * USD cost per 1k input tokens for gemini-embedding-001 via Vertex AI.
 * Single source of truth — every cost-preview surface reads this constant,
 * so a pricing change is a one-line edit. Update when GCP pricing changes.
 */
export const EMBEDDING_COST_PER_1K_TOKENS = 0.00015;

/** Compute USD cost estimate for embedding `tokens` at current model rate. */
export function estimateEmbeddingCostUsd(tokens: number): number {
  return (tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS;
}
