/**
 * v0.40.2.0 — LongMemEval trajectory routing tests.
 *
 * End-to-end through `runEvalLongMemEval` with BOTH the answer-gen
 * client and the extractor client stubbed. Verifies:
 *   - Trajectory routing fires for temporal/knowledge_update intents.
 *   - Trajectory block lands in the answer-gen prompt.
 *   - `--no-trajectory` bypasses extraction + injection.
 *   - JSON envelope includes the new fields when enabled, omits when
 *     disabled.
 *   - methodology_note appears on every per-question row when on.
 *
 * Hermetic — uses PGLite in-memory + a small fixture file.
 */

import { describe, test, expect, beforeAll, afterAll } from 'bun:test';
import { writeFileSync, mkdtempSync, rmSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';
import { runEvalLongMemEval } from '../src/commands/eval-longmemeval.ts';
import type { ThinkLLMClient } from '../src/core/think/index.ts';
import {
  __setGenerateTextTransportForTests,
  configureGateway,
  resetGateway,
} from '../src/core/ai/gateway.ts';
import type { LongMemEvalQuestion } from '../src/eval/longmemeval/adapter.ts';
import { withEnv } from './helpers/with-env.ts';

let tmpDir: string;
let datasetPath: string;
let outputPath: string;

const ANSWER_GEN_MARKER = '__ANSWER_GEN__';
const EXTRACTOR_MARKER = '__EXTRACTOR__';

beforeAll(() => {
  tmpDir = mkdtempSync(join(tmpdir(), 'lme-trajectory-'));
  datasetPath = join(tmpDir, 'dataset.jsonl');
  outputPath = join(tmpDir, 'output.jsonl');

  const questions: LongMemEvalQuestion[] = [
    {
      question_id: 'q1-temporal',
      question_type: 'temporal-reasoning',
      question: 'When did I last meet with marco?',
      answer: 'placeholder',
      haystack_sessions: [
        { session_id: 'sess-1', turns: [
          { role: 'user', content: 'Met with marco at Blue Bottle for coffee' },
        ]},
      ],
      answer_session_ids: ['sess-1'],
      haystack_dates: ['2026-01-15'],
    },
    {
      question_id: 'q2-other',
      question_type: 'single-session-user',
      question: 'Summarize the conversation',
      answer: 'placeholder',
      haystack_sessions: [
        { session_id: 'sess-2', turns: [
          { role: 'user', content: 'Random open-ended chat' },
        ]},
      ],
      answer_session_ids: ['sess-2'],
      haystack_dates: ['2026-02-01'],
    },
  ];

  writeFileSync(datasetPath, questions.map(q => JSON.stringify(q)).join('\n'));
});

afterAll(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});

interface StubState {
  answerCalls: string[];
  extractorCalls: number;
}

function stubClients(state: StubState): {
  answerClient: ThinkLLMClient;
  extractorClient: ThinkLLMClient;
} {
  const answerClient: ThinkLLMClient = {
    create: async (params) => {
      const userMsg = params.messages[0]?.content;
      state.answerCalls.push(typeof userMsg === 'string' ? userMsg : '');
      return {
        id: 'a', type: 'message', role: 'assistant', model: 's',
        stop_reason: 'end_turn', stop_sequence: null,
        usage: { input_tokens: 1, output_tokens: 1, cache_creation_input_tokens: 0, cache_read_input_tokens: 0, server_tool_use: null, service_tier: null },
        content: [{ type: 'text', text: `${ANSWER_GEN_MARKER} stubbed answer` }],
      } as never;
    },
  };
  const extractorClient: ThinkLLMClient = {
    create: async (params) => {
      state.extractorCalls++;
      const userMsg = params.messages[0]?.content;
      const text = typeof userMsg === 'string' ? userMsg : '';
      // Stubbed extractor returns one event row for sess-1 (marco meeting),
      // empty for sess-2.
      const claims = text.includes('marco at Blue Bottle')
        ? [{
            entity: 'marco',
            metric: null,
            value: null,
            unit: null,
            period: null,
            event_type: 'meeting',
            valid_from: '2026-01-15',
            text: `${EXTRACTOR_MARKER} met marco at Blue Bottle`,
          }]
        : [];
      return {
        id: 'e', type: 'message', role: 'assistant', model: 's',
        stop_reason: 'end_turn', stop_sequence: null,
        usage: { input_tokens: 1, output_tokens: 1, cache_creation_input_tokens: 0, cache_read_input_tokens: 0, server_tool_use: null, service_tier: null },
        content: [{ type: 'text', text: JSON.stringify(claims) }],
      } as never;
    },
  };
  return { answerClient, extractorClient };
}

function readOutput(): Array<Record<string, unknown>> {
  const raw = readFileSync(outputPath, 'utf-8').trim();
  return raw.split('\n').map(line => JSON.parse(line));
}

describe('runEvalLongMemEval — trajectory routing on (default)', () => {
  test('temporal-reasoning question gets the trajectory block in the prompt', async () => {
    const state: StubState = { answerCalls: [], extractorCalls: 0 };
    const { answerClient, extractorClient } = stubClients(state);
    await withEnv({ GBRAIN_MODEL: undefined }, async () => {
      await runEvalLongMemEval(
        [datasetPath, '--keyword-only', '--output', outputPath],
        { client: answerClient, extractorClient },
      );
    });

    // q1 (temporal) → answer-gen call must include trajectory block.
    // q2 (other) → no trajectory block.
    expect(state.answerCalls.length).toBe(2);
    expect(state.answerCalls[0]).toContain('Known trajectory:');
    expect(state.answerCalls[0]).toContain('<trajectory entity="marco"');
    expect(state.answerCalls[1]).not.toContain('Known trajectory:');

    // Extractor fires for both sessions even though only q1 routes.
    // (Extraction happens during import, ahead of intent classification.)
    expect(state.extractorCalls).toBeGreaterThanOrEqual(2);

    // Envelope shape.
    const out = readOutput();
    expect(out.length).toBe(2);
    expect(out[0].intent).toBe('temporal');
    expect(out[0].trajectory_points).toBeGreaterThan(0);
    expect(out[0].entity_resolved).toBe('marco');
    expect(out[0].resolution_source).toBe('fallback_slugify');  // benchmark brain has no people/marco page
    expect(out[0].methodology_note).toBe('extractor=haiku-preprocess-full-haystack-v1');
    expect(out[1].intent).toBe('other');
    expect(out[1].trajectory_points).toBe(0);
    expect(out[1].entity_resolved).toBe(null);
  });
});

describe('runEvalLongMemEval — gateway-backed non-Anthropic clients', () => {
  test('routes both answer generation and extraction through DeepSeek', async () => {
    const model = 'deepseek:deepseek-v4-flash';
    const calls: Array<{ maxOutputTokens?: number; providerOptions?: Record<string, any>; system?: unknown }> = [];

    resetGateway();
    __setGenerateTextTransportForTests(async (args: any) => {
      calls.push(args);
      const text = String(args.system).startsWith('You extract typed claims')
        ? '[]'
        : 'gateway-deepseek-answer';
      return {
        text,
        content: [{ type: 'text', text }],
        finishReason: 'stop',
        usage: { inputTokens: 1, outputTokens: 1 },
      } as any;
    });
    configureGateway({
      chat_model: model,
      provider_chat_options: {
        [model]: { thinking: { type: 'disabled' } },
      },
      env: { DEEPSEEK_API_KEY: 'fake' },
    });

    try {
      await withEnv({ ANTHROPIC_API_KEY: undefined, GBRAIN_MODEL: model }, async () => {
        await runEvalLongMemEval(
          [datasetPath, '--keyword-only', '--limit', '1', '--model', model, '--output', outputPath],
        );
      });

      expect(calls).toHaveLength(2);
      expect(calls.map(call => call.maxOutputTokens)).toEqual([2000, 512]);
      expect(calls.every(call => call.providerOptions?.deepseek?.thinking?.type === 'disabled')).toBe(true);
      expect(calls.some(call => String(call.system).startsWith('You extract typed claims'))).toBe(true);
      expect(calls.some(call => String(call.system).includes('UNTRUSTED user-generated data'))).toBe(true);
      const row = readOutput()[0];
      expect(row.hypothesis).toBe('gateway-deepseek-answer');
      expect(row.methodology_note).toBe(
        'extractor=deepseek:deepseek-v4-flash-preprocess-full-haystack-v2',
      );
    } finally {
      __setGenerateTextTransportForTests(null);
      resetGateway();
    }
  }, 60_000);
});

describe('runEvalLongMemEval — --no-trajectory bypasses both extractor and injection', () => {
  test('--no-trajectory: extractor never called, no trajectory block, envelope omits new fields', async () => {
    const state: StubState = { answerCalls: [], extractorCalls: 0 };
    const { answerClient, extractorClient } = stubClients(state);
    await withEnv({ GBRAIN_MODEL: undefined }, async () => {
      await runEvalLongMemEval(
        [datasetPath, '--keyword-only', '--no-trajectory', '--output', outputPath],
        { client: answerClient, extractorClient },
      );
    });
    expect(state.extractorCalls).toBe(0);
    expect(state.answerCalls.length).toBe(2);
    expect(state.answerCalls[0]).not.toContain('Known trajectory:');
    expect(state.answerCalls[1]).not.toContain('Known trajectory:');

    // Envelope: trajectory fields absent when --no-trajectory.
    const out = readOutput();
    expect(out[0].intent).toBeUndefined();
    expect(out[0].trajectory_points).toBeUndefined();
    expect(out[0].methodology_note).toBeUndefined();
  });
});

describe('runEvalLongMemEval — methodology_note presence', () => {
  test('default run stamps methodology_note on every routed row', async () => {
    const state: StubState = { answerCalls: [], extractorCalls: 0 };
    const { answerClient, extractorClient } = stubClients(state);
    await withEnv({ GBRAIN_MODEL: undefined }, async () => {
      await runEvalLongMemEval(
        [datasetPath, '--keyword-only', '--output', outputPath],
        { client: answerClient, extractorClient },
      );
    });
    const out = readOutput();
    for (const row of out) {
      expect(row.methodology_note).toBe('extractor=haiku-preprocess-full-haystack-v1');
    }
  });
});

describe('runEvalLongMemEval — perf gate preserved', () => {
  // v0.40.10 flake-hardening: the perf assertion's ceiling is mode-aware.
  // Solo run (10s) is the tight gate — catches real harness regressions.
  // Shard run (60s) is the loose gate — CPU contention with 8 parallel
  // shards routinely 3-5x's wallclock, which is contention, not a code
  // regression. `SHARD=N/M` env var is set by scripts/run-unit-parallel.sh
  // when running under the parallel wrapper. Per-test timeout always bumped
  // to outrun bun's 5s default.
  const SHARD_MODE = !!process.env.SHARD;
  const PERF_CEILING_MS = SHARD_MODE ? 60_000 : 10_000;
  test(`run completes for the 2-question fixture in under ${PERF_CEILING_MS / 1000}s with stubs`, async () => {
    const state: StubState = { answerCalls: [], extractorCalls: 0 };
    const { answerClient, extractorClient } = stubClients(state);
    const start = Date.now();
    await withEnv({ GBRAIN_MODEL: undefined }, async () => {
      await runEvalLongMemEval(
        [datasetPath, '--keyword-only', '--output', outputPath],
        { client: answerClient, extractorClient },
      );
    });
    const elapsed = Date.now() - start;
    expect(elapsed).toBeLessThan(PERF_CEILING_MS);
  }, 90_000);
});
