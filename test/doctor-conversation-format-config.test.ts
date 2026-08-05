import { describe, expect, test } from 'bun:test';

import { computeConversationFormatCoverageCheck } from '../src/commands/doctor.ts';
import { ALLOWED_TYPES } from '../src/commands/extract-conversation-facts.ts';
import type { BrainEngine } from '../src/core/engine.ts';

function fakeEngine(configValue: string | null, requestedTypes: string[]): BrainEngine {
  return {
    getConfig: async (key: string) =>
      key === 'cycle.conversation_facts_backfill.types' ? configValue : null,
    listPages: async (filters: { type?: string }) => {
      requestedTypes.push(filters.type ?? '');
      return [];
    },
  } as unknown as BrainEngine;
}

describe('doctor conversation format coverage config', () => {
  test('scans only configured conversation fact types', async () => {
    const requestedTypes: string[] = [];
    const engine = fakeEngine(JSON.stringify(['conversation']), requestedTypes);

    const check = await computeConversationFormatCoverageCheck(engine);

    expect(check.status).toBe('ok');
    expect(check.message).toContain('No configured conversation-type pages');
    expect(requestedTypes).toEqual(['conversation']);
    expect(requestedTypes).not.toContain('meeting');
  });

  test('falls back to the full allowlist for invalid config', async () => {
    const requestedTypes: string[] = [];
    const engine = fakeEngine('{not-json', requestedTypes);

    const check = await computeConversationFormatCoverageCheck(engine);

    expect(check.status).toBe('ok');
    expect(requestedTypes).toEqual([...ALLOWED_TYPES]);
    expect(requestedTypes).toContain('meeting');
  });
});
