// Runtime validation coverage for offline IndexedDB and queued sync persistence.
import 'fake-indexeddb/auto';
import { expect, it } from 'vitest';
import { offlineStore } from './offlineStore.js';

globalThis.window = globalThis;

it('persists local records, preserves queued local changes over refreshes, and removes synced operations', async () => {
  const localPig = { id: crypto.randomUUID(), aiIdentity: 'Pig_1', penTag: '01' };
  const serverPig = { id: crypto.randomUUID(), aiIdentity: 'Pig_2', penTag: '02' };
  const operation = offlineStore.createOperation('create', 'pigs', localPig.id, localPig);

  await offlineStore.saveAndQueue('pigs', localPig, operation);
  const firstSnapshot = await offlineStore.readSnapshot();
  expect(firstSnapshot.pigs).toContainEqual(localPig);
  expect(firstSnapshot.pending).toContainEqual(operation);

  await offlineStore.replaceRecords('pigs', [serverPig]);
  const mergedSnapshot = await offlineStore.readSnapshot();
  expect(mergedSnapshot.pigs).toContainEqual(localPig);
  expect(mergedSnapshot.pigs).toContainEqual(serverPig);
  expect(mergedSnapshot.pending).toContainEqual(operation);

  await offlineStore.removePending(operation.opId);
  await offlineStore.replaceRecords('pigs', [serverPig]);
  const syncedSnapshot = await offlineStore.readSnapshot();
  expect(syncedSnapshot.pigs).toEqual([serverPig]);
  expect(syncedSnapshot.pending).toHaveLength(0);
});
