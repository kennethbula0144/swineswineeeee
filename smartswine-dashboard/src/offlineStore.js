const DATABASE_NAME = 'smart-swine-local';
const DATABASE_VERSION = 1;
const RECORD_STORES = ['pigs', 'observations', 'alerts', 'feedLogs'];
const PENDING_STORE = 'pending';

let databasePromise;
let lastOperationTime = 0;

function openDatabase() {
  if (!('indexedDB' in window)) {
    return Promise.reject(new Error('This browser does not support local offline storage.'));
  }
  if (databasePromise) return databasePromise;

  databasePromise = new Promise((resolve, reject) => {
    const request = window.indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      for (const store of RECORD_STORES) {
        if (!database.objectStoreNames.contains(store)) {
          database.createObjectStore(store, { keyPath: 'id' });
        }
      }
      if (!database.objectStoreNames.contains(PENDING_STORE)) {
        database.createObjectStore(PENDING_STORE, { keyPath: 'opId' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => {
      databasePromise = null;
      reject(request.error || new Error('Could not open local farm storage.'));
    };
    request.onblocked = () => {
      databasePromise = null;
      reject(new Error('Local farm storage is blocked by another open dashboard tab. Close other tabs and retry.'));
    };
  });
  return databasePromise;
}

function transactionComplete(transaction) {
  return new Promise((resolve, reject) => {
    transaction.oncomplete = () => resolve();
    transaction.onerror = () => reject(transaction.error || new Error('Local storage transaction failed.'));
    transaction.onabort = () => reject(transaction.error || new Error('Local storage transaction was aborted.'));
  });
}

function readAll(database, storeName) {
  return new Promise((resolve, reject) => {
    const request = database.transaction(storeName, 'readonly').objectStore(storeName).getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error(`Could not read local ${storeName}.`));
  });
}

async function writeRecordAndQueue(storeName, record, operation) {
  const database = await openDatabase();
  const transaction = database.transaction([storeName, PENDING_STORE], 'readwrite');
  transaction.objectStore(storeName).put(record);
  transaction.objectStore(PENDING_STORE).put(operation);
  await transactionComplete(transaction);
}

async function deleteRecordAndQueue(storeName, id, operation) {
  const database = await openDatabase();
  const transaction = database.transaction([storeName, PENDING_STORE], 'readwrite');
  transaction.objectStore(storeName).delete(id);
  transaction.objectStore(PENDING_STORE).put(operation);
  await transactionComplete(transaction);
}

async function putRecord(storeName, record) {
  const database = await openDatabase();
  const transaction = database.transaction(storeName, 'readwrite');
  transaction.objectStore(storeName).put(record);
  await transactionComplete(transaction);
}

async function replaceRecords(storeName, records) {
  const database = await openDatabase();
  const transaction = database.transaction([storeName, PENDING_STORE], 'readwrite');
  const store = transaction.objectStore(storeName);
  const pendingRequest = transaction.objectStore(PENDING_STORE).getAll();
  pendingRequest.onsuccess = () => {
    const merged = new Map(records.map((record) => [record.id, record]));
    for (const operation of pendingRequest.result) {
      if (operation.entity !== storeName) continue;
      if (operation.type === 'delete') {
        merged.delete(operation.id);
      } else if (operation.type === 'create') {
        merged.set(operation.id, operation.payload);
      } else if (operation.type === 'updateStatus') {
        const current = merged.get(operation.id);
        if (current) merged.set(operation.id, { ...current, status: operation.payload.status });
      }
    }
    store.clear();
    for (const record of merged.values()) store.put(record);
  };
  await transactionComplete(transaction);
}

async function removePending(opId) {
  const database = await openDatabase();
  const transaction = database.transaction(PENDING_STORE, 'readwrite');
  transaction.objectStore(PENDING_STORE).delete(opId);
  await transactionComplete(transaction);
}

function createOperation(type, entity, id, payload = null) {
  lastOperationTime = Math.max(Date.now(), lastOperationTime + 1);
  return {
    opId: crypto.randomUUID(),
    type,
    entity,
    id,
    payload,
    createdAt: new Date(lastOperationTime).toISOString(),
  };
}

export const offlineStore = {
  recordStores: RECORD_STORES,
  createOperation,
  readAll: async (storeName) => readAll(await openDatabase(), storeName),
  readSnapshot: async () => {
    const database = await openDatabase();
    const records = await Promise.all(RECORD_STORES.map((store) => readAll(database, store)));
    const pending = await readAll(database, PENDING_STORE);
    for (const operation of pending) {
      lastOperationTime = Math.max(lastOperationTime, Date.parse(operation.createdAt) || 0);
    }
    return {
      ...Object.fromEntries(RECORD_STORES.map((store, index) => [store, records[index]])),
      pending: pending.sort((left, right) => left.createdAt.localeCompare(right.createdAt)),
    };
  },
  saveAndQueue: writeRecordAndQueue,
  deleteAndQueue: deleteRecordAndQueue,
  saveRecord: putRecord,
  replaceRecords,
  removePending,
  pendingCount: async () => (await readAll(await openDatabase(), PENDING_STORE)).length,
};
