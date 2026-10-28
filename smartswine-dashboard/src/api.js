const apiBaseUrl = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8765').replace(/\/$/, '');

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${apiBaseUrl}${path}`, {
      ...options,
      headers: {
        ...(options.body && !(options.body instanceof FormData) ? { 'Content-Type': 'application/json' } : {}),
        ...options.headers,
      },
    });
  } catch (error) {
    throw new Error(`Cannot reach the Smart Swine API at ${apiBaseUrl}. Start the backend and check VITE_API_URL.`);
  }

  if (!response.ok) {
    const result = await response.json().catch(() => null);
    if (response.status === 404 && result?.detail === 'Not Found') {
      throw new Error(
        `The API endpoint ${path} is unavailable. Restart the backend from the current workspace, then refresh the dashboard.`,
      );
    }
    const error = new Error(result?.detail || `API request failed with status ${response.status}.`);
    error.status = response.status;
    throw error;
  }
  if (response.status === 204) return null;
  return response.json();
}

function mapPig(record) {
  return {
    id: record.id,
    aiIdentity: record.ai_identity || 'Unassigned',
    penTag: String(record.pen_tag).padStart(2, '0'),
    stage: record.growth_stage,
    registeredWeight: Number(record.registered_weight),
    status: record.status,
    notes: record.notes,
  };
}

function mapObservation(record) {
  return {
    id: record.id,
    pigId: record.pig_id,
    timestamp: new Date(record.created_at).toLocaleString(),
    createdAt: record.created_at,
    aiIdentity: record.ai_identity || 'Unassigned',
    penTag: record.pen_tag == null ? '—' : String(record.pen_tag).padStart(2, '0'),
    weight: record.weight_kg == null ? null : Number(record.weight_kg),
    surfaceTemperatureC: record.thermal_surface_c == null ? null : Number(record.thermal_surface_c),
    behavior: record.behavior || 'Unknown',
    confidence: record.confidence == null ? null : Number(record.confidence),
    box: record.bounding_box,
  };
}

function mapAlert(record) {
  return {
    id: record.id,
    pigId: record.pig_id,
    timestamp: new Date(record.created_at).toLocaleString(),
    aiIdentity: record.ai_identity || 'Unassigned',
    penTag: record.pen_tag == null ? '—' : String(record.pen_tag).padStart(2, '0'),
    issue: record.issue,
    severity: record.severity,
    status: record.status,
    details: record.details,
  };
}

function mapFeedLog(record) {
  return {
    id: record.id,
    pigId: record.pig_id,
    date: record.fed_on,
    penTag: String(record.pen_tag).padStart(2, '0'),
    feedType: record.feed_type,
    quantityKg: Number(record.quantity_kg),
    costPerKg: Number(record.cost_per_kg),
  };
}

export const api = {
  health: () => request('/api/health'),
  pigs: {
    list: async () => (await request('/api/pigs')).map(mapPig),
    create: async (pig) => mapPig(await request('/api/pigs', {
      method: 'POST',
      body: JSON.stringify({
        id: pig.id,
        ai_identity: pig.aiIdentity,
        pen_tag: Number(pig.penTag),
        growth_stage: pig.stage,
        registered_weight: Number(pig.registeredWeight),
        status: pig.status,
        notes: pig.notes,
      }),
    })),
    delete: (id) => request(`/api/pigs/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  },
  observations: {
    list: async () => (await request('/api/observations')).map(mapObservation),
    create: async (detection) => mapObservation(await request('/api/observations', {
      method: 'POST',
      body: JSON.stringify({
        id: detection.id,
        pig_id: detection.pig_id,
        pen_tag: detection.pen_tag == null ? undefined : Number(detection.pen_tag),
        ai_identity: detection.identity || detection.class_name || detection.className,
        weight_kg: detection.weight_estimate?.weight_kg,
        thermal_surface_c: detection.surface_temperature_c,
        behavior: detection.posture
          ? `Posture: ${detection.posture.label}`
          : detection.weight_estimate
            ? 'Camera weight estimate'
            : undefined,
        confidence: detection.confidence,
        bounding_box: detection.box,
      }),
    })),
  },
  camera: {
    status: () => request('/api/camera/status'),
    thermalStatus: () => request('/api/thermal/status'),
    tapoWebRTCOffer: (sdp) => request('/api/camera/tapo-c230/offer', {
      method: 'POST',
      body: JSON.stringify({ sdp }),
    }),
    detect: (frame) => {
      const form = new FormData();
      form.append('image', frame, 'camera-frame.jpg');
      return request('/api/camera/detect', { method: 'POST', body: form });
    },
  },
  datasets: {
    list: () => request('/api/datasets'),
    upload: (file) => {
      const form = new FormData();
      form.append('file', file);
      return request('/api/datasets', { method: 'POST', body: form });
    },
    delete: (id) => request(`/api/datasets/${encodeURIComponent(id)}`, { method: 'DELETE' }),
    train: (id) => request(`/api/datasets/${encodeURIComponent(id)}/train`, { method: 'POST' }),
    trainPosture: (id) => request(`/api/datasets/${encodeURIComponent(id)}/train-posture`, { method: 'POST' }),
  },
  training: {
    status: () => request('/api/training/status'),
    postureStatus: () => request('/api/posture-training/status'),
  },
  alerts: {
    list: async () => (await request('/api/alerts')).map(mapAlert),
    updateStatus: async (id, alertStatus) => mapAlert(await request(`/api/alerts/${encodeURIComponent(id)}`, {
      method: 'PATCH',
      body: JSON.stringify({ status: alertStatus }),
    })),
  },
  feedLogs: {
    list: async () => (await request('/api/feed-logs')).map(mapFeedLog),
    create: async (entry) => mapFeedLog(await request('/api/feed-logs', {
      method: 'POST',
      body: JSON.stringify({
        id: entry.id,
        pig_id: entry.pigId,
        pen_tag: Number(entry.penTag),
        feed_type: entry.feedType,
        quantity_kg: Number(entry.quantityKg),
        cost_per_kg: Number(entry.costPerKg),
        fed_on: entry.date,
      }),
    })),
    delete: (id) => request(`/api/feed-logs/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  },
};
