const fs = require("fs");
const path = require("path");
const { parseCsv } = require("./csv");

const STATIONS_URL =
  "http://web.mta.info/developers/data/nyct/subway/Stations.csv";
const CACHE_PATH = path.join(__dirname, "..", "..", "data", "stations.cache.json");
const REFRESH_INTERVAL_MS = 24 * 60 * 60 * 1000; // 24h

let stations = [];
let lastLoadedAt = null;
let loadPromise = null;

function toFloat(v) {
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : null;
}

// Pull whichever header variant MTA is using for a field. The published
// Stations.csv schema has been stable for years but this keeps us from
// hard-failing on a minor rename.
function pick(row, candidates) {
  for (const key of candidates) {
    if (row[key] !== undefined && row[key] !== "") return row[key];
  }
  return "";
}

function normalize(rows) {
  const byId = new Map();
  for (const row of rows) {
    const id = pick(row, ["GTFS Stop ID", "GTFS_Stop_ID", "gtfs_stop_id"]);
    const lat = toFloat(pick(row, ["GTFS Latitude", "GTFS_Latitude"]));
    const lon = toFloat(pick(row, ["GTFS Longitude", "GTFS_Longitude"]));
    if (!id || lat === null || lon === null) continue;

    const routesRaw = pick(row, ["Daytime Routes", "Daytime_Routes"]);
    const routes = routesRaw.split(/\s+/).filter(Boolean);

    byId.set(id, {
      id,
      complexId: pick(row, ["Complex ID", "Complex_ID"]),
      name: pick(row, ["Stop Name", "Stop_Name"]),
      borough: pick(row, ["Borough"]),
      routes,
      lat,
      lon,
      northLabel: pick(row, ["North Direction Label", "North_Direction_Label"]),
      southLabel: pick(row, ["South Direction Label", "South_Direction_Label"]),
    });
  }
  return Array.from(byId.values());
}

function haversineMiles(lat1, lon1, lat2, lon2) {
  const R = 3958.8;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

async function fetchFromMta() {
  const res = await fetch(STATIONS_URL);
  if (!res.ok) {
    throw new Error(`MTA Stations.csv fetch failed: HTTP ${res.status}`);
  }
  const text = await res.text();
  const rows = parseCsv(text);
  const normalized = normalize(rows);
  if (normalized.length === 0) {
    throw new Error("Parsed 0 stations from MTA Stations.csv (unexpected format)");
  }
  return normalized;
}

function readCache() {
  try {
    const raw = fs.readFileSync(CACHE_PATH, "utf8");
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed.stations) && parsed.stations.length > 0) {
      return parsed.stations;
    }
  } catch (_) {
    // no cache yet
  }
  return null;
}

function writeCache(list) {
  try {
    fs.mkdirSync(path.dirname(CACHE_PATH), { recursive: true });
    fs.writeFileSync(
      CACHE_PATH,
      JSON.stringify({ fetchedAt: new Date().toISOString(), stations: list }, null, 2)
    );
  } catch (err) {
    console.warn("[stations] failed to write cache:", err.message);
  }
}

async function load({ force = false } = {}) {
  if (!force && loadPromise) return loadPromise;

  loadPromise = (async () => {
    try {
      const fresh = await fetchFromMta();
      stations = fresh;
      lastLoadedAt = new Date();
      writeCache(fresh);
      console.log(`[stations] loaded ${fresh.length} stations from MTA`);
    } catch (err) {
      console.warn(`[stations] live fetch failed (${err.message}), trying cache`);
      const cached = readCache();
      if (cached) {
        stations = cached;
        lastLoadedAt = new Date();
        console.log(`[stations] loaded ${cached.length} stations from disk cache`);
      } else if (stations.length === 0) {
        throw err;
      }
    }
    return stations;
  })();

  return loadPromise;
}

function ensureFreshInBackground() {
  if (!lastLoadedAt || Date.now() - lastLoadedAt.getTime() > REFRESH_INTERVAL_MS) {
    load({ force: true }).catch((err) =>
      console.warn("[stations] background refresh failed:", err.message)
    );
  }
}

async function getAll() {
  await load();
  ensureFreshInBackground();
  return stations;
}

async function getById(id) {
  const all = await getAll();
  return all.find((s) => s.id === id) || null;
}

async function search(query, limit = 10) {
  const all = await getAll();
  const q = query.trim().toLowerCase();
  if (!q) return [];
  return all
    .filter((s) => s.name.toLowerCase().includes(q))
    .slice(0, limit);
}

async function nearby(lat, lon, limit = 8) {
  const all = await getAll();
  return all
    .map((s) => ({ ...s, distanceMiles: haversineMiles(lat, lon, s.lat, s.lon) }))
    .sort((a, b) => a.distanceMiles - b.distanceMiles)
    .slice(0, limit);
}

module.exports = { load, getAll, getById, search, nearby };
