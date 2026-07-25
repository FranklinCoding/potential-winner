const GtfsRealtimeBindings = require("gtfs-realtime-bindings");

const FEED_BASE = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct";
const FEEDS = {
  "1234567S": `${FEED_BASE}/gtfs`,
  ace: `${FEED_BASE}/gtfs-ace`,
  bdfm: `${FEED_BASE}/gtfs-bdfm`,
  g: `${FEED_BASE}/gtfs-g`,
  jz: `${FEED_BASE}/gtfs-jz`,
  nqrw: `${FEED_BASE}/gtfs-nqrw`,
  l: `${FEED_BASE}/gtfs-l`,
  si: `${FEED_BASE}/gtfs-si`,
};

// Route ID -> feed group key. Covers the standard NYCT subway route set,
// including shuttles, per MTA's published GTFS-realtime feed grouping.
const ROUTE_TO_FEED = {
  1: "1234567S", 2: "1234567S", 3: "1234567S", 4: "1234567S", 5: "1234567S",
  "5X": "1234567S", 6: "1234567S", "6X": "1234567S", 7: "1234567S", "7X": "1234567S",
  GS: "1234567S", S: "1234567S",
  A: "ace", C: "ace", E: "ace", H: "ace", FS: "ace",
  B: "bdfm", D: "bdfm", F: "bdfm", M: "bdfm",
  G: "g",
  J: "jz", Z: "jz",
  N: "nqrw", Q: "nqrw", R: "nqrw", W: "nqrw",
  L: "l",
  SI: "si", SIR: "si",
};

const CACHE_TTL_MS = 20 * 1000;
const cache = new Map(); // feedGroup -> { fetchedAt, entities }

function feedGroupsForRoutes(routes) {
  const groups = new Set();
  for (const r of routes) {
    const group = ROUTE_TO_FEED[r.toUpperCase()];
    if (group) groups.add(group);
  }
  return Array.from(groups);
}

async function fetchFeed(groupKey) {
  const cached = cache.get(groupKey);
  if (cached && Date.now() - cached.fetchedAt < CACHE_TTL_MS) {
    return cached.entities;
  }

  const url = FEEDS[groupKey];
  if (!url) throw new Error(`Unknown feed group: ${groupKey}`);

  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`MTA realtime feed ${groupKey} fetch failed: HTTP ${res.status}`);
  }
  const buffer = Buffer.from(await res.arrayBuffer());
  const feed = GtfsRealtimeBindings.transit_realtime.FeedMessage.decode(buffer);
  cache.set(groupKey, { fetchedAt: Date.now(), entities: feed.entity });
  return feed.entity;
}

// stopId is the base GTFS Stop ID (no N/S suffix), e.g. "127" for Times Sq-42 St.
function extractArrivals(entities, stopId) {
  const results = [];
  for (const entity of entities) {
    const tripUpdate = entity.tripUpdate;
    if (!tripUpdate || !tripUpdate.stopTimeUpdate) continue;
    const routeId = tripUpdate.trip?.routeId;

    for (const stu of tripUpdate.stopTimeUpdate) {
      if (!stu.stopId || !stu.stopId.startsWith(stopId)) continue;
      const direction = stu.stopId.slice(stopId.length); // "N" or "S"
      if (direction !== "N" && direction !== "S") continue;

      const arrivalTime = stu.arrival?.time
        ? Number(stu.arrival.time)
        : stu.departure?.time
        ? Number(stu.departure.time)
        : null;
      if (!arrivalTime) continue;

      results.push({
        routeId,
        direction,
        tripId: tripUpdate.trip?.tripId,
        arrivalEpochSeconds: arrivalTime,
      });
    }
  }
  return results;
}

async function getArrivalsForStation(station) {
  const groups = feedGroupsForRoutes(station.routes);
  const feedResults = await Promise.allSettled(groups.map((g) => fetchFeed(g)));

  const errors = [];
  let arrivals = [];
  feedResults.forEach((r, idx) => {
    if (r.status === "fulfilled") {
      arrivals = arrivals.concat(extractArrivals(r.value, station.id));
    } else {
      errors.push(`${groups[idx]}: ${r.reason.message}`);
    }
  });

  const nowSec = Date.now() / 1000;
  arrivals = arrivals
    .filter((a) => a.arrivalEpochSeconds >= nowSec - 30) // drop just-departed noise
    .sort((a, b) => a.arrivalEpochSeconds - b.arrivalEpochSeconds)
    .map((a) => ({
      ...a,
      minutesAway: Math.max(0, Math.round((a.arrivalEpochSeconds - nowSec) / 60)),
      directionLabel:
        a.direction === "N" ? station.northLabel || "Northbound" : station.southLabel || "Southbound",
    }));

  return { arrivals, errors, feedGroupsQueried: groups };
}

module.exports = { getArrivalsForStation, feedGroupsForRoutes };
