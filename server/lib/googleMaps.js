const DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json";
const ISOCHRONES_URL = "https://isochrones.googleapis.com/v1/isochrones:generate";

function getServerKey() {
  const key = process.env.GOOGLE_MAPS_SERVER_KEY || process.env.GOOGLE_MAPS_API_KEY;
  if (!key) {
    throw new Error(
      "Missing GOOGLE_MAPS_SERVER_KEY (or GOOGLE_MAPS_API_KEY) in server environment"
    );
  }
  return key;
}

async function getTransitDirections({ origin, destination, transitMode, departureTime }) {
  const params = new URLSearchParams({
    origin,
    destination,
    mode: "transit",
    region: "us",
    key: getServerKey(),
  });
  if (transitMode) params.set("transit_mode", transitMode);
  params.set("departure_time", departureTime || "now");

  const res = await fetch(`${DIRECTIONS_URL}?${params.toString()}`);
  if (!res.ok) {
    throw new Error(`Google Directions API HTTP ${res.status}`);
  }
  const data = await res.json();
  if (data.status && data.status !== "OK" && data.status !== "ZERO_RESULTS") {
    throw new Error(`Google Directions API error: ${data.status} ${data.error_message || ""}`.trim());
  }
  return data;
}

// Isochrones API is a newer Google Maps Platform product (walk/drive/bike
// reachable-area polygons). Request/response shape follows the public docs
// at https://developers.google.com/maps/documentation/isochrones as of
// this writing; if Google changes field names, adjust here.
async function generateIsochrone({ lat, lon, travelMode, travelDurationSeconds, travelDirection }) {
  const body = {
    location: { latitude: lat, longitude: lon },
    travelMode: travelMode || "WALK",
    travelDuration: `${travelDurationSeconds || 600}s`,
    travelDirection: travelDirection || "FROM",
    routingPreference: "TRAFFIC_UNAWARE",
    enableSmoothing: true,
  };

  const res = await fetch(ISOCHRONES_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Goog-Api-Key": getServerKey(),
    },
    body: JSON.stringify(body),
  });

  const text = await res.text();
  let data;
  try {
    data = JSON.parse(text);
  } catch (_) {
    throw new Error(`Isochrones API returned non-JSON response (HTTP ${res.status})`);
  }
  if (!res.ok) {
    throw new Error(
      `Isochrones API HTTP ${res.status}: ${data.error?.message || text}`
    );
  }
  return data;
}

module.exports = { getTransitDirections, generateIsochrone };
