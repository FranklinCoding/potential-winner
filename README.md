# NYC Transit Nav

A local web app for getting around New York City: turn-by-turn transit
directions (Google Maps) plus live NYC subway arrival countdowns (MTA
real-time feeds), with an optional "walking radius" overlay around any
station powered by Google's Isochrones API.

## Features

- **Map of every NYC subway station**, color-coded by line, centered on your
  location (or NYC generally if you don't share it).
- **Directions** between any two points, biased toward subway routes, with
  step-by-step instructions (walk → board line X → N stops → walk) and the
  route drawn on the map. Powered by the Google Directions API (transit mode).
- **Live arrivals**: click any station to see real train countdowns in both
  directions, refreshed every 20 seconds, decoded straight from MTA's
  GTFS-realtime feeds (no MTA API key required).
- **Nearby stations panel** that updates as you pan the map.
- **Walking radius (isochrone)**: from a station's detail panel, draw the
  area reachable on foot in 5–20 minutes, using Google's Isochrones API.

## Prerequisites

- Node.js 18+ (uses the built-in `fetch`)
- A Google Cloud project with billing enabled and a Google Maps Platform API
  key

## 1. Get a Google Maps Platform API key

Google's setup guide: <https://developers.google.com/maps/documentation/isochrones/get-api-key>

1. Create (or pick) a project in the [Google Cloud Console](https://console.cloud.google.com/).
2. Enable these APIs for the project:
   - **Maps JavaScript API** (renders the map in the browser)
   - **Places API** (address autocomplete)
   - **Directions API** (transit routing)
   - **Isochrones API** (walking-radius overlay — this is the API from the
     link above; it may need to be requested/enabled separately as it's a
     newer Google Maps Platform product)
3. Create an API key under **APIs & Services → Credentials**.
4. (Recommended) Create **two** keys so each is scoped as narrowly as
   possible:
   - A **browser key** restricted by HTTP referrer (e.g.
     `http://localhost:3000/*`), used for the map + autocomplete.
   - A **server key** restricted by IP (or left unrestricted, since it never
     reaches the browser), used for Directions/Isochrones calls from the
     Node server.

   One key with no restrictions also works for local testing — just set
   `GOOGLE_MAPS_API_KEY` and skip the other two.

## 2. Configure and run

```bash
npm install
cp .env.example .env
# edit .env and paste in your key(s)
npm start
```

Then open <http://localhost:3000>.

## How it works

- `server/index.js` — Express app serving the static frontend and JSON API.
- `server/routes/*` — `/api/config`, `/api/stations`, `/api/arrivals/:id`,
  `/api/directions`, `/api/isochrone`.
- `server/lib/stationsStore.js` — fetches MTA's public
  [`Stations.csv`](http://web.mta.info/developers/data/nyct/subway/Stations.csv)
  (station names, lines, coordinates, GTFS stop IDs), caches it to
  `data/stations.cache.json` so the app still works if that endpoint is
  briefly unavailable, and refreshes it in the background daily.
- `server/lib/gtfsRealtime.js` — fetches and decodes MTA's GTFS-realtime
  protobuf feeds (one per line group, e.g. `gtfs-ace`, `gtfs-nqrw`) with the
  `gtfs-realtime-bindings` package, and extracts next-arrival times for a
  given station's GTFS stop ID. MTA's real-time feeds have not required an
  API key since 2019.
- `server/lib/googleMaps.js` — thin server-side proxy to Google's Directions
  and Isochrones REST APIs, so your server key never reaches the browser.
- `public/` — plain HTML/CSS/JS frontend (no build step) using the Google
  Maps JavaScript API, Places Autocomplete, and `fetch()` against the routes
  above.

## Known limitations

- The Isochrones API is a newer/preview Google Maps Platform product; its
  exact JSON response shape wasn't verified against a live key while
  building this (network access was restricted in the build environment).
  `extractIsochronePolygons()` in `public/app.js` parses a couple of
  plausible shapes defensively — if walking-radius polygons don't render for
  you, check your browser console / server logs for the raw response and
  adjust that function to match.
- Arrivals are matched by a station's single GTFS stop ID; large transfer
  complexes with multiple platform IDs (e.g. Times Sq–42 St) currently only
  show trains for the specific platform whose ID matches, not every line
  passing through the complex.
- No offline/PWA support — this is meant to be run locally with
  `npm start` while you have internet access.

## Troubleshooting

- **Map doesn't load / blank screen**: check the browser console. Usually a
  missing/invalid `GOOGLE_MAPS_BROWSER_KEY`, a referrer restriction that
  doesn't match `http://localhost:3000`, or an API not enabled in Cloud
  Console.
- **"No upcoming trains reported"**: MTA feeds occasionally have gaps for a
  given line/time — try again in a few seconds, or check
  [MTA's service status](https://www.mta.info/status).
- **Directions fail server-side**: check the server logs; the error message
  from `/api/directions` passes through Google's own `status`/`error_message`.
