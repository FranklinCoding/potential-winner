(() => {
  const LINE_COLORS = {
    1: "#EE352E", 2: "#EE352E", 3: "#EE352E",
    4: "#00933C", 5: "#00933C", 6: "#00933C", "6X": "#00933C",
    7: "#B933AD", "7X": "#B933AD",
    A: "#0039A6", C: "#0039A6", E: "#0039A6",
    B: "#FF6319", D: "#FF6319", F: "#FF6319", M: "#FF6319",
    G: "#6CBE45",
    J: "#996633", Z: "#996633",
    L: "#A7A9AC",
    N: "#FCCC0A", Q: "#FCCC0A", R: "#FCCC0A", W: "#FCCC0A",
    S: "#808183", GS: "#808183", FS: "#808183", H: "#808183",
    SI: "#0078C6", SIR: "#0078C6",
  };
  const lineColor = (route) => LINE_COLORS[String(route).toUpperCase()] || "#555b66";
  const textColorFor = (bg) => (bg === "#FCCC0A" ? "#1a1a1a" : "#ffffff");

  const state = {
    map: null,
    userLocation: null,
    stationMarkers: new Map(), // id -> marker
    allStations: [],
    activeStationId: null,
    routePolyline: null,
    walkshedPolygons: [],
    arrivalsPollHandle: null,
    nearbyPollHandle: null,
    nycCenter: { lat: 40.7128, lng: -74.006 },
    nycBounds: null,
  };

  const $ = (id) => document.getElementById(id);

  async function fetchJson(url, opts) {
    const res = await fetch(url, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(data.error || `Request failed: HTTP ${res.status}`);
    }
    return data;
  }

  function minutesLabel(m) {
    if (m <= 0) return "due";
    return `${m} min`;
  }

  // ---------- Boot ----------

  async function boot() {
    let config;
    try {
      config = await fetchJson("/api/config");
    } catch (err) {
      showConfigError(`Couldn't reach the app server: ${err.message}`);
      return;
    }

    state.nycCenter = config.nycCenter || state.nycCenter;
    state.nycBounds = config.nycBounds || null;

    if (!config.hasBrowserKey) {
      showConfigError(
        `No Google Maps API key configured. Add <code>GOOGLE_MAPS_BROWSER_KEY</code> ` +
          `to your <code>.env</code> file (see README.md), then restart the server.`
      );
      return;
    }

    loadGoogleMaps(config.googleMapsBrowserKey);
  }

  function showConfigError(html) {
    const el = $("config-error");
    el.innerHTML = html;
    el.classList.remove("hidden");
  }

  function loadGoogleMaps(key) {
    window.__initMap = initMap;
    const script = document.createElement("script");
    script.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(
      key
    )}&libraries=places,geometry&callback=__initMap&v=weekly`;
    script.async = true;
    script.onerror = () =>
      showConfigError("Failed to load the Google Maps JavaScript API. Check your API key and enabled APIs.");
    document.head.appendChild(script);
  }

  // ---------- Map setup ----------

  function initMap() {
    state.map = new google.maps.Map($("map"), {
      center: state.nycCenter,
      zoom: 13,
      mapId: "NYC_TRANSIT_NAV",
      disableDefaultUI: false,
      clickableIcons: false,
    });

    setupAutocomplete();
    setupUserLocation();
    loadAllStations();

    state.map.addListener("idle", debounce(refreshNearbyForMapCenter, 400));

    $("get-directions-btn").addEventListener("click", handleGetDirections);
    $("refresh-nearby-btn").addEventListener("click", refreshNearbyForMapCenter);
    $("close-station-panel").addEventListener("click", closeStationPanel);
    $("show-walkshed-btn").addEventListener("click", handleShowWalkshed);
    $("clear-walkshed-btn").addEventListener("click", clearWalkshed);
  }

  function debounce(fn, ms) {
    let t;
    return (...args) => {
      clearTimeout(t);
      t = setTimeout(() => fn(...args), ms);
    };
  }

  function setupAutocomplete() {
    const bounds = state.nycBounds
      ? new google.maps.LatLngBounds(
          { lat: state.nycBounds.south, lng: state.nycBounds.west },
          { lat: state.nycBounds.north, lng: state.nycBounds.east }
        )
      : undefined;

    ["origin-input", "destination-input"].forEach((id) => {
      const input = $(id);
      const ac = new google.maps.places.Autocomplete(input, {
        bounds,
        strictBounds: false,
        componentRestrictions: { country: "us" },
        fields: ["formatted_address", "geometry", "name"],
      });
      ac.bindTo("bounds", state.map);
    });
  }

  function setupUserLocation() {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        state.userLocation = { lat: pos.coords.latitude, lng: pos.coords.longitude };
        state.map.setCenter(state.userLocation);
        new google.maps.Marker({
          position: state.userLocation,
          map: state.map,
          title: "You are here",
          icon: {
            path: google.maps.SymbolPath.CIRCLE,
            scale: 7,
            fillColor: "#4285F4",
            fillOpacity: 1,
            strokeColor: "#ffffff",
            strokeWeight: 2,
          },
        });
      },
      () => {
        /* geolocation denied/unavailable — silently fall back to NYC center */
      },
      { timeout: 8000 }
    );
  }

  // ---------- Stations & markers ----------

  async function loadAllStations() {
    try {
      const { stations } = await fetchJson("/api/stations");
      state.allStations = stations;
      stations.forEach(renderStationMarker);
    } catch (err) {
      console.warn("Failed to load stations:", err.message);
    }
  }

  function renderStationMarker(station) {
    if (state.stationMarkers.has(station.id)) return;
    const primaryColor = lineColor(station.routes[0]);
    const marker = new google.maps.Marker({
      position: { lat: station.lat, lng: station.lon },
      map: state.map,
      title: `${station.name} (${station.routes.join(" ")})`,
      icon: {
        path: google.maps.SymbolPath.CIRCLE,
        scale: 5,
        fillColor: primaryColor,
        fillOpacity: 0.95,
        strokeColor: "#0f1115",
        strokeWeight: 1,
      },
    });
    marker.addListener("click", () => openStationPanel(station.id));
    state.stationMarkers.set(station.id, marker);
  }

  function linePillsHtml(routes) {
    return routes
      .map((r) => {
        const bg = lineColor(r);
        return `<span class="line-pill" style="background:${bg};color:${textColorFor(bg)}">${r}</span>`;
      })
      .join("");
  }

  // ---------- Nearby panel ----------

  async function refreshNearbyForMapCenter() {
    if (!state.map) return;
    const center = state.map.getCenter();
    const statusEl = $("nearby-status");
    statusEl.textContent = "Loading…";
    statusEl.classList.remove("error");
    try {
      const { stations } = await fetchJson(
        `/api/stations/nearby?lat=${center.lat()}&lon=${center.lng()}&limit=8`
      );
      renderNearbyList(stations);
      statusEl.textContent = "";
    } catch (err) {
      statusEl.textContent = err.message;
      statusEl.classList.add("error");
    }
  }

  function renderNearbyList(stations) {
    const list = $("nearby-list");
    list.innerHTML = "";
    stations.forEach((s) => {
      const card = document.createElement("div");
      card.className = "station-card";
      card.innerHTML = `
        <div class="name">${s.name}</div>
        <div class="dist">${s.distanceMiles.toFixed(2)} mi away</div>
        <div class="line-pills">${linePillsHtml(s.routes)}</div>
      `;
      card.addEventListener("click", () => {
        state.map.panTo({ lat: s.lat, lng: s.lon });
        openStationPanel(s.id);
      });
      list.appendChild(card);
    });
  }

  // ---------- Station detail panel ----------

  async function openStationPanel(stationId) {
    state.activeStationId = stationId;
    const station = state.allStations.find((s) => s.id === stationId);
    if (!station) return;

    $("station-panel").classList.remove("hidden");
    $("station-panel-name").textContent = station.name;
    $("station-panel-lines").innerHTML = linePillsHtml(station.routes);
    $("station-panel-arrivals").innerHTML = `<div class="status-text">Loading arrivals…</div>`;

    await loadArrivals(stationId);

    if (state.arrivalsPollHandle) clearInterval(state.arrivalsPollHandle);
    state.arrivalsPollHandle = setInterval(() => {
      if (state.activeStationId === stationId) loadArrivals(stationId);
    }, 20000);
  }

  async function loadArrivals(stationId) {
    const container = $("station-panel-arrivals");
    try {
      const { arrivals, errors } = await fetchJson(`/api/arrivals/${encodeURIComponent(stationId)}`);
      if (arrivals.length === 0) {
        container.innerHTML = `<div class="status-text">No upcoming trains reported right now.</div>`;
      } else {
        container.innerHTML = arrivals
          .slice(0, 10)
          .map((a) => {
            const bg = lineColor(a.routeId);
            return `
              <div class="arrival-row">
                <span>
                  <span class="line-pill" style="background:${bg};color:${textColorFor(
              bg
            )};width:18px;height:18px;font-size:0.65rem;display:inline-flex;vertical-align:middle;margin-right:6px;">${
              a.routeId || "?"
            }</span>
                  ${a.directionLabel}
                </span>
                <span class="mins">${minutesLabel(a.minutesAway)}</span>
              </div>`;
          })
          .join("");
      }
      if (errors && errors.length) {
        container.innerHTML += `<div class="status-text error">${errors.join("; ")}</div>`;
      }
    } catch (err) {
      container.innerHTML = `<div class="status-text error">${err.message}</div>`;
    }
  }

  function closeStationPanel() {
    $("station-panel").classList.add("hidden");
    state.activeStationId = null;
    if (state.arrivalsPollHandle) clearInterval(state.arrivalsPollHandle);
  }

  // ---------- Directions ----------

  async function handleGetDirections() {
    const originText = $("origin-input").value.trim();
    const destinationText = $("destination-input").value.trim();
    const statusEl = $("directions-status");
    const resultsEl = $("directions-results");
    statusEl.classList.remove("error");
    resultsEl.innerHTML = "";

    if (!destinationText) {
      statusEl.textContent = "Enter a destination.";
      statusEl.classList.add("error");
      return;
    }

    let origin = originText;
    if (!origin) {
      if (state.userLocation) {
        origin = `${state.userLocation.lat},${state.userLocation.lng}`;
      } else {
        statusEl.textContent = "Enter a starting point (couldn't detect your location).";
        statusEl.classList.add("error");
        return;
      }
    }

    statusEl.textContent = "Finding route…";
    try {
      const params = new URLSearchParams({ origin, destination: destinationText });
      if ($("subway-only").checked) params.set("transitMode", "subway");
      const data = await fetchJson(`/api/directions?${params.toString()}`);

      if (!data.routes || data.routes.length === 0) {
        statusEl.textContent = "No transit route found for that trip.";
        statusEl.classList.add("error");
        return;
      }
      statusEl.textContent = "";
      renderDirections(data.routes[0]);
    } catch (err) {
      statusEl.textContent = err.message;
      statusEl.classList.add("error");
    }
  }

  function renderDirections(route) {
    clearRoutePolyline();
    const resultsEl = $("directions-results");
    const leg = route.legs[0];

    const summary = document.createElement("div");
    summary.className = "route-summary";
    summary.textContent = `${leg.duration.text} · ${leg.distance.text}`;
    resultsEl.appendChild(summary);

    leg.steps.forEach((step) => {
      const stepEl = document.createElement("div");
      stepEl.className = "step";

      if (step.travel_mode === "TRANSIT") {
        const line = step.transit_details.line;
        const routeLabel = line.short_name || line.name || "";
        const bg = line.color || lineColor(routeLabel);
        stepEl.innerHTML = `
          <span class="badge" style="background:${bg};color:${textColorFor(bg)}">${routeLabel}</span>
          <span class="step-detail">
            Board at <strong>${step.transit_details.departure_stop.name}</strong><br/>
            <span class="sub">${step.transit_details.num_stops} stops → ${step.transit_details.arrival_stop.name}</span><br/>
            <span class="sub">${step.duration.text}</span>
          </span>`;
      } else {
        stepEl.innerHTML = `
          <span class="badge walk">Walk</span>
          <span class="step-detail">
            ${step.html_instructions.replace(/<[^>]+>/g, " ")}<br/>
            <span class="sub">${step.duration.text} · ${step.distance.text}</span>
          </span>`;
      }
      resultsEl.appendChild(stepEl);
    });

    if (route.overview_polyline && route.overview_polyline.points) {
      const path = google.maps.geometry.encoding.decodePath(route.overview_polyline.points);
      state.routePolyline = new google.maps.Polyline({
        path,
        strokeColor: "#4285F4",
        strokeWeight: 5,
        strokeOpacity: 0.85,
        map: state.map,
      });
      const bounds = new google.maps.LatLngBounds();
      path.forEach((p) => bounds.extend(p));
      state.map.fitBounds(bounds);
    }
  }

  function clearRoutePolyline() {
    if (state.routePolyline) {
      state.routePolyline.setMap(null);
      state.routePolyline = null;
    }
  }

  // ---------- Isochrone walkshed ----------

  async function handleShowWalkshed() {
    if (!state.activeStationId) return;
    const station = state.allStations.find((s) => s.id === state.activeStationId);
    if (!station) return;
    const minutes = Number($("walk-minutes").value);

    clearWalkshed();
    try {
      const data = await fetchJson("/api/isochrone", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ lat: station.lat, lon: station.lon, travelMode: "WALK", minutes }),
      });
      const polygons = extractIsochronePolygons(data);
      if (polygons.length === 0) {
        alert("The Isochrones API didn't return a walkable area for this station/duration.");
        return;
      }
      polygons.forEach((paths) => {
        const poly = new google.maps.Polygon({
          paths,
          strokeColor: "#ff6319",
          strokeWeight: 2,
          fillColor: "#ff6319",
          fillOpacity: 0.18,
          map: state.map,
        });
        state.walkshedPolygons.push(poly);
      });
    } catch (err) {
      alert(`Couldn't load walking radius: ${err.message}`);
    }
  }

  // The Isochrones API response shape may evolve; handle a couple of
  // plausible layouts defensively rather than assuming one exact schema.
  function extractIsochronePolygons(data) {
    const isochrones = data.isochrones || data.result?.isochrones || [];
    const out = [];
    for (const iso of isochrones) {
      const polygons = iso.polygons || iso.shape?.polygons || (iso.shape ? [iso.shape] : []);
      for (const poly of polygons) {
        const rings = poly.rings || poly.loops || [poly.points || poly.path || []];
        for (const ring of rings) {
          const path = ring
            .map((pt) => ({
              lat: pt.latitude ?? pt.lat,
              lng: pt.longitude ?? pt.lng,
            }))
            .filter((p) => Number.isFinite(p.lat) && Number.isFinite(p.lng));
          if (path.length > 2) out.push(path);
        }
      }
    }
    return out;
  }

  function clearWalkshed() {
    state.walkshedPolygons.forEach((p) => p.setMap(null));
    state.walkshedPolygons = [];
  }

  boot();
})();
