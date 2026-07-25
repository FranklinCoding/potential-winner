require("dotenv").config();
const path = require("path");
const express = require("express");

const configRoute = require("./routes/config");
const stationsRoute = require("./routes/stations");
const arrivalsRoute = require("./routes/arrivals");
const directionsRoute = require("./routes/directions");
const isochroneRoute = require("./routes/isochrone");
const stationsStore = require("./lib/stationsStore");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(express.static(path.join(__dirname, "..", "public")));

app.use("/api/config", configRoute);
app.use("/api/stations", stationsRoute);
app.use("/api/arrivals", arrivalsRoute);
app.use("/api/directions", directionsRoute);
app.use("/api/isochrone", isochroneRoute);

app.get("/healthz", (req, res) => res.json({ ok: true }));

app.listen(PORT, () => {
  console.log(`NYC transit nav app listening on http://localhost:${PORT}`);
  if (!process.env.GOOGLE_MAPS_BROWSER_KEY && !process.env.GOOGLE_MAPS_API_KEY) {
    console.warn(
      "[warn] No GOOGLE_MAPS_BROWSER_KEY/GOOGLE_MAPS_API_KEY set — the map won't load. See .env.example."
    );
  }
  // Warm the station cache in the background so the first search isn't slow.
  stationsStore.load().catch((err) =>
    console.warn(`[warn] initial station load failed: ${err.message}`)
  );
});
