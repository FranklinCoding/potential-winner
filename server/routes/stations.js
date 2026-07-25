const express = require("express");
const stationsStore = require("../lib/stationsStore");

const router = express.Router();

router.get("/", async (req, res) => {
  try {
    const query = String(req.query.query || "");
    if (query) {
      const results = await stationsStore.search(query, Number(req.query.limit) || 10);
      return res.json({ stations: results });
    }
    const all = await stationsStore.getAll();
    return res.json({ stations: all });
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

router.get("/nearby", async (req, res) => {
  try {
    const lat = parseFloat(req.query.lat);
    const lon = parseFloat(req.query.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      return res.status(400).json({ error: "lat and lon query params are required" });
    }
    const limit = Number(req.query.limit) || 8;
    const results = await stationsStore.nearby(lat, lon, limit);
    res.json({ stations: results });
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

router.get("/:id", async (req, res) => {
  try {
    const station = await stationsStore.getById(req.params.id);
    if (!station) return res.status(404).json({ error: "Station not found" });
    res.json({ station });
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

module.exports = router;
