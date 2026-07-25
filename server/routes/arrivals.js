const express = require("express");
const stationsStore = require("../lib/stationsStore");
const { getArrivalsForStation } = require("../lib/gtfsRealtime");

const router = express.Router();

router.get("/:stationId", async (req, res) => {
  try {
    const station = await stationsStore.getById(req.params.stationId);
    if (!station) return res.status(404).json({ error: "Station not found" });

    const { arrivals, errors, feedGroupsQueried } = await getArrivalsForStation(station);
    res.json({ station, arrivals, feedGroupsQueried, errors });
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

module.exports = router;
