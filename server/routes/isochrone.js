const express = require("express");
const { generateIsochrone } = require("../lib/googleMaps");

const router = express.Router();

router.post("/", async (req, res) => {
  try {
    const { lat, lon, travelMode, minutes, travelDirection } = req.body || {};
    const latNum = parseFloat(lat);
    const lonNum = parseFloat(lon);
    if (!Number.isFinite(latNum) || !Number.isFinite(lonNum)) {
      return res.status(400).json({ error: "lat and lon are required in the request body" });
    }
    const data = await generateIsochrone({
      lat: latNum,
      lon: lonNum,
      travelMode: travelMode || "WALK",
      travelDurationSeconds: (Number(minutes) || 10) * 60,
      travelDirection: travelDirection || "FROM",
    });
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

module.exports = router;
