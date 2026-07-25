const express = require("express");
const { getTransitDirections } = require("../lib/googleMaps");

const router = express.Router();

router.get("/", async (req, res) => {
  try {
    const { origin, destination, transitMode, departureTime } = req.query;
    if (!origin || !destination) {
      return res.status(400).json({ error: "origin and destination query params are required" });
    }
    const data = await getTransitDirections({
      origin: String(origin),
      destination: String(destination),
      transitMode: transitMode ? String(transitMode) : undefined,
      departureTime: departureTime ? String(departureTime) : undefined,
    });
    res.json(data);
  } catch (err) {
    res.status(502).json({ error: err.message });
  }
});

module.exports = router;
