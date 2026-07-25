const express = require("express");
const router = express.Router();

// NYC-ish bounding box, used to bias/restrict Places Autocomplete on the client.
const NYC_BOUNDS = {
  north: 40.917577,
  south: 40.477399,
  east: -73.700272,
  west: -74.259865,
};
const NYC_CENTER = { lat: 40.7128, lng: -74.006 };

router.get("/", (req, res) => {
  const browserKey =
    process.env.GOOGLE_MAPS_BROWSER_KEY || process.env.GOOGLE_MAPS_API_KEY || "";

  res.json({
    googleMapsBrowserKey: browserKey,
    hasBrowserKey: Boolean(browserKey),
    hasServerKey: Boolean(
      process.env.GOOGLE_MAPS_SERVER_KEY || process.env.GOOGLE_MAPS_API_KEY
    ),
    nycBounds: NYC_BOUNDS,
    nycCenter: NYC_CENTER,
  });
});

module.exports = router;
