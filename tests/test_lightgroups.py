"""Light groups: grouping keys, main group, filter-mismatch exclusion, pedestal fallback."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lightgroups as lg  # noqa: E402

PRO = {
    "TELESCOP": "S50 Pro_00000000",
    "EXPTIME": 30.0,
    "GAIN": 200,
    "BITPIX": 16,
    "NAXIS": 2,
    "NAXIS1": 2160,
    "NAXIS2": 3840,
    "BAYERPAT": "GRBG",
    "BIAS": 2063,
    "FILTER": "LP",
    "XPIXSZ": 2.9,
    "FOCALLEN": 260.0,
    "RA": 341.0,
    "DEC": 58.1,
    "DATE-OBS": "2026-10-03T20:00:00",
}
S50 = {
    "TELESCOP": "S50_00000000",
    "INSTRUME": "Seestar S50",
    "EXPTIME": 10.0,
    "GAIN": 80,
    "BITPIX": 16,
    "NAXIS": 2,
    "NAXIS1": 1080,
    "NAXIS2": 1920,
    "BAYERPAT": "GRBG",
    "BIAS": 468,
    "FILTER": "LP",
    "XPIXSZ": 2.9,
    "FOCALLEN": 250.0,
    "RA": 341.1,
    "DEC": 58.0,
    "DATE-OBS": "2026-10-20T21:00:00",
}


def lights(*specs):
    """specs: (folder, header, count, overrides) -> describe() records in order."""
    out = []
    for folder, header, count, extra in specs:
        for k in range(count):
            h = {**header, **extra}
            h = {key: v for key, v in h.items() if v is not None}
            out.append(lg.describe(Path(folder) / f"light_{k:03d}.fit", h))
    return out


class GroupingTests(unittest.TestCase):
    def test_single_folder_is_one_used_group(self):
        ls = lights(("/a/lights", PRO, 5, {}))
        p = lg.plan(ls)
        self.assertEqual(len(p["groups"]), 1)
        self.assertEqual(p["used"], list(range(5)))
        g = p["groups"][0]
        self.assertTrue(g["main"] and g["used"])
        self.assertEqual((g["telescope"], g["pedestal"], g["pedestal_source"]), ("S50 Pro", [2063.0], "BIAS header"))
        self.assertEqual(g["total_exposure_s"], 150)

    def test_main_group_and_filter_mismatch(self):
        # NGC 2175-like: Pro IRCUT; the S50 folder mixes IRCUT and LP lights
        ls = lights(
            ("/n/s50pro/lights", PRO, 6, {"FILTER": "IRCUT"}),
            ("/n/lights", S50, 3, {"FILTER": "IRCUT", "BIAS": None}),
            ("/n/lights", S50, 9, {"BIAS": None}),
        )
        p = lg.plan(ls)
        main = [g for g in p["groups"] if g["main"]][0]
        self.assertEqual(main["telescope"], "S50 Pro")  # 6 x 30 s x 8.3 Mpx beats 9 x 10 s x 2.1 Mpx
        lp = [g for g in p["groups"] if g["filter"] == "LP"][0]
        self.assertFalse(lp["used"])
        self.assertIn("filter LP differs from the main group's IRCUT", lp["reason"])
        self.assertEqual(p["used"], list(range(9)))
        # the LP group can never join; leaving out the IRCUT groups stacks the LP lights alone
        ircut = [g["key"] for g in p["groups"] if g["filter"] == "IRCUT"]
        p2 = lg.plan(ls, exclude=ircut)
        self.assertEqual(p2["used"], list(range(9, 18)))
        self.assertEqual(p2["main"], lp["key"])
        p3 = lg.plan(ls, exclude=[p["groups"][1]["key"]])
        self.assertEqual(p3["used"], list(range(6)))
        self.assertEqual(p3["groups"][1]["reason"], "left out by the user")
        with self.assertRaises(ValueError):
            lg.plan(ls, exclude=["no such group"])

    def test_main_group_follows_area_times_exposure(self):
        # many S50 lights can outweigh few Pro lights
        ls = lights(("/x/pro", PRO, 2, {}), ("/x/s50", S50, 40, {}))
        self.assertEqual(lg.plan(ls)["main"], lg.group_key(ls[2]))
        # excluding the main group makes the next one main (and sets the filter)
        ls = lights(("/x/pro", PRO, 6, {"FILTER": "IRCUT"}), ("/x/s50", S50, 9, {}))
        p = lg.plan(ls, exclude=[lg.group_key(ls[0])])
        self.assertEqual(p["main"], lg.group_key(ls[6]))
        self.assertEqual(p["used"], list(range(6, 15)))

    def test_pixel_scale_from_optics(self):
        ls = lights(
            ("/p/pro", PRO, 2, {}),
            ("/p/s50", S50, 2, {}),
            ("/p/s50x", S50, 2, {"FOCALLEN": None}),
            ("/p/s50y", S50, 1, {"FOCALLEN": None}),
            ("/p/s50y", S50, 1, {}),
        )
        p = lg.plan(ls)
        self.assertAlmostEqual(p["pixel_scale"][0], 206.265 * 2.9 / 260, places=6)
        self.assertAlmostEqual(p["pixel_scale"][2] / p["pixel_scale"][0], 260 / 250)
        self.assertEqual(p["pixel_scale"][4], p["pixel_scale"][0])  # no optics data: main instrument
        self.assertEqual(p["pixel_scale"][6], p["pixel_scale"][7])  # group mate's value
        self.assertEqual(p["main_lights"], [0, 1])

    def test_pedestal_fallback(self):
        # S50 lights without BIAS borrow the same telescope's BIAS at the same gain
        ls = lights(
            ("/s/pro", PRO, 2, {}),
            ("/s/s50new", S50, 2, {}),
            ("/s/s50old", S50, 2, {"BIAS": None}),
            ("/s/s50g", S50, 2, {"BIAS": None, "GAIN": 120}),
        )
        p = lg.plan(ls)
        src = {Path(g["folder"]).name: (g["pedestal"], g["pedestal_source"]) for g in p["groups"]}
        self.assertEqual(p["pedestal"][4], 468.0)
        self.assertEqual(src["s50old"], ([468.0], "BIAS of the same telescope and gain in other lights"))
        self.assertEqual(p["pedestal"][6], 0.0)
        self.assertEqual(src["s50g"], ([0.0], "none found (0)"))
        self.assertEqual(p["pedestal"][0], 2063.0)

    def test_groups_split_by_exposure_gain_size_and_bayer(self):
        ls = lights(
            ("/g", PRO, 2, {}),
            ("/g", PRO, 2, {"EXPTIME": 20.0}),
            ("/g", PRO, 2, {"GAIN": 100}),
            ("/g", PRO, 2, {"NAXIS1": 1080, "NAXIS2": 1920}),
            ("/g", PRO, 2, {"BAYERPAT": "RGGB"}),
            ("/g", PRO, 2, {"BAYERPAT": None}),
            ("/g", PRO, 2, {"BAYERPAT": "RGBG"}),
            ("/g", PRO, 2, {"NAXIS": 3}),
            ("/g", PRO, 2, {"BITPIX": 64, "BAYERPAT": "CYGM"}),
        )
        p = lg.plan(ls)
        self.assertEqual(len(p["groups"]), 9)
        rggb = p["groups"][4]
        self.assertTrue(rggb["used"])
        self.assertEqual(rggb["sensor"], [2160, 3840])
        self.assertEqual(
            rggb["conversion"], ["first and last column left out to start on the GRBG layout with even dimensions"]
        )
        self.assertEqual((ls[8]["width"], ls[8]["height"]), (2158, 3840))  # what the engine reads
        self.assertEqual(p["groups"][0]["conversion"], [])
        reasons = [g["reason"] for g in p["groups"][5:]]
        self.assertEqual(
            reasons,
            [
                "no Bayer pattern in the header (BAYERPAT)",
                "Bayer pattern RGBG is not supported",
                "not a single raw image (colour or multi-plane data)",
                "Bayer pattern CYGM is not supported",
            ],
        )
        self.assertEqual(p["used"], list(range(10)))
        with self.assertRaisesRegex(ValueError, "no Bayer pattern in the header"):
            lg.plan(lights(("/g", PRO, 2, {"BAYERPAT": None})))

    def test_paths_and_labels(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "s50pro" / "lights", Path(d) / "s50 lp" / "lights"
            a.mkdir(parents=True), b.mkdir(parents=True)
            for f in (a / "2.fit", a / "1.fit", b / "0.fit", a / "x.fits", a / "y.FTS", a / "z.png"):
                f.write_bytes(b"")
            paths = lg.light_paths([str(a), str(b), str(a)])
            self.assertEqual([p.name for p in paths], ["1.fit", "2.fit", "x.fits", "y.FTS", "0.fit"])
            with self.assertRaises(ValueError):
                lg.light_paths([str(Path(d) / "missing")])
        rec = lg.describe(b / "0.fit", S50)
        self.assertEqual(lg.label(rec, 891), "S50, LP, 891 x 10 s, gain 80, 1080 x 1920 (s50 lp/lights)")
        self.assertEqual(lg.telescope_name("S50 Pro_00000000"), "S50 Pro")
        self.assertTrue(np.isnan(lg.describe(b / "0.fit", {})["pointing"][0]))

    def test_bridge_scan_reports_groups_without_pixels(self):
        from astropy.io import fits

        from worker import bridge

        with tempfile.TemporaryDirectory() as d:
            pro, s50 = Path(d) / "pro", Path(d) / "s50"
            pro.mkdir(), s50.mkdir()
            for folder, header, filt, n in ((pro, PRO, "IRCUT", 3), (s50, S50, "IRCUT", 2), (s50, S50, "LP", 2)):
                for k in range(n):
                    h = fits.Header({key: v for key, v in header.items() if key not in ("NAXIS1", "NAXIS2")})
                    h["FILTER"] = filt
                    fits.writeto(folder / f"{filt}_{k}.fit", np.zeros((8, 4), np.uint16), h)
            out = Path(d) / "out"
            out.mkdir()
            reporter = bridge.Reporter(out, out / "cancel", save_log=False)
            import contextlib, io

            with contextlib.redirect_stdout(io.StringIO()):
                res = bridge.execute({"operation": "scan", "lights_dirs": [str(pro), str(s50)]}, reporter, out)
            self.assertEqual(res["input_frames"], 7)
            self.assertEqual(res["used_frames"], 5)
            self.assertEqual(
                [(g["filter"], g["used"], g["sensor"]) for g in res["groups"]],
                [("IRCUT", True, [4, 8]), ("IRCUT", True, [4, 8]), ("LP", False, [4, 8])],
            )
            json.dumps(bridge.safe_json(res), allow_nan=False)  # report is strict JSON
            with self.assertRaises(ValueError):
                bridge.light_folders({"operation": "scan"})


if __name__ == "__main__":
    unittest.main()
