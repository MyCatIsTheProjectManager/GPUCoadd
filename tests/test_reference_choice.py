"""The measured reference choice (GPUCOADD_REFERENCE=measured) on lights with known qualities."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gpucoadd import choose_reference, star_transparency  # noqa: E402


def stars_with(flux, count=200):
    """A star list (x, y, R, G, B, peak) whose green flux falls off with rank like a real field."""
    s = np.zeros((count, 6))
    s[:, 3] = flux * np.linspace(2.0, 0.2, count)
    return s


class ReferenceChoice(unittest.TestCase):
    def test_single_field_skips_hazy_bright_and_soft_lights(self):
        n = 9
        fwhm = np.full(n, 4.0)
        stars = [stars_with(1.0) for _ in range(n)]
        background = {i: 100.0 for i in range(n)}
        fwhm[2] = 3.0
        stars[2] = stars_with(0.4)  # sharpest, but through haze
        fwhm[3] = 3.2
        background[3] = 400.0  # sharp, but moonlit
        fwhm[6] = 3.5  # sharp, clear, dark: the one to pick
        pointings = [(300.0 + 0.01 * (i % 3), 60.0) for i in range(n)]
        i, choice = choose_reference(range(n), fwhm, stars, background, pointings, 0.7)
        self.assertEqual(i, 6)
        self.assertEqual(choice["candidates"], n, "a single field keeps every light as a candidate")
        self.assertFalse(choice["widened_to_all"])

    def mosaic(self):
        panels = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
        pointings, fwhm = [], []
        for dx, dy in panels:
            for _ in range(5):
                pointings.append((150.0 + dx * 1.0 / np.cos(np.deg2rad(40.0)), 40.0 + dy * 0.6))
                fwhm.append(3.0 if (dx, dy) == (1, 1) else 4.0)  # the sharpest lights sit in a corner panel
        n = len(pointings)
        return panels, pointings, np.array(fwhm), [stars_with(1.0) for _ in range(n)], {i: 100.0 for i in range(n)}

    def test_mosaic_prefers_the_central_panel(self):
        panels, pointings, fwhm, stars, background = self.mosaic()
        i, choice = choose_reference(range(len(fwhm)), fwhm, stars, background, pointings, 0.7)
        self.assertEqual(panels[i // 5], (0, 0), "the reference comes from the central panel")
        self.assertEqual(choice["candidates"], 10, "widened to ten lights around the centre")
        self.assertLess(choice["centre_offset_deg"], 0.05)

    def test_mosaic_skips_a_sharp_twilight_light(self):
        panels, pointings, fwhm, stars, background = self.mosaic()
        centre = [k for k in range(len(fwhm)) if panels[k // 5] == (0, 0)]
        fwhm[centre[0]] = 3.2
        background[centre[0]] = 190.0  # the sharpest central light was taken in twilight
        fwhm[centre[1]] = 3.6
        i, choice = choose_reference(range(len(fwhm)), fwhm, stars, background, pointings, 0.7)
        self.assertEqual(i, centre[1])

    def test_when_nothing_near_the_centre_qualifies_all_lights_are_weighed(self):
        panels, pointings, fwhm, stars, background = self.mosaic()
        for k in range(len(fwhm)):
            if panels[k // 5] in ((0, 0), (0, -1), (0, 1)):
                background[k] = 200.0  # every light near the centre under a bright sky
        i, choice = choose_reference(range(len(fwhm)), fwhm, stars, background, pointings, 0.7)
        self.assertTrue(choice["widened_to_all"])
        self.assertEqual(background[i], 100.0)

    def test_without_pointings_every_light_is_a_candidate(self):
        fwhm = np.array([4.0, np.nan, 3.6, 4.2])
        stars = [stars_with(1.0) for _ in range(4)]
        i, choice = choose_reference(range(4), fwhm, stars, {}, None, None)
        self.assertEqual(i, 2, "a light without a measured FWHM is never preferred")
        self.assertEqual(choice["candidates"], 4)

    def test_transparency_counts_missing_stars(self):
        self.assertLess(star_transparency(stars_with(1.0, 30)), star_transparency(stars_with(1.0, 200)))
        self.assertEqual(star_transparency(np.zeros((0, 6))), 0.0)


if __name__ == "__main__":
    unittest.main()
