"""Tile budgeting includes the persistent stack buffer and phase-specific scratch."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tileplan import GIB, memory_estimate


class MemoryPlanTests(unittest.TestCase):
    def test_buffer_reserved_for_busiest_tile(self):
        estimate = memory_estimate(512, 2000, 2000, [10, 344, 0, 21])
        self.assertEqual(estimate["buffer_bytes"], 344 * 3 * 512 * 512 * 4)
        self.assertEqual(estimate["required_bytes"], estimate["buffer_bytes"] + estimate["scratch_bytes"] + GIB)

    def test_larger_tiles_need_more_memory(self):
        estimates = [memory_estimate(t, 4000, 4000, [344])["required_bytes"] for t in (256, 512, 768, 1024)]
        self.assertEqual(estimates, sorted(estimates))
        self.assertEqual(len(set(estimates)), 4)

    def test_numpy_counts_produce_serializable_estimates(self):
        import json
        import numpy as np

        estimate = memory_estimate(512, 4000, 4000, [np.int64(344)])
        self.assertEqual(json.loads(json.dumps(estimate)), estimate)

    def test_crop_padding_and_rejection_are_budgeted(self):
        small = memory_estimate(512, 2000, 2000, [344], crop_pixels=1)
        large = memory_estimate(512, 2000, 2000, [344], crop_pixels=8_000_000)
        self.assertGreater(large["required_bytes"], small["required_bytes"])
        sigma = memory_estimate(256, 2000, 2000, [587], crop_pixels=1)
        noise = memory_estimate(256, 2000, 2000, [587], crop_pixels=1, rejection="noise")
        self.assertGreater(noise["scratch_bytes"], sigma["scratch_bytes"])

    def test_empty_tiles_do_not_need_a_stack(self):
        plan = memory_estimate(1024, 4000, 4000, [0, 0])
        self.assertEqual(plan["buffer_bytes"], 0)
        self.assertEqual(plan["scratch_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
