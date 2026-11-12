"""The cubin cache: a damaged, empty, old-format or driver-refused file is compiled again and loaded
from memory, and replaced when it can be; a cache that can't be written still works."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE = r"""
extern "C" __global__ void add_one(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}
"""


class KernelCache(unittest.TestCase):
    def setUp(self):
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        import cuda_kernels

        self.ck, self.torch = cuda_kernels, torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        p = patch.object(cuda_kernels, "CACHE", Path(self.temp.name) / "cubin_cache")
        p.start()
        self.addCleanup(p.stop)
        self.source = SOURCE + f"// {self.id()}\n"  # a cache entry of its own per test

    def module(self):
        return self.ck.Module(self.source)

    def launches(self, module):
        x = self.torch.arange(8, dtype=self.torch.float32, device="cuda")
        module.launch("add_one", 1, 32, [x, 8])
        self.torch.cuda.synchronize()
        self.assertEqual(x.cpu().tolist(), [float(v + 1) for v in range(8)])

    def cache_file(self):
        files = list(self.ck.CACHE.glob("*.cubin"))
        self.assertEqual(len(files), 1, files)
        return files[0]

    def test_first_build_is_cached_with_its_checksum_and_reused(self):
        self.launches(self.module())
        path = self.cache_file()
        cubin = self.ck._read_cache(path)
        self.assertIsNotNone(cubin)
        self.assertTrue(path.read_bytes().startswith(self.ck.MAGIC))
        with patch.object(self.ck, "_compile", side_effect=AssertionError("compiled again")):
            self.launches(self.module())

    def test_damaged_empty_and_old_format_files_are_rebuilt_and_replaced(self):
        self.module()
        path = self.cache_file()
        good = path.read_bytes()
        cubin = self.ck._read_cache(path)
        for name, content in (
            ("random bytes", os.urandom(len(good))),
            ("empty", b""),
            ("truncated", good[: len(good) // 2]),
            ("one byte changed", good[:-1] + bytes([good[-1] ^ 1])),
            ("old format (headerless cubin)", cubin),
        ):
            with self.subTest(name):
                path.write_bytes(content)
                self.launches(self.module())
                self.assertEqual(path.read_bytes(), good, "the file is replaced by the compiled cubin")

    def test_an_intact_file_the_driver_refuses_is_compiled_again(self):
        self.module()
        path = self.cache_file()
        good = path.read_bytes()
        refused = b"\0" * 256  # not a cubin, but stored with a matching checksum
        path.write_bytes(self.ck.MAGIC + __import__("hashlib").sha256(refused).digest() + refused)
        self.launches(self.module())
        self.assertEqual(path.read_bytes(), good)

    def test_a_file_that_cannot_be_replaced_still_loads_now_and_next_time(self):
        self.module()
        path = self.cache_file()
        path.write_bytes(b"damaged")
        with patch.object(self.ck.os, "replace", side_effect=PermissionError("locked")):
            self.launches(self.module())
            self.launches(self.module())  # the next process finds the same damaged file
        self.assertEqual(path.read_bytes(), b"damaged")
        self.assertEqual(list(self.ck.CACHE.glob("*.tmp")), [], "no temporary file is left")

    def test_a_cache_folder_that_cannot_be_created_compiles_each_time(self):
        blocker = Path(self.temp.name) / "not a folder"
        blocker.write_text("a file where the cache folder would go")
        with patch.object(self.ck, "CACHE", blocker / "cubin_cache"):
            self.launches(self.module())
            self.launches(self.module())


if __name__ == "__main__":
    unittest.main()
