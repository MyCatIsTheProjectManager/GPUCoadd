"""The window on its own (start_gpucoadd.pyw): how saved images open without a host program, and
that each start is its own session."""

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def launcher():
    loader = SourceFileLoader("start_gpucoadd", str(ROOT / "start_gpucoadd.pyw"))
    module = module_from_spec(spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    return module


class StandaloneHostTests(unittest.TestCase):
    def test_opens_with_the_fits_program(self):
        host = launcher().StandaloneHost()
        with patch("os.startfile", create=True) as start, patch("subprocess.run") as run:
            host.open("E:/stacks/m42.fits")
        start.assert_called_once_with("E:/stacks/m42.fits")
        run.assert_not_called()

    def test_shows_the_file_when_no_program_opens_fits(self):
        host = launcher().StandaloneHost()
        with patch("os.startfile", create=True, side_effect=OSError("no association")), \
                patch("subprocess.run") as run:
            host.open("E:/stacks/m42.fits")
        args = run.call_args.args[0]
        self.assertEqual(args[:2], ["explorer.exe", "/select,"])
        self.assertEqual(Path(args[2]), Path("E:/stacks/m42.fits"))

    def test_each_start_is_its_own_session_and_keeps_the_environment(self):
        module = launcher()
        first, second = module.StandaloneHost(), module.StandaloneHost()
        self.assertTrue(first.session.startswith("standalone:"))
        self.assertNotEqual(first.session, second.session)
        self.assertEqual(first.variables, ())
        first.log("anything", error=True)  # the window's own log shows it; nothing else to do


class IconTests(unittest.TestCase):
    def test_icon_has_every_size_windows_asks_for(self):
        from PIL import Image

        with Image.open(ROOT / "GPUCoadd" / "gpucoadd.ico") as icon:
            self.assertTrue({(16, 16), (32, 32), (48, 48), (256, 256)} <= set(icon.info["sizes"]))


if __name__ == "__main__":
    unittest.main()
