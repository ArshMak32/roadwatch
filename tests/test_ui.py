import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class UITests(unittest.TestCase):
    def test_startup(self):
        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "app.py")
        ).run(timeout=30)
        self.assertFalse(app.exception)

    def test_demo_photo(self):
        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "app.py")
        ).run(timeout=30)
        next(
            x
            for x in app.checkbox
            if x.label == "Use the licensed public road-photo demo"
        ).set_value(True).run()
        self.assertFalse(app.exception)
        self.assertTrue(
            any(x.label == "Detect road damage candidates" for x in app.button)
        )
