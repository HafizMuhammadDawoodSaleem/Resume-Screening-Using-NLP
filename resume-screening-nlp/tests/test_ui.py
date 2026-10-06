"""Exercise the front end with the explicit offline baseline."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class FrontEndTests(unittest.TestCase):
    def test_sample_comparison(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                                default_timeout=30).run()
        self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value("TF-IDF · offline baseline")
        next(r for r in app.radio if r.label == "Resume source").set_value("Sample resumes")
        app.run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any(s.value == "Ranked matches" for s in app.subheader))
        self.assertEqual(len(app.metric), 6)


if __name__ == "__main__":
    unittest.main()
