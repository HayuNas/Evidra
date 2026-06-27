import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.settings import Settings


class TestSettings(unittest.TestCase):
    def test_langfuse_base_url_env_alias_populates_host(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("LANGFUSE_")
            }
            env["LANGFUSE_BASE_URL"] = "https://cloud.langfuse.com"

            with patch.dict(os.environ, env, clear=True):
                settings = Settings.from_env(project_root=Path(directory))

        self.assertEqual(settings.langfuse_host, "https://cloud.langfuse.com")


if __name__ == "__main__":
    unittest.main()
