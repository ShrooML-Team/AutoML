import os
import shutil

class Utils:
    """Utility class for project-wide helper functions."""

    @classmethod
    def remove_all_pycache(cls, root_dir: str = None):
        """
        Recursively remove all __pycache__ directories.
        If root_dir is None, automatically detect the project root as the parent of the 'tests' folder.
        """
        if root_dir is None:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        for root, dirs, _ in os.walk(root_dir):
            for dir_name in dirs:
                if dir_name == "__pycache__":
                    cache_path = os.path.join(root, dir_name)
                    shutil.rmtree(cache_path)
                    print(f"[INFO] Removed {cache_path}")
