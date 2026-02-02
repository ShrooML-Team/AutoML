import unittest
from colorama import Fore, Style, init
import time
from tqdm import tqdm

init(autoreset=True)

class ColoredTextTestResult(unittest.TextTestResult):
    def stopTestRun(self):
        elapsed = getattr(self, "_elapsed", 0.0)
        total = self.testsRun
        failed = len(self.failures) + len(self.errors)

        if failed == 0:
            color = Fore.GREEN
            status = "OK"
        elif failed == total:
            color = Fore.RED
            status = "FAILED"
        else:
            color = Fore.YELLOW
            status = "PARTIALLY OK"

        print(f"\n{color}{'-'*70}")
        print(f"{color}Ran {total} tests in {elapsed:.3f}s")
        print(f"{color}RESULT: {status}")
        print(f"{color}{'-'*70}{Style.RESET_ALL}")

class ColoredTextTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return ColoredTextTestResult(self.stream, self.descriptions, self.verbosity)

    def run(self, test):
        result = self._makeResult()
        start_time = time.time()

        tests = list(test)
        for t in tqdm(tests, desc="Running tests", unit="test"):
            t(result)

        result._elapsed = time.time() - start_time
        result.stopTestRun()
        return result

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="*.py")

    runner = ColoredTextTestRunner(verbosity=2)
    runner.run(suite)
