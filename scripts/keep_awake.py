"""Visit the live website in a real browser so Streamlit Community Cloud keeps it awake.

Free Streamlit apps go to sleep after a period with no visitors. A scheduled
GitHub Action (.github/workflows/keep-awake.yml) runs this script every few
hours. If the app is already asleep, it clicks Streamlit's "wake up" button and
waits until the home page has really loaded. It exits with an error if the site
doesn't come up, so GitHub emails the repo owner.

Usage:  python scripts/keep_awake.py https://market-predictor1734.streamlit.app/
Needs:  pip install playwright && python -m playwright install chromium
"""

from __future__ import annotations

import re
import sys
import time

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

WAKE_BUTTON = re.compile(r"get this app back up", re.IGNORECASE)
READY_TEXT = "Understand any US stock"


def visit(url: str) -> bool:
    app_url = url.rstrip("/") + "/~/+/"  # the app itself, without Streamlit's outer frame
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        time.sleep(5)
        wake = page.get_by_role("button", name=WAKE_BUTTON)
        if wake.count() and wake.first.is_visible():
            print("App was asleep: clicking the wake-up button.")
            wake.first.click()
            time.sleep(60)  # the container needs time to boot
        else:
            print("App was awake.")

        for attempt in range(1, 6):
            try:
                page.goto(app_url, wait_until="domcontentloaded", timeout=120_000)
                page.get_by_text(READY_TEXT).first.wait_for(timeout=120_000)
                print(f"Home page loaded (attempt {attempt}).")
                time.sleep(10)  # stay a moment, like a real visitor
                browser.close()
                return True
            except PlaywrightTimeout:
                print(f"Not ready yet (attempt {attempt}); retrying…")
                time.sleep(30)
        browser.close()
        return False


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "https://market-predictor1734.streamlit.app/"
    sys.exit(0 if visit(target) else 1)
