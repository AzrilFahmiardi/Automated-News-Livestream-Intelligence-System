"""
Browser Automation Module - Livestream Capturer
Uses patchright-python for YouTube livestream capture
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from patchright.async_api import async_playwright, Page, Browser

logger = logging.getLogger(__name__)


class StreamCapturer:
    """Captures livestream video frames using patchright browser automation"""

    def __init__(self, config: dict):
        self.config = config
        self.browser_instance: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright = None
        self.is_running = False

    async def initialize(self):
        """Initialize browser and prepare for stream capture"""
        logger.info("Initializing patchright browser...")

        # Start patchright
        self.playwright = await async_playwright().start()

        # Browser launch options with additional stealth
        browser_config = self.config.get("browser", {})
        
        self.browser_instance = await self.playwright.chromium.launch(
            headless=browser_config.get("headless", False),
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-blink-features=AutomationControlled",  # Hide automation
                "--disable-dev-shm-usage",
                "--disable-web-security",
                "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            ],
        )
        
        # Create page directly
        self.page = await self.browser_instance.new_page()

        # Set realistic viewport
        await self.page.set_viewport_size({"width": 1920, "height": 1080})

        # Allow browser network stack to initialize
        await asyncio.sleep(1)

        logger.info("Browser initialized successfully")

    async def open_livestream(self, url: str) -> bool:
        """
        Open YouTube livestream and prepare for capture

        Args:
            url: YouTube livestream URL

        Returns:
            bool: True if successfully opened
        """
        try:
            logger.info(f"Opening livestream: {url}")

            # Navigate to URL
            logger.info("Navigating to URL...")
            await self.page.goto(url, wait_until="domcontentloaded", timeout=60000)
            
            # Inject stealth scripts AFTER page load (avoids DNS issues)
            logger.info("Injecting stealth scripts...")
            await self.page.evaluate("""
                () => {
                    // Hide webdriver property
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    });
                    
                    // Fake plugins
                    Object.defineProperty(navigator, 'plugins', {
                        get: () => [1, 2, 3, 4, 5]
                    });
                    
                    // Fake languages
                    Object.defineProperty(navigator, 'languages', {
                        get: () => ['en-US', 'en', 'id-ID', 'id']
                    });
                    
                    // Remove automation traces
                    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Array;
                    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Promise;
                    delete window.cdc_adoQpoasnfa76pfcZLmcfl_Symbol;
                }
            """)
            
            logger.info("Page loaded, dismissing popups...")
            await self._dismiss_popups()
            
            # Allow page to settle
            await asyncio.sleep(2)
            
            # Click play button if present
            logger.info("Looking for play button...")
            try:
                play_selectors = [
                    "button.ytp-large-play-button",
                    "button.ytp-play-button",
                    ".ytp-large-play-button-red-bg"
                ]
                for selector in play_selectors:
                    play_button = await self.page.query_selector(selector)
                    if play_button and await play_button.is_visible():
                        await play_button.click()
                        logger.info(f"Clicked play button: {selector}")
                        await asyncio.sleep(1)
                        break
            except Exception as e:
                logger.warning(f"Could not click play button: {e}")
            
            logger.info("Waiting for video element...")
            # Force show video element with JavaScript
            try:
                await self.page.evaluate("""
                    () => {
                        const video = document.querySelector('video');
                        if (video) {
                            video.style.visibility = 'visible';
                            video.style.display = 'block';
                            video.style.opacity = '1';
                            video.play().catch(() => {});
                        }
                    }
                """)
                await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"Could not execute video show script: {e}")
            
            # Wait for video element to be attached to DOM
            await self.page.wait_for_selector("video", state="attached", timeout=30000)
            
            logger.info("Video element ready")

            # Allow video to start loading
            await asyncio.sleep(2)

            # Set video quality
            await self._set_video_quality("1080p")

            logger.info("Livestream opened successfully")
            self.is_running = True
            return True

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to open livestream: {error_msg}")
            
            if "ERR_NAME_NOT_RESOLVED" in error_msg:
                logger.error("DNS resolution failed - Check internet connection")
            elif "ERR_CONNECTION_REFUSED" in error_msg:
                logger.error("Connection refused - YouTube might be blocking access")
            elif "Timeout" in error_msg:
                logger.error("Timeout - Page took too long to load")
            
            return False

    async def _set_video_quality(self, quality: str = "1080p"):
        """Set video quality if available"""
        try:
            await self.page.click("button.ytp-settings-button", timeout=5000)
            await asyncio.sleep(0.5)

            quality_menu = await self.page.query_selector('div.ytp-menuitem:has-text("Quality")')
            if quality_menu:
                await quality_menu.click()
                await asyncio.sleep(0.5)

                quality_option = await self.page.query_selector(f'span.ytp-menuitem-label:has-text("{quality}")')
                if quality_option:
                    await quality_option.click()
                    logger.info(f"Video quality set to {quality}")

        except Exception as e:
            logger.warning(f"Could not set video quality: {e}")

    async def _dismiss_popups(self):
        """Dismiss YouTube popups and cookie banners"""
        try:
            # Dismiss cookie consent
            cookie_button = await self.page.query_selector('button[aria-label*="Accept"]')
            if cookie_button:
                await cookie_button.click()
                await asyncio.sleep(0.5)

            # Dismiss any other dialogs
            dismiss_button = await self.page.query_selector('button[aria-label*="Dismiss"]')
            if dismiss_button:
                await dismiss_button.click()

        except:
            pass

    async def capture_frame(self) -> Optional[bytes]:
        """
        Capture current video frame

        Returns:
            bytes: PNG image data or None if failed
        """
        try:
            # Get video element
            video = await self.page.query_selector("video")
            if not video:
                logger.warning("Video element not found")
                return None

            # Take screenshot of video element
            screenshot = await video.screenshot(type="png")
            return screenshot

        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None

    async def keep_alive(self):
        """
        Keep browser session alive with human-like activity
        Prevents bot detection and session timeout
        """
        import random
        
        browser_config = self.config.get("browser", {})
        mouse_interval = browser_config.get("random_mouse_interval", 30)

        logger.info("Starting keep-alive routine...")

        while self.is_running:
            try:
                # More natural random mouse movement
                x = random.randint(200, 1700)
                y = random.randint(200, 900)
                await self.page.mouse.move(x, y)

                # Random small scroll (simulates user watching)
                scroll_amount = random.randint(-20, 20)
                await self.page.evaluate(f"window.scrollBy(0, {scroll_amount})")
                
                # Occasionally move mouse over video (looks more human)
                if random.random() < 0.3:  # 30% chance
                    video = await self.page.query_selector("video")
                    if video:
                        box = await video.bounding_box()
                        if box:
                            vx = box['x'] + random.randint(50, int(box['width']) - 50)
                            vy = box['y'] + random.randint(50, int(box['height']) - 50)
                            await self.page.mouse.move(vx, vy)

                # Vary interval slightly
                await asyncio.sleep(mouse_interval + random.uniform(-5, 5))

            except Exception as e:
                logger.error(f"Keep-alive error: {e}")
                await asyncio.sleep(5)

    async def close(self):
        """Close browser and cleanup"""
        logger.info("Closing browser...")
        self.is_running = False
            
        if self.browser_instance:
            await self.browser_instance.close()

        if self.playwright:
            await self.playwright.stop()

        logger.info("Browser closed")

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
