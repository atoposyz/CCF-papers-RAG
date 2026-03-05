import asyncio
import nodriver as nd

async def main():
    try:
        print("Starting browser...")
        browser = await nd.start(
            browser_executable_path=r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            browser_args=["--headless", "--no-sandbox", "--disable-dev-shm-usage"],
            sandbox=False
        )
        print("Connected!")
        page = await browser.get('https://example.com')
        print(await page.get_content())
        browser.stop()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    nd.loop().run_until_complete(main())
