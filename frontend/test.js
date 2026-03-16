const puppeteer = require('puppeteer');
(async () => {
    const browser = await puppeteer.launch({ headless: 'new' });
    const page = await browser.newPage();
    page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
    page.on('pageerror', err => console.error('BROWSER ERROR:', err.toString()));
    await page.goto('http://localhost:5173');
    await new Promise(r => setTimeout(r, 2000));

    // click station map tab
    const tabs = await page.$$('button');
    for (const btn of tabs) {
        const text = await page.evaluate(el => el.textContent, btn);
        if (text && text.includes('Station Map')) {
            await btn.click();
            console.log('Clicked Station Map');
        }
    }

    // click Allow on geolocation prompt?
    // We can't easily do that, but let's see if we get the crash before allow
    // wait and see if there are any immediate errors
    await new Promise(r => setTimeout(r, 3000));

    // let's explicitly trigger the location found logic if needed?
    // Since we replaced navigator.geolocation, maybe it crashes parsing it?

    await browser.close();
})();
