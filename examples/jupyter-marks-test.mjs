import { 
    scroll, scrollJupyter, launchBrowseAndOpenPage, getInitialHeight,
    selectAllDelete, typeQuery, runQuery, getDistanceToBottom,
    moveToElement, changeCursor
} from '../drive_pinot.mjs'

import {
    executeCell, waitForQueryToFinish,
    newJupyterCell, scrollJupyterDown,
    stylePage, autoScroll, scrollOutput
} from '../drive_jupyter.mjs'

async function run() {
    const zoomLevel = 350;
    const token = "d0cb0f0a9ee2e953b05cc4a0934b1f6b188acdbf4ff399ce";
    const { browser, page } = await launchBrowseAndOpenPage({
        url: `http://localhost:8888/doc/tree/MyTest.ipynb?token=${token}`, 
        zoomLevel: `${zoomLevel}%`
    });
        
    await page.waitForSelector('.jp-Notebook');
    await stylePage(page);
    await autoScroll(page);
    await new Promise(r => setTimeout(r, 1000))

    await newJupyterCell(page);
    let code = `for index in range(0, 50):
print(index)`;
    await page.keyboard.type(code, {delay: 20});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await new Promise(r => setTimeout(r, 500));

    await scrollOutput(page, {longPauseIntervals:0, longPause:500, pause:50});
    

    // await page.keyboard.down('Meta');
    // await page.keyboard.down('Shift'); 
    // await page.keyboard.press('F');
    // await page.keyboard.up('Shift'); 
    // await page.keyboard.up('Meta');

   

    // await browser.close();
}

run();
