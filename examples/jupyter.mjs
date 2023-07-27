import { 
    scroll, scrollJupyter, launchBrowseAndOpenPage, getInitialHeight,
    selectAllDelete, typeQuery, runQuery, getDistanceToBottom,
    moveToElement, changeCursor
} from '../drive_pinot.mjs'


async function run() {
    const zoomLevel = 250;
    const token = "a17191c48278aaf729b8f811ea30094a43685a730991bac4";
    const { browser, page } = await launchBrowseAndOpenPage({
        url: `http://localhost:8888/doc/workspaces/auto-B/tree/SentimentAnalysis.ipynb?token=${token}&simpleInterface=true`, 
        zoomLevel: `${zoomLevel}%`
    });

    await new Promise(r => setTimeout(r, 1000))

    await page.keyboard.down('Meta');
    await page.keyboard.down('Shift'); 
    await page.keyboard.press('f');
    await page.keyboard.up('Shift'); 
    await page.keyboard.up('Meta');

    await new Promise(r => setTimeout(r, 3000))

    const delay = ms => new Promise(res => setTimeout(res, ms));

    for (let i = 0; i < 5; i++) {
        await page.keyboard.press('ArrowDown');
        await delay(200); // Sleep for 100 milliseconds
    }
    
    await scrollJupyter(page, 1000)

    // await browser.close();
}

run();
