import { 
    scroll, launchBrowseAndOpenPage, getInitialHeight,
    selectAllDelete, typeQuery, runQuery, getDistanceToBottom,
    moveToElement, changeCursor
} from './drive_pinot.mjs'


async function run() {
    const zoomLevel = 250;
    const { browser, page } = await launchBrowseAndOpenPage({url: 'http://localhost:9000/#/query', zoomLevel: `${zoomLevel}%`});

    await new Promise(r => setTimeout(r, 3000))

    const codeMirrorTextArea = await page.waitForSelector('.CodeMirror');
    const initialHeight = await getInitialHeight(page);

    // Query traffic table
    const queryTab = await page.waitForSelector('table tbody tr.MuiTableRow-root span');
    
    await moveToElement(page, queryTab)
    changeCursor(page, {type: "hand"});
    await new Promise(r => setTimeout(r, 500))
    await page.evaluate(el => {
        el.click();
    }, queryTab);
    await new Promise(r => setTimeout(r, 1000)); 

    await scroll(page, await getDistanceToBottom(page, zoomLevel*0.8), 3);
    await new Promise(r => setTimeout(r, 1000)); 
    changeCursor(page, {type: "arrow"});
    await runQuery(page);
    await new Promise(r => setTimeout(r, 500)); 
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000)); 

    // // Query one junction
    const textToType = `
    select resolution, count(*)
    from telemetry 
    WHERE country = 'GB'
    GROUP BY resolution
    limit 10`;
    
    changeCursor(page, {type: "arrow"});
    await selectAllDelete(page, codeMirrorTextArea);
    await typeQuery(page, textToType, initialHeight, {charPause: 100});
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000));
    await scroll(page, 500, 3);
    await new Promise(r => setTimeout(r, 1000)); 

    // await browser.close();
}

run();
