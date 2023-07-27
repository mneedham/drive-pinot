import { 
    scroll, launchBrowseAndOpenPage, getInitialHeight,
    selectAllDelete, typeQuery, runQuery, getDistanceToBottom,
    moveToElement, changeCursor
} from '../drive_pinot.mjs'


async function run() {
    const zoomLevel = 250;
    const { browser, page } = await launchBrowseAndOpenPage({url: 'http://localhost:9000/#/query', zoomLevel: `${zoomLevel}%`});

    await new Promise(r => setTimeout(r, 3000))

    const codeMirrorTextArea = await page.waitForSelector('.CodeMirror');
    const initialHeight = await getInitialHeight(page);

    // Click on events
    const eventsTableTab = await page.waitForSelector('table tbody tr.MuiTableRow-root span'); 
    await moveToElement(page, eventsTableTab)
    changeCursor(page, {type: "hand"});
    await new Promise(r => setTimeout(r, 500))
    await page.evaluate(el => {
        el.click();
    }, eventsTableTab);
    await new Promise(r => setTimeout(r, 1000)); 
    await scroll(page, await getDistanceToBottom(page, zoomLevel*0.7), 3);
    await new Promise(r => setTimeout(r, 1000)); 

    // Find the uuid count
    let textToType = `
    select uuid, count(*)
    from events 
    GROUP BY uuid
    ORDER BY count(*) DESC
    limit 10`;    
    changeCursor(page, {type: "arrow"});
    await selectAllDelete(page, codeMirrorTextArea);
    await typeQuery(page, textToType, initialHeight, {charPause: 100});
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000));
    await scroll(page, await getDistanceToBottom(page, zoomLevel*0.7), 3);
    await new Promise(r => setTimeout(r, 1000)); 

    // Click on events dedup
    const eventsDeDupTab = await page.waitForSelector('table tbody tr:nth-child(2) span');
    await moveToElement(page, eventsDeDupTab)
    changeCursor(page, {type: "hand"});
    await new Promise(r => setTimeout(r, 500))
    await page.evaluate(el => {
        el.click();
    }, eventsDeDupTab);
    await new Promise(r => setTimeout(r, 1000)); 

    await scroll(page, await getDistanceToBottom(page, zoomLevel*0.7), 3);
    await new Promise(r => setTimeout(r, 1000)); 


    // Find the uuid count
    textToType = `
    select uuid, count(*)
    from events_dedup
    GROUP BY uuid
    ORDER BY count(*) DESC
    limit 10`;    
    changeCursor(page, {type: "arrow"});
    await selectAllDelete(page, codeMirrorTextArea);
    await typeQuery(page, textToType, initialHeight, {charPause: 100});
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000));
    await scroll(page, await getDistanceToBottom(page, zoomLevel*0.7), 3);
    await new Promise(r => setTimeout(r, 1000)); 

    // await browser.close();
}

run();
