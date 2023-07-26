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

    await scroll(page, await getDistanceToBottom(page, zoomLevel), 3);
    await new Promise(r => setTimeout(r, 1000)); 
    changeCursor(page, {type: "arrow"});
    await runQuery(page);
    await new Promise(r => setTimeout(r, 500)); 
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000)); 

    // Query one junction
    const textToType = `
    select junctionId, timeBucket, minSpeed, maxSpeed, numberOfVehicles
    from traffic
    where junctionId = 1000
    order by timeBucket DESC
    limit 10`;
    
    await selectAllDelete(page, codeMirrorTextArea);
    await typeQuery(page, textToType, initialHeight, {charPause: 100});
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000));
    await scroll(page, await getDistanceToBottom(page, zoomLevel), 3);
    await new Promise(r => setTimeout(r, 1000)); 
    
    await moveToElement(page, codeMirrorTextArea);
    await new Promise(r => setTimeout(r, 700)); 

    await page.evaluate(() => {
        const codeMirror = document.querySelector('.CodeMirror').CodeMirror;
        const lastLine = codeMirror.lineCount() - 1; // Subtract 1 because lines are 0-indexed
        const lastLineText = codeMirror.getLineHandle(lastLine).text;
        codeMirror.setCursor({line: lastLine, ch: lastLineText.length});
        codeMirror.execCommand('newlineAndIndent');
    });
    await typeQuery(page, "option(skipUpsert=true)", initialHeight);

    await page.evaluate(() => {
        const codeMirror = document.querySelector('.CodeMirror').CodeMirror;
        const lastLineText = codeMirror.getLineHandle(2).text;
        codeMirror.setCursor({line: 2, ch: lastLineText.length});
    });
    await typeQuery(page, " AND timeBucket = '2023-07-24 11:29:40.0'", initialHeight, {shouldCleanQuery: false});
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000));
    await scroll(page, await getDistanceToBottom(page, zoomLevel), 3);
    await new Promise(r => setTimeout(r, 1000)); 

    // await browser.close();
}

run();
