import { 
    scroll, launchBrowseAndOpenPage, getInitialHeight,
    selectAllDelete, typeQuery, runQuery, getDistanceToBottom
} from './drive_pinot.mjs'


async function run() {
    const zoomLevel = 250;
    const { browser, page } = await launchBrowseAndOpenPage({url: 'http://localhost:9000/#/query', zoomLevel: `${zoomLevel}%`});

    await new Promise(r => setTimeout(r, 3000))

    const codeMirrorTextArea = await page.waitForSelector('.CodeMirror');
    const initialHeight = await getInitialHeight(page);

    // Query 1
    await selectAllDelete(page, codeMirrorTextArea);
    let textToType = `select * 
    from stocks`;
    await typeQuery(page, textToType);
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000));     
    await scroll(page, await getDistanceToBottom(page, zoomLevel), 3);
    await new Promise(r => setTimeout(r, 1000)); 

    // Query 2
    await selectAllDelete(page, codeMirrorTextArea);
    textToType = `select * 
    from stocks 
    where ticker = 'MSFT'
    order by ts desc`;
    await typeQuery(page, textToType, initialHeight);
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000)); 
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000)); 
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000)); 

    // Query 3
    await selectAllDelete(page, codeMirrorTextArea);
    textToType = `select * 
    from stocks 
    where ticker = 'MSFT'
    order by ts desc
    limit 10
    option(skipUpsert=true)`;
    await typeQuery(page, textToType, initialHeight);
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000));
    await scroll(page, await getDistanceToBottom(page, zoomLevel), 3);
    await new Promise(r => setTimeout(r, 1000)); 

    await browser.close();
}

run();
