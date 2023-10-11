import { 
    scroll, launchBrowseAndOpenPage, getInitialHeight,
    selectAllDelete, typeQuery, runQuery, getDistanceToBottom,
    moveToElement, changeCursor
} from '../drive_pinot.mjs'

import http from 'http';

let proceed = false;

// This function waits until the 'proceed' flag is set to true
async function waitForSignal() {
    return new Promise(resolve => {
        const checkInterval = setInterval(() => {
            if (proceed) {
                clearInterval(checkInterval);
                resolve();
            }
        }, 1000);
    });
}

const server = http.createServer((req, res) => {
    if (req.url === '/continue') {
        proceed = true;
        res.end('Script will proceed');
    } else {
        res.end('Unknown command');
    }
});

server.listen(3000, () => {
    console.log('Control server running on http://localhost:3000');
});


async function run() {
    const zoomLevel = 250;
    const { browser, page } = await launchBrowseAndOpenPage({url: 'http://localhost:9000/#/query', zoomLevel: `${zoomLevel}%`});

    await new Promise(r => setTimeout(r, 3000))

    const codeMirrorTextArea = await page.waitForSelector('.CodeMirror');
    const initialHeight = await getInitialHeight(page);

    // Query events table
    const textToType = `
    select $segmentName, ToDateTime(max(ts), 'YYYY-MM-dd HH:mm:ss') as maxTs, count(*)
    from events
    group by $segmentName
    order by maxTs desc
    limit 100`;
    
    changeCursor(page, {type: "arrow"});
    await selectAllDelete(page, codeMirrorTextArea);
    await typeQuery(page, textToType, initialHeight, {charPause: 20});
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000));
    await scroll(page, 800, 3);
    await new Promise(r => setTimeout(r, 1000)); 

    // await runQuery(page);
    // await new Promise(r => setTimeout(r, 1000));
    // await scroll(page, 1000, 3);
    // await new Promise(r => setTimeout(r, 3000)); 

    // Pause here until signal received
    console.log("Waiting for /continue signal to proceed...");
    await waitForSignal();

    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000));
    await scroll(page, 1000, 3);
    await new Promise(r => setTimeout(r, 3000)); 

    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000));
    await scroll(page, 1000, 3);
    await new Promise(r => setTimeout(r, 3000)); 

    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000));
    await scroll(page, 1000, 3);
    await new Promise(r => setTimeout(r, 3000)); 

    // await browser.close();
}

run();
