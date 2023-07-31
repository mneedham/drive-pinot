import { 
    scroll, scrollJupyter, launchBrowseAndOpenPage, getInitialHeight,
    selectAllDelete, typeQuery, runQuery, getDistanceToBottom,
    moveToElement, changeCursor
} from '../drive_pinot.mjs'


async function executeQuery(page) {
    await page.keyboard.down('Meta');
    await page.keyboard.press('Enter');
    await page.keyboard.up('Meta'); 
}

async function waitForQueryToFinish(page) {
    await page.waitForFunction(
        () => document.querySelector('.jp-mod-active .jp-InputPrompt.jp-InputArea-prompt').textContent.includes('*'),
        {timeout: 5000}
    );
  
    await page.waitForFunction(
        () => !document.querySelector('.jp-mod-active .jp-InputPrompt.jp-InputArea-prompt').textContent.includes('*'),
        {timeout: 60000}
    );
}

async function run() {
    const zoomLevel = 250;
    const token = "9f2ae06c9c9640fc75a1f7b013486924e6a15b76ab2248c4";
    const { browser, page } = await launchBrowseAndOpenPage({
        url: `http://localhost:8888/doc/tree/LLMOwnLaptop2.ipynb?token=${token}`, 
        zoomLevel: `${zoomLevel}%`
    });
        
    await page.waitForSelector('.jp-Notebook');
    await new Promise(r => setTimeout(r, 1000))

    // await page.keyboard.down('Meta');
    // await page.keyboard.down('Shift'); 
    // await page.keyboard.press('F');
    // await page.keyboard.up('Shift'); 
    // await page.keyboard.up('Meta');

    // await page.keyboard.press('Escape');
    // await new Promise(r => setTimeout(r, 500))
    // await page.keyboard.press('b');
    // await page.keyboard.press('Enter');

    // await page.keyboard.type('print("Hello, world!")', {delay: 100});

    // await page.keyboard.down('Meta');
    // await page.keyboard.press('Enter');
    // await page.keyboard.up('Meta');

    await page.keyboard.press('Escape');
    await new Promise(r => setTimeout(r, 500))
    await page.keyboard.press('b');
    await page.keyboard.press('Enter');

    let code = `import time
time.sleep(5)`;

    await page.keyboard.type(code, {delay: 100});

    await executeQuery(page)
    await waitForQueryToFinish(page);

    await page.keyboard.press('Escape');
    await new Promise(r => setTimeout(r, 500))
    await page.keyboard.press('b');
    await page.keyboard.press('Enter');

    await page.keyboard.type('print("Hello, world!")', {delay: 100});

    await executeQuery(page);
    await waitForQueryToFinish(page);

    await page.keyboard.press('Escape');
    await new Promise(r => setTimeout(r, 500))
    await page.keyboard.press('b');
    await page.keyboard.press('Enter');

    await page.keyboard.type('print("Hello, Arya!")', {delay: 100});

    await executeQuery(page)
    await waitForQueryToFinish(page);


    // await new Promise(r => setTimeout(r, 3000))

    // const delay = ms => new Promise(res => setTimeout(res, ms));

    // for (let i = 0; i < 5; i++) {
    //     await page.keyboard.press('ArrowDown');
    //     await delay(200); // Sleep for 100 milliseconds
    // }
    
    // await scrollJupyter(page, 1000)

    // await browser.close();
}

run();
