import puppeteer from 'puppeteer-core';
import { promises as fs } from 'fs';

async function scroll(page, pixels, wait) {
    const scrollStep = 100;
    const iterations = Math.ceil(pixels / scrollStep);

    // Scroll down
    for (let i = 0; i < iterations; i++) {
        await page.evaluate((y) => { window.scrollBy(0, y); }, scrollStep);
        await new Promise(r => setTimeout(r, 50));
    }
    await new Promise(r => setTimeout(r, wait * 1000));

    // Scroll up
    for (let i = 0; i < iterations; i++) {
        await page.evaluate((y) => { window.scrollBy(0, -y); }, scrollStep);
        await new Promise(r => setTimeout(r, 50));
    }
    await new Promise(r => setTimeout(r, wait * 1000));
}


async function selectAllDelete(page, codeMirrorTextArea) {
    await page.evaluate(async element => {
        await document.cursor.moveToTarget(element, speed = 3, offsetX = 0.35, offsetY = 0.20 );
     }, codeMirrorTextArea);

    await page.evaluate(() => {
        return new Promise((resolve, reject) => {
            const codeMirror = document.querySelector('.CodeMirror').CodeMirror;
            codeMirror.execCommand("selectAll");
            setTimeout(() => {
                codeMirror.replaceSelection("");
                resolve();
            }, 500);
        });
    });
}

async function typeQuery(page, query, initialHeight) {
    for(let i = 0; i < query.length; i++) {
        const ch = query.charAt(i);
        await page.evaluate((ch) => {
            return new Promise((resolve) => {
                const codeMirror = document.querySelector('.CodeMirror').CodeMirror;
                codeMirror.replaceRange(ch, codeMirror.getCursor());
                setTimeout(resolve, 100);
            });
        }, ch);
        // Resize the CodeMirror instance after each character is typed
        await resizeCodeMirror(page, query.substring(0, i+1), initialHeight);
    }
}


async function runQuery(page) {
    const runQueryButton = await page.$x("//button[contains(., 'Run Query')]");
    if (runQueryButton.length > 0) {
        let initialBackgroundColor = ''
        await page.evaluate(async (element) => {
            initialBackgroundColor = element.style.backgroundColor;
            await document.cursor.moveToTarget(element, speed = 3, offsetX = 0.35, offsetY = 0.20 );
            element.style.backgroundColor = '#115293';
            await new Promise(r => setTimeout(r, 1000));
        }, runQueryButton[0])
        
        await page.evaluate(async el => {
            el.click();
            await new Promise(r => setTimeout(r, 100));
            el.style.backgroundColor = initialBackgroundColor;
        }, runQueryButton[0])
        

    } else {
        throw new Error("Run Query button not found");
    }
}

async function resizeCodeMirror(page, query, initialHeight) {
    await page.evaluate((query, initialHeight) => {
        const numLines = query.split('\n').length;

        const codeMirror = document.querySelector('.CodeMirror');
        const line = codeMirror.querySelector('.CodeMirror-line');
        const cursor = codeMirror.querySelector('.CodeMirror-cursor');
        
        const style = window.getComputedStyle(line);
        const lineHeight = parseFloat(style.lineHeight);
        const cursorHeight = parseFloat(window.getComputedStyle(cursor).height);

        const header = 48;
        let newHeight = header + (numLines * lineHeight);
        newHeight = Math.max(newHeight + cursorHeight, initialHeight);
        // console.log(numLines, lineHeight, cursorHeight, ((numLines * lineHeight) + cursorHeight), initialHeight, newHeight)

        const codeMirrorTextArea = document.querySelector(".MuiGrid-root .MuiGrid-item .MuiGrid-grid-xs-12 div");
        codeMirrorTextArea.style.height = `${newHeight}px`;
    }, query, initialHeight);
}


async function getInitialHeight(page) {
    return await page.evaluate(() => {
        const codeMirrorTextArea = document.querySelector(".MuiGrid-root .MuiGrid-item .MuiGrid-grid-xs-12 div");
        return parseFloat(window.getComputedStyle(codeMirrorTextArea).height);
    });
}

async function getDistanceToBottom(page) {
    return await page.evaluate(() => {
        const element = document.querySelector('.MuiPaper-root.MuiAccordion-root.Mui-expanded');
        const elementBottom = element.getBoundingClientRect().bottom;
        return elementBottom - window.scrollY;
    });
}

async function run() {
    const browser = await puppeteer.launch({
        headless: false,
        executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',  // Add your path here
        ignoreDefaultArgs: ['--enable-automation'], // exclude this switch
        defaultViewport: null, // required for --window-size
        args: [
            '--start-maximized', // set the window size
            '--disable-infobars',
        ],
    });
    const page = await browser.newPage();    
    await page.goto('http://localhost:9000/#/query');
    // await page.evaluate(() => document.body.style.zoom = "250%" );
    await page.evaluate(() => document.body.style.zoom = "125%" );

    const cursorScript = await fs.readFile('./cursor.js', 'utf8');
    await page.evaluate(cursorScript);  

    const [initPage] = await browser.pages();
    await initPage.close()

    await new Promise(r => setTimeout(r, 1000))

    const codeMirrorTextArea = await page.waitForSelector('.CodeMirror');
    await page.addStyleTag({content: `
        div.CodeMirror-cursors {
            visibility: visible !important;
        }

        .CodeMirror-cursor {
            border-left: 1px solid black;
            animation: blink 1s step-end infinite;
        }

        @keyframes blink {
            50% { visibility: hidden; }
        }

    `});

    const initialHeight = await getInitialHeight(page);
    console.log(initialHeight);

    // Query 1
    await selectAllDelete(page, codeMirrorTextArea);
    let textToType = `select * 
    from stocks`.replace(/^\s+/gm, '');
    await typeQuery(page, textToType);
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000));     

    let distanceToElementBottom = await page.evaluate(() => {
        const element = document.querySelector('.MuiPaper-root.MuiAccordion-root.Mui-expanded');
        const elementBottom = element.getBoundingClientRect().bottom;
        return elementBottom - window.pageYOffset;
    });

    console.log(await getDistanceToBottom(page))
    await scroll(page, await getDistanceToBottom(page), 3);
    await new Promise(r => setTimeout(r, 1000)); 

    // Query 2
    await selectAllDelete(page, codeMirrorTextArea);
    textToType = `select * 
    from stocks 
    where ticker = 'MSFT'
    order by ts desc
    limit 10
    option(skipUpsert=true)`.replace(/^\s+/gm, '');
    await typeQuery(page, textToType, initialHeight);
    await runQuery(page);
    await new Promise(r => setTimeout(r, 1000));     
    console.log(await getDistanceToBottom(page))
    await scroll(page, await getDistanceToBottom(page), 3);
    await new Promise(r => setTimeout(r, 1000)); 

    await browser.close();
}

run();
