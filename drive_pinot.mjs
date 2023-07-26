import puppeteer from 'puppeteer-core';
import { promises as fs } from 'fs';

export async function scroll(page, pixels, wait) {
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

export async function moveToElement(page, element, {speed=3, offsetX=0.35, offsetY=0.20} = {}) {
    await page.evaluate(async (element, speed, offsetX, offsetY) => {
        await document.cursor.moveToTarget(element, speed = speed, offsetX = offsetX, offsetY = offsetY );
     }, element, speed, offsetX, offsetY);
}

export async function selectAllDelete(page, codeMirrorTextArea) {
    await moveToElement(page, codeMirrorTextArea)

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

export async function changeCursor(page, {type="arrow"}= {}) {
    await page.evaluate(async (type)  => {
        document.cursor.changeCursor(type);
     }, type);

    
}

export async function typeQuery(page, query, initialHeight, {charPause=100, shouldCleanQuery=true} = {}) {
    const cleanQuery = shouldCleanQuery ?  query.replace(/^\s+/gm, '') : query;


    for(let i = 0; i < cleanQuery.length; i++) {
        const ch = cleanQuery.charAt(i);
        await page.evaluate((ch, charPause) => {
            return new Promise((resolve) => {
                const codeMirror = document.querySelector('.CodeMirror').CodeMirror;
                codeMirror.replaceRange(ch, codeMirror.getCursor());
                setTimeout(resolve, charPause);
            });
        }, ch, charPause);
        // Resize the CodeMirror instance after each character is typed
        // await resizeCodeMirror(page, cleanQuery.substring(0, i+1), initialHeight);
        await resizeCodeMirror(page, initialHeight);
    }
}


export async function runQuery(page) {
    const runQueryButton = await page.$x("//button[contains(., 'Run Query')]");
    if (runQueryButton.length > 0) {
        let initialBackgroundColor = ''
        await page.evaluate(async (element) => {
            initialBackgroundColor = element.style.backgroundColor;
            await document.cursor.moveToTarget(element, speed = 3, offsetX = 0.35, offsetY = 0.20 );
            element.style.backgroundColor = '#115293';            
        }, runQueryButton[0])

        await changeCursor(page, {type: "hand"})
        await new Promise(r => setTimeout(r, 1000));
        
        await page.evaluate(async el => {
            el.click();
            await new Promise(r => setTimeout(r, 100));
            el.style.backgroundColor = initialBackgroundColor;
        }, runQueryButton[0])
        

    } else {
        throw new Error("Run Query button not found");
    }
}

export async function resizeCodeMirror(page, initialHeight) {
    await page.evaluate((initialHeight) => {
        const codeMirrorInstance = document.querySelector('.CodeMirror').CodeMirror;
        const numLines = codeMirrorInstance.lineCount();

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
    }, initialHeight);
}


export async function resizeCodeMirrorWithQuery(page, query, initialHeight) {
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


export async function getInitialHeight(page) {
    return await page.evaluate(() => {
        const codeMirrorTextArea = document.querySelector(".MuiGrid-root .MuiGrid-item .MuiGrid-grid-xs-12 div");
        return parseFloat(window.getComputedStyle(codeMirrorTextArea).height);
    });
}

export async function getDistanceToBottom(page, zoomLevel) {
    return await page.evaluate((zoomLevel) => {
        const zoomFactor = parseInt(zoomLevel) / 100;
        const element = document.querySelector('.MuiPaper-root.MuiAccordion-root.Mui-expanded');
        const elementBottom = element.getBoundingClientRect().bottom;
        return (elementBottom - window.scrollY) * zoomFactor;
    }, zoomLevel);
}


export async function launchBrowseAndOpenPage({url, zoomLevel="250%"}) {
    const browser = await puppeteer.launch({
        headless: false,
        executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        ignoreDefaultArgs: ['--enable-automation'], // exclude this switch
        defaultViewport: null, // required for --window-size
        args: [
        //   '--start-maximized', // set the window size          
          '--disable-infobars',
          '--window-size=3840,2160', 
          '--window-position=-1080,0' 
        ],
      });
    
      const page = await browser.newPage();
      await page.goto(url);
      await page.evaluate((zoomLevel) => { document.body.style.zoom = zoomLevel; }, zoomLevel);
    
      const cursorScript = await fs.readFile('./cursor.js', 'utf8');
      await page.evaluate(cursorScript);
    
      const [initPage] = await browser.pages();
      await initPage.close();

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
    
  return { browser, page };
}