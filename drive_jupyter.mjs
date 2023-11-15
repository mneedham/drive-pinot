import puppeteer from 'puppeteer-core';
import { promises as fs } from 'fs';

export async function executeCell(page) {
    await page.keyboard.down('Meta');
    await page.keyboard.press('Enter');
    await page.keyboard.up('Meta'); 
}

export async function waitForQueryToFinish(page) {
    try {
        await page.waitForFunction(
            () => document.querySelector('.jp-mod-active .jp-InputPrompt.jp-InputArea-prompt').textContent.includes('*'),
            {timeout: 1000}
        );
    } catch (e) {
        console.log('Cell may have finished executing before we started waiting for it.');
    }

    await page.waitForFunction(
        () => !document.querySelector('.jp-mod-active .jp-InputPrompt.jp-InputArea-prompt').textContent.includes('*'),
        {timeout: 60000}
    );

    await new Promise(r => setTimeout(r, 1000))
}

// export async function waitForQueryToFinish(page, {pause=30} = {}) {
//     try {
//         await page.waitForFunction(
//             () => document.querySelector('.jp-mod-active .jp-InputPrompt.jp-InputArea-prompt').textContent.includes('*'),
//             { timeout: 1000 }
//         );
//     } catch (e) {
//         console.log('Cell may have finished executing before we started waiting for it.');
//     }

//     let isExecutionFinished = false;
    
//     // Start a loop to check for new output and scroll to reveal it
//     const checkForOutput = async () => {
//         while (!isExecutionFinished) {
//             // Check for new output and scroll to reveal it
//             await scrollOutput(page, {pause: pause});  // Assume this function scrolls to reveal the latest output
            
//             // Pause before checking again
//             await new Promise(r => setTimeout(r, 1000));
//         }
//     };
    
//     // Start the loop in a separate async context so it runs in parallel with the following code
//     checkForOutput();
    
//     // Wait for the cell execution to complete
//     await page.waitForFunction(
//         () => {
//             const isFinished = !document.querySelector('.jp-mod-active .jp-InputPrompt.jp-InputArea-prompt').textContent.includes('*');
//             if (isFinished) {
//                 isExecutionFinished = true;  // Set the flag to terminate the loop
//             }
//             return isFinished;
//         },
//         { timeout: 60000 }
//     );

//     // Additional delay to allow any remaining scrolling to complete
//     await new Promise(r => setTimeout(r, 1000));
// }


export async function newJupyterCell(page, {type= "code"} = {}) {
    await page.keyboard.press('Escape');
    await new Promise(r => setTimeout(r, 500))
    await page.keyboard.press('b');

    if(type === "markdown") {
        await page.keyboard.press('m');
    }

    await page.keyboard.press('Enter');
}


// export async function scrollJupyterDown(page, pixels, {wait = 1000, initWait = 0, scrollPause = 50, scrollStep = 10} = {}) { 
//     await new Promise(r => setTimeout(r, initWait));    
//     const iterations = Math.ceil(pixels / scrollStep);

//     const elementHandle = await page.$('div.jp-WindowedPanel-outer'); // Replace this with the selector for your element

//     // Scroll down
//     for (let i = 0; i < iterations; i++) {
//         await page.evaluate((el, y) => { el.scrollBy(0, y); }, elementHandle, scrollStep);
//         await new Promise(r => setTimeout(r, scrollPause));
//     }
//     await new Promise(r => setTimeout(r, wait));
// }

export async function scrollJupyterDown(page, pixels, {wait = 1000, initWait = 0, scrollPause = 50, scrollStep = 10, extraPause = 200, n = 100} = {}) { 
    await new Promise(r => setTimeout(r, initWait));    
    const iterations = Math.ceil(pixels / scrollStep);

    const elementHandle = await page.$('div.jp-WindowedPanel-outer'); // Replace this with the selector for your element

    // Scroll down
    for (let i = 0; i < iterations; i++) {
        await page.evaluate((el, y) => { el.scrollBy(0, y); }, elementHandle, scrollStep);
        await new Promise(r => setTimeout(r, scrollPause));

        // Insert an extra pause every n iterations
        if ((i + 1) % n === 0) {
            await new Promise(r => setTimeout(r, extraPause));
        }
    }
    await new Promise(r => setTimeout(r, wait));
}


export async function stylePage(page) {
    await page.$eval('div.jp-WindowedPanel-outer', element => {
        element.style.top = '0px';
        // element.style.height = '461px';
        // element.style.marginBottom = '1em';
    });

    await page.evaluate(() => {
        const toolbar = document.querySelector('div.jp-NotebookPanel-toolbar');
        if (toolbar) {
            toolbar.style.display = 'none';
        }

        let style = document.createElement('style');
        style.innerHTML = `
            .jp-Notebook.jp-mod-scrollPastEnd::after {
                display: block;
                content: '';
                margin-bottom: 3em;
                min-height:0;
            }
            .jp-WindowedPanel-inner {
                margin-bottom: 3em;
            }
        `;
        document.head.appendChild(style);
    });
}

// export async function autoScroll(page) {
//     return await page.evaluate(async () => {
//         // Promise-based incremental scroll function
//         window.incrementalScroll = (scrollAmount, delayMs) => {
//             return new Promise((resolve) => {
//                 const container = document.querySelector('div.jp-WindowedPanel-outer');
    
//                 function scrollStep() {
//                     if (container.scrollTop + container.clientHeight < container.scrollHeight) {
//                         container.scrollTop += scrollAmount;
//                         setTimeout(scrollStep, delayMs);
//                     } else {
//                         resolve();  // Resolve the promise when scrolling completes
//                     }
//                 }
    
//                 scrollStep();
//             });
//         };

//         window.scrollWhenCaretNearBottom = async () => {
//             const container = document.querySelector('div.jp-WindowedPanel-outer');
//             const activeElement = document.activeElement;
            
//             if (activeElement) {
//                 const caretRect = activeElement.getBoundingClientRect();
//                 const containerRect = container.getBoundingClientRect();
                
//                 const distanceFromBottom = containerRect.bottom - caretRect.bottom;
                
//                 if (distanceFromBottom < 100) {  // adjust the threshold as needed
//                     await window.incrementalScroll(10, 50);  // Await the completion of scrolling
//                 }
//             }
//         };

//         document.addEventListener('keydown', window.scrollWhenCaretNearBottom);

//         // Ensure the initial scrolling is awaited
//         return await window.scrollWhenCaretNearBottom();
//     });
// }


export async function scrollOutput(page, {step=10, pause=30, longPause=1000, longPauseIntervals=5, initialWait=500, finalWait=500} = {}) {
    // Promise-based delay function
    function delay(time) {
        return new Promise(resolve => setTimeout(resolve, time));
    }

    // Initial wait
    await delay(initialWait);

    await page.evaluate(async (step, pause, longPause, longPauseIntervals) => {
        const outputCells = document.querySelectorAll('.jp-OutputArea-output');
        const lastOutputCell = outputCells[outputCells.length - 1];
        const container = document.querySelector('div.jp-WindowedPanel-outer');
        
        let scrollCounter = 0; // Counter to keep track of how many times we've scrolled

        if (lastOutputCell && container) {
            const incrementalScroll = async () => {
                return new Promise(async (resolve) => {
                    const performScroll = () => {
                        const lastOutputRect = lastOutputCell.getBoundingClientRect();
                        const containerRect = container.getBoundingClientRect();
                        const distanceFromBottom = containerRect.bottom - lastOutputRect.bottom;

                        // Increment the counter every time we scroll
                        scrollCounter++;

                        if (distanceFromBottom < 0) {
                            container.scrollTop += step;
                            // Check if it's time for a long pause
                            if (scrollCounter % longPauseIntervals === 0) {
                                setTimeout(performScroll, longPause);
                            } else {
                                setTimeout(performScroll, pause);
                            }
                        } else {
                            resolve();
                        }
                    };
                    performScroll();
                });
            };

            await incrementalScroll();            
        }
    }, step, pause, longPause, longPauseIntervals);

    // Final wait
    await delay(finalWait);
}

export async function autoScroll(page) {
    await page.evaluate(() => {
        window.isScrolling = false;

        window.incrementalScroll = (scrollAmount, delayMs, maxScrolls = 50) => {
            const container = document.querySelector('div.jp-WindowedPanel-outer');
            let scrollCount = 0;
            
            function scrollStep() {
                window.isScrolling = true;
                if (container.scrollTop + container.clientHeight < container.scrollHeight && scrollCount < maxScrolls) {
                    container.scrollTop += scrollAmount;
                    scrollCount++;
                    setTimeout(scrollStep, delayMs);
                } else {
                    window.isScrolling = false;  // Indicate that scrolling has finished
                }
            }
    
            scrollStep();
        };

        window.triggerAutoScroll = () => {
            return new Promise((resolve) => {
                const container = document.querySelector('div.jp-WindowedPanel-outer');
                const activeElement = document.activeElement;

                if (activeElement) {
                    const caretRect = activeElement.getBoundingClientRect();
                    const containerRect = container.getBoundingClientRect();
                    
                    const desiredCaretPosition = (containerRect.top + containerRect.bottom) / 2;
                    const totalScrollAmount = caretRect.top - desiredCaretPosition;
                    
                    const increment = 10;
                    const delayMs = 20; 
                    let scrolledAmount = 0;

                    function smoothScrollStep() {
                        if (Math.abs(scrolledAmount) < Math.abs(totalScrollAmount)) {
                            const scrollBy = (Math.abs(scrolledAmount + increment) <= Math.abs(totalScrollAmount)) ? increment : totalScrollAmount - scrolledAmount;
                            container.scrollTop += (totalScrollAmount > 0 ? scrollBy : -scrollBy);
                            scrolledAmount += scrollBy;
                            setTimeout(smoothScrollStep, delayMs);
                        } else {
                            resolve();  // Scrolling is done, so resolve the promise
                        }
                    }

                    smoothScrollStep();
                } else {
                    resolve();  // If there's no activeElement, resolve the promise immediately
                }
            });
        };

    });
}



export async function typeAndWaitForScroll(page, text, options) {
    // Trigger the scroll first to position the caret
    await page.evaluate(() => window.triggerAutoScroll && window.triggerAutoScroll());

    // Extended pause to ensure all scrolling completes before typing begins
    await new Promise(resolve => setTimeout(resolve, 500));  // adjust as needed

    // Type the content
    await page.keyboard.type(text, options);

    // Wait a bit to ensure typing has settled
    await new Promise(resolve => setTimeout(resolve, 200));  // adjust as needed

    // Trigger any further scrolling after typing
    await page.evaluate(() => window.triggerAutoScroll && window.triggerAutoScroll());

    // Setup a maximum wait time for the scrolling to complete
    const maxWaitTime = 5000; // 5 seconds, adjust if needed
    const startTime = Date.now();

    // Wait for the scrolling to complete or for maxWaitTime to pass
    while (await page.evaluate(() => window.isScrolling) && Date.now() - startTime < maxWaitTime) {
        await new Promise(resolve => setTimeout(resolve, 100));
    }
}

export async function typeAndWaitForScrollMultiline(page, text, options) {
    // Split the text into lines
    const lines = text.split('\n');

    // Trigger the initial scroll
    await page.evaluate(() => window.triggerAutoScroll && window.triggerAutoScroll());

    // Extended pause for initial scrolling
    await new Promise(resolve => setTimeout(resolve, 500));  // adjust as needed

    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];

        // Move the cursor to the start of the line (Cmd + Left Arrow)
        await page.keyboard.down('Meta'); // 'Meta' is the Command key
        await page.keyboard.press('ArrowLeft');
        await page.keyboard.up('Meta');

        // Type each line individually
        await page.keyboard.type(line, options);

        // Simulate pressing Enter to move to the next line, except for the last line
        if (i < lines.length - 1) {
            await page.keyboard.press('Enter');
        }

        // Wait a bit after each line
        await new Promise(resolve => setTimeout(resolve, 200));  // adjust as needed
    }

    // Trigger any further scrolling after typing
    await page.evaluate(() => window.triggerAutoScroll && window.triggerAutoScroll());

    // Setup a maximum wait time for scrolling
    const maxWaitTime = 5000; // 5 seconds, adjust if needed
    const startTime = Date.now();

    // Wait for the scrolling to complete or timeout
    while (await page.evaluate(() => window.isScrolling) && Date.now() - startTime < maxWaitTime) {
        await new Promise(resolve => setTimeout(resolve, 100));
    }
}






