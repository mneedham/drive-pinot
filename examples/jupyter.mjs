import { 
    scroll, scrollJupyter, launchBrowseAndOpenPage, getInitialHeight,
    selectAllDelete, typeQuery, runQuery, getDistanceToBottom,
    moveToElement, changeCursor
} from '../drive_pinot.mjs'

import {
    executeCell, waitForQueryToFinish
} from '../drive_jupyter.mjs'


async function newJupyterCell(page, {type= "code"} = {}) {
    await page.keyboard.press('Escape');
    await new Promise(r => setTimeout(r, 500))
    await page.keyboard.press('b');

    if(type === "markdown") {
        await page.keyboard.press('m');
    }

    await page.keyboard.press('Enter');
}


export async function scrollJupyterDown(page, pixels, {wait, scrollPause} = {wait: 3000, scrollPause: 50}) {
    const scrollStep = 100;
    const iterations = Math.ceil(pixels / scrollStep);

    const elementHandle = await page.$('div.jp-WindowedPanel-outer'); // Replace this with the selector for your element

    // Scroll down
    for (let i = 0; i < iterations; i++) {
        await page.evaluate((el, y) => { el.scrollBy(0, y); }, elementHandle, scrollStep);
        await new Promise(r => setTimeout(r, scrollPause));
    }
    await new Promise(r => setTimeout(r, wait));
}

async function run() {
    const zoomLevel = 400;
    const token = "9f2ae06c9c9640fc75a1f7b013486924e6a15b76ab2248c4";
    const { browser, page } = await launchBrowseAndOpenPage({
        url: `http://localhost:8888/doc/tree/LLMOwnLaptop.ipynb?token=${token}`, 
        zoomLevel: `${zoomLevel}%`
    });
        
    await page.waitForSelector('.jp-Notebook');

    await page.evaluate(() => {
        let style = document.createElement('style');
        style.type = 'text/css';
        style.innerHTML = `
            .jp-Notebook.jp-mod-scrollPastEnd::after {
                display: block;
                content: '';
                margin-bottom: 7em;
                min-height:0;
            }
        `;
        document.head.appendChild(style);
    });
    

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

    await newJupyterCell(page, {type: "markdown"});
    await page.keyboard.type(`## Download the LLM
We're going to write some code to manually download the model.`, {delay: 50});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    let code = `import os
from huggingface_hub import hf_hub_download`;
    await page.keyboard.type(code, {delay: 100});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    await page.keyboard.type('HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")', {delay: 100});
    await executeCell(page);
    await waitForQueryToFinish(page);


    await newJupyterCell(page);
    code = `model_id = "lmsys/fastchat-t5-3b-v1.0"
filenames = [
    "pytorch_model.bin", "added_tokens.json", "config.json", "generation_config.json", 
    "special_tokens_map.json", "spiece.model", "tokenizer_config.json"
]`
    await page.keyboard.type(code, {delay: 100});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 200);

    await newJupyterCell(page);
    code = `for filename in filenames:
    downloaded_model_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        token=HUGGING_FACE_API_KEY
    )
    print(downloaded_model_path)`
    await page.keyboard.type(code, {delay: 100});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 250);

    // await page.keyboard.press('ArrowDown');
    // await page.keyboard.press('ArrowDown');
    // await page.keyboard.press('ArrowDown');
    // await page.keyboard.press('ArrowDown');
    // await page.keyboard.press('ArrowDown');

    await newJupyterCell(page, {type: "markdown"});    
    await scrollJupyterDown(page, 200);
    await page.keyboard.type(`## Run the LLM
Now let's try running the model. But before we do that, let's disable the Wi-Fi.`, {delay: 50});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    await scrollJupyterDown(page, 200);
    code = `from utils import check_connectivity, toggle_wifi
import time

print(check_connectivity())
toggle_wifi("off")
time.sleep(0.5)
print(check_connectivity())`    
    await page.keyboard.type(code, {delay: 100});
    await executeCell(page)
    await waitForQueryToFinish(page);


    await newJupyterCell(page);
    await scrollJupyterDown(page, 200);
    code = `from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipeline = pipeline("text2text-generation", model=model, device=-1, tokenizer=tokenizer, max_length=1000)`    
    await page.keyboard.type(code, {delay: 100});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 200);

    await newJupyterCell(page);
    await scrollJupyterDown(page, 200);
    code = `pipeline("What are competitors to Apache Kafka?")`    
    await page.keyboard.type(code, {delay: 100});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 200);

    await newJupyterCell(page);
    await scrollJupyterDown(page, 200);
    code = `pipeline("""My name is Mark.
I have brothers called David and John and my best friend is Michael.
Using only the context above. Do you know if I have a sister?    
""")`    
    await page.keyboard.type(code, {delay: 100});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 200);    


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
