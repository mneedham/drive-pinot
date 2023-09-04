import { 
    scroll, scrollJupyter, launchBrowseAndOpenPage, getInitialHeight,
    selectAllDelete, typeQuery, runQuery, getDistanceToBottom,
    moveToElement, changeCursor
} from '../drive_pinot.mjs'

import {
    executeCell, waitForQueryToFinish,
    newJupyterCell, scrollJupyterDown, typeAndWaitForScroll,
    stylePage, autoScroll, scrollOutput
} from '../drive_jupyter.mjs'

async function run() {
    const zoomLevel = 400;
    const token = "26b85e4f4189710a869de9934150f58450321507bd920902";
    const { browser, page } = await launchBrowseAndOpenPage({
        url: `http://localhost:8888/doc/tree/ChromaDb-Tutorial.ipynb?token=${token}`, 
        zoomLevel: `${zoomLevel}%`
    });
        
    await page.waitForSelector('.jp-Notebook');
    await stylePage(page);
    await autoScroll(page);
    await new Promise(r => setTimeout(r, 1000))

    await newJupyterCell(page);
    let code = `import chromadb
import uuid
import pprint
import chromadb.utils.embedding_functions as ef
import csv`;
    await typeAndWaitForScroll(page,code, {delay: 20});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `client = chromadb.PersistentClient(path="vector.db")`
    await typeAndWaitForScroll(page,code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `collection = client.create_collection(
name="Wimbledon",
embedding_function=ef.DefaultEmbeddingFunction() # Default if no function provided
)`
    await typeAndWaitForScroll(page,code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page, {type: "markdown"});
    await typeAndWaitForScroll(page,`## Storing Documents
First, we're going to learn how to store some documents.`, {delay: 20});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `with open("sentences.csv", "r") as sentences_file:
reader = csv.reader(sentences_file, delimiter=",")
documents = [row[0] for row in reader]

documents`
    await typeAndWaitForScroll(page,code, {delay: 20});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollOutput(page, {longPauseIntervals:0, longPause:500, pause:50});

    await newJupyterCell(page);
    code = `collection.add(
documents=documents,
metadatas=[{"source": "Wikipedia"} for _ in documents],
ids=[f"id-{uuid.uuid4()}" for _ in documents]
)`
    await typeAndWaitForScroll(page,code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page, {type: "markdown"});
    await typeAndWaitForScroll(page,`## Where are the vectors? 
We're using a vector database, but haven't seen any vectors yet. Let's dig into what happened when we added those documents.`, {delay: 20});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `embedding_fn = ef.DefaultEmbeddingFunction()`
    await typeAndWaitForScroll(page,code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `embeddings = embedding_fn(documents)
pprint.pprint(embeddings, width=150, compact=True)`
    await typeAndWaitForScroll(page,code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await new Promise(r => setTimeout(r, 500))
    await page.keyboard.press('s'); // Make the output scrollable
    await new Promise(r => setTimeout(r, 500))

    await newJupyterCell(page, {type: "markdown"});
    await typeAndWaitForScroll(page,`## Querying documents
It's time to query the documents that we stored!`, {delay: 20});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `results = collection.query(
query_texts=["Clothing"],
n_results=3
)
pprint.pprint(results)`
    await typeAndWaitForScroll(page,code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollOutput(page, {longPauseIntervals:0, longPause:500, pause:50});

    await newJupyterCell(page);
    code = `query_embeddings = embedding_fn(["2023 dates"])
results = collection.query(
query_embeddings=query_embeddings,
n_results=3
)
pprint.pprint(results)`
    await typeAndWaitForScroll(page,code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollOutput(page, {longPauseIntervals:0, longPause:500, pause:50});

    await newJupyterCell(page, {type: "markdown"});
    await typeAndWaitForScroll(page,`## Using a different embedding algorithm
We don't have to use the default embeddng algorithm, we can use one from OpenAI, HuggingFace, or another provider.`, {delay: 20});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `import inspect
[
cls_name
for cls_name in dir(ef) 
if inspect.isclass(getattr(ef,cls_name)) and ef.EmbeddingFunction in getattr(ef,cls_name).__bases__
]`
    await typeAndWaitForScroll(page,code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollOutput(page, {longPauseIntervals:0, longPause:500, pause:50});

    await newJupyterCell(page);
    code = `collection2 = client.create_collection(
name="Wimbledon-DistiRoberta",
embedding_function=ef.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-distilroberta-v1")
)`
    await typeAndWaitForScroll(page,code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `collection2.add(
documents=documents,
metadatas=[{"source": "Wikipedia"} for _ in documents],
ids=[f"id-{uuid.uuid4()}" for _ in documents]
)`
    await typeAndWaitForScroll(page,code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `results = collection2.query(
query_texts=["Clothing"],
n_results=3,
include=["documents", "distances"]
)
pprint.pprint(results)`
    await typeAndWaitForScroll(page,code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollOutput(page, {longPauseIntervals:0, longPause:500, pause:50});

    await newJupyterCell(page);
    code = `query_embeddings = ef.DefaultEmbeddingFunction()(["2023 dates"])
results = collection2.query(
query_embeddings=query_embeddings,
n_results=3
)
pprint.pprint(results)`
    await typeAndWaitForScroll(page,code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollOutput(page, {longPauseIntervals:0, longPause:500, pause:50});
    


//     await newJupyterCell(page);
//     code = `model_id = "lmsys/fastchat-t5-3b-v1.0"
// filenames = [
//     "pytorch_model.bin", "added_tokens.json", "config.json", "generation_config.json", 
//     "special_tokens_map.json", "spiece.model", "tokenizer_config.json"
// ]`
//     await typeAndWaitForScroll(page,code, {delay: 100});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 200);

//     await newJupyterCell(page);
//     code = `for filename in filenames:
//     downloaded_model_path = hf_hub_download(
//         repo_id=model_id,
//         filename=filename,
//         token=HUGGING_FACE_API_KEY
//     )
//     print(downloaded_model_path)`
//     await typeAndWaitForScroll(page,code, {delay: 100});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 250);

    // await page.keyboard.press('ArrowDown');
    // await page.keyboard.press('ArrowDown');
    // await page.keyboard.press('ArrowDown');
    // await page.keyboard.press('ArrowDown');
    // await page.keyboard.press('ArrowDown');

//     await newJupyterCell(page, {type: "markdown"});    
//     await scrollJupyterDown(page, 200);
//     await typeAndWaitForScroll(page,`## Run the LLM
// Now let's try running the model. But before we do that, let's disable the Wi-Fi.`, {delay: 50});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     await scrollJupyterDown(page, 200);
//     code = `from utils import check_connectivity, toggle_wifi
// import time

// print(check_connectivity())
// toggle_wifi("off")
// time.sleep(0.5)
// print(check_connectivity())`    
//     await typeAndWaitForScroll(page,code, {delay: 100});
//     await executeCell(page)
//     await waitForQueryToFinish(page);


//     await newJupyterCell(page);
//     await scrollJupyterDown(page, 200);
//     code = `from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

// tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
// model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

// pipeline = pipeline("text2text-generation", model=model, device=-1, tokenizer=tokenizer, max_length=1000)`    
//     await typeAndWaitForScroll(page,code, {delay: 100});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 200);

//     await newJupyterCell(page);
//     await scrollJupyterDown(page, 200);
//     code = `pipeline("What are competitors to Apache Kafka?")`    
//     await typeAndWaitForScroll(page,code, {delay: 100});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 200);

//     await newJupyterCell(page);
//     await scrollJupyterDown(page, 200);
//     code = `pipeline("""My name is Mark.
// I have brothers called David and John and my best friend is Michael.
// Using only the context above. Do you know if I have a sister?    
// """)`    
//     await typeAndWaitForScroll(page,code, {delay: 100});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 200);    


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
