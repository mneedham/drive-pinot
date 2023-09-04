import { 
    scroll, scrollJupyter, launchBrowseAndOpenPage, getInitialHeight,
    selectAllDelete, typeQuery, runQuery, getDistanceToBottom,
    moveToElement, changeCursor
} from '../drive_pinot.mjs'

import {
    executeCell, waitForQueryToFinish,
    newJupyterCell, scrollJupyterDown
} from '../drive_jupyter.mjs'

async function run() {
    const zoomLevel = 350;
    const token = "d0cb0f0a9ee2e953b05cc4a0934b1f6b188acdbf4ff399ce";
    const { browser, page } = await launchBrowseAndOpenPage({
        url: `http://localhost:8888/doc/tree/RAG-Tutorial.ipynb?token=${token}`, 
        zoomLevel: `${zoomLevel}%`
    });
        
    await page.waitForSelector('.jp-Notebook');

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
                margin-bottom: 7em;
                min-height:0;
            }
            .jp-WindowedPanel-inner {
                margin-bottom: 5em;
            }
        `;
        document.head.appendChild(style);
    });

    await page.evaluate(() => {
        window.incrementalScroll = (scrollAmount, delayMs) => {
            const container = document.querySelector('div.jp-WindowedPanel-outer');
    
            function scrollStep() {
                if (container.scrollTop + container.clientHeight < container.scrollHeight) {
                    container.scrollTop += scrollAmount;
                    setTimeout(scrollStep, delayMs);
                }
            }
    
            scrollStep();
        };

        window.scrollWhenCaretNearBottom = () => {
            const container = document.querySelector('div.jp-WindowedPanel-outer');
            const activeElement = document.activeElement;
            
            if (activeElement) {
                const caretRect = activeElement.getBoundingClientRect();
                const containerRect = container.getBoundingClientRect();
                
                const distanceFromBottom = containerRect.bottom - caretRect.bottom;
                
                if (distanceFromBottom < 100) {  // adjust the threshold as needed
                    window.incrementalScroll(10, 20); 
                }
            }
        };
        document.addEventListener('keydown', window.scrollWhenCaretNearBottom);
    });

    await new Promise(r => setTimeout(r, 1000))
    // await page.keyboard.down('Meta');
    // await page.keyboard.down('Shift'); 
    // await page.keyboard.press('F');
    // await page.keyboard.up('Shift'); 
    // await page.keyboard.up('Meta');

    await newJupyterCell(page, {type: "markdown"});
    await page.keyboard.type(`## Load data from Wikipedia
We're going to first extract data from the Wimbledon 2023 Wikipedia page.`, {delay: 20});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await new Promise(r => setTimeout(r, 500));
    // await scrollJupyterDown(page, 250, {initWait:500, wait: 1000});
    
    await newJupyterCell(page);
    let code = `from langchain.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter`;
    await page.keyboard.type(code, {delay: 20});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await new Promise(r => setTimeout(r, 500));
    // await scrollJupyterDown(page, 250, {initWait:500, wait: 1000});

    await newJupyterCell(page);
    code = `search_term = "2023 Wimbledon Championships"
docs = WikipediaLoader(query=search_term, load_max_docs=1).load()`
    await page.keyboard.type(code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await new Promise(r => setTimeout(r, 500));
    // await scrollJupyterDown(page, 250, {initWait:500, wait: 1000});

    await newJupyterCell(page);
    code = `text_splitter = RecursiveCharacterTextSplitter(
chunk_size = 100,
chunk_overlap  = 20,
length_function = len,
is_separator_regex = False,
)

data = text_splitter.split_documents(docs)
data[:3]`
    await page.keyboard.type(code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await new Promise(r => setTimeout(r, 500));
    // await scrollJupyterDown(page, 400, {initWait:500, wait: 1000});

    await newJupyterCell(page, {type: "markdown"});
    await page.keyboard.type(`## Storing embeddings in ChromaDB
Next, let's store those chunks of text as embeddings in ChromaDB`, {delay: 20});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await new Promise(r => setTimeout(r, 500));
    // await scrollJupyterDown(page, 250, {initWait:500, wait: 1000});

    await newJupyterCell(page);
    code = `from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings`
    await page.keyboard.type(code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await new Promise(r => setTimeout(r, 500));
    // await scrollJupyterDown(page, 250, {initWait:500, wait: 1000});

    await newJupyterCell(page);
    code = `embeddings = OpenAIEmbeddings()`
    await page.keyboard.type(code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await new Promise(r => setTimeout(r, 500));
    // await scrollJupyterDown(page, 250, {initWait:500, wait: 1000});


    await newJupyterCell(page);
    code = `store = Chroma.from_documents(
data, 
embeddings, 
ids = [f"{item.metadata['source']}-{index}" for index, item in enumerate(data)],
collection_name="Wimbledon-Embeddings", 
persist_directory='db',
)
store.persist()`
    await page.keyboard.type(code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await new Promise(r => setTimeout(r, 500));
    // await scrollJupyterDown(page, 250, {initWait:500, wait: 1000});

    await newJupyterCell(page, {type: "markdown"});
    await page.keyboard.type(`## Asking questions about Wimbledon 2023
Now let's use OpenAI, augmented by ChromaDB, to ask some questions about the tournament.`, {delay: 20});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await new Promise(r => setTimeout(r, 500));
    // await scrollJupyterDown(page, 250, {initWait:500, wait: 1000});

    await newJupyterCell(page);
    code = `from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import pprint`
    await page.keyboard.type(code, {delay: 20});
    await executeCell(page);
    await waitForQueryToFinish(page);
    // await scrollJupyterDown(page, 250, {initWait:500, wait: 1000});

    await newJupyterCell(page);
    code = `template = """You are a bot that answers questions about Wimbledon 2023, using only the context provided.
If you don't know the answer, simply state that you don't know.

{context}

Question: {question}"""

PROMPT = PromptTemplate(
template=template, input_variables=["context", "question"]
)`
    await page.keyboard.type(code, {delay: 20});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await new Promise(r => setTimeout(r, 500));

    await newJupyterCell(page);
    code = `llm = ChatOpenAI(temperature=0, model="gpt-4")`
    await page.keyboard.type(code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await new Promise(r => setTimeout(r, 500));

    await newJupyterCell(page);
    code = `qa_with_source = RetrievalQA.from_chain_type(
llm=llm,
chain_type="stuff",
retriever=store.as_retriever(),
chain_type_kwargs={"prompt": PROMPT, },
return_source_documents=True,
)`
    await page.keyboard.type(code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await new Promise(r => setTimeout(r, 500));

    await newJupyterCell(page);
    code = `pprint.pprint(
qa_with_source("When and where was Wimbledon 2023 held?")
)`
    await page.keyboard.type(code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 150, {initWait:500, wait: 500});

    await newJupyterCell(page);
    code = `pprint.pprint(
qa_with_source("Who won the mens' singles title and what was the score?")
)`
    await page.keyboard.type(code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 150, {initWait:500, wait: 500});

    await newJupyterCell(page);
    code = `pprint.pprint(
qa_with_source("Were Russian players allowed to play?")
)`
    await page.keyboard.type(code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 150, {initWait:500, wait: 500});

    await newJupyterCell(page);
    code = `pprint.pprint(
qa_with_source("Did Russian players play?")
)`
    await page.keyboard.type(code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 150, {initWait:500, wait: 500});

    await newJupyterCell(page);
    code = `pprint.pprint(
qa_with_source("Did British players play?")
)`
    await page.keyboard.type(code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 150, {initWait:500, wait: 500});

    await newJupyterCell(page);
    code = `pprint.pprint(
qa_with_source("Were any extra events held?")
)`
    await page.keyboard.type(code, {delay: 50});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 150, {initWait:500, wait: 500});

//     await newJupyterCell(page, {type: "markdown"});
//     await page.keyboard.type(`## Storing Documents
// First, we're going to learn how to store some documents.`, {delay: 20});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `with open("sentences.csv", "r") as sentences_file:
// reader = csv.reader(sentences_file, delimiter=",")
// documents = [row[0] for row in reader]

// documents`
//     await page.keyboard.type(code, {delay: 20});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `collection.add(
// documents=documents,
// metadatas=[{"source": "Wikipedia"} for _ in documents],
// ids=[f"id-{uuid.uuid4()}" for _ in documents]
// )`
//     await page.keyboard.type(code, {delay: 50});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page, {type: "markdown"});
//     await page.keyboard.type(`## Where are the vectors? 
// We're using a vector database, but haven't seen any vectors yet. Let's dig into what happened when we added those documents.`, {delay: 20});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `embedding_fn = ef.DefaultEmbeddingFunction()`
//     await page.keyboard.type(code, {delay: 50});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `embeddings = embedding_fn(documents)
// pprint.pprint(embeddings, width=150, compact=True)`
//     await page.keyboard.type(code, {delay: 50});
//     await executeCell(page);
//     await waitForQueryToFinish(page);
//     await page.keyboard.press('s'); // Make the output scrollable

//     await newJupyterCell(page, {type: "markdown"});
//     await page.keyboard.type(`## Querying documents
// It's time to query the documents that we stored!`, {delay: 20});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `results = collection.query(
// query_texts=["Clothing"],
// n_results=3
// )
// pprint.pprint(results)`
//     await page.keyboard.type(code, {delay: 50});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `query_embeddings = embedding_fn(["2023 dates"])
// results = collection.query(
// query_embeddings=query_embeddings,
// n_results=3
// )
// pprint.pprint(results)`
//     await page.keyboard.type(code, {delay: 50});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page, {type: "markdown"});
//     await page.keyboard.type(`## Using a different embedding algorithm
// We don't have to use the default embeddng algorithm, we can use one from OpenAI, HuggingFace, or another provider.`, {delay: 20});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `import inspect
// [
// cls_name
// for cls_name in dir(ef) 
// if inspect.isclass(getattr(ef,cls_name)) and ef.EmbeddingFunction in getattr(ef,cls_name).__bases__
// ]`
//     await page.keyboard.type(code, {delay: 50});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `collection2 = client.create_collection(
// name="Wimbledon-DistiRoberta",
// embedding_function=ef.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-distilroberta-v1")
// )`
//     await page.keyboard.type(code, {delay: 50});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `collection2.add(
// documents=documents,
// metadatas=[{"source": "Wikipedia"} for _ in documents],
// ids=[f"id-{uuid.uuid4()}" for _ in documents]
// )`
//     await page.keyboard.type(code, {delay: 50});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `results = collection2.query(
// query_texts=["Clothing"],
// n_results=3,
// include=["documents", "distances"]
// )
// pprint.pprint(results)`
//     await page.keyboard.type(code, {delay: 50});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `query_embeddings = ef.DefaultEmbeddingFunction()(["2023 dates"])
// results = collection2.query(
// query_embeddings=query_embeddings,
// n_results=3
// )
// pprint.pprint(results)`
//     await page.keyboard.type(code, {delay: 50});
//     await executeCell(page);
//     await waitForQueryToFinish(page);
    


//     await newJupyterCell(page);
//     code = `model_id = "lmsys/fastchat-t5-3b-v1.0"
// filenames = [
//     "pytorch_model.bin", "added_tokens.json", "config.json", "generation_config.json", 
//     "special_tokens_map.json", "spiece.model", "tokenizer_config.json"
// ]`
//     await page.keyboard.type(code, {delay: 100});
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
//     await page.keyboard.type(code, {delay: 100});
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
//     await page.keyboard.type(`## Run the LLM
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
//     await page.keyboard.type(code, {delay: 100});
//     await executeCell(page)
//     await waitForQueryToFinish(page);


//     await newJupyterCell(page);
//     await scrollJupyterDown(page, 200);
//     code = `from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

// tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
// model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

// pipeline = pipeline("text2text-generation", model=model, device=-1, tokenizer=tokenizer, max_length=1000)`    
//     await page.keyboard.type(code, {delay: 100});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 200);

//     await newJupyterCell(page);
//     await scrollJupyterDown(page, 200);
//     code = `pipeline("What are competitors to Apache Kafka?")`    
//     await page.keyboard.type(code, {delay: 100});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 200);

//     await newJupyterCell(page);
//     await scrollJupyterDown(page, 200);
//     code = `pipeline("""My name is Mark.
// I have brothers called David and John and my best friend is Michael.
// Using only the context above. Do you know if I have a sister?    
// """)`    
//     await page.keyboard.type(code, {delay: 100});
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
