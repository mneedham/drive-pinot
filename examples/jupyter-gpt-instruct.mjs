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
    const token = "942cb35c78fa657c9978edb7d2c4a292d4715aa59d42c04c";
    const { browser, page } = await launchBrowseAndOpenPage({
        url: `http://localhost:8888/doc/tree/GPT-Instruct-Tutorial.ipynb?token=${token}`,
        zoomLevel: `${zoomLevel}%`
    });

    const normalSpeed = 50;
    const fastSpeed = 20;

    // const normalSpeed = 1;
    // const fastSpeed = 0;
        
    await page.waitForSelector('.jp-Notebook');
    await stylePage(page);
    await autoScroll(page);
    await new Promise(r => setTimeout(r, 3000))

    const pause = 150;
    await page.keyboard.press('ArrowDown');
    await new Promise(r => setTimeout(r, pause))
    await page.keyboard.press('ArrowDown');
    await new Promise(r => setTimeout(r, pause))
    await page.keyboard.press('ArrowDown');
    await new Promise(r => setTimeout(r, 1000))
    
    let code = ``;

    await newJupyterCell(page, {type: "markdown"});
    await typeAndWaitForScroll(page,`## Configuring the LLMs ⚙️
First, let's setup our models`, {delay: fastSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `import dotenv
dotenv.load_dotenv()`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `from langchain.chat_models import ChatOpenAI
turbo = ChatOpenAI(model_name="gpt-3.5-turbo")`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `from langchain.llms import OpenAI
turbo_instruct = OpenAI(model_name="gpt-3.5-turbo-instruct")`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    // First question
    await newJupyterCell(page, {type: "markdown"});
    await typeAndWaitForScroll(page,`## Answering factual questions 🤓
First up, how do they answer a factual question?`, {delay: fastSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `question = "Did Microsoft acquire OpenAI?"`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `%%time
print(turbo.predict(question))`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `%%time
print(turbo_instruct.predict(question))`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    // Second question
    await newJupyterCell(page, {type: "markdown"});
    await typeAndWaitForScroll(page,`## Replying with empathy 💕
How well will they do cheering me up?`, {delay: fastSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `question = "I'm feeling sad today. Can you cheer me up?"`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `%%time
print(turbo.predict(question))`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 350);

    await newJupyterCell(page);
    code = `%%time
print(turbo_instruct.predict(question))`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 350);

    // Third question
    await newJupyterCell(page, {type: "markdown"});
    await typeAndWaitForScroll(page,`## Code generation 💻 
A simple coding task is next.`, {delay: fastSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `question = "Can you write a Python function that reads a CSV file and prints out every other row?"`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `%%time
print(turbo.predict(question))`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 350);

    await newJupyterCell(page);
    code = `%%time
print(turbo_instruct.predict(question))`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 350);

    // Forth question
    await newJupyterCell(page, {type: "markdown"});
    await typeAndWaitForScroll(page,`## Analysing sentiment 😊 
Let's see how the models analyse the sentiment of a sentence.`, {delay: fastSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template("Analyse the sentiment of the following text: {text}")
question = prompt.format(text="The DuckDB team is happy to announce the latest DuckDB release (0.9.0).")`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `%%time
for _ in range(0,3):
print(turbo.predict(question))`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 350);

    await newJupyterCell(page);
    code = `%%time
for _ in range(0,3):
print(turbo_instruct.predict(question))`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 350);

    // Fifth question
    await newJupyterCell(page, {type: "markdown"});
    await typeAndWaitForScroll(page,`## Summarising a Wikipedia page 📄 
How well can the models summarise a Wiki page?`, {delay: fastSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `from langchain.document_loaders import WikipediaLoader`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `docs = WikipediaLoader(query="Laver Cup", load_max_docs=1).load()
docs[0].page_content`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 350);
    await page.keyboard.type("s");

    await newJupyterCell(page);
    code = `prompt = PromptTemplate.from_template("Give me 5 interesting things from this text: {text}")
question = prompt.format(text=docs[0].page_content)`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `%%time
print(turbo.predict(question))`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 350);

    await newJupyterCell(page);
    code = `%%time
print(turbo_instruct.predict(question))`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 350);
 

//     await newJupyterCell(page);
//     code = `llm_gpt35 = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);


//     await newJupyterCell(page, {type: "markdown"});
//     await typeAndWaitForScroll(page,`## Let the summarisation begin! 📝 `, {delay: fastSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `from langchain.chains.summarize import load_summarize_chain`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);


//     await newJupyterCell(page);
//     code = `summarize_chain_gpt35 = load_summarize_chain(llm_gpt35, chain_type="stuff")`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `%%time
// summarize_chain_gpt35.run(duck_email)`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 100);

//     await newJupyterCell(page, {type: "markdown"});
//     await typeAndWaitForScroll(page,`## Can we change the way it does the summary? 📇 `, {delay: fastSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `from langchain.prompts import PromptTemplate`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);


//     await newJupyterCell(page);
//     code = `prompt_template = """Write a concise summary of the following:
// "{text}"
// CONCISE SUMMARY:"""
// prompt = PromptTemplate.from_template(prompt_template)

// chain = load_summarize_chain(llm_gpt35, chain_type="stuff", prompt=prompt)`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await page.keyboard.press('Enter');
//     await page.keyboard.down('Meta'); // 'Meta' is the key code for Cmd on Mac
//     await page.keyboard.press('ArrowUp');
//     await page.keyboard.up('Meta');

//     await page.keyboard.down('Alt');
//     await page.keyboard.press('ArrowRight');
//     await page.keyboard.press('ArrowRight');
//     await page.keyboard.press('ArrowRight');
//     await page.keyboard.up('Alt');

//     await page.keyboard.down('Meta');
//     await page.keyboard.down('Shift');
//     await page.keyboard.press('ArrowRight');
//     await page.keyboard.up('Meta');
//     await page.keyboard.up('Shift');
//     await page.keyboard.press('Delete');

//     code = `Pull out a maximum of 5 interesting things as bullet points from the following:`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});

//     await page.keyboard.press('ArrowDown');
//     await page.keyboard.press('ArrowDown');
//     await page.keyboard.press('ArrowLeft');
//     await page.keyboard.press('ArrowLeft');
//     await page.keyboard.press('ArrowLeft');

//     await page.keyboard.down('Meta');
//     await page.keyboard.down('Shift');
//     await page.keyboard.press('ArrowLeft');
//     await page.keyboard.up('Meta');
//     await page.keyboard.up('Shift');
//     await page.keyboard.press('Delete');


//     code = `The 5 things are:`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `print(chain.run(duck_email))`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 250);

//     await newJupyterCell(page, {type: "markdown"});
//     await typeAndWaitForScroll(page,`## How much does it cost? 💰
// We probably want to know how much it'll cost to create these summaries. `, {delay: fastSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `from langchain.callbacks import get_openai_callback`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);


//     await newJupyterCell(page);
//     code = `with get_openai_callback() as cb:
// print(chain.run(duck_email))  
// print(cb)`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 250);

//     await newJupyterCell(page);
//     code = `import time

// def run_with_cost(chain, data):
// start = time.time()
// with get_openai_callback() as cb:
// summary = chain.run(data)
// end = time.time()
    
// return {
// "model": chain.llm_chain.llm.model_name,
// "summary": summary,
// "cost": cb.total_cost,
// "tokens": cb.total_tokens,
// "promptTokens": cb.prompt_tokens,
// "completionTokens": cb.completion_tokens,
// "timeTaken": end - start
// }`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 250);

//     await newJupyterCell(page);
//     code = `result = run_with_cost(chain, duck_email)
// result`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 250);

//     await newJupyterCell(page, {type: "markdown"});
//     await typeAndWaitForScroll(page,`## Comparing different models 🅰️🆚🅱️
// Let's have a look at how summaries vary with different models.`, {delay: fastSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `models = ["gpt-3.5-turbo-16k", "gpt-3.5-turbo", "gpt-4"]
// results = []
// for model_name in models:
// print(f"Running {model_name}")
// llm = ChatOpenAI(temperature=0, model_name=model_name)
// summary_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
// result = run_with_cost(summary_chain, duck_email)
// results.append(result)
// print(f"Finished in  {result['timeTaken']} seconds")`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await scrollJupyterDown(page, 200);
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 250);

//     await newJupyterCell(page);
//     code = `import pandas as pd
// from style_pandas import style_dataframe`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 250);

//     await newJupyterCell(page);
//     code = `style_dataframe(pd.DataFrame(results))`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 250);
//     await new Promise(r => setTimeout(r, 1000))
//     await scrollJupyterDown(page, 550);


//     await newJupyterCell(page);
//     code = `dimensions = 2
// number_of_vectors = 10_000
// vectors = np.random.random((number_of_vectors, dimensions)).astype(np.float32)`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `fig = create_plot()
// plot_points(fig, points=vectors, color='#CCCCCC', label="Data")
// fig`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 300)

//     await newJupyterCell(page);
//     code = `search_vector = np.array([[0.5, 0.5]])`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `plot_points(fig, points=search_vector, color='black', label="Search Vector", symbol="x", size=10)
// fig`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 300)

//     await newJupyterCell(page, {type: "markdown"});
//     await typeAndWaitForScroll(page,`## Creating a cell probe index
// When creating the index, we need to specify how many partitions (or cells) we want to divide the vector space into.`, {delay: fastSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `cells = 10

// quantizer = faiss.IndexFlatL2(dimensions)
// index = faiss.IndexIVFFlat(quantizer, dimensions, cells)`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `index.train(vectors)`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);    

//     await newJupyterCell(page);
//     code = `centroids = index.quantizer.reconstruct_n(0, index.nlist)
// centroids`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 300)

//     await newJupyterCell(page, {type: "markdown"});
//     await typeAndWaitForScroll(page,`## Visualising cells and centroids
// Let's update our chart to show the centroids and to which cell each vector will be assigned.`, {delay: fastSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `_, cell_ids = index.quantizer.search(vectors, k=1)
// cell_ids = cell_ids.flatten()
// cell_ids[:10]`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 300)

//     await newJupyterCell(page);
//     code = `color_map = generate_distinct_colors(index.nlist)

// fig_cells = create_plot()

// unique_ids = np.unique(cell_ids)
// for uid in unique_ids:
// mask = (cell_ids == uid)
// masked_vectors = vectors[mask]
// plot_points(fig_cells, masked_vectors, color_map[uid], "Cell {}".format(uid))

// plot_points(fig_cells, centroids, symbol="diamond-tall", color="black", size=10, showlegend=False)
// plot_points(fig_cells, search_vector, symbol="x", color="black", size=10, label="Search Vector")

// fig_cells`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 300)

//     await newJupyterCell(page, {type: "markdown"});
//     await typeAndWaitForScroll(page,`## Searching for our vector
// Let's add the vectors to the index and look for our search vector.`, {delay: fastSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `index.add(vectors)`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page, {type: "markdown"});
//     await typeAndWaitForScroll(page,`When using a cell probe index, we can specify how many cells we want to use in the search. More cells will mean a slower, but potentially more accurate search.`, {delay: fastSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `index.nprobe`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `%%time 
// distances, indices = index.search(search_vector, k=10)

// df_ann = pd.DataFrame({
// "id": indices[0],
// "vector": [vectors[id] for id in indices[0]],
// "distance": distances[0],
// })
// df_ann`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 300)

//     await newJupyterCell(page);
//     code = `_, search_vectors_cell_ids = index.quantizer.search(search_vector, k=1)
// unique_searched_ids = search_vectors_cell_ids[0]
// unique_searched_ids`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 300)

//     await newJupyterCell(page);
//     code = `fig_search = create_plot()

// for uid in unique_searched_ids:
// mask = (cell_ids == uid)
// masked_vectors = vectors[mask]
// plot_points(fig_search, masked_vectors, color_map[uid], label="Cell {}".format(uid))
// plot_points(fig_search, centroids[uid].reshape(1, -1), symbol="diamond-tall", color="black", size=10, label="Centroid for Cell {}".format(uid), showlegend=False)

// plot_points(fig_search, points=search_vector, color='black', label="Search Vector", symbol="x", size=10)

// ann_vectors = np.array(df_ann["vector"].tolist())
// plot_points(fig_search, points=ann_vectors, color='black', label="Approx Nearest Neighbors")

// fig_search`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 300)

//     await newJupyterCell(page, {type: "markdown"});
//     await typeAndWaitForScroll(page,`# Brute Force Nearest Neighbours
// How well did this approach work compared to a brute force one?`, {delay: fastSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `brute_force_index = faiss.IndexFlatL2(dimensions)
// brute_force_index.add(vectors)`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `%%time
// distances, indices = brute_force_index.search(search_vector, k=10)

// pd.DataFrame({
// "id": indices[0],
// "vector": [vectors[id] for id in indices[0]],
// "distance": distances[0],
// "cell": [cell_ids[id] for id in indices[0]]
// })`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 300)

//     await newJupyterCell(page);
//     code = `index.nprobe = 2`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `index.quantizer.search(search_vector, k=2)`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 300)

//     await newJupyterCell(page);
//     code = `%%time
// distances, indices = index.search(search_vector, k=10)

// pd.DataFrame({
// "id": indices[0],
// "vector": [vectors[id] for id in indices[0]],
// "distance": distances[0],
// "cell": [cell_ids[id] for id in indices[0]]
// })`;
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 300)

//     await newJupyterCell(page, {type: "markdown"});
//     await typeAndWaitForScroll(page,`## Creating vectors
// Let's create some vectors to work with.`, {delay: fastSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `import numpy as np`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `vectors = np.random.randint(low=1, high=100, size=(20, 2), dtype='int32')
// vectors[:5]`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 100)

//     await newJupyterCell(page);
//     code = `import plotly.graph_objects as go`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `fig = go.Figure()
// fig.add_trace(go.Scatter(x=vectors[:, 0], y=vectors[:, 1], mode='markers', marker=dict(size=8, color='blue'), name="Data"))
// fig.update_layout(template="plotly_white", margin=dict(t=50, b=50, l=50, r=50), xaxis=dict(range=[0, 100]),  yaxis=dict(range=[0, 100]))
// fig`
//     await typeAndWaitForScroll(page,code, {delay: fastSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 300)

    
//     await newJupyterCell(page, {type: "markdown"});
//     await typeAndWaitForScroll(page,`## Finding nearest neighbours
// Next, we're going to use sk-learn to find the nearest neighbour for a search vector.`, {delay: fastSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `search_vector = np.array([[10,10]])`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `fig.add_trace(go.Scatter(x=search_vector[:, 0], y=search_vector[:, 1], mode='markers', marker=dict(size=8, color='red'), name="Search Vector"))
// fig`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 300)

//     await newJupyterCell(page);
//     code = `from sklearn.neighbors import NearestNeighbors`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `nn = NearestNeighbors(algorithm="brute", metric="minkowski")
// nbrs = nn.fit(vectors)`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `distances, indices = nbrs.kneighbors(search_vector, n_neighbors=3)
// print("Distances:", distances)
// print("Indices:", indices)`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `for id, distance in zip(indices[0], distances[0]):
// print("Vector:", vectors[id])
// print("Distance:", distance)
// print("")`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 100)
    

//     await newJupyterCell(page, {type: "markdown"});
//     await typeAndWaitForScroll(page,`## More and bigger vectors
// Let's have a look what happens when we use more and bigger vectors.`, {delay: fastSpeed});
//     await executeCell(page)
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `dimensions = 64
// number_of_vectors = 5_000_000    
// vectors = np.random.randint(low=1, high=100_000, size=(number_of_vectors, dimensions), dtype='int32')`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `brute = NearestNeighbors(algorithm="brute", metric="minkowski")
// brute_nbrs = brute.fit(vectors)`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `kd_tree = NearestNeighbors(algorithm="kd_tree", metric="minkowski")
// kd_tree_nbrs = kd_tree.fit(vectors)`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `ball_tree = NearestNeighbors(algorithm="ball_tree", metric="minkowski")
// ball_tree_nbrs = ball_tree.fit(vectors)`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `search_vector = np.random.randint(1, 100_000, (1, dimensions), dtype='int32')
// search_vector`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 200)

    
//     // for (let i = 0; i < 19; i++) {
//     //     await page.keyboard.press('ArrowDown');
//     // }

//     // let code = ``;
//     await newJupyterCell(page);
//     code = `import time
// def nearest_neighbours(algorithm, number_of_neighbours):
// start = time.time()
// distances, indices = algorithm.kneighbors(search_vector, n_neighbors=number_of_neighbours)
// end = time.time()

//     print(f"Time: {end - start:.4f} seconds")

//     print("Results:")
// return pd.DataFrame({
//     "id": indices[0],
//     "vector": [vectors[id] for id in indices[0]],
//     "distance": distances[0]
// })`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);

//     await newJupyterCell(page);
//     code = `nearest_neighbours(brute_nbrs, 3)`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 200)

//     await newJupyterCell(page);
//     code = `nearest_neighbours(kd_tree_nbrs, 3)`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 200)


//     await newJupyterCell(page);
//     code = `nearest_neighbours(ball_tree_nbrs, 3)`
//     await typeAndWaitForScroll(page,code, {delay: normalSpeed});
//     await executeCell(page);
//     await waitForQueryToFinish(page);
//     await scrollJupyterDown(page, 200)

    
    

    // old


    // await browser.close();
}

run();
