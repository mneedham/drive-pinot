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
    const token = "76e2c9a08844df9ebbd4251027766bb1ea9a7e0f73b155b9";
    const { browser, page } = await launchBrowseAndOpenPage({
        url: `http://localhost:8888/doc/tree/VectorSearch-Tutorial.ipynb?token=${token}`, 
        zoomLevel: `${zoomLevel}%`
    });

    // const normalSpeed = 50;
    // const fastSpeed = 20;

    const normalSpeed = 40;
    const fastSpeed = 20;
        
    await page.waitForSelector('.jp-Notebook');
    await stylePage(page);
    await autoScroll(page);
    await new Promise(r => setTimeout(r, 1000))
    
    await newJupyterCell(page);
    let code = `!pip install numpy scikit-learn plotly pandas`;
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 300)

    await newJupyterCell(page, {type: "markdown"});
    await typeAndWaitForScroll(page,`## Creating vectors
Let's create some vectors to work with.`, {delay: fastSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `import numpy as np`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `vectors = np.random.randint(low=1, high=100, size=(20, 2), dtype='int32')
vectors[:5]`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 100)

    await newJupyterCell(page);
    code = `import plotly.graph_objects as go`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `fig = go.Figure()
fig.add_trace(go.Scatter(x=vectors[:, 0], y=vectors[:, 1], mode='markers', marker=dict(size=8, color='blue'), name="Data"))
fig.update_layout(template="plotly_white", margin=dict(t=50, b=50, l=50, r=50), xaxis=dict(range=[0, 100]),  yaxis=dict(range=[0, 100]))
fig`
    await typeAndWaitForScroll(page,code, {delay: fastSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 300)

    
    await newJupyterCell(page, {type: "markdown"});
    await typeAndWaitForScroll(page,`## Finding nearest neighbours
Next, we're going to use sk-learn to find the nearest neighbour for a search vector.`, {delay: fastSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `search_vector = np.array([[10,10]])`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `fig.add_trace(go.Scatter(x=search_vector[:, 0], y=search_vector[:, 1], mode='markers', marker=dict(size=8, color='red'), name="Search Vector"))
fig`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 300)

    await newJupyterCell(page);
    code = `from sklearn.neighbors import NearestNeighbors`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `nn = NearestNeighbors(algorithm="brute", metric="minkowski")
nbrs = nn.fit(vectors)`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `distances, indices = nbrs.kneighbors(search_vector, n_neighbors=3)
print("Distances:", distances)
print("Indices:", indices)`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `for id, distance in zip(indices[0], distances[0]):
print("Vector:", vectors[id])
print("Distance:", distance)
print("")`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 100)
    

    await newJupyterCell(page, {type: "markdown"});
    await typeAndWaitForScroll(page,`## More and bigger vectors
Let's have a look what happens when we use more and bigger vectors.`, {delay: fastSpeed});
    await executeCell(page)
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `dimensions = 64
number_of_vectors = 5_000_000    
vectors = np.random.randint(low=1, high=100_000, size=(number_of_vectors, dimensions), dtype='int32')`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `brute = NearestNeighbors(algorithm="brute", metric="minkowski")
brute_nbrs = brute.fit(vectors)`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `kd_tree = NearestNeighbors(algorithm="kd_tree", metric="minkowski")
kd_tree_nbrs = kd_tree.fit(vectors)`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `ball_tree = NearestNeighbors(algorithm="ball_tree", metric="minkowski")
ball_tree_nbrs = ball_tree.fit(vectors)`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `search_vector = np.random.randint(1, 100_000, (1, dimensions), dtype='int32')
search_vector`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 200)

    
    // for (let i = 0; i < 19; i++) {
    //     await page.keyboard.press('ArrowDown');
    // }

    // let code = ``;
    await newJupyterCell(page);
    code = `import time
def nearest_neighbours(algorithm, number_of_neighbours):
start = time.time()
distances, indices = algorithm.kneighbors(search_vector, n_neighbors=number_of_neighbours)
end = time.time()

    print(f"Time: {end - start:.4f} seconds")

    print("Results:")
return pd.DataFrame({
    "id": indices[0],
    "vector": [vectors[id] for id in indices[0]],
    "distance": distances[0]
})`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);

    await newJupyterCell(page);
    code = `nearest_neighbours(brute_nbrs, 3)`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 200)

    await newJupyterCell(page);
    code = `nearest_neighbours(kd_tree_nbrs, 3)`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 200)


    await newJupyterCell(page);
    code = `nearest_neighbours(ball_tree_nbrs, 3)`
    await typeAndWaitForScroll(page,code, {delay: normalSpeed});
    await executeCell(page);
    await waitForQueryToFinish(page);
    await scrollJupyterDown(page, 200)

    
    

    // old


    // await browser.close();
}

run();
