# Institutional Data Initiative
🤖 Take home exercise for IDI's engineering and data science roles, Q2 2025.

---

## Summary

- [Foreword](#foreword)
- [Instructions](#instructions)

---

## Foreword

Thank you so much for taking the time to complete this coding exercise. 

We tried our best to come up with an exercise that lets you show us how you go about solving the types of problems we encounter on a regular basis at the lab. 


[👆 Back to summary](#summary)

---

## Instructions

- **Please do not spend more than 4 hours on this exercise**. These four hours do not need to be consecutive, take as many breaks as you need. 
- **Please submit your code as a pull request**. Include notes to help us understand the solution you've designed and implemented.
- **Focus on key goals**, try to complete at least 1 stretch goal.
- **Do not hesitate to reach out**. If you are encountering an issue or have any question. `mcargnelutti@law.harvard.edu`, `gleppert@law.harvard.edu`

[👆 Back to summary](#summary)

---

## Exercise 

### Scenario
We would like to evaluate how "good" the summaries generated as part of the [`ccdv/govreport-summarization`](https://huggingface.co/datasets/ccdv/govreport-summarization/) dataset are.

We have very limited compute, and would like to get a sense of how semantically "close" these AI-generated summaries are from their original reports. We are going to distill and use a [static embedding model](https://huggingface.co/blog/static-embeddings) for that purpose, with the help of [Model2Vec](https://github.com/MinishLab/model2vec). 

In order to run repeatable experiments that will help us solve that problem, we will create a Python pipeline invokable via CLI. Please make sure to leave necessary instructions to install and run the pipeline.

### Key goals
- [ ] **The pipeline must have a `distill` command to create distilled models with [Model2Vec](https://github.com/MinishLab/model2vec).**
  - By default, the pipeline distills [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), which is the model we are going to focus on for this experiment. 
  - The distilled models are stored locally.
- [ ] **The pipeline must have an `analyze` command which:**
  - Pulls the [`ccdv/govreport-summarization`](https://huggingface.co/datasets/ccdv/govreport-summarization/) dataset
  - Encodes `report` and `summary` for each row with one of the models we created with the `distill` command, as a way to assess their semantic similarity. 
  - Measures and stores the distance (cosine similarity) of each resulting vector pair. This data can be stored in a local CSV file, or any other suitable format. Said file should keep trace of when that experiment was run and of which model was used. 
- [ ] **The pipeline must have a `report` command which:**
  - Generates a report (which could be one or multiple CSV files) containing statistics about a given `analyze` run. 
  - That report should feature:
    - Key statistics about the run itself: how many rows were analyzed, how many could not be analyzed, which model was used ... 
    - Information about the "quality" of the summaries such as: 
      - Average distance
      - Min and max distance
      - Distance distribution (in tranches)
- [ ] **Reflect on findings:** 
  - In your pull request, briefly reflect on the results the pipeline yielded and what we've learned in the process about: 
    - The qualities of the dataset we've analyzed
    - The potential suitability of the tools and techniques we used to analyze it

### Stretch goals
> Pick and choose!
- [ ] Optimize `analyze` so it makes **controlled** use of multiprocessing without loading the whole dataset in memory. Add `--max-workers` parameters to `analyze` to control the pipeline's use of resources. 
- [ ] Optimize `analyze` so it can work on any sub-sample of [`ccdv/govreport-summarization`](https://huggingface.co/datasets/ccdv/govreport-summarization/)
- [ ] Optimize `report` so it also generates a plot for the distance distribution by tranches.

### Hints and notes
- ⚠️ Model2Vec's `model.encode()` default arguments may cause unexpected behaviors. 
- HuggingFace datasets can often be streamed. 
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) generates vectors in 1024 dimensions. Model2Vec can reduce that dimensionality. 
