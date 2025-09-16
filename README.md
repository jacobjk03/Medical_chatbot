# Medical-Chatbot-using-Llama2

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/jacobjk03/Medical_chatbot.git
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medicalchatbot python=3.10.14 -y
```

```bash
conda activate medicalchatbot OR source activate medicalchatbot
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


### Install and run Llama 3 with Ollama

```ini
## Install Ollama from here
https://ollama.com/download

## Pull the Llama 3 model:
ollama pull llama3

## Start the Ollama server:
ollama serve

## (Optional) Test the model:
ollama run llama3

```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama3 (Via Ollama)
- Pinecone
- Agentic AI Pipeline (classification, reranking, safety nodes)


