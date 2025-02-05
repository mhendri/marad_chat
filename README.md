# Chatbot Implementations with Langchain + Streamlit

## Running locally

Copy files from [MARAD AI Project Files/literature](https://drive.google.com/drive/folders/1gfWqp2dlKiT4xWRGJk-6LPkZ0hPHDLMD?usp=share_link) into a local directory called 'corpus'

```shell
# create environment
conda create -n marad_chat python=3.9 -y
conda activate marad_chat
pip install -r requirements.txt
```

```shell
# setup local docs and API Key
mkdir corpus
mkdir .streamlit
echo 'OPENAI_API_KEY = "your-api-key-here"' > secrets.toml

# Run main streamlit app
$ python -m streamlit run chat_with_documents.py
```


## Running with Docker - not tested yet
```shell
# To generate image
$ docker build -t langchain-chatbot .

# To run the docker container
$ docker run -p 8501:8501 langchain-chatbot
```
