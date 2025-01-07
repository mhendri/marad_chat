# Chatbot Implementations with Langchain + Streamlit

## Running locally
```shell
# Run main streamlit app
$ python -m streamlit run chat_with_documents.py
```

## Running with Docker
```shell
# To generate image
$ docker build -t langchain-chatbot .

# To run the docker container
$ docker run -p 8501:8501 langchain-chatbot
```
