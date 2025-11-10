# Genkit Voice Assistant

## Setup

1. Get OpenAI API Key from the [OpenAI Plaform](https://platform.openai.com/api-keys).

2. Get Google AI API Key from the [Google AI Studio](https://aistudio.google.com/api-keys).

3. Get ElevenLabs API Key from [ElevenLabs Developers page](https://elevenlabs.io/app/developers/api-keys).

4. Run Chroma
    
    4.1. Either locally by using Docker
    ```
    docker run -d --name chromadb -p 8000:8000 -v ./chroma:/chroma/chroma -e IS_PERSISTENT=TRUE chromadb/chroma:0.4.24
    ```
    or in [Chroma Cloud](https://www.trychroma.com/).

5. Copy .env.example 
    ```
    cp .env.example .env
    ```
    and set the variables in there.

## Commands

```bash
# Start Genkit Developer UI
npm run genkit:ui
```
