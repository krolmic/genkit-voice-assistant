import { ElevenLabsClient } from '@elevenlabs/elevenlabs-js';
import { openAI } from '@genkit-ai/compat-oai/openai';
import { googleAI } from '@genkit-ai/google-genai';
import { config } from 'dotenv';
import { z } from 'genkit';
import { genkit } from 'genkit/beta';
import { chroma, chromaIndexerRef, chromaRetrieverRef } from 'genkitx-chromadb';
import { createChatSession, defaultSystemInstructions, deleteSession, sendMessagesToSession } from './chat.js';
import { extractTextFromPdf, extractTextFromPdfBuffer, getDocumentsFromPdf, getDocumentsFromPdfBuffer } from './pdf-extraction.js';
import { getTextFromSpeech } from './speech-to-text.js';
import { getSpeechFromText } from './text-to-speech.js';

config();

if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not set');
}
if (!process.env.ELEVENLABS_API_KEY) {
    throw new Error('ELEVENLABS_API_KEY is not set');
}
if (!process.env.GOOGLE_GENAI_API_KEY) {
    throw new Error('GOOGLE_GENAI_API_KEY is not set');
}

const assistantCollectionName = 'assistant-collection';

const ai = genkit({
    plugins: [
        openAI({ apiKey: process.env.OPENAI_API_KEY }),
        googleAI(),
        chroma([
            {
                collectionName: assistantCollectionName,
                embedder: googleAI.embedder('gemini-embedding-001'),
                clientParams: {
                    path: "http://localhost:8000",
                },
                createCollectionIfMissing: true,
            },
        ]),
    ],
});

const assistantIndexer = chromaIndexerRef({
    collectionName: assistantCollectionName,
});
const assistantRetriever = chromaRetrieverRef({
    collectionName: assistantCollectionName,
});

const elevenLabsClient = new ElevenLabsClient({ apiKey: process.env.ELEVENLABS_API_KEY! });

export const indexPdfFromUrlFlow = ai.defineFlow(
    {
        name: 'indexPdfFromUrl',
        inputSchema: z.object({
            url: z.string().describe('PDF file URL'),
            metadata: z.record(z.string(), z.any()).optional().describe('Metadata to attach to indexed documents'),
        }),
        outputSchema: z.object({
            documentsIndexed: z.number(),
            error: z.string().optional(),
        }),
    },
    async ({ url, metadata }) => {
        try {
            const documents = await getDocumentsFromPdf(ai, url, metadata);

            await ai.index({
                indexer: assistantIndexer,
                documents,
            });

            return {
                documentsIndexed: documents.length,
            };
        } catch (err) {
            return {
                documentsIndexed: 0,
                error: err instanceof Error ? err.message : String(err),
            };
        }
    },
);

export const indexPdfFromBase64Flow = ai.defineFlow(
    {
        name: 'indexPdfFromBase64',
        inputSchema: z.object({
            base64Pdf: z.string().describe('Base64 encoded PDF file'),
            metadata: z.record(z.string(), z.any()).optional().describe('Metadata to attach to indexed documents'),
        }),
        outputSchema: z.object({
            documentsIndexed: z.number(),
            error: z.string().optional(),
        }),
    },
    async ({ base64Pdf, metadata }) => {
        try {
            const buffer = Uint8Array.from(Buffer.from(base64Pdf, 'base64'));
            const documents = await getDocumentsFromPdfBuffer(ai, buffer, metadata);

            await ai.index({
                indexer: assistantIndexer,
                documents,
            });

            return {
                documentsIndexed: documents.length,
            };
        } catch (err) {
            return {
                documentsIndexed: 0,
                error: err instanceof Error ? err.message : String(err),
            };
        }
    },
);

export const createChatFlow = ai.defineFlow(
    {
        name: "createChat",
        inputSchema: z.object({
            systemInstructions: z.string().default("You are friendly and helpful."),
            maxTokens: z.number().optional(),
            temperature: z.number().optional(),
            stopSequences: z.array(z.string()).optional(),
        }),
        outputSchema: z.object({
            sessionId: z.string(),
        }),
    },
    async ({ systemInstructions, maxTokens, temperature, stopSequences }) => {
        const sessionId = await createChatSession(ai, systemInstructions, maxTokens, temperature, stopSequences);
        return { sessionId };
    }
);

export const sendSpeechMessageToChatFlow = ai.defineFlow(
    {
        name: 'sendSpeechMessageToChat',
        inputSchema: z.object({
            sessionId: z.string(),
            base64Audio: z.string(),
            contentType: z.string().optional(),
            systemInstructions: z.string().default(defaultSystemInstructions),
            maxTokens: z.number().optional(),
            temperature: z.number().optional(),
            stopSequences: z.array(z.string()).optional(),
            generateAudio: z.boolean().optional(),
            voiceId: z.string().optional(),
            modelId: z.string().optional(),
        }),
        outputSchema: z.object({
            response: z.string(),
            audioResponse: z.string().optional(),
            audioResponseContentType: z.string().optional(),
        }),
    },
    async ({
        sessionId,
        base64Audio,
        contentType,
        systemInstructions,
        maxTokens,
        temperature,
        stopSequences,
        generateAudio,
        voiceId,
        modelId,
    }) => {
        const messageText = await getTextFromSpeech(ai, base64Audio, contentType ?? 'audio/mp3');
        const docs = await ai.retrieve({ retriever: assistantRetriever, query: messageText });
        const chatResponse = await sendMessagesToSession(
            ai,
            sessionId,
            [messageText],
            systemInstructions,
            maxTokens,
            temperature,
            stopSequences,
            docs,
        );

        let audioResponse: string | undefined;
        let audioResponseContentType: string | undefined;

        if (generateAudio) {
            const audioResult = await getSpeechFromText(elevenLabsClient, chatResponse, voiceId, modelId);
            audioResponse = audioResult.base64Audio;
            audioResponseContentType = audioResult.contentType;
        }

        return {
            response: chatResponse ?? '',
            audioResponse,
            audioResponseContentType,
        };
    }
);

export const sendTextMessageToChatFlow = ai.defineFlow(
    {
        name: 'sendTextMessageToChat',
        inputSchema: z.object({
            sessionId: z.string(),
            messageText: z.string(),
            systemInstructions: z.string().default(defaultSystemInstructions),
            maxTokens: z.number().optional(),
            temperature: z.number().optional(),
            stopSequences: z.array(z.string()).optional(),
            generateAudio: z.boolean().optional(),
            voiceId: z.string().optional(),
            modelId: z.string().optional(),
        }),
        outputSchema: z.object({
            textResponse: z.string(),
            audioResponse: z.string().optional(),
            audioResponseContentType: z.string().optional(),
        }),
    },
    async ({
        sessionId,
        messageText,
        systemInstructions,
        maxTokens,
        temperature,
        stopSequences,
        generateAudio,
        voiceId,
        modelId,
    }) => {
        const docs = await ai.retrieve({ retriever: assistantRetriever, query: messageText });
        const textResponse = await sendMessagesToSession(
            ai,
            sessionId,
            [messageText],
            systemInstructions,
            maxTokens,
            temperature,
            stopSequences,
            docs,
        );

        let audioResponse: string | undefined;
        let audioResponseContentType: string | undefined;

        if (generateAudio) {
            const audioResult = await getSpeechFromText(elevenLabsClient, textResponse, voiceId, modelId);
            audioResponse = audioResult.base64Audio;
            audioResponseContentType = audioResult.contentType;
        }

        return {
            textResponse,
            audioResponse,
            audioResponseContentType,
        };
    }
);

export const deleteChatFlow = ai.defineFlow(
    {
        name: "deleteChat",
        inputSchema: z.object({
            sessionId: z.string(),
        }),
        outputSchema: z.void(),
    },
    async ({ sessionId }) => {
        await deleteSession(sessionId);
        return;
    }
);

export const getTextFromSpeechFlow = ai.defineFlow(
    {
        name: 'getTextFromSpeech',
        inputSchema: z.object({
            base64Audio: z.string(),
            contentType: z.string().optional(),
        }),
        outputSchema: z.string(),
    },
    async ({ base64Audio, contentType }) => {
        return await getTextFromSpeech(ai, base64Audio, contentType ?? 'audio/mp3');
    }
);

export const getSpeechFromTextFlow = ai.defineFlow(
    {
        name: 'getSpeechFromText',
        inputSchema: z.object({
            text: z.string(),
            voiceId: z.string().optional(),
            modelId: z.string().optional(),
        }),
        outputSchema: z.object({
            base64Audio: z.string(),
            contentType: z.string(),
        }),
    },
    async ({ text, voiceId, modelId }) => {
        return await getSpeechFromText(elevenLabsClient, text, voiceId, modelId);
    }
);

export const getTextFromPdfUrlFlow = ai.defineFlow(
    {
        name: 'getTextFromPdfUrl',
        inputSchema: z.object({ url: z.string().describe('PDF file URL') }),
        outputSchema: z.string(),
    },
    async ({ url }) => {
        return await extractTextFromPdf(url);
    }
);

export const getTextFromBase64PdfFlow = ai.defineFlow(
    {
        name: 'getTextFromBase64Pdf',
        inputSchema: z.object({
            base64Pdf: z.string().describe('Base64 encoded PDF file'),
        }),
        outputSchema: z.string(),
    },
    async ({ base64Pdf }) => {
        const buffer = Uint8Array.from(Buffer.from(base64Pdf, 'base64'));
        return await extractTextFromPdfBuffer(buffer);
    }
);
