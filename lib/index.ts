import { ElevenLabsClient } from '@elevenlabs/elevenlabs-js';
import { openAI } from '@genkit-ai/compat-oai/openai';
import { config } from 'dotenv';
import { z } from 'genkit';
import { genkit } from 'genkit/beta';
import { createChatSession, defaultSystemInstructions, deleteSession, sendMessagesToSession } from './chat.js';
import { getTextFromSpeech } from './speech-to-text.js';
import { getSpeechFromText } from './text-to-speech.js';

config();

if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not set');
}
if (!process.env.ELEVENLABS_API_KEY) {
    throw new Error('ELEVENLABS_API_KEY is not set');
}

const ai = genkit({
    plugins: [
        openAI({ apiKey: process.env.OPENAI_API_KEY }),
    ],
});

const elevenLabsClient = new ElevenLabsClient({ apiKey: process.env.ELEVENLABS_API_KEY! });

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
        const chatResponse = await sendMessagesToSession(ai, sessionId, [messageText], systemInstructions, maxTokens, temperature, stopSequences);

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
        const textResponse = await sendMessagesToSession(ai, sessionId, [messageText], systemInstructions, maxTokens, temperature, stopSequences);

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