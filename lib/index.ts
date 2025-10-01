import { openAI } from '@genkit-ai/compat-oai/openai';
import { config } from 'dotenv';
import { z } from 'genkit';
import { genkit } from 'genkit/beta';
import { createChatSession, defaultSystemInstructions, deleteSession, sendMessagesToSession } from './chat.js';
import { transcribeVoiceMessage } from './transcription.js';

config();

if (!process.env.OPENAI_API_KEY) {
    throw new Error('OPENAI_API_KEY is not set');
}

const ai = genkit({
    plugins: [
        openAI({ apiKey: process.env.OPENAI_API_KEY }),
    ],
});

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

export const sendVoiceMessageToChatFlow = ai.defineFlow(
    {
        name: 'sendVoiceMessageToChat',
        inputSchema: z.object({
            sessionId: z.string(),
            base64Audio: z.string(),
            contentType: z.string().optional(),
            systemInstructions: z.string().default(defaultSystemInstructions),
            maxTokens: z.number().optional(),
            temperature: z.number().optional(),
            stopSequences: z.array(z.string()).optional(),
        }),
        outputSchema: z.object({
            response: z.string(),
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
    }) => {
        const messageText = await transcribeVoiceMessage(ai, base64Audio, contentType ?? 'audio/mp3');
        const chatResponse = await sendMessagesToSession(ai, sessionId, [messageText], systemInstructions, maxTokens, temperature, stopSequences);
        return { response: chatResponse ?? '' };
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
        }),
        outputSchema: z.object({
            response: z.string(),
        }),
    },
    async ({
        sessionId,
        messageText,
        systemInstructions,
        maxTokens,
        temperature,
        stopSequences,
    }) => {
        const chatResponse = await sendMessagesToSession(ai, sessionId, [messageText], systemInstructions, maxTokens, temperature, stopSequences);
        return { response: chatResponse ?? '' };
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

export const transcribeVoiceMessageFlow = ai.defineFlow(
    {
        name: 'transcribeVoiceMessage',
        inputSchema: z.object({
            base64Audio: z.string(),
            contentType: z.string().optional(),
        }),
        outputSchema: z.string(),
    },
    async ({ base64Audio, contentType }) => {
        return await transcribeVoiceMessage(ai, base64Audio, contentType ?? 'audio/mp3');
    }
);