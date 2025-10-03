import { openAI } from '@genkit-ai/compat-oai/openai';
import { unlink } from "fs/promises";
import { type SessionData, type SessionStore } from 'genkit';
import { type GenkitBeta } from 'genkit/beta';
import { Document } from 'genkit/retriever';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

export const defaultSystemInstructions = "You are friendly and helpful.";

const chatModel = openAI.model('gpt-4o-mini');

class PersistentSessionStore<S = any> implements SessionStore<S> {
    private baseDir: string;

    constructor() {
        this.baseDir = join(process.cwd(), 'sessions');
    }

    private async maybeCreateSessionsDirectory(): Promise<void> {
        await mkdir(this.baseDir, { recursive: true });
    }

    private getSessionFilePath(sessionId: string): string {
        return join(this.baseDir, `${sessionId}.json`);
    }

    async get(sessionId: string): Promise<SessionData<S> | undefined> {
        try {
            const s = await readFile(this.getSessionFilePath(sessionId), { encoding: 'utf8' });
            const data = JSON.parse(s);
            return data;
        } catch {
            return undefined;
        }
    }

    async save(sessionId: string, sessionData: SessionData<S>): Promise<void> {
        await this.maybeCreateSessionsDirectory();
        const sessionDataJsonString = JSON.stringify(sessionData);
        await writeFile(this.getSessionFilePath(sessionId), sessionDataJsonString, { encoding: 'utf8' });
    }

    async delete(sessionId: string): Promise<void> {
        await unlink(this.getSessionFilePath(sessionId));
    }
}

export async function createChatSession(
    ai: GenkitBeta,
    systemInstructions: string,
    maxTokens?: number,
    temperature?: number,
    stopSequences?: string[],
): Promise<string> {
    const store = new PersistentSessionStore();
    const session = ai.createSession({ store });

    session.chat({
        model: chatModel,
        system: systemInstructions,
        config: {
            maxOutputTokens: maxTokens ?? 2048,
            temperature: temperature ?? 0.7,
            stopSequences: stopSequences ?? [],
        },
    });

    return session.id;
}

export async function sendMessagesToSession(
    ai: GenkitBeta,
    sessionId: string,
    messages: string[],
    systemInstructions: string,
    maxTokens?: number,
    temperature?: number,
    stopSequences?: string[],
    documents?: Document[],
): Promise<string> {
    const store = new PersistentSessionStore();
    const session = await ai.loadSession(sessionId, { store });
    const chatInstance = session.chat({
        model: chatModel,
        system: systemInstructions,
        config: {
            maxOutputTokens: maxTokens ?? 2048,
            temperature: temperature ?? 0.7,
            stopSequences: stopSequences ?? [],
        },
        docs: documents ?? [],
    });

    let responseText = "";
    for (const msg of messages) {
        const response = await chatInstance.send(msg);
        const text = response.text;
        if (text) {
            responseText += text + "\n";
        }
    }

    return responseText.trim();
}

export async function deleteSession(sessionId: string): Promise<void> {
    const store = new PersistentSessionStore();
    await store.delete(sessionId);
}