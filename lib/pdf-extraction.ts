import type { GenkitBeta } from "genkit/beta";
import { Document } from 'genkit/retriever';
import { chunk } from "llm-chunk";
import fetch from 'node-fetch';
import { extractText, getDocumentProxy } from "unpdf";

const chunkingConfig = {
    minLength: 1000,
    maxLength: 2000,
    splitter: 'sentence',
    overlap: 100,
    delimiters: '',
} as any;

export async function extractTextFromPdf(url: string) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    return await extractTextFromPdfBuffer(new Uint8Array(buffer));
}

export async function getDocumentsFromPdf(
    ai: GenkitBeta,
    url: string,
    metadata?: Record<string, any>,
): Promise<Document[]> {
    const pdfTxt = await ai.run('extract-text', () => extractTextFromPdf(url));
    const chunks = await ai.run('chunk-it', async () => chunk(pdfTxt, chunkingConfig));
    const documents = chunks.map((text) => {
        return Document.fromText(text, { url, ...(metadata ?? {}) });
    });
    return documents;
}

export async function extractTextFromPdfBuffer(buffer: Uint8Array) {
    const pdf = await getDocumentProxy(buffer);
    const { text } = await extractText(pdf, { mergePages: true });
    return text;
}

export async function getDocumentsFromPdfBuffer(
    ai: GenkitBeta,
    buffer: Uint8Array,
    metadata?: Record<string, any>,
): Promise<Document[]> {
    const pdfTxt = await ai.run('extract-text', () => extractTextFromPdfBuffer(buffer));
    const chunks = await ai.run('chunk-it', async () => chunk(pdfTxt, chunkingConfig));
    const documents = chunks.map((text) => {
        return Document.fromText(text, metadata ?? {});
    });
    return documents;
}