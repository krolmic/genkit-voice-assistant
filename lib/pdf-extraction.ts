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
    const pdf = await getDocumentProxy(new Uint8Array(buffer));
    const { text } = await extractText(pdf, { mergePages: true });
    return text;
}

export async function getDocumentsFromPdf(
    ai: GenkitBeta,
    url: string,
): Promise<Document[]> {
    const pdfTxt = await ai.run('extract-text', () => extractTextFromPdf(url));
    const chunks = await ai.run('chunk-it', async () => chunk(pdfTxt, chunkingConfig));
    const documents = chunks.map((text) => {
        return Document.fromText(text, { url });
    });
    return documents;
}