import openAI from "@genkit-ai/compat-oai/openai";
import { type GenkitBeta } from "genkit/beta";

const speechToTextModel = openAI.model('whisper-1');

export async function getTextFromSpeech(
    ai: GenkitBeta,
    base64Audio: string,
    contentType: string = 'audio/mp3',
): Promise<string> {
    const result = await ai.generate({
        model: speechToTextModel,
        prompt: [
            {
                media: {
                    contentType,
                    url: `data:${contentType};base64,${base64Audio}`,
                },
            },
        ],
    });
    return result.text;
}