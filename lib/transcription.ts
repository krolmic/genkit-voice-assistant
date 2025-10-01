import openAI from "@genkit-ai/compat-oai/openai";
import { type GenkitBeta } from "genkit/beta";

const transcriptionModel = openAI.model('whisper-1');

export async function transcribeVoiceMessage(
    ai: GenkitBeta,
    base64Audio: string,
    contentType: string = 'audio/mp3',
): Promise<string> {
    const result = await ai.generate({
        model: transcriptionModel,
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