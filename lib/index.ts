import { openAI } from '@genkit-ai/compat-oai/openai';
import { genkit, z } from 'genkit';

const ai = genkit({
    plugins: [openAI()],
});

const whisper = openAI.model('whisper-1');

export async function transcribeAudio(audioBase64: string, contentType: string = 'audio/mp3'): Promise<string> {
    const result = await ai.generate({
        model: whisper,
        prompt: [
            {
                media: {
                    contentType,
                    url: `data:${contentType};base64,${audioBase64}`,
                },
            },
        ],
    });
    return result.text;
}

export const transcribeAudioFlow = ai.defineFlow(
    {
        name: 'transcribeAudio',
        inputSchema: z.object({
            audioBase64: z.string(),
            contentType: z.string().optional(),
        }),
        outputSchema: z.string(),
    },
    async ({ audioBase64, contentType }) => {
        return await transcribeAudio(audioBase64, contentType ?? 'audio/mp3');
    }
);