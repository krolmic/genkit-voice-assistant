import { ElevenLabsClient } from '@elevenlabs/elevenlabs-js';

const defaultVoiceId = 'cgSgspJ2msm6clMCkdW9';
const defaultModelId = 'eleven_turbo_v2';

export async function getSpeechFromText(
    client: ElevenLabsClient,
    text: string,
    voiceId: string = defaultVoiceId,
    modelId: string = defaultModelId,
): Promise<{ base64Audio: string; contentType: string }> {
    const response = await client.textToSpeech.convert(voiceId, {
        modelId: modelId,
        text,
        outputFormat: "mp3_44100_128",
    });

    const base64Audio = await getBase64AudioFromStream(response);

    return {
        base64Audio,
        contentType: 'audio/mpeg',
    };
}

async function getBase64AudioFromStream(stream: AsyncIterable<Uint8Array>): Promise<string> {
    const chunks: Uint8Array[] = [];
    for await (const chunk of stream) {
        chunks.push(chunk);
    }

    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const audioBuffer = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
        audioBuffer.set(chunk, offset);
        offset += chunk.length;
    }

    return Buffer.from(audioBuffer).toString('base64');
}
