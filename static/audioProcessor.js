class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        const output = outputs[0];

        const pcmData = new Float32Array(input[0]);
        this.port.postMessage(pcmData);

        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);
