class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
    }
  
    process(inputs, outputs, parameters) {
      const input = inputs[0];
      if (!input || input.length === 0 || input[0].length === 0) {
        return true; 
      }
  
      const monoInput = input[0]; 
      const pcmData = new Float32Array(monoInput); 
  
      this.port.postMessage(pcmData);
  
      return true;
    }
  }
  
  registerProcessor('audio-processor', AudioProcessor);
  