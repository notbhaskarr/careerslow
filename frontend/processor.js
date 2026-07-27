class PCMProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super();
        this.targetSampleRate = 16000;
        this.vadThreshold = 0.01; // Minimum RMS to be considered "speech"
        
        // Accumulator for downsampling
        this.bufferSize = 4096;
        this.outBuffer = new Float32Array(this.bufferSize);
        this.outBufferIndex = 0;
        
        // Native sample rate comes from options if provided, default 48k
        this.nativeSampleRate = (options.processorOptions && options.processorOptions.sampleRate) || 48000;
        this.downsampleRatio = this.nativeSampleRate / this.targetSampleRate;
        this.sampleCount = 0;
        this.nextSampleAt = 0;
    }

    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (!input || !input.length) return true;
        
        const channelData = input[0];
        
        // Calculate RMS for VAD (Voice Activity Detection)
        let sum = 0;
        for (let i = 0; i < channelData.length; i++) {
            sum += channelData[i] * channelData[i];
        }
        const rms = Math.sqrt(sum / channelData.length);
        
        // We must send all frames (even silent ones) to the server 
        // so that the cloud STT's Voice Activity Detection (VAD) 
        // can detect the end of speech.

        // Basic linear interpolation downsampling
        for (let i = 0; i < channelData.length; i++) {
            this.sampleCount++;
            if (this.sampleCount >= this.nextSampleAt) {
                this.outBuffer[this.outBufferIndex] = channelData[i];
                this.outBufferIndex++;
                this.nextSampleAt += this.downsampleRatio;
                
                // When we fill our target buffer size, process and flush
                if (this.outBufferIndex >= this.bufferSize) {
                    this.flushBuffer();
                }
            }
        }
        
        return true;
    }
    
    flushBuffer() {
        // Convert Float32 to Int16
        const int16Buffer = new Int16Array(this.bufferSize);
        for (let i = 0; i < this.bufferSize; i++) {
            const s = Math.max(-1, Math.min(1, this.outBuffer[i]));
            int16Buffer[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        
        // Send Int16 array buffer to the main thread
        this.port.postMessage(int16Buffer.buffer, [int16Buffer.buffer]);
        
        // Reset buffer
        this.outBufferIndex = 0;
    }
}

registerProcessor('pcm-processor', PCMProcessor);
