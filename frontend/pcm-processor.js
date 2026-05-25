/**
 * pcm-processor.js
 * AudioWorklet processor: accumulates float32 samples,
 * converts to Int16 PCM (pcm_s16le), emits fixed-size chunks.
 *
 * Why fixed chunks: Sarvam STT streaming expects consistently-sized
 * binary frames. Variable-size frames cause VAD misfires.
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    // Default chunk size: 4096 samples = 256ms @ 16kHz
    // Smaller = lower latency but more overhead; larger = more stable
    this._chunkSize = (options.processorOptions && options.processorOptions.chunkSize) || 4096;
    this._buffer = new Float32Array(this._chunkSize);
    this._bufferFill = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0]; // Float32Array, 128 samples per quantum

    let offset = 0;
    while (offset < samples.length) {
      const spaceInBuffer = this._chunkSize - this._bufferFill;
      const available = samples.length - offset;
      const toCopy = Math.min(spaceInBuffer, available);

      this._buffer.set(samples.subarray(offset, offset + toCopy), this._bufferFill);
      this._bufferFill += toCopy;
      offset += toCopy;

      if (this._bufferFill === this._chunkSize) {
        this._emit();
      }
    }

    return true; // keep processor alive
  }

  _emit() {
    // Convert Float32 [-1,1] → Int16 [-32768,32767]
    const int16 = new Int16Array(this._chunkSize);
    for (let i = 0; i < this._chunkSize; i++) {
      // Clamp, then scale
      const s = Math.max(-1, Math.min(1, this._buffer[i]));
      int16[i] = s < 0 ? s * 32768 : s * 32767;
    }

    // Transfer the underlying buffer (zero-copy)
    this.port.postMessage(int16, [int16.buffer]);

    // Reset fill
    this._bufferFill = 0;
  }
}

registerProcessor('pcm-processor', PCMProcessor);