/**
 * pcm-processor.js
 * AudioWorklet processor: accumulates float32 samples,
 * converts to Int16 PCM (pcm_s16le), emits fixed-size chunks.
 *
 * INTERRUPT ADDITION:
 * Detects speech energy locally and posts { type: 'speech_energy_detected' }
 * to the main thread BEFORE Sarvam VAD fires — saves 200-400ms interrupt latency.
 * Energy threshold tunable via processorOptions.energyThreshold (default 0.002).
 */
class PCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions || {};

    // PCM chunk config — unchanged
    this._chunkSize  = opts.chunkSize  || 4096;
    this._buffer     = new Float32Array(this._chunkSize);
    this._bufferFill = 0;

    // Energy-based speech detection
    // Tune energyThreshold between 0.001 (sensitive) and 0.005 (less sensitive)
    this._energyThreshold      = opts.energyThreshold      || 0.002;
    this._silenceThreshold     = opts.silenceThreshold      || 0.0005;
    this._speaking             = false;

    // Consecutive-frame gating: avoids single-frame noise spikes
    // speech fires after N consecutive loud frames; silence after M quiet frames
    this._speechFrameGate      = opts.speechFrameGate      || 3;   // ~6ms at 16kHz
    this._silenceFrameGate     = opts.silenceFrameGate     || 20;  // ~40ms
    this._loudFrameCount       = 0;
    this._quietFrameCount      = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const samples = input[0]; // Float32Array, 128 samples per quantum

    // ── Energy detection for interrupt signalling ──────────────────────────
    let energy = 0;
    for (let i = 0; i < samples.length; i++) energy += samples[i] * samples[i];
    energy /= samples.length;

    if (!this._speaking) {
      if (energy > this._energyThreshold) {
        this._loudFrameCount++;
        this._quietFrameCount = 0;
        if (this._loudFrameCount >= this._speechFrameGate) {
          this._speaking = true;
          this._loudFrameCount = 0;
          // Post immediately — main thread uses this to interrupt AI audio
          this.port.postMessage({ type: 'speech_energy_detected' });
        }
      } else {
        this._loudFrameCount = 0;
      }
    } else {
      // Already speaking — look for silence to reset
      if (energy < this._silenceThreshold) {
        this._quietFrameCount++;
        if (this._quietFrameCount >= this._silenceFrameGate) {
          this._speaking = false;
          this._quietFrameCount = 0;
          this._loudFrameCount  = 0;
        }
      } else {
        this._quietFrameCount = 0;
      }
    }
    // ──────────────────────────────────────────────────────────────────────

    // ── PCM buffering + emit (unchanged) ──────────────────────────────────
    let offset = 0;
    while (offset < samples.length) {
      const spaceInBuffer = this._chunkSize - this._bufferFill;
      const available     = samples.length - offset;
      const toCopy        = Math.min(spaceInBuffer, available);

      this._buffer.set(samples.subarray(offset, offset + toCopy), this._bufferFill);
      this._bufferFill += toCopy;
      offset           += toCopy;

      if (this._bufferFill === this._chunkSize) {
        this._emit();
      }
    }

    return true; // keep processor alive
  }

  _emit() {
    const int16 = new Int16Array(this._chunkSize);
    for (let i = 0; i < this._chunkSize; i++) {
      const s  = Math.max(-1, Math.min(1, this._buffer[i]));
      int16[i] = s < 0 ? s * 32768 : s * 32767;
    }
    // Transfer the underlying buffer (zero-copy)
    this.port.postMessage(int16, [int16.buffer]);
    this._bufferFill = 0;
  }
}

registerProcessor('pcm-processor', PCMProcessor);