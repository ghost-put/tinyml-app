#include <PDM.h>
#include "audio_provider.h"
#include "micro_features_micro_model_settings.h"
#include <cstdint>

#define DEFAULT_BUFFER_SIZE DEFAULT_PDM_BUFFER_SIZE
extern PDMClass PDM;
static volatile int samplesRead;

void CaptureSamples();

namespace Audio {
bool is_initialized = false;

constexpr int kCaptureBufferSize = DEFAULT_BUFFER_SIZE * 16;
int16_t capture_buffer[kAudioCaptureBufferSize];
int16_t output_buffer[kMaxAudioSampleSize];

volatile int32_t latest_timestamp = 0;
volatile int16_t recording_buffer[DEFAULT_BUFFER_SIZE];

volatile int max_audio = -32768, min_audio = 32768;

// IRQ handler
static void onPDMdata() {
    int bytesAvailable = PDM.available();
    PDM.read((int16_t *) recording_buffer, bytesAvailable);
    samplesRead = bytesAvailable / 2;
}

void TIMER_CALLBACK() {
  static bool ledtoggle = false;
  static uint32_t audio_idx = 0;
  int32_t sample = 0;

  PDM.IrqHandler();    // wait for samples to be read
  if (samplesRead) {
    max_audio = -32768;
    min_audio = 32768;

    for (int i=0; i<samplesRead; ++i) {
      min_audio = min(recording_buffer[i], min_audio);
      max_audio = max(recording_buffer[i], max_audio);
    }
    CaptureSamples();
    samplesRead = 0;
  }
  // we did a whole buffer at once, so we're done
  return;
  
  if (audio_idx >= DEFAULT_BUFFER_SIZE) {
    CaptureSamples();
    max_audio = -32768, min_audio = 32768;
    audio_idx = 0;
  }

#if defined(USE_EXTERNAL_MIC)
    sample = analogRead(USE_EXTERNAL_MIC);
    sample -= 2047;
#endif
#if defined(AUDIO_OUT)
    analogWrite(AUDIO_OUT, sample+2048); 
#endif

  recording_buffer[audio_idx++] = sample;
  min_audio = min(min_audio, sample);
  max_audio = max(max_audio, sample);
}

TfLiteStatus InitRecording(tflite::ErrorReporter* error_reporter) {
    Serial.begin(115200);
    Serial.println("Initializing Audio"); delay(10);

    // Hook up the callback that will be called with each sample
    PDM.onReceive(onPDMdata);
    if (!PDM.begin(1, 16000)) {
        Serial.println("Failed to start PDM!");
        while (true) yield();
    }
    // Block until we have our first audio sample
    while (!latest_timestamp) delay(1);
    return kTfLiteOk;
}

void CaptureSamples() {
    const int number_of_samples = DEFAULT_BUFFER_SIZE;
    // Calculate what timestamp the last audio sample represents
    const int32_t time_in_ms = latest_timestamp 
            + (number_of_samples / (kAudioSampleFrequency / 1000));
    // Determine the index, in the history of all samples, of the last sample
    const int32_t start_sample_offset = latest_timestamp * (kAudioSampleFrequency / 1000);
    // Determine the index of this sample in our ring buffer
    const int capture_index = start_sample_offset % kAudioCaptureBufferSize;
    // Read the data to the correct place in our buffer, note 2 bytes per buffer entry
    memcpy(audio_capture_buffer + capture_index, (void *) recording_buffer, DEFAULT_BUFFER_SIZE*2);
    // This is how we let the outside world know that new audio data has arrived.
    latest_timestamp = time_in_ms;
    int peak = (max_audio - min_audio);
    Serial.printf("pp %d\n", peak);
}

TfLiteStatus GetSamples(tflite::ErrorReporter* error_reporter,
                        int start_ms, int duration_ms,
                        int* audio_samples_size, int16_t** audio_samples) {
    if (!is_initialized) {
        TfLiteStatus init_status = InitializeRecording(error_reporter);
        if (!is_status_ok(init_status)) return init_status;
        is_initialized = true;
    }
    // This next part should only be called when the main thread notices that the
    // latest audio sample data timestamp has changed, so that there's new data
    // in the capture ring buffer. The ring buffer will eventually wrap around and
    // overwrite the data, but the assumption is that the main thread is checking
    // often enough and the buffer is large enough that this call will be made
    // before that happens.

    // Determine the index, in the history of all samples, of the first sample we want
    const int start_offset = start_ms * (kAudioSampleFrequency / 1000);
    // Determine how many samples we want in total
    const int duration_sample_count = duration_ms * (kAudioSampleFrequency / 1000);

    for (int i = 0; i < duration_sample_count; ++i) {
        // For each sample, transform its index in the history of all samples into
        // its index in capture_buffer
        const int capture_index = (start_offset + i) % kAudioCaptureBufferSize;
        // Write the sample to the output buffer
        output_buffer[i] = capture_buffer[capture_index];
    }

    *samples_size = kMaxAudioSampleSize;
    *samples = output_buffer;
    return kTfLiteOk;
}

int32_t LatestAudioTimestamp() { return latest_timestamp; }
}
