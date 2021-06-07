#include "core_imports.h"

bool is_status_ok(const TfLiteStatus& status, const char* error_message=nullptr);

namespace {
tflite::ErrorReporter* error_reporter = nullptr;

bool is_status_ok(const TfLiteStatus& status, const char* error_message=nullptr) {
    if (status != kTfLiteOk) {
        error_reporter->Report(error_message);
        return false;
    }
    return true;
}
}

namespace Audio {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* model_input = nullptr;

FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

void setup() {
    model = tflite::GetModel(model);

    static tflite::MicroMutableOpResolver<5> micro_op_resolver(error_reporter);
    if (!is_status_ok(micro_op_resolver.AddDepthwiseConv2D())) return;
    if (!is_status_ok(micro_op_resolver.AddFullyConnected()) return;
    if (!is_status_ok(micro_op_resolver.AddSoftmax())) return;
    if (!is_status_ok(micro_op_resolver.AddReshape())) return;

    static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();

    if (!is_status_ok(allocate_status,"AllocateTensors() failed")) return;
    
    static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                   model_input->data.uint8);
    feature_provider = &static_feature_provider;
}
}

namespace Image {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;

constexpr int kTensorArenaSize = 136 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];

void setup() {
    model = tflite::GetModel(g_mask_detect_model_data);

    static tflite::MicroMutableOpResolver<5> micro_op_resolver;
    if (!is_status_ok(micro_op_resolver.AddAveragePool2D())) return;
    if (!is_status_ok(micro_op_resolver.AddConv2D())) return;
    if (!is_status_ok(micro_op_resolver.AddDepthwiseConv2D())) return;
    if (!is_status_ok(micro_op_resolver.AddReshape())) return;
    if (!is_status_ok(micro_op_resolver.AddSoftmax())) return;

    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (!is_status_ok(allocate_status, "AllocateTensors() failed")) return;
    
    input = interpreter->input(0);
}
}

void setup() {
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    Audio::setup();
    Image::setup();
}

void loop() {
    const int32_t current_time = LatestAudioTimestamp();
    int slice_count = 0;

    TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
        error_reporter, previous_time, current_time, &slice_count);
    if (!is_status_ok(feature_status, "Feature generation failed")) return;

    previous_time = current_time;
    if (slice_count == 0) return;

    /// [Interpreter]
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (!is_status_ok(invoke_status, "Invocation failed")) return;

    /// [Output]
    TfLiteTensor* output = interpreter->output(0);
    const char* found_command = nullptr;
    uint8_t score = 0;

    bool is_command_new = false;
    TfLiteStatus process_status = recognizer->ProcessLatestResults(
        output, current_time, &found_command, &score, &is_command_new);
    
    if (!is_status_ok(process_status, "RecognizeCommands::ProcessLatestResults() failed")) return;
    RespondToCommand(error_reporter, current_time, found_command, score, is_command_new);

    /// [Image] 
    KTfLite image_status = GetImage(error_reporter, kNumCols, kNumRows, kNumChannels, input->data.int8);
    if (!is_status_ok(image_status, "Image capture failed.")) return;

    KTfLite invoke_status = interpreter->Invoke();
    if (!is_status_ok(image_status, "Invocation failed.")) return;

    TfLiteTensor* mask_output = mask_interpreter->output(0);

    int8_t mask_score = output->data.uint8[kMaskIndex];
    int8_t no_mask_score = output->data.uint8[kNotAMaskIndex];
    RespondToDetection(error_reporter, mask_score, no_mask_score);
}
