/* mbed */
#include "mbed.h"
/* Lab9 RPC */
#include "mbed_rpc.h"
/* HW2 uLCD */
#include "uLCD_4DGL.h"
/* Lab8 ML */
#include "config.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "accelerometer_handler.h"


////////////////////////////////////////////////////////////
/* Global Thread */
Thread gesture_thread;
Thread detection_thread;

////////////////////////////////////////////////////////////
/* Global EventQueue */
EventQueue gesture_queue;
EventQueue detection_queue;

////////////////////////////////////////////////////////////
/* RPC variable */
RpcDigitalOut myled2(LED1,"gestureMode");
RpcDigitalOut myled3(LED2,"detectionMode");
BufferedSerial pc(USBTX, USBRX);

////////////////////////////////////////////////////////////
/* enable variable */
DigitalOut if_gesture_mode(LED1);
DigitalOut if_detection_mode(LED2);
DigitalOut led3(LED3);

////////////////////////////////////////////////////////////
/* uLCD variable */
uLCD_4DGL uLCD(D1, D0, D2);

////////////////////////////////////////////////////////////
/* ML variable */
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

////////////////////////////////////////////////////////////
/* angel table */
int thres_angel_mode = 0;
int thres_angel_table[3] = {30, 45, 60};

void readRPCCommand();
void gestureMode();
void gestureMode_gestureVerify();
int  PredictGesture(float* output);
void detectionMode();
void led1_blink(uint32_t period);

int main() {
   gesture_queue.call(&gestureMode);
   detection_queue.call(&detectionMode);
   gesture_thread.start(callback(&gesture_queue, &EventQueue::dispatch_forever));
   detection_thread.start(callback(&detection_queue, &EventQueue::dispatch_forever));
   readRPCCommand();
}

void readRPCCommand() {
   char buf[256], outbuf[256];
   FILE *devin = fdopen(&pc, "r");
   FILE *devout = fdopen(&pc, "w");
   

   while(1) {
      /* clear buffer */
      memset(buf, 0, 256);
      /* read command */      
      for(int i=0; ; i++) {
            char recv = fgetc(devin);
            if (recv == '\n') {
               printf("\r\n");
               break;
            }
            buf[i] = fputc(recv, devout);
      }
      /* Call the static call method on the RPC class */
      RPC::call(buf, outbuf);
      printf("%s\r\n", outbuf);
   }
}
void gestureMode() {
   while (1) {
      if (if_gesture_mode) {
         if_detection_mode = 0;
         gestureMode_gestureVerify();
      }
   }
}
void detectionMode() {
   while (1) {
      if (if_detection_mode) {
         if_gesture_mode = 0;
         led1_blink(100);
      }
   }
}
void led1_blink(uint32_t peroid) {
   for (int i = 1; i <= 5; i++) {
      ThisThread::sleep_for(peroid);
      led3 = 1;
      ThisThread::sleep_for(peroid);
      led3 = 0;
   }
}
void gestureMode_gestureVerify() {
   // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction
  int gesture_index;

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(), 1);

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return;
  }

  error_reporter->Report("Set up successful...\n");

  while (if_gesture_mode) {

    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;

    // Produce an output
    if (gesture_index < label_num) {
      error_reporter->Report(config.output_message[gesture_index]);
    }
  }
}
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}
void uLCDDisplay() {
   if (!if_gesture_mode && )
   uLCD.background_color(BLACK);
   uLCD.cls();
   uLCD.locate(0, 15);
   uLCD.printf(" Down "); 
   uLCD.locate(7, 15);
   uLCD.printf("Select");
   uLCD.locate(14, 15);
   uLCD.printf(" Up "); 
   uLCD.text_width(3);
   uLCD.text_height(3);
   uLCD.color(WHITE);
   uLCD.textbackground_color(BLACK);
   uLCD.locate(0, 2);
   uLCD.printf("%4dHz", freqTable[freqMode]);
}