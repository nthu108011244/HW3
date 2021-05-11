/* mbed */
#include "mbed.h"
#include <iostream>
#include <math.h>
#define PI 3.14159265
using namespace std;
/* Lab9 RPC */
#include "mbed_rpc.h"
/* HW2 uLCD */
#include "uLCD_4DGL.h"
/* Lab8 ML */
#include "tfconfig.h"
#include "magic_wand_model_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "accelerometer_handler.h"
/* Lab10 MQTT & Accelerometer */
#include "stm32l475e_iot01_accelero.h"
#include "MQTTNetwork.h"
#include "MQTTmbed.h"
#include "MQTTClient.h"


////////////////////////////////////////////////////////////
/* Global Thread */
Thread gesture_thread;
Thread detection_thread;
Thread mqtt_thread(osPriorityHigh);
Thread publish_thread;

////////////////////////////////////////////////////////////
/* Global EventQueue */
EventQueue gesture_queue;
EventQueue detection_queue;
EventQueue mqtt_queue;
EventQueue publish_queue;

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
/* accelerometer variable */
int16_t acc_data_XYZ[3] = {0};

////////////////////////////////////////////////////////////
/* MQTT variable */
WiFiInterface *wifi;
InterruptIn btn(USER_BUTTON);
volatile int message_num = 0;
volatile int arrivedcount = 0;
volatile bool closed = false;
const char* topic = "Mbed";

////////////////////////////////////////////////////////////
/* thres angel */
#define thres_angle_mode_max 3
int thres_angle_mode = 0;
int thres_angle_table[thres_angle_mode_max] = {30, 45, 60};
int thres_over_counter = 0;

void readRPCCommand();
void uLCDInit();
void uLCDDisplay(double inform);
void publish_MQTT();
void publish_message(MQTT::Client<MQTTNetwork, Countdown>* client);
void messageArrived(MQTT::MessageData& md);
void close_mqtt();
void gestureMode();
void gestureMode_gestureVerify();
int  PredictGesture(float* output);
void detectionMode();

int main() {
   gesture_queue.call(&gestureMode);
   detection_queue.call(&detectionMode);
   publish_queue.call(&publish_MQTT);
   gesture_thread.start(callback(&gesture_queue, &EventQueue::dispatch_forever));
   detection_thread.start(callback(&detection_queue, &EventQueue::dispatch_forever));
   publish_thread.start(callback(&publish_queue, &EventQueue::dispatch_forever));
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
void uLCDInit() {
   uLCD.background_color(BLACK);
   uLCD.cls();
   uLCD.text_width(3);
   uLCD.text_height(3);
}
void uLCDDisplay(double inform) {
   if (if_gesture_mode) {
      uLCD.color(GREEN);
      uLCD.locate(1, 1);
      uLCD.printf("%d", int(inform));
   }
   else if (if_detection_mode) {
      int count = thres_over_counter;
      uLCD.color(BLUE);
      uLCD.locate(1, 2);
      uLCD.printf("%2d", int(inform));
      uLCD.color(RED);
      uLCD.locate(1, 3);
      uLCD.printf("%d", count);
   }
   else {
      uLCDInit();
   }
}
void gestureMode() {
   while (1) {
      if (if_gesture_mode) {
         if_detection_mode = 0;
         uLCDInit();
         uLCDDisplay(thres_angle_table[thres_angle_mode]);
         gestureMode_gestureVerify();
      }
   }
}
void detectionMode() {
   bool acc_init = 0;
   double acc_stanZ = 0;
   double curr_angel;
   

   while (1) {
      if (if_detection_mode) {
         if_gesture_mode = 0;

         if (!acc_init) {
            acc_init = 1;
            uLCDInit();
            BSP_ACCELERO_Init();
            thres_over_counter = 0;
            for (int i = 1; i <= 10; i++) {
               BSP_ACCELERO_AccGetXYZ(acc_data_XYZ);
               acc_stanZ += acc_data_XYZ[2];
            }
            acc_stanZ /= 10;
         }

         BSP_ACCELERO_AccGetXYZ(acc_data_XYZ);
         curr_angel = acos(acc_data_XYZ[2] / acc_stanZ) * 180 / PI;
         if (curr_angel >= thres_angle_table[thres_angle_mode]) thres_over_counter++;
         
         cout << "[Tilt Angle Detection Mode]: " << curr_angel << " (over thres angel: " << thres_over_counter << " times)" << endl;
         uLCDDisplay(curr_angel);

         if (thres_over_counter >= 15) {
            thres_over_counter = 0;
            if_detection_mode = 0;
            acc_init = 0;
            acc_stanZ = 0;
            uLCDInit();
         }

         ThisThread::sleep_for(10ms);
      }
   }
}
void publish_MQTT()  {
    wifi = WiFiInterface::get_default_instance();
    if (!wifi) {
        ("ERROR: No WiFiInterface found.\r\n");
            return;
    }
    printf("\nConnecting to %s...\r\n", MBED_CONF_APP_WIFI_SSID);
    int ret = wifi->connect(MBED_CONF_APP_WIFI_SSID, MBED_CONF_APP_WIFI_PASSWORD, NSAPI_SECURITY_WPA_WPA2);
    if (ret != 0) {
            printf("\nConnection error: %d\r\n", ret);
            return;
    }
    NetworkInterface* net = wifi;
    MQTTNetwork mqttNetwork(net);
    MQTT::Client<MQTTNetwork, Countdown> client(mqttNetwork);

    //TODO: revise host to your IP
    const char* host = "192.168.31.205";
    printf("Connecting to TCP network...\r\n");

    SocketAddress sockAddr;
    sockAddr.set_ip_address(host);
    sockAddr.set_port(1883);
    printf("address is %s/%d\r\n", (sockAddr.get_ip_address() ? sockAddr.get_ip_address() : "None"),  (sockAddr.get_port() ? sockAddr.get_port() : 0) ); //check setting
    int rc = mqttNetwork.connect(sockAddr);//(host, 1883);
    if (rc != 0) {
            printf("Connection error.");
            return;
    }
    printf("Successfully connected!\r\n");
    MQTTPacket_connectData data = MQTTPacket_connectData_initializer;
    data.MQTTVersion = 3;
    data.clientID.cstring = "Mbed";
    if ((rc = client.connect(data)) != 0){
            printf("Fail to connect MQTT\r\n");
    }
    if (client.subscribe(topic, MQTT::QOS0, messageArrived) != 0){
            printf("Fail to subscribe\r\n");
    }

    mqtt_thread.start(callback(&mqtt_queue, &EventQueue::dispatch_forever));
    btn.rise(mqtt_queue.event(&publish_message, &client));

    int num = 0;
    while (num != 5) {
            client.yield(100);
            ++num;
    }

    while (1) {
            if (closed) break;
            client.yield(500);
            ThisThread::sleep_for(500ms);
    }

    printf("Ready to close MQTT Network......\n");
    if ((rc = client.unsubscribe(topic)) != 0) {
            printf("Failed: rc from unsubscribe was %d\n", rc);
    }
    if ((rc = client.disconnect()) != 0) {
    printf("Failed: rc from disconnect was %d\n", rc);
    }
    //mqttNetwork.disconnect();
    //printf("Successfully closed!\n");
}
void publish_message(MQTT::Client<MQTTNetwork, Countdown>* client) {
    MQTT::Message message;
    char buff[100];
    if (if_gesture_mode) sprintf(buff, "0\n");
    else if (if_detection_mode) sprintf(buff, "%d\n", thres_over_counter);
    else sprintf(buff, "???\n");
   
    message.qos = MQTT::QOS0;
    message.retained = false;
    message.dup = false;
    message.payload = (void*) buff;
    message.payloadlen = strlen(buff) + 1;
    int rc = client->publish(topic, message);

    printf("rc:  %d\r\n", rc);
    printf("Puslish message: %s\r\n", buff);
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
      
      //change thres angel
      if (gesture_index == 0) {
         if (thres_angle_mode < thres_angle_mode_max - 1) thres_angle_mode++;
         else thres_angle_mode = thres_angle_mode_max - 1;
         cout << "[Gesture UI Mode]: threshold angel = " << thres_angle_table[thres_angle_mode];
         uLCDDisplay(thres_angle_table[thres_angle_mode]);
      }
      else if (gesture_index == 1) {
         if (thres_angle_mode > 0) thres_angle_mode--;
         else thres_angle_mode = 0;
         cout << "[Gesture UI Mode]: threshold angel = " << thres_angle_table[thres_angle_mode];
         uLCDDisplay(thres_angle_table[thres_angle_mode]);
      }
      
    // Produce an output
    if (gesture_index < label_num) {
      error_reporter->Report(config.output_message[gesture_index]);
    }

      
  }
}
void messageArrived(MQTT::MessageData& md) {
    MQTT::Message &message = md.message;
    char msg[300];
    sprintf(msg, "Message arrived: QoS%d, retained %d, dup %d, packetID %d\r\n", message.qos, message.retained, message.dup, message.id);
    //printf(msg);
    ThisThread::sleep_for(1000ms);
    char payload[300];
    sprintf(payload, "Payload %.*s\r\n", message.payloadlen, (char*)message.payload);
    //printf(payload);
    ++arrivedcount;
}
void close_mqtt() {
    closed = true;
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