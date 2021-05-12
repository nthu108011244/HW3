# HW3
## RPC loop
``` bash
#include "mbed_rpc.h"

void readRPCCommand();
```

## uLCD display
``` bash
#include "uLCD_4DGL.h"

uLCD_4DGL uLCD(D1, D0, D2);

void uLCDInit();
void uLCDDisplay(double inform);
```
## Tensor Flow ML
```bash
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

void gestureMode_gestureVerify();
int  PredictGesture(float* output);
```

## MQTT & Accelerometer
```bash
#include "stm32l475e_iot01_accelero.h"
#include "MQTTNetwork.h"
#include "MQTTmbed.h"
#include "MQTTClient.h"

Thread mqtt_thread(osPriorityHigh);
Thread publish_thread;
EventQueue mqtt_queue;
EventQueue publish_queue;

int16_t acc_data_XYZ[3] = {0};
WiFiInterface *wifi;
InterruptIn btn2(USER_BUTTON);
volatile int message_num = 0;
volatile int arrivedcount = 0;
volatile bool closed = false;
const char* topic = "Mbed";
MQTT::Client<MQTTNetwork, Countdown> *global_client;

void publish_MQTT();
void publish_message(MQTT::Client<MQTTNetwork, Countdown>* client);
void messageArrived(MQTT::MessageData& md);
void close_mqtt();
```

## Gesture UI Mode
``` bash
RpcDigitalOut myled2(LED1,"gestureMode");
DigitalOut if_gesture_mode(LED1);

void gestureMode();
```

## Tilt Angel Detection Mode
```bash
#define thres_angle_mode_max 3

Thread detection_thread;
EventQueue detection_queue;

int thres_angle_mode = 0;
int thres_angle_table[thres_angle_mode_max] = {30, 45, 60};
int thres_over_counter = 0;

void detectionMode();
```

## Main function
```bash
int main() {
   gesture_queue.call(&gestureMode);
   detection_queue.call(&detectionMode);
   publish_queue.call(&publish_MQTT);
   gesture_thread.start(callback(&gesture_queue, &EventQueue::dispatch_forever));
   detection_thread.start(callback(&detection_queue, &EventQueue::dispatch_forever));
   publish_thread.start(callback(&publish_queue, &EventQueue::dispatch_forever));
   uLCDDisplay(0);
   readRPCCommand();
}
```
