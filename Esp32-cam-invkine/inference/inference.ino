#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>
#include "invkine_model.h"

// Input size
#define INPUT_SIZE 3

// Output size
#define OUTPUT_SIZE 2

#define BAUDRATE 115200

struct transfer {
  float theta0;
  float theta1;
};

char myData[30] = { 0 };

// Create a TensorFlow model wrapper object
Eloquent::TinyML::TensorFlow::AllOpsTensorFlow<INPUT_SIZE, OUTPUT_SIZE, invkine_model_len> tf;

void setup() {
  // Begin the TensorFlow model
  Serial.begin(BAUDRATE);
//  Serial.setTimeout(1);
  tf.begin(invkine_model);

}
void loop() {

  if (Serial.available() >= INPUT_SIZE*sizeof(float)) {

    // Read input data from serial
    float input_data[INPUT_SIZE];
    Serial.readBytes((char*)input_data, INPUT_SIZE*sizeof(float));

    // Make prediction
    float output_data[OUTPUT_SIZE];
    tf.predict(input_data, output_data);

    Serial.print("Output data: ");
    Serial.print(output_data[0]);
    Serial.print(", ");
    Serial.println(output_data[1]);

    // Send output data to serial
    Serial.write((char*)output_data, OUTPUT_SIZE*sizeof(float));
  }
}