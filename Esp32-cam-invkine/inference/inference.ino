#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>

#include "invkine_model.h"

// Input size
#define INPUT_SIZE 3

// Output size
#define OUTPUT_SIZE 2

// Create a TensorFlow model wrapper object
Eloquent::TinyML::TensorFlow::AllOpsTensorFlow<INPUT_SIZE, OUTPUT_SIZE, invkine_model_len> tf;

void setup() {
  // Begin the TensorFlow model
  Serial.begin(115200);
  tf.begin(invkine_model);

}

void loop() {

  
  // Input data

//    float x = 3.14 * random(100) / 100;
//    float y = 3.14 * random(100) / 100;
//    float z = 3.14 * random(100) / 100;

  // float input_data[INPUT_SIZE] = {0.5, 2.3, -0.9};

  // // Output data
  // float output_data[OUTPUT_SIZE] = { 0.0, 0.0 };

  // // Make a prediction using the input data
  // tf.predict(input_data, output_data);

  
  // Serial.print("Input: ");
  // for (int i = 0; i < INPUT_SIZE; i++) {
  //   Serial.print(input_data[i]);
  //   if (i < INPUT_SIZE - 1) {
  //     Serial.print(", ");
  //   }
  // }
  // Serial.println();

  // // Print the output data
  // Serial.print("Output: ");
  // for (int i = 0; i < OUTPUT_SIZE; i++) {
  //   Serial.print(output_data[i]);
  //   if (i < OUTPUT_SIZE - 1) {
  //     Serial.print(", ");
  //   }
  // }
  // Serial.println();

  while (Serial.available() < sizeof(float) * 3);

  float input_data[INPUT_SIZE];
  float output_data[OUTPUT_SIZE] = { 0.0, 0.0 };
  Serial.readBytes((char *)input_data, sizeof(input_data));

  tf.predict(input_data, output_data);

  Serial.write((const uint8_t *)output_data, sizeof(output_data));
// Send a newline character to separate responses
  Serial.write('\n');
  
}
