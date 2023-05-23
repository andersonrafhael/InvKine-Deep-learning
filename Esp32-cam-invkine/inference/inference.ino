#include <EloquentTinyML.h>
#include <eloquent_tinyml/tensorflow.h>
#include "invkine_model.h"

#define INPUT_SIZE 3
#define OUTPUT_SIZE 3
#define BAUDRATE 115200

Eloquent::TinyML::TensorFlow::AllOpsTensorFlow<INPUT_SIZE, OUTPUT_SIZE, invkine_model_len> tf;

void setup() {
  
  Serial.begin(BAUDRATE);
  tf.begin(invkine_model);
  
}
void loop() {

    int i = 0;
    float input_data[INPUT_SIZE];
    float output_data[OUTPUT_SIZE];

    if (Serial.available() > 0) {

        String receivedString = Serial.readString(); 
        char* token = strtok(const_cast<char*>(receivedString.c_str()), ",");
        
        while (token != NULL && i < INPUT_SIZE) {
          input_data[i] = atof(token);
          token = strtok(NULL, ",");
          i++;
        }
  
        tf.predict(input_data, output_data);
        Serial.print(String(output_data[0]) + "," + String(output_data[1]) + "," + String(output_data[2]));
    }
}
