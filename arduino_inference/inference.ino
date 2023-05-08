#include <EloquentTinyML.h>
#include "invikine_model.h"

#define NUMBER_OF_INPUTS 3
#define NUMBER_OF_OUTPUTS 2
// in future projects you may need to tweek this value: it's a trial and error process
#define TENSOR_ARENA_SIZE 5*1024

Eloquent::TinyML::TfLite<NUMBER_OF_INPUTS, NUMBER_OF_OUTPUTS, TENSOR_ARENA_SIZE> ml;


void setup() {
    Serial.begin(115200);
    ml.begin(invkine_model);
}

void loop() {
    // pick up a random x and predict its sine
    float x = 3.14 * random(100) / 100;
    float y = 3.14 * random(100) / 100;
    float z = 3.14 * random(100) / 100;

    float input[1] = { x, y, z };
    float output[2] = {0, 0};

    ml.predict(input, output);

    Serial.println("  Pose -> " + "x: " + String(x) + "y: " + String(y) + "z: " + String(z));
    Serial.println("Thetas -> " + "theta0: " + String(output[0]) + "theta1: " + String(output[1]));
    
    delay(1000);
}