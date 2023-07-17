#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUM_INPUTS 2
#define NUM_SAMPLES 4
#define LEARNING_RATE 0.1
#define MAX_ITERATIONS 1000

typedef struct {
    double *weights;
    double bias;
} Neuron;

void initialize_neuron(Neuron *neuron) {
    neuron->weights = (double *)malloc(NUM_INPUTS * sizeof(double));
    for (int i = 0; i < NUM_INPUTS; i++) {
        neuron->weights[i] = ((double)rand() / RAND_MAX) * 2 - 1; // Random value between -1 and 1
    }
    neuron->bias = ((double)rand() / RAND_MAX) * 2 - 1;
}

double neuron_output(Neuron *neuron, double inputs[]) {
    double sum = 0.0;
    for (int i = 0; i < NUM_INPUTS; i++) {
        sum += neuron->weights[i] * inputs[i];
    }
    sum += neuron->bias;
    return 1 / (1 + exp(-sum)); // Sigmoid activation function
}

void train_neuron(Neuron *neuron, double inputs[][NUM_INPUTS], double outputs[]) {
    for (int iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        double total_error = 0.0;
        for (int sample = 0; sample < NUM_SAMPLES; sample++) {
            double predicted_output = neuron_output(neuron, inputs[sample]);
            double error = outputs[sample] - predicted_output;
            total_error += fabs(error);

            for (int i = 0; i < NUM_INPUTS; i++) {
                neuron->weights[i] += LEARNING_RATE * error * inputs[sample][i];
            }
            neuron->bias += LEARNING_RATE * error;
        }

        if (total_error == 0) {
            break; // Converged
        }
    }
}

int main() {
    Neuron neuron;
    initialize_neuron(&neuron);

    double inputs[][NUM_INPUTS] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    double outputs[] = {0, 0, 0, 1};

    train_neuron(&neuron, inputs, outputs);

    for (int i = 0; i < NUM_SAMPLES; i++) {
        double predicted_output = neuron_output(&neuron, inputs[i]);
        printf("Input: %d %d, Predicted Output: %f\n", (int)inputs[i][0], (int)inputs[i][1], predicted_output);
    }

    free(neuron.weights);
    return 0;
}
