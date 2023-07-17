#include <stdio.h>

#define LEARNING_RATE 0.01
#define MAX_ITERATIONS 1000

double calculate_mean(double data[], int num_samples) {
    double sum = 0.0;
    for (int i = 0; i < num_samples; i++) {
        sum += data[i];
    }
    return sum / num_samples;
}

void gradient_descent(double x[], double y[], int num_samples, double* slope, double* intercept) {
    double error, sum_error, prev_sum_error, slope_gradient, intercept_gradient;
    *slope = 0.0;
    *intercept = 0.0;

    for (int i = 0; i < MAX_ITERATIONS; i++) {
        sum_error = 0.0;
        slope_gradient = 0.0;
        intercept_gradient = 0.0;

        for (int j = 0; j < num_samples; j++) {
            error = *slope * x[j] + *intercept - y[j];
            sum_error += error;

            slope_gradient += error * x[j];
            intercept_gradient += error;
        }

        prev_sum_error = sum_error;

        *slope = *slope - (LEARNING_RATE * slope_gradient) / num_samples;
        *intercept = *intercept - (LEARNING_RATE * intercept_gradient) / num_samples;

        if (prev_sum_error == sum_error) {
            break;
        }
    }
}

int main() {
    double x[] = {1, 2, 3, 4, 5};
    double y[] = {3, 5, 7, 9, 11};
    int num_samples = sizeof(x) / sizeof(double);

    double slope, intercept;
    gradient_descent(x, y, num_samples, &slope, &intercept);

    printf("Slope: %f\n", slope);
    printf("Intercept: %f\n", intercept);

    double new_x = 6;
    double prediction = slope * new_x + intercept;
    printf("Prediction for x = %f: %f\n", new_x, prediction);

    return 0;
}
