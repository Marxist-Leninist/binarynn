#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <windows.h>
#include <bcrypt.h>

constexpr int INPUT_SIZE = 16;
constexpr int HIDDEN_SIZE = 32;
constexpr int OUTPUT_SIZE = 1;
constexpr double LEARNING_RATE = 0.01;
constexpr int EPOCHS = 10000;
constexpr int DATASET_SIZE = 10000;
constexpr int TEST_SIZE = 1000;
constexpr double SPARSITY = 0.7;

struct TrueBinarySparseNN {
   std::vector<std::vector<int>> weights;
   std::vector<std::vector<int>> mask;
   std::vector<int> hidden_bias;
   std::vector<int> output_weights;
   int output_bias;

   TrueBinarySparseNN() :
      weights(INPUT_SIZE, std::vector<int>(HIDDEN_SIZE)),
      mask(INPUT_SIZE, std::vector<int>(HIDDEN_SIZE)),
      hidden_bias(HIDDEN_SIZE),
      output_weights(HIDDEN_SIZE),
      output_bias(0) {}
};

static double sigmoid(double x) {
   return 1.0 / (1.0 + std::exp(-x));
}

static int binary_activation(double x) {
   return x > 0 ? 1 : -1;
}

static int win_rand() {
   BCRYPT_ALG_HANDLE hAlg = NULL;
   NTSTATUS status = BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_RNG_ALGORITHM, NULL, 0);
   if (!BCRYPT_SUCCESS(status)) {
      std::cerr << "Failed to open algorithm provider\n";
      return 0;
   }

   unsigned int result;
   status = BCryptGenRandom(hAlg, reinterpret_cast<PUCHAR>(&result), sizeof(result), 0);
   BCryptCloseAlgorithmProvider(hAlg, 0);

   if (!BCRYPT_SUCCESS(status)) {
      std::cerr << "Failed to generate random number\n";
      return 0;
   }

   return static_cast<int>(result);
}

static void initialize_true_binary_sparse_nn(TrueBinarySparseNN& nn) {
   for (int i = 0; i < INPUT_SIZE; i++) {
      for (int j = 0; j < HIDDEN_SIZE; j++) {
         nn.weights[i][j] = (win_rand() % 2) * 2 - 1;  // -1 or 1
         nn.mask[i][j] = (static_cast<double>(win_rand()) / UINT_MAX) < (1 - SPARSITY);
      }
   }
   for (int i = 0; i < HIDDEN_SIZE; i++) {
      nn.hidden_bias[i] = (win_rand() % 2) * 2 - 1;
      nn.output_weights[i] = (win_rand() % 2) * 2 - 1;
   }
   nn.output_bias = (win_rand() % 2) * 2 - 1;
}

static double forward_true_binary_sparse(const TrueBinarySparseNN& nn, const std::vector<int>& input) {
   std::vector<int> hidden_layer(HIDDEN_SIZE);
   for (int i = 0; i < HIDDEN_SIZE; i++) {
      int sum = 0;
      for (int j = 0; j < INPUT_SIZE; j++) {
         if (nn.mask[j][i]) {
            sum += input[j] * nn.weights[j][i];
         }
      }
      hidden_layer[i] = binary_activation(sum + nn.hidden_bias[i]);
   }

   int output_sum = 0;
   for (int i = 0; i < HIDDEN_SIZE; i++) {
      output_sum += hidden_layer[i] * nn.output_weights[i];
   }
   return sigmoid(output_sum + nn.output_bias);
}

static void train_true_binary_sparse(TrueBinarySparseNN& nn, const std::vector<int>& input, double target) {
   std::vector<int> hidden_layer(HIDDEN_SIZE);
   for (int i = 0; i < HIDDEN_SIZE; i++) {
      int sum = 0;
      for (int j = 0; j < INPUT_SIZE; j++) {
         if (nn.mask[j][i]) {
            sum += input[j] * nn.weights[j][i];
         }
      }
      hidden_layer[i] = binary_activation(sum + nn.hidden_bias[i]);
   }

   double output = 0;
   for (int i = 0; i < HIDDEN_SIZE; i++) {
      output += hidden_layer[i] * nn.output_weights[i];
   }
   output = sigmoid(output + nn.output_bias);

   double output_error = target - output;
   double output_delta = output_error * output * (1 - output);

   for (int i = 0; i < HIDDEN_SIZE; i++) {
      int update = static_cast<int>(LEARNING_RATE * output_delta * hidden_layer[i]);
      nn.output_weights[i] += update;
      nn.output_weights[i] = nn.output_weights[i] > 0 ? 1 : -1;  // Binarize

      double hidden_error = output_delta * nn.output_weights[i];
      for (int j = 0; j < INPUT_SIZE; j++) {
         if (nn.mask[j][i]) {
            int update = static_cast<int>(LEARNING_RATE * hidden_error * input[j]);
            nn.weights[j][i] += update;
            nn.weights[j][i] = nn.weights[j][i] > 0 ? 1 : -1;  // Binarize
         }
      }
   }

   int bias_update = static_cast<int>(LEARNING_RATE * output_delta);
   nn.output_bias += bias_update;
   nn.output_bias = nn.output_bias > 0 ? 1 : -1;  // Binarize
}

static void generate_math_problem(std::vector<int>& input, double& target) {
   int op1 = win_rand() % 100;
   int op2 = win_rand() % 100;
   int operation = win_rand() % 4;

   input.assign(INPUT_SIZE, 0);

   for (int i = 0; i < 7; i++) {
      input[i] = (op1 >> i) & 1 ? 1 : -1;  // Convert to -1 or 1
      input[i + 7] = (op2 >> i) & 1 ? 1 : -1;
   }
   input[14 + operation] = 1;

   switch (operation) {
   case 0: target = (op1 + op2) / 200.0; break;
   case 1: target = (op1 - op2 + 100) / 200.0; break;
   case 2: target = (op1 * op2) / 10000.0; break;
   case 3: target = op2 != 0 ? (op1 / static_cast<double>(op2)) / 100.0 : 0.5; break;
   }
}

static double test_network(const TrueBinarySparseNN& nn, int num_tests) {
   int correct = 0;
   for (int i = 0; i < num_tests; i++) {
      std::vector<int> input(INPUT_SIZE);
      double target;
      generate_math_problem(input, target);

      double prediction = forward_true_binary_sparse(nn, input);
      double actual_result = target * 200.0;
      double predicted_result = prediction * 200.0;

      if (std::fabs(actual_result - predicted_result) < 5.0) {
         correct++;
      }
   }
   return static_cast<double>(correct) / num_tests;
}

int main() {
   TrueBinarySparseNN nn;
   initialize_true_binary_sparse_nn(nn);

   std::vector<std::vector<int>> inputs(DATASET_SIZE, std::vector<int>(INPUT_SIZE));
   std::vector<double> targets(DATASET_SIZE);

   for (int i = 0; i < DATASET_SIZE; i++) {
      generate_math_problem(inputs[i], targets[i]);
   }

   LARGE_INTEGER frequency, start, end;
   QueryPerformanceFrequency(&frequency);

   // Train and time True Binary Sparse NN
   QueryPerformanceCounter(&start);
   for (int epoch = 0; epoch < EPOCHS; epoch++) {
      for (int i = 0; i < DATASET_SIZE; i++) {
         train_true_binary_sparse(nn, inputs[i], targets[i]);
      }
      if (epoch % 1000 == 0) {
         std::cout << "Epoch " << epoch << " completed\n";
      }
   }
   QueryPerformanceCounter(&end);
   double cpu_time_used = static_cast<double>(end.QuadPart - start.QuadPart) / frequency.QuadPart;
   std::cout << "True Binary Sparse NN training time: " << cpu_time_used << " seconds\n";

   // Test network
   double accuracy = test_network(nn, TEST_SIZE);
   std::cout << "True Binary Sparse NN Accuracy: " << accuracy * 100 << "%\n";

   // Interactive testing
   std::cout << "\nEnter math problems (e.g., '23 + 45' or '12 * 6'). Type 'exit' to quit.\n";
   std::string input_str;
   while (true) {
      std::cout << "Enter problem: ";
      std::getline(std::cin, input_str);

      if (input_str == "exit") {
         break;
      }

      int op1, op2;
      char operation;
      if (sscanf_s(input_str.c_str(), "%d %c %d", &op1, &operation, 1, &op2) != 3) {
         std::cout << "Invalid input. Please use format: number operation number\n";
         continue;
      }

      std::vector<int> input(INPUT_SIZE, 0);
      for (int i = 0; i < 7; i++) {
         input[i] = (op1 >> i) & 1 ? 1 : -1;
         input[i + 7] = (op2 >> i) & 1 ? 1 : -1;
      }

      switch (operation) {
      case '+': input[14] = 1; input[15] = -1; input[16] = -1; input[17] = -1; break;
      case '-': input[14] = -1; input[15] = 1; input[16] = -1; input[17] = -1; break;
      case '*': input[14] = -1; input[15] = -1; input[16] = 1; input[17] = -1; break;
      case '/': input[14] = -1; input[15] = -1; input[16] = -1; input[17] = 1; break;
      default:
         std::cout << "Unsupported operation. Please use +, -, *, or /\n";
         continue;
      }

      double prediction = forward_true_binary_sparse(nn, input);
      double result = prediction * 200.0;
      std::cout << "Predicted result: " << result << "\n";

      // Calculate actual result for comparison
      double actual_result;
      switch (operation) {
      case '+': actual_result = op1 + op2; break;
      case '-': actual_result = op1 - op2; break;
      case '*': actual_result = op1 * op2; break;
      case '/': actual_result = op2 != 0 ? static_cast<double>(op1) / op2 : 0; break;
      }
      std::cout << "Actual result: " << actual_result << "\n";
      std::cout << "Difference: " << std::fabs(result - actual_result) << "\n";
   }

   return 0;
}