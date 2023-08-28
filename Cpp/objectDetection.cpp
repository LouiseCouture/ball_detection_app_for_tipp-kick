/*
#include <torch/script.h>
#include <iostream>
#include <memory>

#include "objectdetection.h"

int loadModel(const string nameModel) {

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(nameModel);
    }
    catch (const c10::Error& e) {
        std::cerr << "ERROR loading the model" <<endl;
        return -1;
    }
    std::cout << "Model " << nameModel << " loaded fine\n";
    return 0;
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({ 1, 1, 64, 101 }));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward(inputs).toTensor();
    std::cout << output << "\n";
    int y_hat = output.argmax(1).item().toInt();
    std::cout << "Predicted class: " << y_hat << "\n";
    
}
*/