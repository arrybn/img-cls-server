#include "inference_server.h"
#include <openvino/openvino.hpp>
#include <iostream>
#include <fstream>
#include <thread>

namespace {

// class InferenceRequestsPool {
// public:
//     InferenceRequestsPool(std::vector<ov::InferRequest>&& requests) {

//     }


// };

    const char MODEL_DIMS_ORDER[] = "NCHW";

    void printInputAndOutputsInfo(const ov::Model& network) {
        std::cout << "model name: " << network.get_friendly_name() << std::endl;

        const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
        for (const ov::Output<const ov::Node> input : inputs) {
            std::cout << "    inputs" << std::endl;

            const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
            std::cout << "        input name: " << name << std::endl;

            const ov::element::Type type = input.get_element_type();
            std::cout << "        input type: " << type << std::endl;

            const ov::Shape shape = input.get_shape();
            std::cout << "        input shape: " << shape << std::endl;
        }

        const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
        for (const ov::Output<const ov::Node> output : outputs) {
            std::cout << "    outputs" << std::endl;

            const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
            std::cout << "        output name: " << name << std::endl;

            const ov::element::Type type = output.get_element_type();
            std::cout << "        output type: " << type << std::endl;

            const ov::Shape shape = output.get_shape();
            std::cout << "        output shape: " << shape << std::endl;
        }
    }

    template <class T>
    void top_n(unsigned int n, const ov::Tensor& input, std::vector<unsigned>& output) {
        ov::Shape shape = input.get_shape();
        size_t input_rank = shape.size();
        OPENVINO_ASSERT(input_rank != 0 && shape[0] != 0, "Input tensor has incorrect dimensions!");
        size_t batchSize = shape[0];
        std::vector<unsigned> indexes(input.get_size() / batchSize);

        n = static_cast<unsigned>(std::min<size_t>((size_t)n, input.get_size()));
        output.resize(n * batchSize);

        for (size_t i = 0; i < batchSize; i++) {
            const size_t offset = i * (input.get_size() / batchSize);
            const T* batchData = input.data<const T>();
            batchData += offset;

            std::iota(std::begin(indexes), std::end(indexes), 0);
            std::partial_sort(std::begin(indexes),
                              std::begin(indexes) + n,
                              std::end(indexes),
                              [&batchData](unsigned l, unsigned r) {
                                  return batchData[l] > batchData[r];
                              });
            for (unsigned j = 0; j < n; j++) {
                output.at(i * n + j) = indexes.at(j);
            }
        }
    }

    ics::Predictions read_labels(const std::string& labels_path, int num_hint=1000) {
        ics::Predictions labels;
        labels.reserve(num_hint);

        std::ifstream labels_fs(labels_path);
        if (labels_fs.good()) {
            std::string class_name;
            while (getline(labels_fs, class_name)) {
                if (class_name.size()) {
                    labels.push_back(class_name);
                }
            }
        }

        return labels;
    }
}

namespace ics
{

class InferenceServer::InferenceServerImpl {
public:
    InferenceServerImpl(const std::string& model_path, const std::string& labels_path) {
        labels_ptr_ = std::make_shared<Predictions>(read_labels(labels_path));

        core_ptr_ = std::make_unique<ov::Core>();
        std::shared_ptr<ov::Model> model = core_ptr_->read_model(model_path);

        printInputAndOutputsInfo(*model);
        // assert the model has only one input
        // assert the model has only one output of size 1000
        // TODO: assert num channels in image
        // TDOD: assert num labels == out shape

        OPENVINO_ASSERT(model->inputs().size() == 1, "Sample supports models with 1 input only");
        OPENVINO_ASSERT(model->outputs().size() == 1, "Sample supports models with 1 output only");

        const std::string input_name = model->inputs()[0].get_any_name();
        ov::preprocess::PrePostProcessor ppp(model);
        ov::preprocess::InputInfo& input = ppp.input(input_name);
        input.tensor()
            .set_spatial_dynamic_shape()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        
        input.model().set_layout(MODEL_DIMS_ORDER);
        input.preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);

        model = ppp.build();

        compiled_model_ptr_ = std::make_unique<ov::CompiledModel>(core_ptr_->compile_model(model, "CPU"));

        auto num_requests = compiled_model_ptr_->get_property(ov::optimal_number_of_infer_requests);
        for (int i = 0; i < num_requests; i++) {
            available_requests_queue_.push(compiled_model_ptr_->create_infer_request());
        }
    }

    InferenceServerImpl(const InferenceServerImpl&) = delete;
    InferenceServerImpl& operator=(const InferenceServerImpl&) = delete;

    void queue_processing_loop() {
        while (true) {
            auto request = available_requests_queue_.pop();
            auto data = processing_queue_->pop();

            auto token = data.first;
            auto image = data.second;

            ov::Tensor in_t = ov::Tensor(ov::element::u8, {1, (unsigned)image.rows, (unsigned)image.cols, (unsigned)image.channels()}, image.data);
            request.set_input_tensor(in_t);

            // TODO: how to pass members to lambda by value
            request.set_callback([request, token, image, this](std::exception_ptr ex) mutable {
                if (ex) {
                    std::rethrow_exception(ex);
                }

                const ov::Tensor& output_tensor = request.get_output_tensor();
                std::vector<unsigned> top_5_indices;
                // TODO: get data type from model
                top_n<float>(5, output_tensor, top_5_indices);

                Predictions predictions;
                predictions.reserve(top_5_indices.size());

                std::for_each(top_5_indices.begin(), top_5_indices.end(), [&predictions, this](unsigned i){predictions.push_back(labels_ptr_->at(i));});

                ready_hashtable_->push(std::make_pair(token, std::move(predictions)));

                available_requests_queue_.push(std::move(request));
            });

            request.start_async();
        }
    }

    void run() {
        queue_porcessing_thread_ = std::thread([this](){this->queue_processing_loop();});
    }

    void set_processing_queue(ProcessingQueuePtr processing_queue) {
        processing_queue_ = processing_queue;
    }

    void set_ready_hashtable(ReadyHashtablePtr ready_hashtable) {
        ready_hashtable_ = ready_hashtable;
    }

private:
    std::unique_ptr<ov::CompiledModel> compiled_model_ptr_;
    std::unique_ptr<ov::Core> core_ptr_;
    std::shared_ptr<Predictions> labels_ptr_;
    PopBlockQueue<ov::InferRequest> available_requests_queue_;
    ProcessingQueuePtr processing_queue_;
    ReadyHashtablePtr ready_hashtable_;
    std::thread queue_porcessing_thread_;
};

InferenceServer::InferenceServer(const std::string& model_path, const std::string& labels_path): 
        impl_ptr_(std::make_unique<InferenceServerImpl>(model_path, labels_path)) {}

void InferenceServer::run() {
    impl_ptr_->run();
}

void InferenceServer::set_processing_queue(ProcessingQueuePtr processing_queue) {
    impl_ptr_->set_processing_queue(processing_queue);
}

void InferenceServer::set_ready_hashtable(ReadyHashtablePtr ready_hashtable) {
    impl_ptr_->set_ready_hashtable(ready_hashtable);
}

InferenceServer::~InferenceServer() {}
// Predictions InferenceServer::predict(cv::Mat image) {
//     // TODO: assert num channels in image
//     // TDOD: assert num labels == out shape
//     ov::InferRequest infer_request = compiled_model_ptr_->create_infer_request();

//     ov::Tensor in_t = ov::Tensor(ov::element::u8, {1, image.rows, image.cols, image.channels()}, image.data);
//     infer_request.set_input_tensor(in_t);
//     infer_request.infer();
//     const ov::Tensor& output_tensor = infer_request.get_output_tensor();
//     std::vector<unsigned> top_5_indices;
//     top_n<float>(5, output_tensor, top_5_indices);

//     Predictions predictions;
//     predictions.reserve(top_5_indices.size());

//     std::for_each(top_5_indices.begin(), top_5_indices.end(), [&predictions, this](unsigned i){predictions.push_back(labels_[i]);});

//     for (const auto s : predictions) {
//         std::cout << s << " ";
//     }
//     std::cout << std::endl;
//     std::cout << output_tensor.get_shape() << std::endl;
    
//     return predictions;
// }
    
} // namespace ics