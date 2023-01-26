#ifndef INCLUDE_INFERENCE_SERVER
#define INCLUDE_INFERENCE_SERVER

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include <opencv2/core/mat.hpp>

#include "pop_blocking_queue.h"
#include "push_pop_map.h"

namespace ov {
class Core;
class CompiledModel;
}  // namespace ov

namespace ics {
using Predictions = std::vector<std::string>;
using ProcessingQueue = PopBlockQueue<std::pair<std::string, cv::Mat>>;
using ProcessingQueuePtr = std::shared_ptr<ProcessingQueue>;
using ReadyHashtable = PushPopMap<std::string, Predictions>;
using ReadyHashtablePtr = std::shared_ptr<ReadyHashtable>;

class InferenceServer final {
 public:
    InferenceServer(const std::string& model_path, const std::string& labels_path);
    InferenceServer(const InferenceServer&) = delete;
    InferenceServer& operator=(const InferenceServer&) = delete;
    ~InferenceServer();

    void run();

    void set_processing_queue(ProcessingQueuePtr processing_queue);
    void set_ready_hashtable(ReadyHashtablePtr ready_hashtable);

 private:
    class InferenceServerImpl;
    std::unique_ptr<InferenceServerImpl> impl_ptr_;
};

}  // namespace ics

#endif /* INCLUDE_INFERENCE_SERVER */
