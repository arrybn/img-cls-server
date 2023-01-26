#ifndef INCLUDE_POP_BLOCKING_QUEUE
#define INCLUDE_POP_BLOCKING_QUEUE

#include <optional>
#include <memory>
#include <mutex>
#include <queue>
#include <condition_variable>

namespace ics {

template <typename T>
class PopBlockQueue {
public:
    PopBlockQueue() = default;
    PopBlockQueue(const PopBlockQueue<T> &) = delete ;
    PopBlockQueue& operator=(const PopBlockQueue<T> &) = delete ;

    virtual ~PopBlockQueue() {}

    size_t size() const {
        std::scoped_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty()) {
            cv_pop_.wait(lock);
        }
        T tmp = queue_.front();
        queue_.pop();
        
        return tmp;
    }

    void push(const T &item) {
        std::scoped_lock<std::mutex> lock(mutex_);
        queue_.push(item);
        cv_pop_.notify_one();
    }

private:
    std::queue<T> queue_;
  mutable std::mutex mutex_;
  mutable std::condition_variable cv_pop_;
};

}

#endif /* INCLUDE_POP_BLOCKING_QUEUE */
