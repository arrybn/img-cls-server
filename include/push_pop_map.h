#ifndef INCLUDE_PUSH_POP_MAP
#define INCLUDE_PUSH_POP_MAP

#include <mutex>
#include <unordered_map>
#include <utility>

namespace ics {

template <typename K, typename V>
class PushPopMap {
 public:
    PushPopMap() = default;
    PushPopMap(const PushPopMap<K, V>&) = delete;
    PushPopMap& operator=(const PushPopMap<K, V>&) = delete;

    bool pop(const K& k, V& v) {
        std::scoped_lock<std::mutex> lck(mutex_);

        if (hash_map_.contains(k)) {
            auto node_handle = hash_map_.extract(k);
            v = std::move(node_handle.mapped());
            return true;
        } else {
            return false;
        }
    }

    void push(std::pair<K, V>&& elem) {
        std::scoped_lock<std::mutex> lck(mutex_);

        hash_map_.insert_or_assign(elem.first, elem.second);
    }

 private:
    std::unordered_map<K, V> hash_map_;
    mutable std::mutex mutex_;
};

}  // namespace ics

#endif /* INCLUDE_PUSH_POP_MAP */
