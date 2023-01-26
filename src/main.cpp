#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <Poco/Net/HTTPRequestHandler.h>
#include <Poco/Net/HTTPRequestHandlerFactory.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/Net/HTTPServer.h>
#include <Poco/Net/HTTPServerRequest.h>
#include <Poco/Net/HTTPServerResponse.h>
#include <Poco/Net/MessageHeader.h>
#include <Poco/Net/MultipartReader.h>
#include <Poco/Net/ServerSocket.h>
#include <Poco/StreamCopier.h>
#include <Poco/UUIDGenerator.h>
#include <Poco/Util/ServerApplication.h>

#include "inference_server.h"
#include "pop_blocking_queue.h"
#include "push_pop_map.h"

class RequestHandler : public Poco::Net::HTTPRequestHandler {
 public:
    RequestHandler(ics::ProcessingQueuePtr processing_queue_ptr,
                   ics::ReadyHashtablePtr ready_hashtable_ptr)
        : processing_queue_ptr_(processing_queue_ptr),
          ready_hashtable_ptr_(ready_hashtable_ptr) {}

    virtual void handleRequest(Poco::Net::HTTPServerRequest& req,
                               Poco::Net::HTTPServerResponse& resp) {
        std::ostringstream oss;

        Poco::Net::MultipartReader mpr(req.stream());
        while (mpr.hasNextPart()) {
            Poco::Net::MessageHeader mh;
            mpr.nextPart(mh);
            for (auto it : mh) {
                std::cout << it.first << " " << it.second << std::endl;
            }

            Poco::StreamCopier::copyStream(mpr.stream(), oss);
        }

        std::string content = oss.str();
        std::vector<char> data(content.begin(), content.end());
        cv::Mat image = cv::imdecode(data, cv::IMREAD_COLOR);

        if (image.empty()) {
            resp.setStatus(Poco::Net::HTTPResponse::HTTP_BAD_REQUEST);
            resp.setContentType("text/html");
            resp.send() << "Error while decoding file: not an image file";
            return;
        }

        auto token = Poco::UUIDGenerator().createRandom().toString();
        processing_queue_ptr_->push(std::make_pair(token, image));

        std::vector<std::string> result;
        bool got_result = false;
        while (!got_result) {
            got_result = ready_hashtable_ptr_->pop(token, result);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        resp.setStatus(Poco::Net::HTTPResponse::HTTP_OK);
        resp.setContentType("text/html");

        std::ostream& out = resp.send();
        for (const auto& l : result) {
            out << l << ";";
        }
        out.flush();
    }

 private:
    ics::ProcessingQueuePtr processing_queue_ptr_;
    ics::ReadyHashtablePtr ready_hashtable_ptr_;
};

class RequestHandlerFactory : public Poco::Net::HTTPRequestHandlerFactory {
 public:
    RequestHandlerFactory(ics::ProcessingQueuePtr processing_queue_ptr,
                          ics::ReadyHashtablePtr ready_hashtable_ptr)
        : processing_queue_ptr_(processing_queue_ptr),
          ready_hashtable_ptr_(ready_hashtable_ptr) {}

    virtual Poco::Net::HTTPRequestHandler* createRequestHandler(
        const Poco::Net::HTTPServerRequest&) {
        return new RequestHandler(processing_queue_ptr_, ready_hashtable_ptr_);
    }

 private:
    ics::ProcessingQueuePtr processing_queue_ptr_;
    ics::ReadyHashtablePtr ready_hashtable_ptr_;
};

class ServerApp : public Poco::Util::ServerApplication {
 protected:
    int main(const std::vector<std::string>&) {
        auto queue_ptr = std::make_shared<ics::ProcessingQueue>();
        auto ready_map_ptr = std::make_shared<ics::ReadyHashtable>();
        auto inference_server_ptr = std::make_unique<ics::InferenceServer>(
            "data/mobilenet-v2-pytorch/FP32/mobilenet-v2-pytorch.xml",
            "data/imagenet_classes.txt");

        inference_server_ptr->set_processing_queue(queue_ptr);
        inference_server_ptr->set_ready_hashtable(ready_map_ptr);
        inference_server_ptr->run();

        Poco::Net::HTTPServer s(
            new RequestHandlerFactory(queue_ptr, ready_map_ptr),
            Poco::Net::ServerSocket(9090), new Poco::Net::HTTPServerParams);

        s.start();
        std::cout << std::endl
                  << "Server started" << std::endl;

        waitForTerminationRequest();

        std::cout << std::endl
                  << "Shutting down..." << std::endl;
        s.stop();

        return Poco::Util::Application::EXIT_OK;
    }
};

int main(int argc, char** argv) {
    ServerApp app;
    return app.run(argc, argv);
}
