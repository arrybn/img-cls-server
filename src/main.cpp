#include <Poco/Net/ServerSocket.h>
#include <Poco/Net/HTTPServer.h>
#include <Poco/Net/HTTPRequestHandler.h>
#include <Poco/Net/HTTPRequestHandlerFactory.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/Net/HTTPServerRequest.h>
#include <Poco/Net/HTTPServerResponse.h>
#include <Poco/Net/MultipartReader.h>
#include <Poco/Net/MessageHeader.h>
#include <Poco/Util/ServerApplication.h>
#include <Poco/StreamCopier.h>
#include <Poco/UUIDGenerator.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <iterator>
#include <optional>
#include <chrono>
#include <thread>
// TODO: remove extra includes

#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>

#include "inference_server.h"
#include "pop_blocking_queue.h"
#include "push_pop_map.h"

using namespace Poco::Net;
using namespace Poco::Util;

class MyRequestHandler : public HTTPRequestHandler
{
public:
  MyRequestHandler(ics::ProcessingQueuePtr processing_queue_ptr, 
                   ics::ReadyHashtablePtr ready_hashtable_ptr):
    processing_queue_ptr_(processing_queue_ptr),
    ready_hashtable_ptr_(ready_hashtable_ptr)
  {}

  virtual void handleRequest(HTTPServerRequest &req, HTTPServerResponse &resp)
  {
    std::ostringstream oss;

      MultipartReader mpr(req.stream());
      while (mpr.hasNextPart()) {
        MessageHeader mh;
        mpr.nextPart(mh);
        for (auto it: mh) {
          std::cout << it.first << " " << it.second << std::endl;
        }

        Poco::StreamCopier::copyStream(mpr.stream(), oss);
      }

      std::string content = oss.str();
      std::vector<char> data(content.begin(), content.end());
      // TODO: decoding exception
      cv::Mat m = cv::imdecode(data, cv::IMREAD_COLOR);

      auto token = Poco::UUIDGenerator().createRandom().toString();
      processing_queue_ptr_->push(std::make_pair(token, m));

      std::vector<std::string> result;
      bool got_result = false;
      while (!got_result)
      {
        got_result = ready_hashtable_ptr_->pop(token, result);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }

    resp.setStatus(HTTPResponse::HTTP_OK);
    resp.setContentType("text/html");

    std::ostream& out = resp.send();
    for (const auto& l: result) {
      out << l << ";"; 
    }
    out.flush();
  }

private:
  ics::ProcessingQueuePtr processing_queue_ptr_;
  ics::ReadyHashtablePtr ready_hashtable_ptr_;
};

class MyRequestHandlerFactory : public HTTPRequestHandlerFactory
{
public:
  MyRequestHandlerFactory(ics::ProcessingQueuePtr processing_queue_ptr, 
                          ics::ReadyHashtablePtr ready_hashtable_ptr):
    processing_queue_ptr_(processing_queue_ptr),
    ready_hashtable_ptr_(ready_hashtable_ptr)
  {}

  virtual HTTPRequestHandler* createRequestHandler(const HTTPServerRequest &)
  {
    return new MyRequestHandler(processing_queue_ptr_, ready_hashtable_ptr_);
  }

private:
  ics::ProcessingQueuePtr processing_queue_ptr_;
  ics::ReadyHashtablePtr ready_hashtable_ptr_;
};

class MyServerApp : public ServerApplication
{
protected:
  int main(const std::vector<std::string> &)
  {
    auto queue_ptr = std::make_shared<ics::ProcessingQueue>();
    auto ready_map_ptr = std::make_shared<ics::ReadyHashtable>();
    auto inference_server_ptr = std::make_unique<ics::InferenceServer>("mobilenet-v2-pytorch/FP32/mobilenet-v2-pytorch.xml", 
                                                                                        "imagenet_classes.txt");

    inference_server_ptr->set_processing_queue(queue_ptr);
    inference_server_ptr->set_ready_hashtable(ready_map_ptr);
    inference_server_ptr->run();

    HTTPServer s(new MyRequestHandlerFactory(queue_ptr, ready_map_ptr), ServerSocket(9090), new HTTPServerParams);

    s.start();
    std::cout << std::endl << "Server started" << std::endl;

    waitForTerminationRequest();  // wait for CTRL-C or kill

    std::cout << std::endl << "Shutting down..." << std::endl;
    s.stop();

    return Application::EXIT_OK;
  }
};

int main(int argc, char** argv)
{
  MyServerApp app;
  return app.run(argc, argv);
}