#include <iostream>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>

#include "inference.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{

    bool runOnGPU = true;
    
    Inference inf("../source/models/yolov5s.onnx", cv::Size(640, 480),
                  "../source/classes/classes.txt", runOnGPU);

    rs2::pipeline p;
    rs2::config cfg;

    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    p.start(cfg);

    while (true)
    {

        rs2::frameset frames = p.wait_for_frames();
        rs2::frame color_frame = frames.get_color_frame();

        cv::Mat frame(Size(640, 480), CV_8UC3, const_cast<void *>(color_frame.get_data()), Mat::AUTO_STEP);

        std::vector<Detection> output = inf.runInference(frame);

        int detections = output.size();

        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;

            cv::rectangle(frame, box, color, 2);

            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, nullptr);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }

        float scale = 0.8;
        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        cv::imshow("Inference", frame);

        if (cv::waitKey(1) == 'q') break;
    }

    p.stop();
    return 0;
}
