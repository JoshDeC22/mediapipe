#ifndef RUN_GRAPH_H
#define RUN_GRAPH_H

#include <cstdlib>
#include <string>
#include <opencv2/opencv.hpp>

// Hand landmark enum object with the names of the landmarks and their indices
enum HandLandmark {
    WRIST = 0,
    THUMB_CMC = 1,
    THUMB_MCP = 2,
    THUMB_IP = 3,
    THUMB_TIP = 4,
    INDEX_FINGER_MCP = 5,
    INDEX_FINGER_PIP = 6,
    INDEX_FINGER_DIP = 7,
    INDEX_FINGER_TIP = 8,
    MIDDLE_FINGER_MCP = 9,
    MIDDLE_FINGER_PIP = 10,
    MIDDLE_FINGER_DIP = 11,
    MIDDLE_FINGER_TIP = 12,
    RING_FINGER_MCP = 13,
    RING_FINGER_PIP = 14,
    RING_FINGER_DIP = 15,
    RING_FINGER_TIP = 16,
    PINKY_MCP = 17,
    PINKY_PIP = 18,
    PINKY_DIP = 19,
    PINKY_TIP = 20
};

// Define the type to store the landmark pairs (for displaying the connections)
typedef std::pair<HandLandmark, HandLandmark> LandmarkPair;

// Create a vector of hand connections
const std::vector<LandmarkPair> landmarkConnections = {
    {HandLandmark::WRIST, HandLandmark::THUMB_CMC},
    {HandLandmark::THUMB_CMC, HandLandmark::THUMB_MCP},
    {HandLandmark::THUMB_MCP, HandLandmark::THUMB_IP},
    {HandLandmark::THUMB_IP, HandLandmark::THUMB_TIP},
    {HandLandmark::WRIST, HandLandmark::INDEX_FINGER_MCP},
    {HandLandmark::INDEX_FINGER_MCP, HandLandmark::INDEX_FINGER_PIP},
    {HandLandmark::INDEX_FINGER_PIP, HandLandmark::INDEX_FINGER_DIP},
    {HandLandmark::INDEX_FINGER_DIP, HandLandmark::INDEX_FINGER_TIP},
    {HandLandmark::INDEX_FINGER_MCP, HandLandmark::MIDDLE_FINGER_MCP},
    {HandLandmark::MIDDLE_FINGER_MCP, HandLandmark::MIDDLE_FINGER_PIP},
    {HandLandmark::MIDDLE_FINGER_PIP, HandLandmark::MIDDLE_FINGER_DIP},
    {HandLandmark::MIDDLE_FINGER_DIP, HandLandmark::MIDDLE_FINGER_TIP},
    {HandLandmark::MIDDLE_FINGER_MCP, HandLandmark::RING_FINGER_MCP},
    {HandLandmark::RING_FINGER_MCP, HandLandmark::RING_FINGER_PIP},
    {HandLandmark::RING_FINGER_PIP, HandLandmark::RING_FINGER_DIP},
    {HandLandmark::RING_FINGER_DIP, HandLandmark::RING_FINGER_TIP},
    {HandLandmark::RING_FINGER_MCP, HandLandmark::PINKY_MCP},
    {HandLandmark::WRIST, HandLandmark::PINKY_MCP},
    {HandLandmark::PINKY_MCP, HandLandmark::PINKY_PIP},
    {HandLandmark::PINKY_PIP, HandLandmark::PINKY_DIP},
    {HandLandmark::PINKY_DIP, HandLandmark::PINKY_TIP}
};

// Struct to store the list of landmarks
struct LandmarkList {
    std::vector<cv::Point3f> landmarks;
    std::vector<float> presence;
    std::vector<float> visibility;
};

// Class definition for the mediapipe graph wrapper
class MPPGraph {
    private:
        void* runner_void;

    public:
        // Constructor
        MPPGraph();

        // Destructor
        ~MPPGraph();

        // This function initialises the graph
        bool init_graph(std::string config_file);

        // This function runs the graph
        bool run_graph(cv::Mat input_frame, cv::Mat &output_frame_mat, std::vector<LandmarkList> &landmarks, size_t timestamp, bool &landmark_presence);
};

#endif