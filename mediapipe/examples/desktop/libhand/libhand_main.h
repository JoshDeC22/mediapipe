#ifndef LIBHAND_H
#define LIBHAND_H

#include <cstdlib>
#include <string>
#include <opencv2/opencv.hpp>

// Windows DLL exports
#if _WIN32
    #define LIBHAND_EXPORT __declspec(dllexport)
    #if defined(COMPILING_DLL)
        #define LIBHAND_API __declspec(dllexport)
    #else
        #define LIBHAND_API __declspec(dllimport)
    #endif

// Linux SO and Mac DYLIB exports
#else
    #define LIBHAND_EXPORT __attribute__((visibility("default")))
    #define LIBHAND_API
#endif

// Define the namespace
namespace libhand{
    // Hand landmark enum object with the names of the landmarks and their indices
    enum LIBHAND_API HandLandmark {
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
    typedef LIBHAND_API std::pair<HandLandmark, HandLandmark> LandmarkPair;

    // Create a vector of hand connections
    extern LIBHAND_API const std::vector<LandmarkPair> landmarkConnections;

    // Struct to store the list of landmarks
    struct LIBHAND_API LandmarkList {
        std::vector<cv::Point3f> landmarks;
        std::vector<float> presence;
        std::vector<float> visibility;
    };

    // Class definition for the mediapipe graph wrapper
    class LIBHAND_API HandProcessor {
        private:
            void* runner_void;

        public:
            // Constructor
            HandProcessor();

            // Destructor
            ~HandProcessor();

            // This function initialises the graph
            bool init(std::string config_file);

            // This function runs the graph to process a frame
            bool process(cv::Mat input_frame, cv::Mat &output_frame_mat, std::vector<LandmarkList> &landmarks, size_t timestamp, bool &landmark_presence);
    };
}

#endif