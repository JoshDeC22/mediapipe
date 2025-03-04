#include "libhand_main.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

#define NormalizedLandmarkList ::mediapipe::NormalizedLandmarkList // Define the namespace for the mediapipe NormalizedLandmarkList

// Define the names of the data streams
constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarkStream[] = "hand_landmarks";
constexpr char kLandmarkPresenceStream[] = "landmark_presence";

// Flag to show the name of the file containing the information on the graph
//ABSL_FLAG(std::string, calculator_graph_config_file, "", "Name of file containing text format CalculatorGraphConfig proto.");

// This function creates a mediapipe graph from a file
absl::Status create_graph_from_file(std::string config_file, mediapipe::CalculatorGraph &graph) {
    // Initialise the graph contents string
    std::string graph_contents;

    // Load the graph contents into the absl flag
    //MP_RETURN_IF_ERROR(mediapipe::file::GetContents(absl::GetFlag(FLAGS_calculator_graph_config_file), &graph_contents));
    mediapipe::file::GetContents(config_file, &graph_contents);

    // Log the result
    ABSL_LOG(INFO) << "Get calculator graph config contents: " << graph_contents;

    // Load the configuration
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph_contents);

    // Initialise the graph
    MP_RETURN_IF_ERROR(graph.Initialize(config));
}

// This is the helper class to run the graph
class GraphRunner {
    private:
        mediapipe::CalculatorGraph graph; // the graph to be run

        std::unique_ptr<mediapipe::OutputStreamPoller> poller_video; // the poller for the video stream

        std::unique_ptr<mediapipe::OutputStreamPoller> poller_landmarks; // the poller for the hand landmarks

        std::unique_ptr<mediapipe::OutputStreamPoller> poller_landmark_presence; // the poller for the landmark presence

    public:
        // This function initialises the graph
        absl::Status init_graph(std::string config_file) {
            // Initialise the graph from a file
            MP_RETURN_IF_ERROR(create_graph_from_file(config_file, graph));

            // Assign the stream pollers to temporary variables
            MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_video_tmp, graph.AddOutputStreamPoller(kOutputStream));
            MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmarks_tmp, graph.AddOutputStreamPoller(kLandmarkStream));
            MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmarks_presence_tmp, graph.AddOutputStreamPoller(kLandmarkPresenceStream));

            // Convert the temporary variables to unique pointers
            poller_video = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_video_tmp));
            poller_landmarks = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_landmarks_tmp));
            poller_landmark_presence = std::make_unique<mediapipe::OutputStreamPoller>(std::move(poller_landmarks_presence_tmp));

            // Start running the graph
            MP_RETURN_IF_ERROR(graph.StartRun({}));

            // If everything works return Ok
            return absl::OkStatus();
        }

        // This function uses the graph to process each video frame
        absl::Status process_frame(cv::Mat &input_raw, size_t timestamp, cv::Mat &output_frame_mat, std::vector<NormalizedLandmarkList> &landmarks, bool &landmark_presence) {
            // First preprocess the input image
            cv::Mat input;
            cv::cvtColor(input_raw, input, cv::COLOR_BGR2RGB);
            cv::flip(input, input, 1);

            // Wrap the image in an ImageFrame
            auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
                mediapipe::ImageFormat::SRGB, input.cols, input.rows,
                mediapipe::ImageFrame::kDefaultAlignmentBoundary
            );

            // Get the original frame back
            cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
            input_raw.copyTo(input_frame_mat);

            // Send the image packet to the graph
            MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(kInputStream, mediapipe::Adopt(input_frame.release()).At(mediapipe::Timestamp(timestamp))));

            // Get the graph result
            mediapipe::Packet packet_video, packet_landmarks, packet_landmark_presence;
            poller_video->Next(&packet_video);

            poller_landmark_presence->Next(&packet_landmark_presence);
            landmark_presence = packet_landmark_presence.Get<bool>();
            if (landmark_presence) {
                poller_landmarks->Next(&packet_landmarks);
                landmarks = packet_landmarks.Get<std::vector<NormalizedLandmarkList>>();
            }

            // Initialise the output frame
            std::unique_ptr<mediapipe::ImageFrame> output_frame;

            // Convert back to opencv
            output_frame_mat = mediapipe::formats::MatView(output_frame.get());
            if (output_frame_mat.channels() == 4) {
                cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGBA2BGR);
            }
            else {
                cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
            }
            return absl::OkStatus();
        }

        // This function closes the graph
        absl::Status close() {
            MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
            return graph.WaitUntilDone();
        }
};

namespace libhand {
    LIBHAND_EXPORT const std::vector<LandmarkPair> landmarkConnections = {
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

    LIBHAND_EXPORT HandProcessor::HandProcessor() {}

    LIBHAND_EXPORT bool HandProcessor::init(std::string config_file) {
        // Create a new graph runner instance
        runner_void = (void *)new GraphRunner();
        GraphRunner &runner = *(GraphRunner *)runner_void;

        // Initialise the graph and get the result
        absl::Status result = runner.init_graph(config_file);

        // If there is an error, show it
        if (!result.ok()) {
            std::cout << "Failed to initialize the graph: " << result.message() << std::endl;
        }
        return result.ok();
    }

    LIBHAND_EXPORT bool HandProcessor::process(cv::Mat input_frame, cv::Mat &output_frame_mat, std::vector<LandmarkList> &landmarks, size_t timestamp, bool &landmark_presence) {
        // Create a new graph runner instance and the temporary vector of landmarks
        GraphRunner &runner = *(GraphRunner *)runner_void;
        std::vector<NormalizedLandmarkList> landmarks_temp;

        // Process the frame
        absl::Status result = runner.process_frame(input_frame, timestamp, output_frame_mat, landmarks_temp, landmark_presence);

        // If there is an error show it
        if (!result.ok()) {
            std::cout << "Failed to process frame: " << result.message() << std::endl;
            return false;
        }

        // Add the normalized landmarks to the landmark list
        landmarks.resize(landmarks_temp.size());
        for (int i = 0; i < landmarks_temp.size(); i++) {
            landmarks[i].landmarks.resize(landmarks_temp[i].landmark_size());
            landmarks[i].presence.resize(landmarks_temp[i].landmark_size());
            landmarks[i].visibility.resize(landmarks_temp[i].landmark_size());
            for (int j = 0; j < landmarks_temp[i].landmark_size(); j++) {
                landmarks[i].landmarks[j].x = landmarks_temp[i].landmark(j).x();
                landmarks[i].landmarks[j].y = landmarks_temp[i].landmark(j).y();
                landmarks[i].landmarks[j].z = landmarks_temp[i].landmark(j).z();
                landmarks[i].presence[j] = landmarks_temp[i].landmark(j).presence();
                landmarks[i].visibility[j] = landmarks_temp[i].landmark(j).visibility();
            }
        }

        return result.ok();
    }

    LIBHAND_EXPORT HandProcessor::~HandProcessor() {
        delete (GraphRunner *)runner_void;
    }
}