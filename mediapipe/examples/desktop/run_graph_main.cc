#include "run_graph_main.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/util/resource_util.h"

#define NormalizedLandmarkList ::mediapipe::NormalizedLandmarkList // Define the namespace for the mediapipe NormalizedLandmarkList

// These strings are used for displaying the results (can be removed)
constexpr char input_stream[] = "input_video";
constexpr char window_name[] = "MediaPipe";

// This function creates a mediapipe graph from a file
absl::Status create_graph_from_file(std::string config_file, mediapipe::CalculatorGraph &graph) {
    // Initialise the graph contents
    std::string graph_contents;

    // Load the graph from the config file
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(config_file, &graph_contents));

    // Create the graph config object
    mediapipe::CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(graph_contents);

    // Initialise the graph
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    return absl::OkStatus();
}

// Setup the different window names (can be removed later)
const char output_stream[] = "output_video";
const char landmarks_output[] = "hand_landmarks";
const char landmark_presence_output[] = "landmark_presence";

// This is a helper class to help run the graph
class GraphRunner {
    private:
        mediapipe::CalculatorGraph graph; // the graph to process the frames

        std::unique_ptr<mediapipe::OutputStreamPoller> poller_video; // the poller for the video stream

        std::unique_ptr<mediapipe::OutputStreamPoller> poller_landmarks; // the poller for the hand landmarks

        std::unique_ptr<mediapipe::OutputStreamPoller> poller_landmark_presence; // the poller for the landmark presence

    public:
        // This function initialises the graph from a configuration file
        absl::Status init_graph(std::string config_file) {
            // Create the graph from the file
            MP_RETURN_IF_ERROR(create_graph_from_file(config_file, graph));

            // Create the stream pollers for the video, landmarks, and landmark presence
            MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller vid_poller_temp, graph.AddOutputStreamPoller(output_stream));
            poller_video = std::make_unique<mediapipe::OutputStreamPoller>(std::move(vid_poller_temp));
            MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmark_poller_temp, graph.AddOutputStreamPoller(landmarks_output));
            poller_landmarks = std::make_unique<mediapipe::OutputStreamPoller>(std::move(landmark_poller_temp));
            MP_ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller presence_poller_temp, graph.AddOutputStreamPoller(landmark_presence_output));
            poller_landmark_presence = std::make_unique<mediapipe::OutputStreamPoller>(std::move(presence_poller_temp));

            // Start running the graph
            MP_RETURN_IF_ERROR(graph.StartRun({}));

            return absl::OkStatus();
        }

        // This function processes each video frame
        absl::Status process_frame(cv::Mat &input_frame, size_t timestamp, cv::Mat &output_frame_mat, std::vector<NormalizedLandmarkList> &landmarks, bool &landmark_presence) {
            // Wrap the cv::Mat object into an image frame
            auto frame = absl::make_unique<mediapipe::ImageFrame>(mediapipe::ImageFormat::SRGBA, input_frame.cols, input_frame.rows, mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);

            cv::Mat frame_mat = mediapipe::formats::MatView(frame.get()); // The opencv Mat frame

            // Copy the input frame to the frame object
            input_frame.copyTo(frame_mat);

            // Send the image packet to the graph
            MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(input_stream, mediapipe::Adopt(frame.release()).At(mediapipe::Timestamp(timestamp))));

            // Get the graph result or stop if it fails
            mediapipe::Packet packet_video, packet_landmarks, packet_landmark_presence;
            poller_video->Next(&packet_video);

            poller_landmark_presence->Next(&packet_landmark_presence);
            landmark_presence = packet_landmark_presence.Get<bool>();
            if (landmark_presence) {
                poller_landmarks->Next(&packet_landmarks);
                landmarks = packet_landmarks.Get<std::vector<NormalizedLandmarkList>>();
            }

            // Initialise the mediapipe output frame
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
            MP_RETURN_IF_ERROR(graph.CloseInputStream(input_stream));
            return graph.WaitUntilDone();
        }
};

MPPGraph::MPPGraph() {}

bool MPPGraph::init_graph(std::string config_file) {
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

bool MPPGraph::run_graph(cv::Mat input_frame, cv::Mat &output_frame_mat, std::vector<LandmarkList> &landmarks, size_t timestamp, bool &landmark_presence) {
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

MPPGraph::~MPPGraph() {
    delete (GraphRunner *)runner_void;
}