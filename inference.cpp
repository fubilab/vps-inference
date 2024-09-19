#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include "dsacstar/dsacstar_types.h"
#include "dsacstar/dsacstar_util.h"
#include "dsacstar/dsacstar_util_rgbd.h"
#include "dsacstar/dsacstar_loss.h"
#include "dsacstar/dsacstar_derivative.h"
#include "dsacstar/stop_watch.h"
#include "dsacstar/thread_rand.h"

#define MAX_REF_STEPS 100 // max pose refienment iterations
#define MAX_HYPOTHESES_TRIES 1000000 // repeat sampling x times hypothesis if hypothesis is invalid

void dsacstar_rgb_forward(
	at::Tensor sceneCoordinatesSrc, 
	at::Tensor outPoseSrc,
	int ransacHypotheses, 
	float inlierThreshold,
	float focalLength,
	float ppointX,
	float ppointY,
	float inlierAlpha,
	float maxReproj,
	int subSampling)
{
	ThreadRand::init();

	// access to tensor objects
	dsacstar::coord_t sceneCoordinates = 
		sceneCoordinatesSrc.accessor<float, 4>();

	// dimensions of scene coordinate predictions
	int imH = sceneCoordinates.size(2);
	int imW = sceneCoordinates.size(3);

	// internal camera calibration matrix
	cv::Mat_<float> camMat = cv::Mat_<float>::eye(3, 3);
	camMat(0, 0) = focalLength;
	camMat(1, 1) = focalLength;
	camMat(0, 2) = ppointX;
	camMat(1, 2) = ppointY;	

	// calculate original image position for each scene coordinate prediction
	cv::Mat_<cv::Point2i> sampling = 
		dsacstar::createSampling(imW, imH, subSampling, 0, 0);

	std::cout << BLUETEXT("Sampling " << ransacHypotheses << " hypotheses.") << std::endl;
	StopWatch stopW;

	// sample RANSAC hypotheses
	std::vector<dsacstar::pose_t> hypotheses;
	std::vector<std::vector<cv::Point2i>> sampledPoints;  
	std::vector<std::vector<cv::Point2f>> imgPts;
	std::vector<std::vector<cv::Point3f>> objPts;

	dsacstar::sampleHypotheses(
		sceneCoordinates,
		sampling,
		camMat,
		ransacHypotheses,
		MAX_HYPOTHESES_TRIES,
		inlierThreshold,
		hypotheses,
		sampledPoints,
		imgPts,
		objPts);

	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;	
	std::cout << BLUETEXT("Calculating scores.") << std::endl;
    
	// compute reprojection error images
	std::vector<cv::Mat_<float>> reproErrs(ransacHypotheses);
	cv::Mat_<double> jacobeanDummy;

	#pragma omp parallel for 
	for(unsigned h = 0; h < hypotheses.size(); h++)
    	reproErrs[h] = dsacstar::getReproErrs(
		sceneCoordinates,
		hypotheses[h], 
		sampling, 
		camMat,
		maxReproj,
		jacobeanDummy);

    // soft inlier counting
	std::vector<double> scores = dsacstar::getHypScores(
    	reproErrs,
    	inlierThreshold,
    	inlierAlpha);

	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
	std::cout << BLUETEXT("Drawing final hypothesis.") << std::endl;	

	// apply soft max to scores to get a distribution
	std::vector<double> hypProbs = dsacstar::softMax(scores);
	double hypEntropy = dsacstar::entropy(hypProbs); // measure distribution entropy
	int hypIdx = dsacstar::draw(hypProbs, false); // select winning hypothesis

	std::cout << "Soft inlier count: " << scores[hypIdx] << " (Selection Probability: " << (int) (hypProbs[hypIdx]*100) << "%)" << std::endl; 
	std::cout << "Entropy of hypothesis distribution: " << hypEntropy << std::endl;


	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;
	std::cout << BLUETEXT("Refining winning pose:") << std::endl;

	// refine selected hypothesis
	cv::Mat_<int> inlierMap;

	dsacstar::refineHyp(
		sceneCoordinates,
		reproErrs[hypIdx],
		sampling,
		camMat,
		inlierThreshold,
		MAX_REF_STEPS,
		maxReproj,
		hypotheses[hypIdx],
		inlierMap);

	std::cout << "Done in " << stopW.stop() / 1000 << "s." << std::endl;

	// write result back to PyTorch
	dsacstar::trans_t estTrans = dsacstar::pose2trans(hypotheses[hypIdx]);

	auto outPose = outPoseSrc.accessor<float, 2>();
	for(unsigned x = 0; x < 4; x++)
	for(unsigned y = 0; y < 4; y++)
		outPose[y][x] = estTrans(y, x);	
}


// Function to convert cv::Mat to torch::Tensor
torch::Tensor CVMatToTensor(const cv::Mat& image) {
    cv::Mat img_float;
    image.convertTo(img_float, CV_32F, 1.0 / 255);
    auto tensor = torch::from_blob(img_float.data, {1, image.rows, image.cols, 1});
    tensor = tensor.permute({0, 3, 1, 2});
    return tensor.clone();
}

// Function to resize the image while maintaining aspect ratio
cv::Mat ResizeImage(const cv::Mat& image, int size) {
    int width = image.cols;
    int height = image.rows;
    int new_width, new_height;

    if (width < height) {
        new_width = size;
        new_height = static_cast<int>(height * (static_cast<float>(size) / width));
    } else {
        new_height = size;
        new_width = static_cast<int>(width * (static_cast<float>(size) / height));
    }

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(new_width, new_height));
    return resized_image;
}

// Function to convert image to grayscale
cv::Mat ConvertToGrayscale(const cv::Mat& image) {
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    return gray_image;
}

// Main function to apply transformations
torch::Tensor ApplyTransforms(const cv::Mat& image) {
    // Resize
    cv::Mat resized_image = ResizeImage(image, 480);

    // Convert to grayscale
    cv::Mat gray_image = ConvertToGrayscale(resized_image);
    auto tensor = CVMatToTensor(gray_image);

    // Normalize
    tensor = tensor.sub(0.44).div(0.25);
    return tensor;
}

int main() {
    // Load the model
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("traced_model.pt");
        std::cout << "Loaded the model" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "Error loading the model\n";
        return -1;
    }

    // Print the model's layers
    //for (const auto& module : module.named_modules()) {
    //    std::cout << "Layer: " << module.name << std::endl;
    //}

    // Load an image using OpenCV
    cv::Mat image = cv::imread("1721396597.png");
    // Apply transformations
    torch::Tensor img_tensor = ApplyTransforms(image);

    //std::cout << "Tensor Dimensions: " << img_tensor.sizes() << std::endl;
    std::cout << "Created tensor" << std::endl;

    // Forward pass
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(img_tensor);
    at::Tensor output = module.forward(inputs).toTensor();

    // Print the output
    std::cout << "Output" << std::endl;
    //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';



    // Prepare parameters for dsacstar.forward_rgb
    at::Tensor scene_coordinates = output; // Assuming output is the scene coordinates tensor
    at::Tensor out_pose = torch::zeros({4, 4}); // Placeholder for the output pose
    int hypotheses = 64; // Example value
    float threshold = 10.0; // Example value
    float focal = 565.0; // Example value
    float principal_point_x = 384;
    float principal_point_y = 240;
    float inlier_alpha = 100.0; // Example value
    float max_pixel_error = 100.0; // Example value
    int output_subsample = 8; // Example value


    // Call dsacstar.forward_rgb
    dsacstar_rgb_forward(
        scene_coordinates,
        out_pose,
        hypotheses,
        threshold,
        focal,
        principal_point_x,
        principal_point_y,
        inlier_alpha,
        max_pixel_error,
        output_subsample
    );

    std::cout << "Pose: " << out_pose << std::endl;

    return 0;



}
