//DMITRII BEZRUKOV
#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <memory>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::unique_ptr;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

const double PI_MATH = atan(1) * 4;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

vector<vector<float>> grayScaleImage(BMP *img, int newHeight, int newWidth)
{
    static float RED_TO_GRAY = 0.299;
    static float GREEN_TO_GRAY = 0.587;
    static float BLUE_TO_GRAY = 0.114;
    static float SUM_TO_GRAY = RED_TO_GRAY + GREEN_TO_GRAY + BLUE_TO_GRAY;
    
    vector<vector<float>> grayImage;
    grayImage.assign(newHeight, vector<float>(newWidth));
    
    for (int x = 0; x < newHeight; x++) {
        for (int y = 0; y < newWidth; y++) {
            grayImage[x][y] = ((*img)(y, x)->Red * RED_TO_GRAY + 
                               (*img)(y, x)->Green * GREEN_TO_GRAY +
                               (*img)(y, x)->Blue * BLUE_TO_GRAY) / SUM_TO_GRAY;
        }
    }
    
    return grayImage;
}

pair<
vector<vector<float>>,
vector<vector<float>>
> getMagnitudesAndAngles(const vector<vector<float>> &img)
{
    if (img.size() == 0 || img[0].size() == 0) {
        return make_pair(vector<vector<float>> (), vector<vector<float>> ());
    }
    
    vector<vector<float>> magnitudes, angles;
    magnitudes.assign(img.size(), vector<float>(img[0].size()));
    angles.assign(img.size(), vector<float>(img[0].size()));
    
    for (unsigned x = 1; x + 1 < img.size(); x++) {
        for (unsigned y = 1; y + 1 < img[0].size(); y++) {
            float sumVertical =  img[x + 1][y] - img[x - 1][y];
            float sumHorizon = img[x][y + 1] - img[x][y - 1];
            
            magnitudes[x][y] = sqrt(sumHorizon * sumHorizon + sumVertical * sumVertical);
            
            angles[x][y] = atan2(sumHorizon, sumVertical) * 180.0 / PI_MATH;
            angles[x][y] = angles[x][y] > 0 ? angles[x][y] : angles[x][y] + 180;
        }
    }
    
    return make_pair(magnitudes, angles);
}

vector<float> transformXiSquare(vector<float> features)
{
    vector<float> ans;
    const float L = 0.25;
    const float SECH = 1.0 / cosh(L * PI_MATH);
    const float EPS = 1e-6;
    for (auto elem: features) {
        if (elem > EPS) {
            float value = cos(L * log(elem)) * sqrt(elem * SECH);
            ans.push_back(value);
            value = sin(L * log(elem)) * sqrt(elem * SECH);
            ans.push_back(value);
        } else {
            ans.push_back(0.0);
            ans.push_back(0.0);
        }
    }
    return ans;
}

vector<float> hogFeatures(vector<vector<float>> grayImage, int numCells, int numBlocks, int numBins)
{
    //int newHeight = grayImage.size() / numCells * numCells;
    //int newWidth = grayImage[0].size() / numCells * numCells;
    
    auto magAndAngle = getMagnitudesAndAngles(grayImage);
    auto magnitude = magAndAngle.first;
    auto angles = magAndAngle.second;
    
    int numHeight = grayImage.size() / numCells;
    int numWidth = grayImage[0].size() / numCells;
    float angleDelta = 180 / numBins;
    float eps = 1e-6;
    
    vector<vector<vector<float>>> features(numCells, vector<vector<float>>(numCells, vector<float>()));
    
    for (int row = 0; row < numCells; row++) {
        for (int col = 0; col < numCells; col++) {
            vector<float> curBins(numBins, 0);
            int xMin = row * numHeight;
            int xMax = (row + 1) * numHeight;
            int yMin = col * numWidth;
            int yMax = (col + 1) * numWidth;
            
            for (int angleNum = 0; angleNum < numBins; angleNum++) {
                for (int x = xMin; x < xMax; x++) {
                    for (int y = yMin; y < yMax; y++) {
                        if (angles[x][y] >= angleNum * angleDelta && 
                        angles[x][y] < (angleNum + 1) * angleDelta) {
                            curBins[angleNum] += magnitude[x][y];
                        }
                    }
                }
            }
            
            double sum = std::accumulate(curBins.begin(), curBins.end(), 0.0);
            for (auto &bin: curBins) {
                bin /= sum;
            }
            features[row][col] = curBins;
        }
    }
    
    vector<float> ansVector(features.size() * features[0].size() * features[0][0].size());
    int curFeature = 0;
    
    for (size_t row = 0; row < features.size(); row++) {
        for (size_t col = 0; col < features[0].size(); col++) {
            for (float elem: features[row][col]) {
                ansVector[curFeature] = elem;
                curFeature++;
            }
        }
    }
    
    numBlocks = numBlocks;
    eps = eps;
      
    return ansVector;
}

vector<vector<float>> cutImage(vector<vector<float>> img, int widthMin, int widthMax, int heightMin, int heightMax)
{
    vector<vector<float>> imgNew(heightMax - heightMin, vector<float>(widthMax - widthMin));
    for (int x = heightMin; x < heightMax; x++) {
        for (int y = widthMin; y < widthMax; y++) {
            imgNew[x - heightMin][y - widthMin] = img[x][y];
        }
    }
    return imgNew;
}

vector<float> concat(const vector<float> &a, const vector<float> &b)
{
    vector<float> ans;
    for (float item: a) {
        ans.push_back(item);
    }
    for (float item: b) {
        ans.push_back(item);
    }
    
    return ans;
}

vector<float> extractColorHist(BMP *img)
{
    int heightBlock = img->TellHeight() / 8;
    int widthBlock = img->TellWidth() / 8;
    vector<float> ans;
    
    for (int blockX = 0; blockX < 8; blockX++) {
        for (int blockY = 0; blockY < 8; blockY++) {
            float red = 0, green = 0, blue = 0;
            for (int x = 0; x < heightBlock; x++) {
                for (int y = 0; y < widthBlock; y++) {
                    red += img->GetPixel(y, x).Red;
                    green += img->GetPixel(y, x).Green;
                    blue += img->GetPixel(y, x).Blue;
                }
            }
            ans.push_back(red / heightBlock / widthBlock / 255);
            ans.push_back(green / heightBlock / widthBlock / 255);
            ans.push_back(blue / heightBlock / widthBlock / 255);
        }
    }
    return ans;
}

// Exatract features from dataset.
// You should implement this function by yourself =)
void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        
        vector<vector<float>> grayImage = grayScaleImage(data_set[image_idx].first, 
                                                         data_set[image_idx].first->TellHeight(),
                                                         data_set[image_idx].first->TellWidth());
        
        vector<float> one_image_features;
        one_image_features = hogFeatures(grayImage, 8, 1, 8);
        
        vector<vector<float>> tmp = cutImage(grayImage, 0, grayImage[0].size() / 2, 0, grayImage.size() / 2);
        one_image_features = concat(one_image_features, hogFeatures(tmp, 6, 1, 8));
        
        tmp = cutImage(grayImage, grayImage[0].size() / 2, grayImage[0].size(), 0, grayImage.size() / 2);
        one_image_features = concat(one_image_features, hogFeatures(tmp, 6, 1, 8));
        
        tmp = cutImage(grayImage, 0, grayImage[0].size() / 2, 
                                  grayImage.size() / 2, grayImage.size());
        one_image_features = concat(one_image_features, hogFeatures(tmp, 6, 1, 8));
        
        tmp = cutImage(grayImage, grayImage[0].size() / 2, grayImage[0].size(), 
                                  grayImage.size() / 2, grayImage.size());
        one_image_features = concat(one_image_features, hogFeatures(tmp, 6, 1, 8));
        
        one_image_features = concat(one_image_features, extractColorHist(data_set[image_idx].first));
        
        one_image_features = transformXiSquare(one_image_features);

        features->push_back(make_pair(one_image_features, data_set[image_idx].second));
        // End of sample code

    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.1;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}
