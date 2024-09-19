# Instructions

## Requirements:

### Install gdown

```
pip install gdown
```

### Download and unzip libraries

libtorch

```
gdown --fuzzy https://drive.google.com/file/d/1tOg1FicMdZ67UcE3mfYGUkmOdL8BhrHM/view
unzip libtorch.zip
rm libtorch.zip
```

opencv 3.4.2
```
gdown --fuzzy https://drive.google.com/file/d/1mef8VFm9AkIjy-QhNMXCkUy71PfQUn8j/view?usp=sharing
unzip opencv-3.4.2.zip
rm opencv-3.4.2.zip
```

## Build the application 
```
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Prepare model and input data
When a new model has been trained, serialize the model before using it during the inference.
Replace [model] and [test_image] with your data.
```
python serialize.py --tiny [model].net [test_image].png
```
The script generates a serialized model `traced_model.pt`. Copy this file to the `build` directory to use it for inference.

### Sample data
You may use the files `traced_model.pt` and `1721396597.png` in the `sample_data` directory to test inference. Simply copy them to the `build` directory.

## Executing the inference
```
./inference
```
If all goes well, your output should look like this:
```
pzhine@fubintlab-GP66:~/fubilab/vps-cpp/build$ ./inference
Loaded the model
Created tensor
Output
Sampling 64 hypotheses.
Done in 0.0168231s.
Calculating scores.
Done in 0.00849331s.
Drawing final hypothesis.
Soft inlier count: 1.98958 (Selection Probability: 4%)
Entropy of hypothesis distribution: 5.78862
Done in 4.6869e-05s.
Refining winning pose:
Done in 0.00269026s.
Pose: -0.3304 -0.3988 -0.8555  0.6897
-0.9389  0.0459  0.3412 -0.9717
-0.0968  0.9159 -0.3896  0.1742
 0.0000  0.0000  0.0000  1.0000
[ CPUFloatType{4,4} ]
```



