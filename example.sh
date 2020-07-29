output_path=example_inference
dataset_path=data/synthetic/se2_randomwalk3
initial_sample=$output_path/initial_sample
sample_directory=$output_path/samples
visualization_directory=$output_path/visualization
numSamples=100
let drawIndex=$numSamples-1

mkdir $output_path
python script/runInitialization.py $dataset_path $initial_sample

mkdir $sample_directory
python script/runSampler.py $initial_sample $sample_directory $numSamples

python script/drawSample.py $sample_directory --sampleIdx $drawIndex --save $visualization_directory
