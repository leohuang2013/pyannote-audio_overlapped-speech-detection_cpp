# Compile
```
$> mkdir build && cd build
- Debug
$> cmake -DCMAKE_BUILD_TYPE=DEBUG ..
- Debug with address santinizer
$> cmake -DCMAKE_BUILD_TYPE=DEBUG -DADDR_SANT=ON ..

$> cmake ..
$> make
```
If want build GPU version
```
$> cmake -DGPU=ON ..
$> make
```

# Run
./speakerDiarizer ../../segment/segment2.onnx ../../embeddings/emd4.onnx ~/storage/sharedFolderVirtualbox/audioForTesting/english_15s16k.wav

# onnxruntime to GPU
It seems no need change code, instead set cuda when convert to model. For segment.onnx model, change source 
a) add following line to model = ....
model.cuda() 
b) change 
dummy_input = torch.zeros(3, 1, 32000)
-->
dummy_input = torch.zeros(3, 1, 32000).cuda()

change onnx.cmake to download gpu version of onnxruntime

# Verification
Since whole project is to translate pyannote-audio speaker diarization pipleline from python to C++, strategy I adopted here is 
write input/output of each small function in python to txt file, and do same for C++, then load txt file into python to compare 
and check difference. Target is to make each input and output is same.
For this purpose, script/verifyEveryStepResult.py is created.
``` bash
$> python verifyEveryStepResult.py
```
Above command is to compare txt files generated /tmp. and command below is to delete all the txt files.
``` bash
$> python verifyEveryStepResult.py clean
```

# Performance issue
Most time spent on embedding,
-----------
Segmenations time: 1061ms
-----------
Embedding time: 62759ms
-----------
Clustering time: 5ms
----Summary----
Time cost: 63869ms

and in embedding process, most time spent on STFT
----------------------------------------------------
STFT time: 31991ms
input shape:32x501x201x2
output shape:32x1x192
----------------------------------------------------
Embedding time: 2874ms
----------------------------------------------------
STFT time: 31033ms
input shape:32x501x201x2
output shape:32x1x192
----------------------------------------------------
Embedding time: 1579ms


# For hierichical clustering
tried following
- hclust-cpp/fastcluster
https://github.com/cdalitz/hclust-cpp
result is wrong, including distance of clusters and result 'fcluster'

- agglomerative-hierarchical-clustering
https://github.com/gyaikhom/agglomerative-hierarchical-clustering/tree/master
centroid_linkage is empty

- alglib
https://www.alglib.net/dataanalysis/clustering.php
does not support centriod
