// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iomanip>
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <limits>
#include <algorithm>
#include <chrono>
#include <sstream>

#include "gflags/gflags.h"

#include "diarization/diarization_model.h"
#include "frontend/resampler.h"
#include "frontend/wav.h"

//#include "kmeans.h"


//#include <torch/torch.h>

#include "diarization/onnx_model.h"

#define SAMPLE_RATE 16000
#define NP_NAN std::numeric_limits<float>::infinity()
#define WRITE_DATA 1

// python: min_num_samples = self._embedding.min_num_samples
size_t min_num_samples = 640;
int g_sample_rate = 16000;



std::chrono::time_point<std::chrono::high_resolution_clock> timeNow()
{
    return std::chrono::high_resolution_clock::now();
}

void timeCost( std::chrono::time_point<std::chrono::high_resolution_clock> beg,
        std::string label )
{
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - beg);
    std::cout<<"-----------"<<std::endl;
    std::cout <<label<<": "<< duration.count()<<"ms"<<std::endl;
}

template<class T>
void debugPrint2D( const std::vector<std::vector<T>>& data, std::string dataInfo )
{
    std::cout<<"--- "<<dataInfo<<" ---"<<std::endl;
    for( const auto& a : data )
    {
        for( T b : a )
        {
            std::cout<<b<<",";
        }
        std::cout<<std::endl;
    }
}

template<class T>
void debugPrint( const std::vector<T>& data, std::string dataInfo )
{
    std::cout<<"--- "<<dataInfo<<" ---"<<std::endl;
    for( T b : data )
    {
        std::cout<<b<<",";
    }
    std::cout<<std::endl;
}

template<class T>
void debugWrite( const std::vector<T>& data, std::string name, bool writeDecimalforZero = false )
{
    std::string fileName = "/tmp/";
    fileName += name;
    fileName += ".txt";
    std::ofstream f( fileName );
    for( T b : data )
    {
        if( std::is_same<T, bool>::value )
        {
            std::string tmp = b ? "True" : "False";
            f<<tmp<<",";
        }
        else if( std::is_same<T, float>::value )
        {
            if( writeDecimalforZero )
            {
                std::ostringstream oss;
                oss << std::setprecision(1) << b;
                std::string result = oss.str();
                result = result == "1" ? "1.0" : result;
                result = result == "0" ? "0.0" : result;
                f<<result<<",";
            }
            else
            {
                f<<b<<",";
            }
        }
        else
        {
            f<<b<<",";
        }
    }
    f.close();
}

template<class T>
void debugWrite2d( const std::vector<std::vector<T>>& data, std::string name, bool writeDecimalforZero = false )
{
    std::string fileName = "/tmp/";
    fileName += name;
    fileName += ".txt";
    std::ofstream f( fileName );
    for( const auto& a : data )
    {
        for( T b : a )
        {
            if( std::is_same<T, bool>::value )
            {
                std::string tmp = b ? "True" : "False";
                f<<tmp<<",";
            }
            else if( std::is_same<T, float>::value )
            {
                if( writeDecimalforZero )
                {
                    std::ostringstream oss;
                    oss << std::setprecision(1) << b;
                    std::string result = oss.str();
                    result = result == "1" ? "1.0" : result;
                    result = result == "0" ? "0.0" : result;
                    f<<result<<",";
                }
                else
                {
                    f<<b<<",";
                }
            }
            else
            {
                f<<b<<",";
            }
        }
        f<<"\n";
    }
    f.close();
}

template<class T>
void debugWrite3d( const std::vector<std::vector<std::vector<T>>>& data, std::string name )
{
    std::string fileName = "/tmp/";
    fileName += name;
    fileName += ".txt";
    std::ofstream f( fileName );
    for( const auto& a : data )
    {
        for( const auto& b : a )
        {
            for( T c : b )
            {
                if( std::is_same<T, bool>::value )
                {
                    std::string tmp = c ? "True" : "False";
                    f<<tmp<<",";
                }
                else if( std::is_same<T, float>::value )
                {
                    std::ostringstream oss;
                    oss << std::setprecision(1) << c;
                    std::string result = oss.str();
                    result = result == "1" ? "1.0" : result;
                    result = result == "0" ? "0.0" : result;
                    f<<result<<",";
                }
                else
                {
                    f<<c<<",";
                }
            }
            f<<"\n";
        }
    }
    f.close();
}

class Segment
{
public:
    float start;
    float end;
    Segment( float start, float end )
        : start( start )
        , end( end )
    {}

    Segment& operator=( const Segment& other )
    {
        this->start = start;
        this->end = end;

        return *this;
    }

    float duration() const 
    {
        return end - start;
    }

    float gap( const Segment& other )
    {
        if( this->start <= other.end || 
                other.start <= this->end )
        {
            return 0.0f;
        }
        else
        {
            if( this->end < other.start )
            {
                return other.start - this->end;
            }
            else
            {
                return this->start - other.end;
            }
        }
    }

    Segment merge( const Segment& other )
    {
        Segment seg(0.0f, 0.0f);
        if( start <= other.start )
            seg.start = start;
        else
            seg.start = other.start;
        if( end >= other.end )
            seg.end = end;
        else
            seg.end = other.end;

        return seg;
    }
};

// Define a struct to represent annotations
struct Annotation 
{
    std::vector<Segment> segments;
    std::vector<int> labels;

    Annotation()
        : segments()
        , labels()
    {
    }

    void addSegment(float start, float end, int label) 
    {
        segments.push_back(Segment(start, end));
        labels.push_back(label);
    }

    Annotation& operator=( const Annotation& other )
    {
        segments = other.segments;
        labels = other.labels;

        return *this;
    }

    Annotation( Annotation&& other )
    {
        segments = std::move( other.segments );
        labels = std::move( other.labels );
    }

    // pyannote/core/annotation.py:1350
    void support( float collar )
    {
        assert( segments.size() == labels.size());
        if( segments.size() == 0 )
            return;

        std::vector<Segment> merged_segments;
        std::vector<int> merged_labels;
        Segment curSeg = segments[0];
        int curLabel = labels[0];
        for( size_t i = 1; i < labels.size(); ++i )
        {
            if( curLabel == labels[i] )
            {
                float gap = curSeg.gap( segments[i] );
                if( gap < collar )
                {
                    curSeg = curSeg.merge( segments[i] );
                }
                else
                {
                    merged_segments.push_back( curSeg );
                    merged_labels.push_back( curLabel );
                    curSeg = segments[i];
                    curLabel = labels[i];
                }
            }
            else
            {
                merged_segments.push_back( curSeg );
                merged_labels.push_back( curLabel );
                curSeg = segments[i];
                curLabel = labels[i];
            }
        }

        // Process remaining
        merged_segments.push_back( curSeg );
        merged_labels.push_back( curLabel );

        segments.swap( merged_segments );
        labels.swap( merged_labels );
    }

    void remove( size_t n )
    {
        segments.erase(segments.begin() + n);
        labels.erase(labels.begin() + n);
    }

    size_t size()
    {
        return segments.size();
    }
};

class SlidingWindow
{
public:
    float start;
    float step;
    float duration;
    size_t num_samples;
    float sample_rate;
    SlidingWindow()
        : start( 0.0f )
        , step( 0.0f )
        , duration( 0.0f )
        , num_samples( 0 )
        , sample_rate( 16000 )
    {
    }

    SlidingWindow( size_t num_samples )
        : start( 0.0f )
        , step( 0.0f )
        , duration( 0.0f )
        , num_samples( num_samples )
        , sample_rate( 16000 )
    {
    }

    SlidingWindow( float start, float step, float duration, size_t num_samples = 0 )
        : start( start )
        , step( step )
        , duration( duration )
        , num_samples( num_samples )
        , sample_rate( 16000 )
    {
    }

    SlidingWindow( const SlidingWindow& other )
    {
        start = other.start;
        step = other.step;
        duration = other.duration;
        num_samples = other.num_samples;
        sample_rate = other.sample_rate;
    }

    SlidingWindow& operator=( const SlidingWindow& other )
    {
        start = other.start;
        step = other.step;
        duration = other.duration;
        num_samples = other.num_samples;
        sample_rate = other.sample_rate;

        return *this;
    }

    size_t closest_frame( float start )
    {
        return std::round(( start - this->start - .5 * duration ) / step );
    }

    Segment operator[]( int pos ) const
    {
        int window_size = std::round(duration * sample_rate); // 80000
        int step_size = std::round(step * sample_rate); // 8000
        float start = 0.0;
        size_t cur_frames = 0;
        int index = 0;
        while( true )
        {
            if( index == pos )
                return Segment( start, start + duration );
            if( cur_frames + window_size >= num_samples )
            {
                break;
            }
            start += step;
            cur_frames += step_size;
            index++;
        }

        return Segment(0.0f, 0.0f);
    }

    std::vector<Segment> data()
    {
        std::vector<Segment> segments;
        int window_size = std::round(duration * sample_rate); // 80000
        int step_size = std::round(step * sample_rate); // 8000
        float start = 0.0;
        size_t cur_frames = 0;
        while( true )
        {
            Segment seg( start, start + duration );
            segments.push_back( seg );
            if( cur_frames + window_size >= num_samples )
            {
                break;
            }
            start += step;
            cur_frames += step_size;
        }

        return segments;
    }
};

class SlidingWindowFeature
{
public:
    std::vector<std::vector<std::vector<float>>> data;
    std::vector<std::pair<float, float>> slidingWindow;

    SlidingWindowFeature& operator=( const SlidingWindowFeature& other )
    {
        data = other.data;
        slidingWindow = other.slidingWindow;

        return *this;
    }

    SlidingWindowFeature( const SlidingWindowFeature& other )
    {
        data = other.data;
        slidingWindow = other.slidingWindow;
    }
};

class Helper 
{
public:

    static std::vector<int> argsort(const std::vector<float> &v) 
    {

        // initialize original index locations
        std::vector<int> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // sort indexes based on comparing values in v
        // using std::stable_sort instead of std::sort
        // to avoid unnecessary index re-orderings
        // when v contains elements of equal values
        std::stable_sort(idx.begin(), idx.end(),
                [&v](int i1, int i2) {return v[i1] < v[i2];});

        return idx;
    }

    // Define a helper function to find non-zero indices in a vector
    static std::vector<int> nonzeroIndices(const std::vector<bool>& input) {
        std::vector<int> indices;
        for (int i = 0; i < input.size(); ++i) {
            if (input[i]) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    // Function to compute the L2 norm of a vector
    static float L2Norm(const std::vector<float>& vec) {
        float sum = 0.0f;
        for (float val : vec) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }

    // Function to normalize a 2D vector
    static void normalizeEmbeddings(std::vector<std::vector<float>>& embeddings) {
        for (std::vector<float>& row : embeddings) {
            float norm = L2Norm(row);
            if (norm != 0.0f) {
                for (float& val : row) {
                    val /= norm;
                }
            }
        }
    }

    // Function to calculate the Euclidean distance between two vectors
    static float euclideanDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
        float sum = 0.0;
        for (size_t i = 0; i < vec1.size(); ++i) {
            float diff = static_cast<float>(vec1[i]) - static_cast<float>(vec2[i]);
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // Function to perform hierarchical clustering using the centroid method
    static std::vector<std::vector<float>> hierarchicalClustering(const std::vector<std::vector<float>>& embeddings) 
    {
        size_t numSamples = embeddings.size();
        size_t numFeatures = embeddings[0].size();

        std::vector<std::vector<float>> dendrogram( numSamples - 1, std::vector<float>( 4, 0.0 ));

        std::vector<std::pair<size_t, float>> clusterSizes(numSamples);
        for (size_t i = 0; i < numSamples; ++i) {
            clusterSizes[i] = {i, 1.0};
        }

        size_t nextClusterIndex = numSamples;
        for (size_t step = 0; step < numSamples - 1; ++step) {
            float minDistance = std::numeric_limits<float>::max();
            size_t minCluster1 = 0;
            size_t minCluster2 = 0;

            for (size_t i = 0; i < numSamples; ++i) {
                for (size_t j = i + 1; j < numSamples; ++j) {
                    float distance = 0.0;
                    for (size_t k = 0; k < numFeatures; ++k) {
                        distance += euclideanDistance(embeddings[i], embeddings[j]);
                    }
                    distance /= static_cast<float>(numFeatures);

                    if (distance < minDistance) {
                        minDistance = distance;
                        minCluster1 = i;
                        minCluster2 = j;
                    }
                }
            }

            // Update the dendrogram
            dendrogram[step][0] = static_cast<float>(minCluster1);
            dendrogram[step][1] = static_cast<float>(minCluster2);
            dendrogram[step][2] = minDistance;
            dendrogram[step][3] = clusterSizes[minCluster1].second + clusterSizes[minCluster2].second;

            // Merge clusters
            clusterSizes.push_back({nextClusterIndex, dendrogram[step][3]});
            nextClusterIndex++;

            // Remove old clusters
            clusterSizes.erase(std::remove_if(clusterSizes.begin(), clusterSizes.end(),
                [minCluster1, minCluster2](const std::pair<size_t, float>& cluster) {
                    return cluster.first == minCluster1 || cluster.first == minCluster2;
                }), clusterSizes.end());
        }

        return dendrogram;
    }

    // Function to extract clusters from dendrogram based on a distance threshold
    static std::vector<int> extractClusters(const std::vector<std::vector<float>>& dendrogram, double threshold) 
    {
        std::vector<int> clusters(dendrogram.size(), -1);

        int nextClusterIndex = 0;

        for (size_t i = 0; i < dendrogram.size(); ++i) {
            int cluster1 = static_cast<int>(dendrogram[i][0]);
            int cluster2 = static_cast<int>(dendrogram[i][1]);

            if (cluster1 >= 0 && cluster1 < dendrogram.size()) {
                clusters[cluster1] = nextClusterIndex;
            }
            if (cluster2 >= 0 && cluster2 < dendrogram.size()) {
                clusters[cluster2] = nextClusterIndex;
            }

            nextClusterIndex++;
        }

        for (size_t i = 0; i < dendrogram.size(); ++i) {
            if (clusters[i] == -1) {
                clusters[i] = nextClusterIndex++;
            }
        }

        return clusters;
    }

    // Function to calculate the mean of embeddings for large clusters
    static std::vector<std::vector<float>> calculateClusterMeans(const std::vector<std::vector<float>>& embeddings,
                                                 const std::vector<int>& clusters,
                                                 const std::vector<int>& largeClusters) 
    {
        std::vector<std::vector<float>> clusterMeans;

        for (int large_k : largeClusters) {
            std::vector<float> meanEmbedding( embeddings[0].size(), 0.0f );
            int count = 0;
            for (size_t i = 0; i < clusters.size(); ++i) {
                if (clusters[i] == large_k) {
                    // Add the embedding to the mean
                    for (size_t j = 0; j < meanEmbedding.size(); ++j) {
                        meanEmbedding[j] += embeddings[i][j];
                    }
                    count++;
                }
            }

            // Calculate the mean by dividing by the count
            if (count > 0) {
                for (size_t j = 0; j < meanEmbedding.size(); ++j) {
                    meanEmbedding[j] /= static_cast<float>(count);
                }
            }

            clusterMeans.push_back(meanEmbedding);
        }

        return clusterMeans;
    }

    // Function to calculate the cosine distance between two vectors
    static float cosineDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) 
    {
        if (vec1.size() != vec2.size()) {
            throw std::runtime_error("Vector sizes must be equal.");
        }

        float dotProduct = 0.0;
        float magnitude1 = 0.0;
        float magnitude2 = 0.0;

        for (size_t i = 0; i < vec1.size(); ++i) {
            dotProduct += static_cast<float>(vec1[i]) * static_cast<float>(vec2[i]);
            magnitude1 += static_cast<float>(vec1[i]) * static_cast<float>(vec1[i]);
            magnitude2 += static_cast<float>(vec2[i]) * static_cast<float>(vec2[i]);
        }

        if (magnitude1 == 0.0 || magnitude2 == 0.0) {
            throw std::runtime_error("Vectors have zero magnitude.");
        }

        return 1.0 - (dotProduct / (std::sqrt(magnitude1) * std::sqrt(magnitude2)));
    }

    // Calculate cosine distances between large and small cluster means
    static std::vector<std::vector<float>> cosineSimilarity( std::vector<std::vector<float>>& largeClusterMeans,
            std::vector<std::vector<float>>& smallClusterMeans )
    {

        std::vector<std::vector<float>> centroidsCdist( largeClusterMeans.size(),
                std::vector<float>( smallClusterMeans.size()));
        for (size_t i = 0; i < largeClusterMeans.size(); ++i) {
            for (size_t j = 0; j < smallClusterMeans.size(); ++j) {
                float distance = cosineDistance(largeClusterMeans[i], smallClusterMeans[j]);
                centroidsCdist[i][j] = distance;
            }
        }

        return centroidsCdist;
    }

    // Function to find unique clusters and return the inverse mapping
    static std::vector<int> findUniqueClusters(const std::vector<int>& clusters,
                                        std::vector<int>& uniqueClusters) 
    {
        std::vector<int> inverseMapping(clusters.size(), -1);
        int nextClusterIndex = 0;

        for (size_t i = 0; i < clusters.size(); ++i) {
            if (inverseMapping[clusters[i]] == -1) {
                inverseMapping[clusters[i]] = nextClusterIndex;
                uniqueClusters.push_back(clusters[i]);
                nextClusterIndex++;
            }
        }

        return inverseMapping;
    }

    // python: embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)
    static std::vector<std::vector<std::vector<float>>> rearrange_up( const std::vector<std::vector<float>>& input, int c )
    {
        assert( input.size() > c );
        assert( input.size() % c == 0 );
        size_t dim1 = c;
        size_t dim2 = input.size() / c;
        size_t dim3 = input[0].size();
        std::vector<std::vector<std::vector<float>>> output(c, 
                std::vector<std::vector<float>>( dim2, std::vector<float>(dim3)));
        int rowNum = 0;
        for( size_t i = 0; i < dim1; i += dim2 )
        {
            for( size_t j = 0; j < dim2; ++j )
            {
                for( size_t k = 0; k < dim3; ++k )
                {
                    output[rowNum][j][k] = input[i+j][k];
                }
            }
            rowNum++;
        }

        return output;
    }

    // Imlemenation of einops.rearrange c f k -> (c k) f
    static std::vector<std::vector<float>> rearrange_down( const std::vector<std::vector<std::vector<float>>>& input )
    {
        int num_chunks = input.size();
        int num_frames = input[0].size();
        int num_classes = input[0][0].size();
        std::vector<std::vector<float>> data(num_chunks * num_classes, std::vector<float>(num_frames));
        int rowNum = 0;
        for ( const auto& row : input ) 
        {
            // Create a new matrix with swapped dimensions
            std::vector<std::vector<float>> transposed(num_classes, std::vector<float>(num_frames));

            for (int i = 0; i < num_frames; ++i) {
                for (int j = 0; j < num_classes; ++j) {
                    data[rowNum * num_classes + j][i] = row[i][j];
                }
            }

            rowNum++;
        }

        return data;
    }

    static std::vector<std::vector<int>> wellDefinedIndex( const std::vector<std::vector<bool>>& off_or_on ) 
    {
        // Find the indices of True values in each row and store them in a vector of vectors
        size_t max_indices = 0;
        std::vector<std::vector<int>> nonzero_indices;
        for (const auto& row : off_or_on) {
            std::vector<int> indices = nonzeroIndices(row);
            if( indices.size() > max_indices )
                max_indices = indices.size();
            nonzero_indices.push_back(indices);
        }

        // Fill missing indices with -1 and create the well_defined_idx vector of vectors
        std::vector<std::vector<int>> well_defined_idx;
        for (const auto& indices : nonzero_indices) {
            if( indices.size() < max_indices )
            {
                std::vector<int> filled_indices(max_indices, -1);
                std::copy(indices.begin(), indices.end(), filled_indices.begin());
                well_defined_idx.push_back(filled_indices);
            }
            else
            {
                well_defined_idx.push_back( indices );
            }
        }

        return well_defined_idx;
    }

    // Function to calculate cumulative sum along axis=1
    static std::vector<std::vector<int>> cumulativeSum(const std::vector<std::vector<bool>>& input) 
    {
        std::vector<std::vector<int>> cumsum;

        for (const auto& row : input) {
            std::vector<int> row_cumsum;
            int running_sum = 0;

            for (bool val : row) {
                running_sum += val ? 1 : 0;
                row_cumsum.push_back(running_sum);
            }

            cumsum.push_back(row_cumsum);
        }

        return cumsum;
    }

    // Define a helper function to calculate np.where
    static std::vector<std::vector<bool>> numpy_where(const std::vector<std::vector<int>>& same_as,
                                    const std::vector<std::vector<bool>>& on,
                                    const std::vector<std::vector<int>>& well_defined_idx,
                                    const std::vector<std::vector<bool>>& initial_state,
                                    const std::vector<std::vector<int>>& samples) 
    {
        assert( same_as.size() == on.size());
        assert( same_as.size() == well_defined_idx.size());
        assert( same_as.size() == initial_state.size());
        assert( same_as.size() == samples.size());
        assert( same_as[0].size() == on[0].size());
        assert( same_as[0].size() == well_defined_idx[0].size());
        assert( same_as[0].size() == initial_state[0].size());
        assert( same_as[0].size() == samples[0].size());
        std::vector<std::vector<bool>> result( same_as.size(), std::vector<bool>( same_as[0].size(), false ));
        for( size_t i = 0; i < same_as.size(); ++i )
        {
            for( size_t j = 0; j < same_as[0].size(); ++j )
            {
                if( same_as[i][j] > 0 )
                {
                    int x = samples[i][j];
                    int y = well_defined_idx[x][same_as[i][j]-1];
                    result[i][j] = on[x][y];
                }
                else
                {
                    result[i][j] = initial_state[i][j];
                }
            }
        }


        return result;
    }

    static std::vector<std::vector<std::vector<float>>> cleanSegmentations(const std::vector<std::vector<std::vector<float>>>& data)
    {
        size_t numRows = data.size();
        size_t numCols = data[0].size();
        size_t numChannels = data[0][0].size();

        // Initialize the result with all zeros
        std::vector<std::vector<std::vector<float>>> result(numRows, std::vector<std::vector<float>>(numCols, std::vector<float>(numChannels, 0.0)));
        for (int i = 0; i < numRows; ++i) 
        {
            for (int j = 0; j < numCols; ++j) 
            {
                float sum = 0.0;
                for (int k = 0; k < numChannels; ++k) 
                {
                    sum += data[i][j][k];
                }
                bool keep = false;
                if( sum < 2.0 )
                {
                    keep = true;
                }
                for (int k = 0; k < numChannels; ++k) 
                {
                    if( keep )
                        result[i][j][k] = data[i][j][k];
                }
            }
        }

        return result;
    }

    // Define a function to interpolate 2D arrays (nearest-neighbor interpolation)
    static std::vector<std::vector<bool>> interpolate(const std::vector<std::vector<float>>& masks, int num_samples, float threshold ) 
    {
        int inputHeight = masks.size();
        int inputWidth = masks[0].size();

        std::vector<std::vector<bool>> output(inputHeight, std::vector<bool>(num_samples, false));
        assert( num_samples > inputWidth );
        int scale = num_samples / inputWidth;

        for (int i = 0; i < inputHeight; ++i) 
        {
            for (int j = 0; j < num_samples; ++j) 
            {
                int src_y = j * inputWidth / num_samples;
                if( masks[i][src_y] > threshold )
                    output[i][j] = true;
            }
        }

        return output;
    }

    // Define a function to perform pad_sequence
    static std::vector<std::vector<float>> padSequence(const std::vector<std::vector<float>>& waveforms,
                                                const std::vector<std::vector<bool>>& imasks) {
        // Find the maximum sequence length
        size_t maxLen = 0;
        for (const std::vector<bool>& mask : imasks) 
        {
            maxLen = std::max(maxLen, mask.size());
        }

        // Initialize the padded sequence with zeros
        std::vector<std::vector<float>> paddedSequence(waveforms.size(), std::vector<float>(maxLen, 0.0f));

        // Copy the valid data from waveforms based on imasks
        for (size_t i = 0; i < waveforms.size(); ++i) 
        {
            size_t validIndex = 0;
            for (size_t j = 0; j < imasks[i].size(); ++j) 
            {
                if (imasks[i][j]) 
                {
                    paddedSequence[i][validIndex++] = waveforms[i][j];
                }
            }
        }

        return paddedSequence;
    }

    // pyannote/audio/core/inference.py:411
    // we ignored warm_up parameter since our case use default value( 0.0, 0.0 )
    // so hard code warm_up
    static std::vector<std::vector<float>> aggregate( 
            const std::vector<std::vector<std::vector<float>>>& scores, 
            const SlidingWindow& scores_frames, SlidingWindow& count_frames,
            float epsilon = 1e-9, bool hamming = false, float missing = 0.0, 
            bool skip_average = false )
    {
        size_t num_chunks = scores.size(); 
        size_t num_frames_per_chunk = scores[0].size(); 
        size_t num_classes = scores[0][0].size(); 
        size_t num_samples = scores_frames.num_samples;
        assert( num_samples > 0 );

        // dont copy data here, just 
        std::vector<std::vector<std::vector<float>>> masks( num_chunks, 
                std::vector<std::vector<float>>( num_frames_per_chunk, std::vector<float>( num_classes, 1.0 )));

        // Replace NaN values in scores with 0 and update masks
        // **********************
        // Here probally got problem, because in python there are specially value:
        // Nan to represent 'No Value', but in c++, previous steps, what we put 
        // for those values, need check....
        // if need address that issue, also need convert following python code,
        // scores.data = np.nan_to_num(scores.data, copy=True, nan=0.0)
        // **********************
        for (size_t i = 0; i < scores.size(); ++i) 
        {
            for (size_t j = 0; j < scores[0].size(); ++j) 
            {
                for( size_t k = 0; k < scores[0][0].size(); ++k )
                {
                    if (std::isnan(scores[i][j][k])) 
                    {
                        masks[i][j][k] = 0.0;
                    }
                }
            }
        }

        if( !hamming )
        {
            // python np.ones((num_frames_per_chunk, 1))
            // no need create it, later will directly apply 1 to computation
        }
        else
        {
            // python: np.hamming(num_frames_per_chunk).reshape(-1, 1)
            assert( false ); // no implemented
        }

        // Get frames, we changed this part. In pyannote, it calc frames(self._frames) before calling
        // this function, but in this function, it creates new frames and use it.
        // step = (self.inc_num_samples / self.inc_num_frames) / sample_rate
        // pyannote/audio/core/model.py:243
        // currently cannot find where self.inc_num_samples / self.inc_num_frames from
        int inc_num_samples = 270; // <-- this may not be correct
        int inc_num_frames = 1; // <-- this may not be correct
        float frames_step = ( inc_num_samples * 1.0f / inc_num_frames) / g_sample_rate;
        float frames_duration = frames_step;
        float frames_start = scores_frames.start;
        /*
        float start = scores_frames.start;
        std::vector<std::pair<float, float>> frames;
        while( true )
        {
            start += frames_step;
            if( start * g_sample_rate >= num_samples )
                break;
            float end = start + frames_duration;
            frames.emplace_back(std::make_pair(start, end));
        }
        */

        // aggregated_output[i] will be used to store the sum of all predictions
        // for frame #i
        // python: num_frames = ( frames.closest_frame(...)) + 1
        float frame_target = scores_frames.start + scores_frames.duration + (num_chunks - 1) * scores_frames.step;
        SlidingWindow frames( frames_start, frames_step, frames_duration );
        //size_t num_frames = (frame_target - frames_start - .5 * frames_duration) / frames_step + 1;
        size_t num_frames = frames.closest_frame( frame_target ) + 1;

        // python: aggregated_output: np.ndarray = np.zeros(...
        std::vector<std::vector<float>> aggregated_output(num_frames, std::vector<float>( num_classes, 0.0f ));

        // overlapping_chunk_count[i] will be used to store the number of chunks
        // that contributed to frame #i
        std::vector<std::vector<float>> overlapping_chunk_count(num_frames, std::vector<float>( num_classes, 0.0f ));

        // aggregated_mask[i] will be used to indicate whether
        // at least one non-NAN frame contributed to frame #i
        std::vector<std::vector<float>> aggregated_mask(num_frames, std::vector<float>( num_classes, 0.0f ));
        
        // for our use case, warm_up_window and hamming_window all 1
        float start = scores_frames.start;
        for( size_t i = 0; i < scores.size(); ++i )
        {
            size_t start_frame = frames.closest_frame( start );
            std::cout<<"start_frame: "<<start_frame<<std::endl;
            start += scores_frames.step; // python: chunk.start
            for( size_t j = start_frame; j < start_frame + num_frames_per_chunk; ++j )
            {
                size_t _j = j - start_frame;
                for( size_t k = 0; k < num_classes; ++k )
                {
                    // score * mask * hamming_window * warm_up_window
                    aggregated_output[j][k] += scores[i][_j][k] * masks[i][_j][k];
                    overlapping_chunk_count[j][k] += masks[i][_j][k];
                    if( masks[i][_j][k] > aggregated_mask[j][k] )
                    {
                        aggregated_mask[j][k] = masks[i][_j][k];
                    }
                }
            }
        }

        count_frames.start = frames_start;
        count_frames.step = frames_step;
        count_frames.duration = frames_duration;
        count_frames.num_samples = num_samples;
        if( !skip_average )
        {
            for( size_t i = 0; i < aggregated_output.size(); ++i )
            {
                for( size_t j = 0; j < aggregated_output[0].size(); ++j )
                {
                    aggregated_output[i][j] /= std::max( overlapping_chunk_count[i][j], 1e-9f );
                }
            }
        }
        else
        {
            // do nothing
        }

        // average[aggregated_mask == 0.0] = missing
        for( size_t i = 0; i < aggregated_output.size(); ++i )
        {
            for( size_t j = 0; j < aggregated_output[0].size(); ++j )
            {
                if( abs( aggregated_mask[i][j] ) < 1e-9 )
                {
                    aggregated_output[i][j] = 1e-9;
                }
            }
        }

        return aggregated_output;

    }

};

class SegmentModel : public OnnxModel 
{
private:
    float m_duration = 5.0;
    float m_step = 0.5;
    int m_batch_size = 32;
    int m_sample_rate = 16000;
    float m_diarization_segmentation_threashold = 0.4442333667381752;
    float m_diarization_segmentation_min_duration_off = 0.5817029604921046;
    size_t m_num_samples = 0;


public:
    SegmentModel(const std::string& model_path)
        : OnnxModel(model_path) {
    }


    // input: batch size x channel x samples count, for example, 32 x 1 x 80000
    // output: batch size x 293 x 3
    std::vector<std::vector<std::vector<float>>> infer( const std::vector<std::vector<float>>& waveform )
    {
        // Create a std::vector<float> with the same size as the tensor
        std::vector<float> audio( waveform.size() * waveform[0].size());
        for( size_t i = 0; i < waveform.size(); ++i )
        {
            for( size_t j = 0; j < waveform[0].size(); ++j )
            {
                audio[i*waveform[0].size() + j] = waveform[i][j];
            }
        }

        // batch_size * num_channels (1 for mono) * num_samples
        const int64_t batch_size = waveform.size();
        const int64_t num_channels = 1;
        int64_t input_node_dims[3] = {batch_size, num_channels,
            static_cast<int64_t>(waveform[0].size())};
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
                memory_info_, const_cast<float*>(audio.data()), audio.size(),
                input_node_dims, 3);
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.emplace_back(std::move(input_ort));

        auto ort_outputs = session_->Run(
                Ort::RunOptions{nullptr}, input_node_names_.data(), ort_inputs.data(),
                ort_inputs.size(), output_node_names_.data(), output_node_names_.size());

        const float* outputs = ort_outputs[0].GetTensorData<float>();
        auto outputs_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int len1 = outputs_shape[0];
        int len2 = outputs_shape[1];
        int len3 = outputs_shape[2];
        std::cout<<"output shape:"<<len1<<"x"<<len2<<"x"<<len3<<std::endl;

        std::vector<std::vector<std::vector<float>>> res( len1, std::vector<std::vector<float>>( len2, std::vector<float>( len3 )));
        for( int i = 0; i < len1; ++i )
        {
            for( int j = 0; j < len2; ++j )
            {
                for( int k = 0; k < len3; ++k )
                {
                    res[i][j][k] = *( outputs + i * len2 * len3 + j * len3 + k );
                }
            }
        }

        return res;
    }

    // segment is 2 dimensional segment which indicates start and end of each segment. e.g. (0.0 - 5.0), (0.5 - 5.5) ...
    // pyannote/audio/core/inference.py:202
    std::vector<std::vector<std::vector<float>>> slide(const std::vector<float>& waveform, 
            SlidingWindow& res_frames )
    {
        int sample_rate = 16000;
        int window_size = std::round(m_duration * sample_rate); // 80000
        int step_size = std::round(m_step * sample_rate); // 8000
        int num_channels = 1;
        size_t num_samples = waveform.size();
        int num_frames_per_chunk = 293; // Need to check with multiple wave files
        size_t i = 0;
        std::vector<std::vector<float>> chunks;
        std::vector<std::vector<std::vector<float>>> outputs;
        while( i + window_size < num_samples )
        {
            // Starting and Ending iterators
            auto start = waveform.begin() + i;
            auto end = start + window_size;

            // To store the sliced vector
            std::vector<float> chunk( window_size, 0.0 );

            // Copy vector using copy function()
            std::copy(start, end, chunk.begin());
            chunks.push_back( chunk ); 
            if( chunks.size() == m_batch_size )
            {
                auto tmp = infer( chunks );
                for( const auto& a : tmp )
                {
                    outputs.push_back( a );
                }
                chunks.clear();
            }

            i += step_size;
        }

        // Process remaining chunks
        if( chunks.size() > 0 )
        {
            auto tmp = infer( chunks );
            for( const auto& a : tmp )
            {
                outputs.push_back( a );
            }
            chunks.clear();
        }

        // Process last chunk if have, last chunk may not equal window_size
        // Make sure at least we have 1 element remaining
        if( i + 1 < num_samples )
        {
            // Starting and Ending iterators
            auto start = waveform.begin() + i;
            auto end = waveform.end();

            // To store the sliced vector, always window_size, for last chunk we pad with 0.0
            std::vector<float> chunk( end - start, 0.0 );

            // Copy vector using copy function()
            std::copy(start, end, chunk.begin());
            chunks.push_back( chunk ); 
            auto tmp = infer( chunks );
            assert( tmp.size() == 1 );

            // Padding
            auto a = tmp[0];
            for( size_t i = a.size(); i < num_frames_per_chunk;  ++i )
            {
                std::vector<float> pad( a[0].size(), 0.0f );
                a.push_back( pad );
            }
            outputs.push_back( a );
        }

        // Calc segments
        res_frames.start = 0.0;
        res_frames.step = m_step;
        res_frames.duration = m_duration;
        res_frames.num_samples = num_samples;
        /*
        float start = 0.0;
        size_t cur_frames = 0;
        while( true )
        {
            std::pair<float, float> seg = { start, start + m_duration };
            segments.push_back( seg );
            if( cur_frames + window_size >= num_samples )
            {
                break;
            }
            start += m_step;
            cur_frames += step_size;
        }
        */

        return outputs;
    }

    std::vector<std::vector<std::vector<float>>> binarize_swf(
        const std::vector<std::vector<std::vector<float>>> scores,
        bool initial_state = false ) 
    {
        float onset = m_diarization_segmentation_threashold;

        // TODO: use hlper::rerange_down
        // Imlemenation of einops.rearrange c f k -> (c k) f
        int num_chunks = scores.size();
        int num_frames = scores[0].size();
        int num_classes = scores[0][0].size();
        std::vector<std::vector<float>> data(num_chunks * num_classes, std::vector<float>(num_frames));
        int rowNum = 0;
        for ( const auto& row : scores ) 
        {
            // Create a new matrix with swapped dimensions
            std::vector<std::vector<float>> transposed(num_classes, std::vector<float>(num_frames));

            for (int i = 0; i < num_frames; ++i) {
                for (int j = 0; j < num_classes; ++j) {
                    data[rowNum * num_classes + j][i] = row[i][j];
                }
            }

            rowNum++;
        }
        /*
        for( const auto& d : data )
        {
            for( float e : d )
            {
                std::cout<<e<<",";
            }
            std::cout<<std::endl;
        }
        */

        auto binarized = binarize_ndarray( data, onset, initial_state);

        // TODO: use help::rerange_up
        // Imlemenation of einops.rearrange (c k) f -> c f k - restore
        std::vector<std::vector<std::vector<float>>> restored(num_chunks, std::vector<std::vector<float>>( num_frames, std::vector<float>(num_classes)));
        rowNum = 0;
        for( size_t i = 0; i < binarized.size(); i += num_classes )
        {
            for( size_t j = 0; j < num_classes; ++j )
            {
                for( size_t k = 0; k < num_frames; ++k )
                {
                    restored[rowNum][k][j] = binarized[i+j][k];
                }
            }
            rowNum++;
        }

        return restored;
    }

    std::vector<std::vector<bool>> binarize_ndarray(
        const std::vector<std::vector<float>>& scores,
        float onset = 0.5,
        bool initialState = false
    ) {

        // Scores shape like 2808x293
        size_t rows = scores.size();
        size_t cols = scores[0].size();

        // python: on = scores > onset
        // on is same shape as scores, with true or false inside
        std::vector<std::vector<bool>> on( rows, std::vector<bool>( cols, false ));
        for( size_t i = 0; i < rows; ++i )
        {
            for( size_t j = 0; j < cols; ++j )
            {
                if( scores[i][j] > onset )
                    on[i][j] = true;
            }
        }

        // python: off_or_on = (scores < offset) | on
        // off_or_on is same shape as scores, with true or false inside
        // Since onset and offset is same value, it should be true unless score[i][j] == onset
        std::vector<std::vector<bool>> off_or_on( rows, std::vector<bool>( cols, true ));
        for( size_t i = 0; i < rows; ++i )
        {
            for( size_t j = 0; j < cols; ++j )
            {
                if(abs( scores[i][j] - onset ) < 1e-9 )
                    off_or_on[i][j] = false;
            }
        }

        // python: # indices of frames for which the on/off state is well-defined
        // well_defined_idx = np.array(
        //     list(zip_longest(*[np.nonzero(oon)[0] for oon in off_or_on], fillvalue=-1))
        // ).T
        auto well_defined_idx = Helper::wellDefinedIndex( off_or_on );

        // same_as same shape of as scores
        // python: same_as = np.cumsum(off_or_on, axis=1)
        auto same_as = Helper::cumulativeSum( off_or_on );

        // python: samples = np.tile(np.arange(batch_size), (num_frames, 1)).T
        std::vector<std::vector<int>> samples( rows, std::vector<int>( cols, 0 ));
        for( size_t i = 0; i < rows; ++i )
        {
            for( size_t j = 0; j < cols; ++j )
            {
                samples[i][j] = i;
            }
        }

        // create same shape of initial_state as scores.
        std::vector<std::vector<bool>> initial_state( rows, std::vector<bool>( cols, initialState ));


        // python: return np.where( same_as, on[samples, well_defined_idx[samples, same_as - 1]], initial_state)
        // TODO: delete tmp, directly return
#ifdef WRITE_DATA
        debugWrite2d( same_as, "cpp_same_as" );
        debugWrite2d( on, "cpp_on" );
        debugWrite2d( well_defined_idx, "cpp_well_defined_idx" );
        debugWrite2d( initial_state, "cpp_initial_state" );
        debugWrite2d( samples, "cpp_samples" );
#endif // WRITE_DATA
        auto tmp = Helper::numpy_where( same_as, on, well_defined_idx, initial_state, samples );
#ifdef WRITE_DATA
        debugWrite2d( tmp, "cpp_binary_ndarray" );
#endif // WRITE_DATA
        return tmp;
    }

    std::vector<float> crop( const std::vector<float>& waveform, std::pair<float, float> segment) 
    {
        int start_frame = static_cast<int>(std::floor(segment.first * m_sample_rate));
        int frames = static_cast<int>(waveform.size());

        int num_frames = static_cast<int>(std::floor(m_duration * m_sample_rate));
        int end_frame = start_frame + num_frames;

        int pad_start = -std::min(0, start_frame);
        int pad_end = std::max(end_frame, frames) - frames;
        start_frame = std::max(0, start_frame);
        end_frame = std::min(end_frame, frames);
        num_frames = end_frame - start_frame;

        std::vector<float> data(waveform.begin() + start_frame, waveform.begin() + end_frame);

        // Pad with zeros
        data.insert(data.begin(), pad_start, 0.0f);
        data.insert(data.end(), pad_end, 0.0f);

        return data;
    }

    // pyannote/audio/pipelines/utils/diarization.py:108
    std::vector<int> speaker_count( const std::vector<std::vector<std::vector<float>>>& segmentations,
            const std::vector<std::vector<std::vector<float>>>& binarized,
            SlidingWindow& count_frames,
            int num_samples )
    {
        // Get frames first - python: self._frames
        // step = (self.inc_num_samples / self.inc_num_frames) / sample_rate
        // pyannote/audio/core/model.py:243
        // currently cannot where where self.inc_num_samples / self.inc_num_frames from
        //int inc_num_samples = 270; // <-- this may not be correct
        //int inc_num_frames = 1; // <-- this may not be correct
        //float step = ( inc_num_samples * 1.0f / inc_num_frames) / m_sample_rate;
        //float window = step;
        //std::vector<std::pair<float, float>> frames;
        //float start = 0.0f;
        //while( true )
        //{
        //    start += step;
        //    if( start * m_sample_rate >= num_samples )
        //        break;
        //    float end = start + window;
        //    frames.emplace_back(std::make_pair<float, float>(start, end));
        //}

        // python: trimmed = Inference.trim
        SlidingWindow trimmed_frames;
        SlidingWindow frames( 0.0, m_step, m_duration );
        auto trimmed = trim( binarized, 0.1, 0.1, frames, trimmed_frames );

#ifdef WRITE_DATA
        debugWrite3d( trimmed, "cpp_trimmed" );
#endif // WRITE_DATA

        // python: count = Inference.aggregate(
        // python: np.sum(trimmed, axis=-1, keepdims=True)
        std::vector<std::vector<std::vector<float>>> sum_trimmed( trimmed.size(), 
                std::vector<std::vector<float>>( trimmed[0].size(), std::vector<float>( 1 )));
        for( size_t i = 0; i < trimmed.size(); ++i )
        {
            for( size_t j = 0; j < trimmed[0].size(); ++j )
            {
                float sum = 0.0f;
                for( size_t k = 0; k < trimmed[0][0].size(); ++k )
                {
                    sum += trimmed[i][j][k];
                }
                sum_trimmed[i][j][0] = sum;
            }
        }
#ifdef WRITE_DATA
        debugWrite3d( sum_trimmed, "cpp_sum_trimmed" );
#endif // WRITE_DATA
       
        auto count_data = Helper::aggregate( sum_trimmed, trimmed_frames, count_frames );

#ifdef WRITE_DATA
        debugWrite2d( count_data, "cpp_count_data" );
#endif // WRITE_DATA
       
        // count_data is Nx1, so we convert it to 1d array
        assert( count_data[0].size() == 1 );

        // python: count.data = np.rint(count.data).astype(np.uint8)
        //std::vector<std::vector<int>> res( count_data.size(), std::vector<int>( count_data[0].size()));
        std::vector<int> res( count_data.size());
        for( size_t i = 0; i < res.size(); ++i )
        {
            res[i] = static_cast<int>( count_data[i][0] );
        }

        return res;
    }

    // pyannote/audio/core/inference.py:540
    // use after_trim_step, after_trim_duration to calc sliding_window later 
    std::vector<std::vector<std::vector<float>>> trim(
            const std::vector<std::vector<std::vector<float>>>& binarized, 
            float left, float right, 
            const SlidingWindow& before_trim, SlidingWindow& trimmed_frames )
    {
        float before_trim_start = before_trim.start;
        float before_trim_step = before_trim.step;
        float before_trim_duration = before_trim.duration;
        size_t chunkSize = binarized.size();
        size_t num_frames = binarized[0].size();

        // python: num_frames_left = round(num_frames * warm_up[0])
        size_t num_frames_left = floor(num_frames * left);

        // python: num_frames_right = round(num_frames * warm_up[1])
        size_t num_frames_right = floor(num_frames * right);
        size_t num_frames_step = floor(num_frames * before_trim_step / before_trim_duration);

        // python: new_data = scores.data[:, num_frames_left : num_frames - num_frames_right]
        std::vector<std::vector<std::vector<float>>> trimed( binarized.size(), 
                std::vector<std::vector<float>>((num_frames - num_frames_right - num_frames_left), 
                std::vector<float>( binarized[0][0].size())));
        for( size_t i = 0; i < binarized.size(); ++i )
        {
            for( size_t j = num_frames_left; j < num_frames - num_frames_right; ++j )
            {
                for( size_t k = 0; k < binarized[0][0].size(); ++k )
                {
                    trimed[i][j - num_frames_left][k] = binarized[i][j][k];
                }
            }
        }

        trimmed_frames.start = before_trim_start + left * before_trim_duration;
        trimmed_frames.step = before_trim_step;
        trimmed_frames.duration = ( 1 - left - right ) * before_trim_duration;
        trimmed_frames.num_samples = num_frames - num_frames_right - num_frames_left;

        return trimed;
    }

}; // SegmentModel


class EmbeddingModel : public OnnxModel 
{
private:


public:
    EmbeddingModel(const std::string& model_path)
        : OnnxModel(model_path) {
    }

    // input: batch size x samples count, and wav_lens is 1d array. for example, 32 x 80000,
    // output: batch size x 192 
    std::vector<std::vector<float>> infer( const std::vector<std::vector<float>>& waveform,
            const std::vector<float>& wav_lens )
    {
        assert( waveform.size() == wav_lens.size());
        // Create a std::vector<float> with the same size as the tensor
        std::vector<float> audio( waveform.size() * waveform[0].size());
        for( size_t i = 0; i < waveform.size(); ++i )
        {
            for( size_t j = 0; j < waveform[0].size(); ++j )
            {
                audio[i*waveform[0].size() + j] = waveform[i][j];
            }
        }

        // batch_size * num_samples
        const int64_t batch_size = waveform.size();
        int64_t input_node_dims[2] = {batch_size, static_cast<int64_t>(waveform[0].size())};
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
                memory_info_, const_cast<float*>(audio.data()), audio.size(),
                input_node_dims, 2);
        int64_t input_node_dims1[1] = {batch_size};
        Ort::Value input_ort1 = Ort::Value::CreateTensor<float>(
                memory_info_, const_cast<float*>(wav_lens.data()), wav_lens.size(),
                input_node_dims1, 1);
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(input_ort1));

        auto ort_outputs = session_->Run(
                Ort::RunOptions{nullptr}, input_node_names_.data(), ort_inputs.data(),
                ort_inputs.size(), output_node_names_.data(), output_node_names_.size());

        const float* outputs = ort_outputs[0].GetTensorData<float>();
        auto outputs_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int len1 = outputs_shape[0];
        int len2 = outputs_shape[1];
        int len3 = outputs_shape[2];
        std::cout<<"output shape:"<<len1<<"x"<<len2<<"x"<<len3<<std::endl;

        // here len2 is always 1, len1 is batch size, len3 is always 192, which is 
        // size each embedding for each input.
        std::vector<std::vector<float>> res( len1, std::vector<float>( len3 ));
        for( int i = 0; i < len1; ++i )
        {
            for( int j = 0; j < len3; ++j )
            {
                res[i][j] = *( outputs + i * len2 + j );
            }
        }

        return res;
    }
};

class Cluster
{
private:
    // Those 2 values extracted from config.yaml under
    // ~/.cache/torch/pyannote/models--pyannote--speaker-diarization/snapshots/xxx/
    float m_threshold = 0.7153814381597874;
    size_t m_min_cluster_size: 15;

public:
    Cluster()
    {
    }

    /*
     * pyannote/audio/pipelines/clustering.py:215, __call__(...
     * embeddings: num of chunks x 3 x 192, where 192 is size each embedding
     * segmentations: num of chunks x 293 x 3, where 293 is size each segment model out[0]
     * and 3 is each segment model output[1]
     * */
    void clustering( const std::vector<std::vector<std::vector<float>>>& embeddings, 
            const std::vector<std::vector<std::vector<float>>>& segmentations, 
            std::vector<std::vector<int>>& hard_clusters, 
            int num_clusters = -1, int min_clusters = -1, int max_clusters = -1 )
    {
        // python: train_embeddings, train_chunk_idx, train_speaker_idx = self.filter_embeddings
        std::vector<int> chunk_idx;
        std::vector<int> speaker_idx;
        auto filteredEmbeddings = filter_embeddings( embeddings, segmentations, chunk_idx, speaker_idx );

        size_t num_embeddings = filteredEmbeddings.size();
        set_num_clusters( static_cast<int>( num_embeddings ), num_clusters, min_clusters, max_clusters );

        // do NOT apply clustering when min_clusters = max_clusters = 1
        if( max_clusters < 2 )
        {
            size_t num_chunks = embeddings.size();
            size_t num_speakers = embeddings[0].size();
            std::vector<std::vector<int>> hcluster( num_chunks, std::vector<int>( num_speakers, 0 ));
            std::vector<std::vector<std::vector<float>>> scluster( num_chunks,
                    std::vector<std::vector<float>>( num_speakers, std::vector<float>( 1, 1.0f )));
            hard_clusters.swap( hcluster );
            return;
        }

        // python: train_clusters = self.cluster(
        auto clusterRes = cluster( filteredEmbeddings, min_clusters, max_clusters, num_clusters );

        // python: hard_clusters, soft_clusters = self.assign_embeddings(
        assign_embeddings( embeddings, chunk_idx, speaker_idx, clusterRes, hard_clusters );
    }

    // Assign embeddings to the closest centroid
    void assign_embeddings(const std::vector<std::vector<std::vector<float>>>& embeddings,
            const std::vector<int>& chunk_idx, 
            const std::vector<int>& speaker_idx,
            const std::vector<int>& clusterRes,
            std::vector<std::vector<int>>& hard_clusters )
    {
        assert( chunk_idx.size() == speaker_idx.size());

        // python: num_clusters = np.max(train_clusters) + 1
        int num_clusters = *std::max_element(clusterRes.begin(), clusterRes.end()) + 1;
        size_t num_chunks = embeddings.size();
        size_t num_speakers = embeddings[0].size();
        size_t dimension = embeddings[0][0].size();

        // python: train_embeddings = embeddings[train_chunk_idx, train_speaker_idx]
        std::vector<std::vector<float>> filtered_embeddings( num_chunks, std::vector<float>( dimension, 0.0f ));
        for( size_t i = 0; i < num_chunks; ++i )
        {
            auto tmp = embeddings[chunk_idx[i]][speaker_idx[i]];
            for( size_t j = 0; j < dimension; ++j )
            {
                filtered_embeddings[i][j] = tmp[j];
            }
        }

        // python: centroids = np.vstack([np.mean(train_embeddings[train_clusters == k], axis=0)
        std::vector<std::vector<float>> centroids( num_clusters, std::vector<float>( dimension, 0.0f ));
        assert( filtered_embeddings.size() == clusterRes.size());
        for( int i = 0; i < num_clusters; ++i )
        {
            for( size_t j = 0; j < clusterRes.size(); ++i )
            {
                if( i == clusterRes[j] )
                {
                    for( size_t k = 0; k < dimension; ++k )
                    {
                        centroids[i][k] += filtered_embeddings[j][k];
                    }
                }
            }
        }
        for( int i = 0; i < num_clusters; ++i )
        {
            for( size_t k = 0; k < dimension; ++k )
            {
                centroids[i][k] /= dimension;
            }
        }

        // compute distance between embeddings and clusters
        // python: rearrange(embeddings, "c s d -> (c s) d"), where d =192
        auto r1 = Helper::rearrange_down( embeddings );

        // python: cdist(
        auto dist = Helper::cosineSimilarity( r1, centroids );

        // python: e2k_distance = rearrange(
        // N x 3 x 4 for example
        auto soft_clusters  = Helper::rearrange_up( dist, num_chunks );

        // python: soft_clusters = 2 - e2k_distance
        for( size_t i = 0; i < soft_clusters.size(); ++i )
        {
            for( size_t j = 0; j < soft_clusters[0].size(); ++j )
            {
                for( size_t k = 0; k < soft_clusters[0].size(); ++k )
                {
                    soft_clusters[i][j][k] = 2.0 - soft_clusters[i][j][k];
                }
            }
        }

        // python: hard_clusters = np.argmax(soft_clusters, axis=2)
        //  N x 3
        hard_clusters.resize( soft_clusters.size(), std::vector<int>( soft_clusters[0].size()));
        for( size_t i = 0; i < soft_clusters.size(); ++i )
        {
            for( size_t j = 0; j < soft_clusters[0].size(); ++j )
            {
                int max_index = -1;
                float max_value = -1.0 * std::numeric_limits<float>::max();
                for( size_t k = 0; k < soft_clusters[0].size(); ++k )
                {
                    if( soft_clusters[i][j][k] > max_value )
                    {
                        max_index = k;
                        max_value = soft_clusters[i][j][k];
                    }
                }
                hard_clusters[i][j] = max_index;
            }
        }
    }

    std::vector<std::vector<float>> filter_embeddings( const std::vector<std::vector<std::vector<float>>>& embeddings, 
            const std::vector<std::vector<std::vector<float>>>& segmentations,
            std::vector<int>& chunk_idx, std::vector<int>& speaker_idx)
    {
        std::vector<std::vector<float>> filteredEmbeddings;


        return filteredEmbeddings;
    }

    void set_num_clusters(int num_embeddings, int& num_clusters, int& min_clusters, int& max_clusters)
    {
        if( num_clusters != -1 )
        {
            min_clusters = num_clusters;
        }
        else
        {
            if( min_clusters == -1 )
            {
                min_clusters = 1;
            }
        }
        min_clusters = std::max(1, std::min(num_embeddings, min_clusters));

        if( num_clusters != -1 )
        {
            max_clusters == num_clusters;
        }
        else
        {
            if( max_clusters == -1 )
            {
                max_clusters = num_embeddings;
            }
        }
        max_clusters = std::max(1, std::min(num_embeddings, max_clusters));
        if( min_clusters > max_clusters )
        {
            min_clusters = max_clusters;
        }
        if( min_clusters == max_clusters )
        {
            num_clusters = min_clusters;
        }
    }

    // pyannote/audio/pipelines/clustering.py:426, cluster(...
    // AgglomerativeClustering
    std::vector<int> cluster( const std::vector<std::vector<float>>& embeddings, int min_clusters, int max_clusters, int num_clusters )
    {
        // python: num_embeddings, _ = embeddings.shape
        size_t num_embeddings = embeddings.size();

        // heuristic to reduce self.min_cluster_size when num_embeddings is very small
        // (0.1 value is kind of arbitrary, though)
        size_t min_cluster_size = std::min( m_min_cluster_size, std::max(static_cast<size_t>( 1 ), static_cast<size_t>( round(0.1 * num_embeddings))));

        // linkage function will complain when there is just one embedding to cluster
        //if( num_embeddings == 1 ) 
        //     return np.zeros((1,), dtype=np.uint8)

        // self.metric == "cosine" and self.method == "centroid"
        // python:
        //    with np.errstate(divide="ignore", invalid="ignore"):
        //        embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True)
        auto normalizedEmbeddings( embeddings );
        Helper::normalizeEmbeddings( normalizedEmbeddings );

        // python:
        //    dendrogram: np.ndarray = linkage(
        //        embeddings, method=self.method, metric="euclidean"
        //    )
        // dendrogram is [N x 4] array
        auto dendrogram = Helper::hierarchicalClustering( embeddings );

        // apply the predefined threshold
        // python: clusters = fcluster(dendrogram, self.threshold, criterion="distance") - 1
        // clusters [N] array
        auto clusters = Helper::extractClusters( dendrogram, m_threshold );

        // split clusters into two categories based on their number of items:
        // large clusters vs. small clusters
        // python: cluster_unique, cluster_counts = np.unique(...
        std::unordered_map<int, int> clusterCountMap;
        for (int cluster : clusters) 
        {
            clusterCountMap[cluster]++;
        }

        // python: large_clusters = cluster_unique[cluster_counts >= min_cluster_size]
        // python: small_clusters = cluster_unique[cluster_counts < min_cluster_size]
        std::vector<int> large_clusters;
        std::vector<int> small_clusters;
        for (const auto& entry : clusterCountMap) 
        {
            if ( entry.second >= m_min_cluster_size) 
            {
                large_clusters.push_back( entry.first );
            }
            else
            {
                small_clusters.push_back( entry.first );
            }
        }
        size_t num_large_clusters = large_clusters.size();

        // force num_clusters to min_clusters in case the actual number is too small
        if( num_large_clusters < min_clusters )
            num_clusters = min_clusters;

        // force num_clusters to max_clusters in case the actual number is too large
        if( num_large_clusters > max_clusters )
            num_clusters = max_clusters;

        if( num_clusters != -1 )
            assert( false ); // this branch is not implemented

        if( num_large_clusters == 0)
        {
            clusters.assign(clusters.size(),0);
            return clusters;
        }

        if( small_clusters.size() == 0 )
        {
            return clusters;
        }

        // re-assign each small cluster to the most similar large cluster based on their respective centroids
        auto large_centroids = Helper::calculateClusterMeans(embeddings, clusters, large_clusters);
        auto small_centroids = Helper::calculateClusterMeans(embeddings, clusters, small_clusters);

        // python: centroids_cdist = cdist(large_centroids, small_centroids, metric=self.metric)
        auto centroids_cdist = Helper::cosineSimilarity( large_centroids, small_centroids );

        // Update clusters based on minimum distances
        for (size_t small_k = 0; small_k < small_clusters.size(); ++small_k) {
            int large_k = std::distance(
                centroids_cdist[small_k].begin(),
                std::min_element(centroids_cdist[small_k].begin(), centroids_cdist[small_k].end())
            );

            for (size_t i = 0; i < clusters.size(); ++i) {
                if (clusters[i] == small_clusters[small_k]) {
                    clusters[i] = large_clusters[large_k];
                }
            }
        }

        // Find unique clusters and return inverse mapping
        std::vector<int> uniqueClusters;
        std::vector<int> inverseMapping = Helper::findUniqueClusters(clusters, uniqueClusters);

        return inverseMapping;
    }


};

// class SpeakerDiarization{ // TODO: add this 

int embedding_batch_size = 32;

// pyannote/audio/pipelines/speaker_verification.py:281 __call__() 
std::vector<std::vector<float>> getEmbedding( EmbeddingModel& em, const std::vector<std::vector<float>>& dataChunks, 
        const std::vector<std::vector<float>>& masks )
{
    assert( dataChunks.size() == masks.size());

    // Debug
    static int number = 0;
#ifdef WRITE_DATA
    debugWrite2d( dataChunks, std::string( "cpp_batch_waveform" ) + std::to_string( number ));
    debugWrite2d( masks, std::string( "cpp_batch_masks" ) + std::to_string( number ), true );
#endif // WRITE_DATA

    size_t batch_size = dataChunks.size();
    size_t num_samples = dataChunks[0].size();

    // python: imasks = F.interpolate(... ) and imasks = imasks > 0.5
    auto imasks = Helper::interpolate( masks, num_samples, 0.5 );
#ifdef WRITE_DATA
    debugWrite2d( imasks, std::string("cpp_imasks") + std::to_string( number ) );
#endif // WRITE_DATA
    
    assert( imasks.size() == batch_size );
    assert( imasks[0].size() == num_samples );
    //masks is [32x293] imask is [32x80000], dataChunks is [32x80000] as welll
    
    // python: signals = pad_sequence(...)
    // ************** Bug in padSequnce, because dataChunks verified, they are same
    auto signals = Helper::padSequence( dataChunks, imasks );
    assert( signals.size() == batch_size );
    assert( signals[0].size() == num_samples );

    // python: wav_lens = imasks.sum(dim=1)
    std::vector<float> wav_lens( batch_size, 0.0 );
    float max_len = 0;
    int index = 0;
    for( const auto& a : imasks )
    {
        float tmp = std::accumulate(a.begin(), a.end(), 0.0);
        wav_lens[index++] = tmp;
        if( tmp > max_len )
            max_len = tmp;
    }

#ifdef WRITE_DATA
    // Debug
    std::string fn = std::string( "/tmp/cpp_wav_lens" ) + std::to_string( number ) + ".txt";
    std::ofstream f( fn );
    for( const auto& a : wav_lens )
    {
        f<<a<<",";
    }
    f.close();
#endif // WRITE_DATA

    // python: if max_len < self.min_num_samples: return np.NAN * np.zeros(...
    if( max_len < min_num_samples )
    {
        // TODO: don't call embedding process, direct return
        // batch_size x 192, where 192 is size of length embedding result for each waveform
        // python: return np.NAN * np.zeros((batch_size, self.dimension))
    }

    // python:         
    //      too_short = wav_lens < self.min_num_samples
    //      wav_lens = wav_lens / max_len
    //      wav_lens[too_short] = 1.0 
    for( size_t i = 0; i < wav_lens.size(); ++i )
    {
        if( wav_lens[i] < min_num_samples )
        {
            wav_lens[i] = 1.0;
        }
        else
        {
            wav_lens[i] /= max_len;
        }
    }

#ifdef WRITE_DATA
    // Debug
    std::string fn1 = std::string( "/tmp/cpp_final_wav_lens" ) + std::to_string( number ) + ".txt";
    std::ofstream f1( fn1 );
    for( const auto& a : wav_lens )
    {
        std::ostringstream oss;
        oss << std::setprecision(5) << a;
        std::string result = oss.str();
        result = result == "1" ? "1.0" : result;
        f1<<result<<",";
    }
    f1.close();

    // Debug
    std::string fn2 = std::string( "/tmp/cpp_signals" ) + std::to_string( number ) + ".txt";
    std::ofstream f2( fn2 );
    for( const auto& a : signals )
    {
        for( float b : a )
        {
            f2<<b<<",";
        }
        f2<<"\n";
    }
    f2.close();
    
    // Debug, should be deleted
    number++;
#endif // WRITE_DATA

    // signals is [32x80000], wav_lens is of length 32 of 1d array, an example for wav_lens
    // [1.0000, 1.0000, 1.0000, 0.0512, 1.0000, 1.0000, 0.1502, ...] 
    // Now call embedding model to get embeddings of batches
    // speechbrain/pretrained/interfaces.py:903
    return em.infer( signals, wav_lens );
}

// pyannote/core/feature.py
// +
// pyannote/core/segment.py
// mode='loose', fixed=None
template<typename T>
std::vector<std::vector<T>> crop_segment( const std::vector<std::vector<T>> data,
        const SlidingWindow& src, const Segment& focus, SlidingWindow& resFrames )
{
    size_t n_samples = data.size();
    // python: ranges = self.sliding_window.crop(
    // As we pass in Segment, so there would on range returned, here we use following
    // block code to simulate sliding_window.crop <-- TODO: maybe move following block into SlidingWindow class
    // { --> start
        // python: i_ = (focus.start - self.duration - self.start) / self.step
        float i_ = (focus.start - src.duration - src.start) / src.step;

        // python: i = int(np.ceil(i_))
        int rng_start = ceil(i_);
        if( rng_start < 0 )
            rng_start = 0;

        // find largest integer j such that
        // self.start + j x self.step <= focus.end
        float j_ = (focus.end - src.start) / src.step;
        int rng_end = floor(j_) + 1;
    // } <-- end 
    //size_t cropped_num_samples = ( rng_end - rng_start ) * m_sample_rate;
    float start = src[rng_start].start;
    SlidingWindow res( start, src.step, src.duration, n_samples );
    //auto segments = res.data();
    std::vector<Segment> segments;
    segments.push_back( Segment( rng_start, rng_end ));
    
    int n_dimensions = 1;
    // python: for start, end in ranges:
    // ***** Note, I found ranges is always 1 element returned from self.sliding_window.crop
    // if this is not true, then need change following code. Read code:
    // pyannote/core/feature.py:196
    std::vector<std::pair<int, int>> clipped_ranges;
    for( auto segment : segments )
    {
        size_t start = segment.start;
        size_t end = segment.end;

        // if all requested samples are out of bounds, skip
        if( end < 0 || start >= n_samples)
        {
            continue;
        }
        else
        {
            // keep track of non-empty clipped ranges
            // python: clipped_ranges += [[max(start, 0), min(end, n_samples)]]
            clipped_ranges.emplace_back( std::make_pair( std::max( start, 0ul ), std::min( end, n_samples )));
        }
    }
    resFrames = res;
    std::vector<std::vector<T>> cropped_data;

    // python: data = np.vstack([self.data[start:end, :] for start, end in clipped_ranges])
    for( const auto& pair : clipped_ranges )
    {
        for( int i = pair.first; i < pair.second; ++i )
        {
            std::vector<T> tmp;
            for( int j = 0; j < data[i].size(); ++j )
                tmp.push_back( data[i][j] );
            cropped_data.push_back( tmp );
        }
    }

    return cropped_data;
}

// pyannote/audio/pipelines/utils/diarization.py:187
std::vector<std::vector<float>> to_diarization( std::vector<std::vector<std::vector<float>>>& segmentations, 
        const SlidingWindow& segmentations_frames,
        const std::vector<int>& count,
        const SlidingWindow& count_frames, SlidingWindow& activations_frames )
{
    // python: activations = Inference.aggregate(...
    SlidingWindow frames;
    auto activations = Helper::aggregate( segmentations, 
            segmentations_frames, frames, 1e-9, false, 0.0, true );

    // python: _, num_speakers = activations.data.shape
    size_t num_speakers = 1;

    // python: count.data = np.minimum(count.data, num_speakers)
    // here also convert 1d to 2d later need pass to crop_segment
    std::vector<std::vector<int>> converted_count( count.size(), std::vector<int>( 1 ));
    for( size_t i = 0; i < count.size(); ++i )
    {
        if( count[i] > num_speakers )
            converted_count[i][0] = num_speakers;
        else
            converted_count[i][0] = count[i];
    }

    // python: extent = activations.extent & count.extent
    // get extent then calc intersection, check extent() of 
    // SlidingWindowFeature and __and__() of Segment
    // Get activations.extent
    float tmpStart = frames.start + (0 - .5) * frames.step + .5 * frames.duration;
    float duration = activations.size() * frames.step;
    float activations_end = tmpStart + duration;
    float activations_start = frames.start;

    // Get count.extent
    tmpStart = count_frames.start + (0 - .5) * count_frames.step + .5 * count_frames.duration;
    duration = count.size() * count_frames.step;
    float count_end = tmpStart + duration;
    float count_start = count_frames.start;

    // __and__(), max of start, min of end
    float intersection_start = std::max( activations_start, count_start );
    float intersection_end = std::min( activations_end, count_end );
    Segment focus( intersection_start, intersection_end );
    SlidingWindow cropped_activations_frames;
    auto cropped_activations = crop_segment( activations, frames, focus, 
            cropped_activations_frames );

    SlidingWindow cropped_count_frames;
    auto cropped_count = crop_segment( converted_count, count_frames, focus, 
            cropped_count_frames );

    // python: sorted_speakers = np.argsort(-activations, axis=-1)
    std::vector<std::vector<int>> sorted_speakers( cropped_activations.size(),
            std::vector<int>( cropped_activations[0].size()));
    int ss_index = 0;
    for( auto& a : cropped_activations )
    {
        auto indices = Helper::argsort( a );
        sorted_speakers[ss_index++].swap( indices );
    }

    assert( cropped_activations.size() > 0 );
    assert( cropped_activations[0].size() > 0 );

    // python: binary = np.zeros_like(activations.data)
    std::vector<std::vector<float>> binary( cropped_activations.size(),
        std::vector<float>( cropped_activations[0].size(), 0.0f ));

    // python: for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
    assert( cropped_count.size() <= sorted_speakers.size());
    for( size_t i = 0; i < cropped_count.size(); ++i )
    {
        int k = cropped_count[i][0];
        assert( k <= binary[0].size());
        for( size_t j = 0; j < k; ++j )
        {
            assert( sorted_speakers[i][j] < cropped_count.size());
            binary[i][sorted_speakers[i][j]] = 1.0f;
        }
    }

    activations_frames = cropped_activations_frames;

    return binary;
}

// np.max( segmentation[:, cluster == k], axis=1 )
std::vector<float> max_segmentation_cluster(const std::vector<std::vector<float>>& segmentation,
                                       const std::vector<int>& cluster, int k) 
{
    std::vector<float> maxValues;
    /*for (size_t i = 0; i < segmentation.size(); ++i) {
        if (cluster[i] == k) {
            float maxValue = -std::numeric_limits<float>::infinity();
            for (size_t j = 0; j < segmentation[i].size(); ++j) {
                maxValue = std::max(maxValue, segmentation[i][j]);
            }
            maxValues.push_back(maxValue);
        }
    }*/

    for (size_t i = 0; i < segmentation.size(); ++i) 
    {
        float maxValue = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < segmentation.size(); ++j) 
        {
            if (cluster[j] == k) 
            {
                maxValue = std::max(maxValue, segmentation[j][i]);
            }
        }
        maxValues.push_back(maxValue);
    }

    return maxValues;
}

// pyannote/audio/pipelines/speaker_diarization.py:403, def reconstruct(
std::vector<std::vector<float>> reconstruct( 
        const std::vector<std::vector<std::vector<float>>>& segmentations,
        const SlidingWindow& segmentations_frames,
        const std::vector<std::vector<int>> hard_clusters, 
        const std::vector<int>& count_data,
        const SlidingWindow& count_frames,
        SlidingWindow& activations_frames)
{
    size_t num_chunks = segmentations.size();
    size_t num_frames = segmentations[0].size();
    size_t local_num_speakers = segmentations[0][0].size();

    // python: num_clusters = np.max(hard_clusters) + 1
    // Note, element in hard_clusters have negative number, don't define num_cluster as size_t
    int num_clusters = 0;
    for( size_t i = 0; i < hard_clusters.size(); ++i )
    {
        for( size_t j = 0; j < hard_clusters[0].size(); ++j )
        {
            if( hard_clusters[i][j] > num_clusters )
                num_clusters = hard_clusters[i][j];
        }
    }
    num_clusters++;
    assert( num_clusters > 0 );

    // python: for c, (cluster, (chunk, segmentation)) in enumerate(...
    std::vector<std::vector<std::vector<float>>> clusteredSegmentations( num_chunks, 
            std::vector<std::vector<float>>( num_frames, std::vector<float>( num_clusters, NP_NAN)));
    for( size_t i = 0; i < num_chunks; ++i ) 
    {
        const auto& cluster = hard_clusters[i];
        const auto& segmentation = segmentations[i];
        for(size_t j = 0; j < cluster.size(); ++j )
        {
            if( abs( cluster[j] + 2 ) < 1e-9 ) // check if it equals -2
            {
                continue;
            }

            int k = cluster[j];
            auto max_sc = max_segmentation_cluster( segmentation, cluster, k );
            assert( k < num_clusters );
            assert( max_sc.size() > 0 );
            assert( max_sc.size() == num_frames );
            for( size_t m = 0; m < num_frames; ++m )
            {
                clusteredSegmentations[i][m][k] = max_sc[m];
            }
        }
    }

    return to_diarization( clusteredSegmentations, segmentations_frames, count_data, count_frames, activations_frames );
}


// pyannote/audio/pipelines/utils/diarization.py:155
Annotation to_annotation( const std::vector<std::vector<float>>& scores,
        const SlidingWindow& frames,
        float onset, float offset, 
        float min_duration_on, float min_duration_off)
{
    // call binarize : pyannote/audio/utils/signal.py: 287
    size_t num_frames = scores.size();
    size_t num_classes = scores[0].size();

    // python: timestamps = [frames[i].middle for i in range(num_frames)]
    std::vector<float> timestamps( num_frames );
    for( size_t i = 0; i < num_frames; ++i )
    {
        float start = frames.start + i * frames.step;
        float end = start + frames.duration;
        timestamps[i] = ( start + end ) / 2;
    }

    // python: socre.data.T
    std::vector<std::vector<float>> inversed( num_classes, std::vector<float>( num_frames ));
    for( size_t i = 0; i < num_frames; ++i )
    {
        for( size_t j = 0; j < num_classes; ++j )
        {
            inversed[j][i] = scores[i][j];
        }
    }

    Annotation active;
    float pad_onset = 0.0f;
    float pad_offset = 0.0f;
    for( size_t i = 0; i< num_classes; ++i )
    {
        int label = i;
        float start = timestamps[0];
        bool is_active = false;
        if( inversed[i][0] > onset )
        {
            is_active = true;
        }
        for( size_t j = 1; j < num_frames; ++j )
        {
            // currently active
            if( is_active )
            {
                // switching from active to inactive
                if( inversed[i][j] < offset )
                {
                    Segment region(start - pad_onset, timestamps[j] + pad_offset);
                    active.addSegment(region.start, region.end, label);
                    start = timestamps[j];
                    is_active = false;
                }
            }
            else
            {
                if( inversed[i][j] > onset )
                {
                    start = timestamps[j];
                    is_active = true;
                }
            }
        }

        if( is_active )
        {
            Segment region(start - pad_onset, timestamps.back() + pad_offset);
            active.addSegment(region.start, region.end, label);
        }
    }

    // because of padding, some active regions might be overlapping: merge them.
    // also: fill same speaker gaps shorter than min_duration_off
    if( pad_offset > 0.0 || pad_onset > 0.0  || min_duration_off > 0.0 )
        active.support( min_duration_off );

    // remove tracks shorter than min_duration_on
    if( min_duration_on > 0 )
    {
        for( size_t i = 0; i < active.size(); ++i )
        {
            if( active.segments[i].duration() < min_duration_on )
            {
                active.remove( i );
                i--;
            }
        }
    }

    return active;
}

// for string delimiter
std::vector<std::string> split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

Annotation speakerDiarization( const std::string& waveFile, const std::string& segmentModel, const std::string& embeddingModel )
{
    wav::WavReader wav_reader( waveFile );
    int num_channels = wav_reader.num_channels();
    CHECK_EQ(num_channels, 1) << "Only support mono (1 channel) wav!";
    int bits_per_sample = wav_reader.bits_per_sample();
    int sample_rate = wav_reader.sample_rate();
    const float* audio = wav_reader.data();
    int num_samples = wav_reader.num_samples();
    std::vector<float> input_wav{audio, audio + num_samples};

    // Print audio samples
    /*
    for( int i = 0; i < num_samples; ++i )
    {
        input_wav[i] = input_wav[i]*1.0f/32768.0;
    }
    */

    //*************************************
    // 1. Segmentation stage
    //*************************************
    auto beg_seg = timeNow();
    std::cout<<"\n---segmentation ---"<<std::endl;
    SegmentModel mm( segmentModel );
    std::vector<std::pair<float, float>> segments;
    SlidingWindow res_frames;
    auto segmentations = mm.slide( input_wav, res_frames );
    auto segment_data = res_frames.data();
    for( auto seg : segment_data )
    {
        segments.emplace_back( std::make_pair( seg.start, seg.end ));
    }
    std::cout<<segmentations.size()<<"x"<<segmentations[0].size()<<"x"<<segmentations[0][0].size()<<std::endl;
    // estimate frame-level number of instantaneous speakers
    //std::vector<std::vector<std::vector<float>>> test = {{{1,2,3},{4,5,6},{7,8,9},{10,11,12}},{{13,14,15},{16,17,18},{19,20,21},{22,23,24}}};
    auto binarized = mm.binarize_swf( segmentations, false );
    assert( binarized.size() == segments.size());

#ifdef WRITE_DATA
    debugWrite3d( binarized, "cpp_binarized_segmentations" );
#endif

    // estimate frame-level number of instantaneous speakers
    // In python code, binarized in speaker_count function is cacluated with 
    // same parameters as we did above, so we reuse it by passing it into speaker_count
    SlidingWindow count_frames( num_samples );
    auto count = mm.speaker_count( segmentations, binarized, count_frames, num_samples );

    // python: duration = binary_segmentations.sliding_window.duration
    float duration = 5.0;
    size_t num_chunks = binarized.size();
    size_t num_frames = binarized[0].size(); 

    // python: num_samples = duration * self._embedding.sample_rate
    size_t min_num_frames = ceil(num_frames * min_num_samples / ( duration * 16000 ));

    // python: clean_frames = 1.0 * ( np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2 )
    // python: clean_segmentations = SlidingWindowFeature( 
    //                               binary_segmentations.data * clean_frames, binary_segmentations.sliding_window )
    auto clean_segmentations = Helper::cleanSegmentations( binarized );

    assert( binarized.size() == clean_segmentations.size());
    std::vector<std::vector<float>> batchData;
    std::vector<std::vector<float>> batchMasks;

    timeCost( beg_seg, "Segmenations time" );
#ifdef WRITE_DATA
    debugWrite3d( segmentations, "cpp_segmentations" );
#endif // WRITE_DATA

    //*************************************
    // 2. Embedding
    //*************************************
    std::cout<<"\n---generating embeddings---"<<std::endl;
    auto beg_emb = timeNow();

    // Create embedding model
    EmbeddingModel em( embeddingModel );
    std::vector<std::vector<float>> embeddings;

    // This for loop processes python: batchify() and zip(*filter(lambda
    for( size_t i = 0; i < binarized.size(); ++i )
    {
        auto chunkData = mm.crop( input_wav, segments[i] );
        auto& masks = binarized[i];
        auto& clean_masks = clean_segmentations[i];
        assert( masks[0].size() == 3 );
        assert( clean_masks[0].size() == 3 );

        // python: for mask, clean_mask in zip(masks.T, clean_masks.T):
        for( size_t j = 0; j < clean_masks[0].size(); ++j )
        {
            std::vector<float> used_mask;
            float sum = 0.0;
            std::vector<float> reversed_clean_mask(clean_masks.size());
            std::vector<float> reversed_mask(masks.size());

            // python: np.sum(clean_mask)
            for( size_t k = 0; k < clean_masks.size(); ++k )
            {
                sum += clean_masks[k][j];
                reversed_clean_mask[k] = clean_masks[k][j];
                reversed_mask[k] = masks[k][j];
            }

            if( sum > min_num_frames )
            {
                used_mask = std::move( reversed_clean_mask );
            }
            else
            {
                used_mask = std::move( reversed_mask );
            }

            // batchify
            batchData.push_back( chunkData );
            batchMasks.push_back( std::move( used_mask ));
            if( batchData.size() == embedding_batch_size )
            {
                auto embedding = getEmbedding( em, batchData, batchMasks );
                batchData.clear();
                batchMasks.clear();

                for( auto& a : embedding )
                {
                    embeddings.push_back( std::move( a ));
                }
            }
        }
    }

    // Process remaining
    if( batchData.size() > 0 )
    {
        auto embedding = getEmbedding( em, batchData, batchMasks );
        for( auto& a : embedding )
        {
            embeddings.push_back( std::move( a ));
        }
    }

    // TODO: use Helper::rerange_up
    // python: embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)
    size_t num_classes = binarized[0][0].size();
    size_t embeddingDim = embeddings[0].size();
    assert( num_classes == 3 );
    assert( embeddingDim = 192 );
    std::vector<std::vector<std::vector<float>>> embeddings1(num_chunks, 
            std::vector<std::vector<float>>( num_classes, std::vector<float>(embeddingDim)));
    int rowNum = 0;
    for( size_t i = 0; i < embeddings.size(); i += num_classes )
    {
        for( size_t j = 0; j < num_classes; ++j )
        {
            for( size_t k = 0; k < embeddingDim; ++k )
            {
                embeddings1[rowNum][j][k] = embeddings[i+j][k];
            }
        }
        rowNum++;
    }
    timeCost( beg_emb, "Embedding time" );

    //*************************************
    // 3. Clustering
    //*************************************
    // Cluster stage
    std::cout<<"\n---clustering---"<<std::endl;
    auto beg_cst = timeNow();
    Cluster cst;
    std::vector<std::vector<int>> hard_clusters; // output 1 for clustering
    cst.clustering( embeddings1, binarized, hard_clusters );

#ifdef WRITE_DATA
    debugWrite2d( hard_clusters, "cpp_hard_clusters" );
#endif // WRITE_DATA

    // keep track of inactive speakers
    //   shape: (num_chunks, num_speakers)
    // python: inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
    // python: hard_clusters[inactive_speakers] = -2
    assert( hard_clusters.size() == binarized.size());
    assert( hard_clusters[0].size() == binarized[0][0].size());
    std::vector<std::vector<float>> inactive_speakers( binarized.size(),
            std::vector<float>( binarized[0][0].size(), 0.0f ));
    for( size_t i = 0; i < binarized.size(); ++i )
    {
        for( size_t j = 0; j < binarized[0].size(); ++j )
        {
            for( size_t k = 0; k < binarized[0][0].size(); ++k )
            {
                inactive_speakers[i][k] += binarized[i][j][k];
            }
        }
    }
    for( size_t i = 0; i < inactive_speakers.size(); ++i )
    {
        for( size_t j = 0; j < inactive_speakers[0].size(); ++j )
        {
            if( abs( inactive_speakers[i][j] ) < 1e-9 )
                hard_clusters[i][j] = -2;
        }
    }

#ifdef WRITE_DATA
    //debugWrite2d( hard_clusters, "cpp_hard_clusters" );
    debugWrite( count, "cpp_count" );
#endif // WRITE_DATA


    // python: discrete_diarization = self.reconstruct(
    // N x 4
    SlidingWindow activations_frames;
    auto discrete_diarization = reconstruct( segmentations, res_frames, 
            hard_clusters, count, count_frames, activations_frames );

#ifdef WRITE_DATA
    debugWrite2d( discrete_diarization, "cpp_discrete_diarization" );
#endif // WRITE_DATA

    // convert to continuous diarization
    // python: diarization = self.to_annotation(
    float diarization_segmentation_min_duration_off = 0.5817029604921046; // see SegmentModel
                                                                          
    // for testing
    /*
    std::ifstream fd("/tmp/py_discrete_diarization.txt"); //taking file as inputstream
    if(fd) {
        std::ostringstream ss;
        ss << fd.rdbuf(); // reading data
        std::string str = ss.str();
        std::string delimiter = ",";
        std::vector<std::string> v = split (str, delimiter);
        assert( v.size() - 1 == discrete_diarization.size());
        for( size_t i = 0; i < discrete_diarization.size(); ++i )
        {
            discrete_diarization[i][0] = std::stof( v[i] );
        }
    }
    */
    auto diarization = to_annotation( discrete_diarization, activations_frames, 0.5, 0.5,
            0.0, diarization_segmentation_min_duration_off );
    timeCost( beg_cst, "Clustering time" );

    return diarization;
}

int main(int argc, char* argv[]) 
{
    if( argc < 4 )
    {
        std::cout<<"program [segment model file] [embeding model file] [wave file]"<<std::endl;
        return 0;
    }

    auto beg = timeNow();
    std::string segmentModel( argv[1] );
    std::string embeddingModel( argv[2] );
    std::string waveFile( argv[3] );
    auto res = speakerDiarization( waveFile, segmentModel, embeddingModel );

    std::cout<<"\n----Summary----"<<std::endl;
    timeCost( beg, "Time cost" );
    std::cout<<"----------------------------------------------------"<<std::endl;
    for( size_t i = 0; i < res.segments.size(); ++i )
    {
        std::cout<<"["<<res.segments[i].start<<" -- "<<res.segments[i].end<<"]"<<" --> Speaker_"<<res.labels[i]<<std::endl;
    }
    std::cout<<"----------------------------------------------------"<<std::endl;
}
