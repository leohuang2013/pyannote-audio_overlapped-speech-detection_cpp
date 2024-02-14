// Copyright (c) 2023 Huang Liyi (webmaster@360converter.com)
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
#include <optional>
#include <unistd.h>

#include "frontend/wav.h"

//#include "kmeans.h"


//#include <torch/torch.h>
#include <torch/script.h>

#include "onnxModel/onnx_model.h"

#define SAMPLE_RATE 16000

#define WRITE_DATA 0

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

/*
 * writeDecimalforZero: if enabled, write 0 as 0.0
 * */
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
                if (std::isnan( b )) 
                {
                    f<<"nan,";
                }
                else
                {
                    if( writeDecimalforZero )
                    {
                        std::ostringstream oss;
                        oss << std::setprecision(6) << b;
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

/*
 * writeDecimalforZero: if enabled, write 0 as 0.0
 * haveNan: if has Nan, which will be written as 'nan'
 * later easier to compare with python result or load into python with loadtxt()
 * */
template<class T>
void debugWrite3d( const std::vector<std::vector<std::vector<T>>>& data, std::string name, 
        bool writeDecimalforZero = false )
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
                else if( std::is_same<T, float>::value ||
                    std::is_same<T, double>::value )
                {
                    if (std::isnan( c )) 
                    {
                        f<<"nan,";
                    }
                    else
                    {
                        if( writeDecimalforZero )
                        {
                            std::ostringstream oss;
                            oss << std::setprecision(6) << c;
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

class Helper 
{
public:

    // for string delimiter
    static std::vector<std::string> split(std::string s, std::string delimiter) 
    {
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

    // Mimic python np.rint
    // For values exactly halfway between rounded decimal values, 
    // NumPy rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0, -0.5 and 0.5 round to 0.0, etc.
    template <typename T>
    static int np_rint( T val )
    {
        if(abs( val - int( val ) - 0.5 * ( val > 0 ? 1 : -1 )) < std::numeric_limits<double>::epsilon())
        {
            int tmp = std::round( val );
            if( tmp % 2 == 0 )
                return tmp;
            else
                return tmp - 1 * ( val > 0 ? 1 : -1 );
        }
        return std::round( val );
    }

    template <typename T>
    static std::vector<int> argsort(const std::vector<T> &v) 
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

    // python: hard_clusters = np.argmax(soft_clusters, axis=2)
    template <typename T>
    static std::vector<std::vector<int>> argmax( std::vector<std::vector<std::vector<T>>>& data )
    {
        std::vector<std::vector<int>> res( data.size(), std::vector<int>( data[0].size()));
        for( size_t i = 0; i < data.size(); ++i )
        {
            for( size_t j = 0; j < data[0].size(); ++j )
            {
                int max_index = 0;
                double max_value = -1.0 * std::numeric_limits<double>::max();
                for( size_t k = 0; k < data[0][0].size(); ++k )
                {
                    if( data[i][j][k] > max_value )
                    {
                        max_index = k;
                        max_value = data[i][j][k];
                    }
                }
                res[i][j] = max_index;
            }
        }

        return res;
    }

    // Define a helper function to find non-zero indices in a vector
    static std::vector<int> nonzeroIndices(const std::vector<bool>& input) 
    {
        std::vector<int> indices;
        for (int i = 0; i < input.size(); ++i) {
            if (input[i]) {
                indices.push_back(i);
            }
        }
        return indices;
    }

    // Function to compute the L2 norm of a vector
    template <typename T>
    static float L2Norm(const std::vector<T>& vec) 
    {
        T sum = 0.0;
        for (T val : vec) 
        {
            sum += val * val;
        }
        return std::sqrt(sum);
    }

    // Function to normalize a 2D vector
    template <typename T>
    static void normalizeEmbeddings(std::vector<std::vector<T>>& embeddings) 
    {
        for (std::vector<T>& row : embeddings) 
        {
            T norm = L2Norm(row);
            if (norm != 0.0) 
            {
                for (T& val : row) 
                {
                    val /= norm;
                }
            }
        }
    }

    // Function to calculate the Euclidean distance between two vectors
    template <typename T>
    static T euclideanDistance(const std::vector<T>& vec1, const std::vector<T>& vec2) {
        T sum = 0.0;
        for (size_t i = 0; i < vec1.size(); ++i) {
            T diff = static_cast<T>(vec1[i]) - static_cast<T>(vec2[i]);
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // scipy.cluster.hierarchy.linkage with method: single
    template <typename T>
    static T clusterDistance_single( const std::vector<std::vector<T>>& embeddings,
            const std::vector<std::vector<T>>& distances,
            const std::vector<int>& cluster1, 
            const std::vector<int>& cluster2 )
    {
        T minDistance = 1e9;
        for( size_t i = 0; i < cluster1.size(); ++i )
        {
            if( cluster1[i] == -1 )
                return minDistance;
            // calc each cluster distance
            for( size_t j = 0; j < cluster2.size(); ++j )
            {
                if( cluster2[j] == -1 )
                    break;
                int _i = cluster1[i];
                int _j = cluster2[j];
                if( _i > _j )
                {
                    _i = cluster2[j];
                    _j = cluster1[i];
                }
                assert( _i != _j ); // same point cannot be in cluster1 and cluster2 at same time
                T dis = distances[_i][_j];
                if( dis < minDistance )
                    minDistance = dis;
            }
        }

        return minDistance;
    }
            
    // scipy.cluster.hierarchy.linkage with method: centroid
    template <typename T>
    static T clusterDistance_centroid( const std::vector<std::vector<T>>& embeddings,
            const std::vector<std::vector<T>>& distances,
            const std::vector<int>& cluster1, 
            const std::vector<int>& cluster2 )
    {
        T minDistance = 1e9;
        // d(i ∪ j, k) = αid(i, k) + αjd(j, k) + βd(i, j)
        // αi = |i| / ( |i|+|j| ), αj = |j| / ( |i|+|j| )
        // β = − |i||j| / (|i|+|j|)^2
        for( size_t i = 0; i < cluster1.size(); ++i )
        {
            if( cluster1[i] == -1 )
                return minDistance;
            // calc each cluster distance
            for( size_t j = 0; j < cluster2.size(); ++j )
            {
                if( cluster2[j] == -1 )
                    return minDistance;
                int _i = cluster1[i];
                int _j = cluster2[j];
                if( _i > _j )
                {
                    _i = cluster2[j];
                    _j = cluster1[i];
                }
                assert( _i != _j ); // same point cannot be in cluster1 and cluster2 at same time
                T dis = distances[_i][_j];
                if( dis < minDistance )
                    minDistance = dis;
            }
        }

        return minDistance;
    }

    // Function to calculate the mean of embeddings for large clusters
    template <typename T>
    static std::vector<std::vector<T>> calculateClusterMeans(const std::vector<std::vector<T>>& embeddings,
                                                 const std::vector<int>& clusters,
                                                 const std::vector<int>& largeClusters) 
    {
        std::vector<std::vector<T>> clusterMeans;

        for (int large_k : largeClusters) {
            std::vector<T> meanEmbedding( embeddings[0].size(), 0.0 );
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
                    meanEmbedding[j] /= static_cast<T>(count);
                }
            }

            clusterMeans.push_back(meanEmbedding);
        }

        return clusterMeans;
    }

    // Function to calculate the cosine distance between two vectors
    template <typename T>
    static T cosineDistance(const std::vector<T>& vec1, const std::vector<T>& vec2) 
    {
        if (vec1.size() != vec2.size()) {
            throw std::runtime_error("Vector sizes must be equal.");
        }

        T dotProduct = 0.0;
        T magnitude1 = 0.0;
        T magnitude2 = 0.0;

        for (size_t i = 0; i < vec1.size(); ++i) {
            dotProduct += static_cast<T>(vec1[i]) * static_cast<T>(vec2[i]);
            magnitude1 += static_cast<T>(vec1[i]) * static_cast<T>(vec1[i]);
            magnitude2 += static_cast<T>(vec2[i]) * static_cast<T>(vec2[i]);
        }

        if (magnitude1 == 0.0 || magnitude2 == 0.0) {
            throw std::runtime_error("Vectors have zero magnitude.");
        }

        return 1.0 - (dotProduct / (std::sqrt(magnitude1) * std::sqrt(magnitude2)));
    }

    // Calculate cosine distances between large and small cluster means
    template <typename T>
    static std::vector<std::vector<T>> cosineSimilarity( std::vector<std::vector<T>>& largeClusterMeans,
            std::vector<std::vector<T>>& smallClusterMeans )
    {

        std::vector<std::vector<T>> centroidsCdist( largeClusterMeans.size(),
                std::vector<T>( smallClusterMeans.size()));
        for (size_t i = 0; i < largeClusterMeans.size(); ++i) {
            for (size_t j = 0; j < smallClusterMeans.size(); ++j) {
                T distance = cosineDistance(largeClusterMeans[i], smallClusterMeans[j]);
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

        // Find unique
        for (size_t i = 0; i < clusters.size(); ++i) 
        {
            std::vector<int>::iterator position = std::find( uniqueClusters.begin(), uniqueClusters.end(), clusters[i] );
            if (position == uniqueClusters.end()) 
            {
                uniqueClusters.push_back(clusters[i]);
            }
        }

        // Sort, python implementation like this
        std::sort(uniqueClusters.begin(), uniqueClusters.end());

        for (size_t i = 0; i < clusters.size(); ++i) 
        {
            std::vector<int>::iterator position = std::find( uniqueClusters.begin(), uniqueClusters.end(), clusters[i] );
            if (position != uniqueClusters.end()) 
            {
                inverseMapping[i] = position - uniqueClusters.begin();
            }
        }

        return inverseMapping;
    }

    // python: embeddings = rearrange(embedding_batches, "(c s) d -> c s d", c=num_chunks)
    template <typename T>
    static std::vector<std::vector<std::vector<T>>> rearrange_up( const std::vector<std::vector<T>>& input, int c )
    {
        assert( input.size() > c );
        assert( input.size() % c == 0 );
        size_t dim1 = c;
        size_t dim2 = input.size() / c;
        size_t dim3 = input[0].size();
        std::vector<std::vector<std::vector<T>>> output(c, 
                std::vector<std::vector<T>>( dim2, std::vector<T>(dim3, -1.0f)));
        for( size_t i = 0; i < dim1; ++i )
        {
            for( size_t j = 0; j < dim2; ++j )
            {
                for( size_t k = 0; k < dim3; ++k )
                {
                    output[i][j][k] = input[i*dim2+j][k];
                }
            }
        }

        return output;
    }

    // Imlemenation of einops.rearrange c s d -> (c s) d
    template <typename T>
    static std::vector<std::vector<T>> rearrange_down( const std::vector<std::vector<std::vector<T>>>& input )
    {
        int num_chunks = input.size();
        int num_frames = input[0].size();
        int num_classes = input[0][0].size();
        std::vector<std::vector<T>> data(num_chunks * num_frames, std::vector<T>(num_classes));
        for( size_t i = 0; i < num_chunks; ++i )
        {
            for( size_t j = 0; j < num_frames; ++j )
            {
                for( size_t k = 0; k < num_classes; ++k )
                {
                    data[ i * num_frames + j][k] = input[i][j][k];
                }
            }
        }

        return data;
    }

    // Imlemenation of einops.rearrange c f k -> (c k) f
    template <typename T>
    static std::vector<std::vector<T>> rearrange_other( const std::vector<std::vector<std::vector<T>>>& input )
    {
        int num_chunks = input.size();
        int num_frames = input[0].size();
        int num_classes = input[0][0].size();
        std::vector<std::vector<T>> data(num_chunks * num_classes, std::vector<T>(num_frames));
        int rowNum = 0;
        for ( const auto& row : input ) 
        {
            // Create a new matrix with swapped dimensions
            std::vector<std::vector<T>> transposed(num_classes, std::vector<T>(num_frames));

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

    static std::vector<std::vector<std::vector<double>>> cleanSegmentations(
            const std::vector<std::vector<std::vector<double>>>& data)
    {
        size_t numRows = data.size();
        size_t numCols = data[0].size();
        size_t numChannels = data[0][0].size();

        // Initialize the result with all zeros
        std::vector<std::vector<std::vector<double>>> result(numRows, 
                std::vector<std::vector<double>>(numCols, std::vector<double>(numChannels, 0.0)));
        for (int i = 0; i < numRows; ++i) 
        {
            for (int j = 0; j < numCols; ++j) 
            {
                double sum = 0.0;
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
    static std::vector<std::vector<bool>> interpolate(const std::vector<std::vector<float>>& masks, 
            int num_samples, float threshold ) 
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
                                                const std::vector<std::vector<bool>>& imasks) 
    {
        // Find the maximum sequence length
        size_t maxLen = 0;
        for (const std::vector<bool>& mask : imasks) 
        {
            maxLen = std::max(maxLen, mask.size());
        }

        // Initialize the padded sequence with zeros
        std::vector<std::vector<float>> paddedSequence(waveforms.size(), std::vector<float>(maxLen, 0.0));

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

};


class Segment
{
public:
    double start;
    double end;
    Segment( double start, double end )
        : start( start )
        , end( end )
    {}

    Segment( const Segment& other )
    {
        start = other.start;
        end = other.end;
    }

    Segment& operator=( const Segment& other )
    {
        start = other.start;
        end = other.end;

        return *this;
    }

    double duration() const 
    {
        return end - start;
    }

    double gap( const Segment& other )
    {
        if( start < other.start )
        {
            if( end >= other.start )
            {
                return 0.0;
            }
            else
            {
                return other.start - end;
            }
        }
        else
        {
            if( start <= other.end )
            {
                return 0.0;
            }
            else
            {
                return start - other.end;
            }
        }
    }

    Segment merge( const Segment& other )
    {
        return Segment(std::min( start, other.start ), std::max( end, other.end ));
    }
};

// Define a struct to represent annotations
struct Annotation 
{
    struct Result
    {
        double start;
        double end;
        int label;
        Result( double start, double end, int label )
            : start( start )
            , end( end )
            , label( label )
        {}
    };
    struct Track 
    {
        std::vector<Segment> segments;
        int label;

        Track( int label )
            : label( label )
        {}

        Track& operator=( const Track& other )
        {
            segments = other.segments;
            label = other.label;

            return *this;
        }

        Track( const Track& other )
        {
            segments = other.segments;
            label = other.label;
        }

        Track( Track&& other )
        {
            segments = std::move( other.segments );
            label = other.label;
        }

        void addSegment(double start, double end ) 
        {
            segments.push_back( Segment( start, end ));
        }

        void support( double collar )
        {
            // Must sort first
            std::sort( segments.begin(), segments.end(), []( const Segment& s1, const Segment& s2 ){
                        return s1.start < s2.start;
                    });
            if( segments.size() == 0 )
                return;
            std::vector<Segment> merged_segments;
            Segment curSeg = segments[0];
            bool merged = true;
            for( size_t i = 1; i < segments.size(); ++i )
            {
                // WHYWHY must assign to tmp object, otherwise
                // in gap function, its value like random
                auto next = segments[i];
                double gap = curSeg.gap( next );
                if( gap < collar )
                {
                    curSeg = curSeg.merge( segments[i] );
                }
                else
                {
                    merged_segments.push_back( curSeg );
                    curSeg = segments[i];
                }
            }
            merged_segments.push_back( curSeg );

            segments.swap( merged_segments );
        }

        void removeShort( double min_duration_on )
        {
            for( size_t i = 1; i < segments.size(); ++i )
            {
                if( segments[i].duration() < min_duration_on )
                {
                    segments.erase( segments.begin() + i );
                    i--;
                }
            }
        }
    };

    std::vector<Track> tracks;

    Annotation()
        : tracks()
    {}

    std::vector<Result> finalResult()
    {
        std::vector<Result> results;
        for( const auto& track : tracks )
        {
            for( const auto& segment : track.segments )
            {
                Result res( segment.start, segment.end, track.label );
                results.push_back( res );
            }
        }
        std::sort( results.begin(), results.end(), []( const Result& s1, const Result& s2 ){
                    return s1.start < s2.start;
                });

        return results;
    }

    void addSegment(double start, double end, int label) 
    {
        for( auto& tk : tracks )
        {
            if( tk.label == label )
            {
                tk.addSegment( start, end );
                return;
            }
        }

        // Not found, create new track
        Track tk( label );
        tk.addSegment( start, end );
        tracks.push_back( tk );
    }

    Annotation& operator=( const Annotation& other )
    {
        tracks = other.tracks;

        return *this;
    }

    Annotation( Annotation&& other )
    {
        tracks = std::move( tracks );
    }

    void removeShort( double min_duration_on )
    {
        for( auto& track : tracks )
        {
            track.removeShort( min_duration_on );
        }
    }

    // pyannote/core/annotation.py:1350
    void support( double collar )
    {
        // python: timeline = timeline.support(collar)
        // pyannote/core/timeline.py:845
        for( auto& track : tracks )
        {
            track.support( collar );
        }
    }
};

class SlidingWindow
{
public:
    double start;
    double step;
    double duration;
    size_t num_samples;
    double sample_rate;
    SlidingWindow()
        : start( 0.0 )
        , step( 0.0 )
        , duration( 0.0 )
        , num_samples( 0 )
        , sample_rate( 16000 )
    {
    }

    SlidingWindow( size_t num_samples )
        : start( 0.0 )
        , step( 0.0 )
        , duration( 0.0 )
        , num_samples( num_samples )
        , sample_rate( 16000 )
    {
    }

    SlidingWindow( double start, double step, double duration, size_t num_samples = 0 )
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

    size_t closest_frame( double start )
    {
        double closest = ( start - this->start - .5 * duration ) / step;
        if( closest < 0.0 )
            closest = 0.0;
        return Helper::np_rint( closest );
    }

    Segment operator[]( int pos ) const
    {
        int window_size = std::round(duration * sample_rate); // 80000
        int step_size = std::round(step * sample_rate); // 8000
        // python: start = self.__start + i * self.__step
        //double start = this->start + pos * this->step;
        double start = 0.0;
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

        return Segment(0.0, 0.0);
    }

    std::vector<Segment> data()
    {
        std::vector<Segment> segments;
        int window_size = std::round(duration * sample_rate); // 80000
        int step_size = std::round(step * sample_rate); // 8000
        double start = 0.0;
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
    //std::vector<std::vector<float>> data; // this change just for osd
    std::vector<std::pair<double, double>> slidingWindow;

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

class PipelineHelper
{
public:
    // pyannote/audio/core/inference.py:411
    // we ignored warm_up parameter since our case use default value( 0.0, 0.0 )
    // so hard code warm_up
    static std::vector<std::vector<double>> aggregate( 
            const std::vector<std::vector<std::vector<double>>>& scoreData, 
            const SlidingWindow& scores_frames, 
            const SlidingWindow& pre_frames, 
            SlidingWindow& post_frames,
            bool hamming = false, 
            double missing = NAN, 
            bool skip_average = false,
            double epsilon = std::numeric_limits<double>::epsilon())
    {
        size_t num_chunks = scoreData.size(); 
        size_t num_frames_per_chunk = scoreData[0].size(); 
        size_t num_classes = scoreData[0][0].size(); 
        size_t num_samples = scores_frames.num_samples;
        assert( num_samples > 0 );
        assert( num_classes == 1 ); // note, if this is not 1, then hamming window multiplication part is wrong

        // create masks 
        std::vector<std::vector<std::vector<double>>> masks( num_chunks, 
                std::vector<std::vector<double>>( num_frames_per_chunk, std::vector<double>( num_classes, 1.0 )));
        auto scores = scoreData;

        // Replace NaN values in scores with 0 and update masks
        // python: masks = 1 - np.isnan(scores)
        // python: scores.data = np.nan_to_num(scores.data, copy=True, nan=0.0)
        for (size_t i = 0; i < num_chunks; ++i) 
        {
            for (size_t j = 0; j < num_frames_per_chunk; ++j) 
            {
                for( size_t k = 0; k < num_classes; ++k )
                {
                    if (std::isnan(scoreData[i][j][k])) 
                    {
                        masks[i][j][k] = 0.0;
                        scores[i][j][k] = 0.0;
                    }
                }
            }
        }

        std::vector<std::vector<float>> hamming_window(num_frames_per_chunk, std::vector<float>(1));  // Reshape as 2D vector
        if( !hamming )
        {
            // python np.ones((num_frames_per_chunk, 1))
            // no need create it, later will directly apply 1 to computation
        }
        else
        {
            // python: np.hamming(num_frames_per_chunk).reshape(-1, 1)
            std::vector<float> hamming_vector(num_frames_per_chunk);
            for (int i = 0; i < num_frames_per_chunk; ++i) 
            {
                hamming_vector[i] = 0.54 - 0.46 * cos(2 * M_PI * i / (num_frames_per_chunk - 1));
            }

            for (int i = 0; i < num_frames_per_chunk; ++i) 
            {
                hamming_window[i][0] = hamming_vector[i];
            }
        }

        // Get frames, we changed this part. In pyannote, it calc frames(self._frames) before calling
        // this function, but in this function, it creates new frames and use it.
        // step = (self.inc_num_samples / self.inc_num_frames) / sample_rate
        // pyannote/audio/core/model.py:243
        // currently cannot find where self.inc_num_samples / self.inc_num_frames from
        /*int inc_num_samples = 270; // <-- this may not be correct
        int inc_num_frames = 1; // <-- this may not be correct
        float frames_step = ( inc_num_samples * 1.0f / inc_num_frames) / g_sample_rate;
        float frames_duration = frames_step;
        float frames_start = scores_frames.start;
        */

        // aggregated_output[i] will be used to store the sum of all predictions
        // for frame #i
        // python: num_frames = ( frames.closest_frame(...)) + 1
        double frame_target = scores_frames.start + scores_frames.duration + (num_chunks - 1) * scores_frames.step;
        SlidingWindow frames( scores_frames.start, pre_frames.step, pre_frames.duration );
        size_t num_frames = frames.closest_frame( frame_target ) + 1;

        // python: aggregated_output: np.ndarray = np.zeros(...
        std::vector<std::vector<double>> aggregated_output(num_frames, std::vector<double>( num_classes, 0.0 ));

        // overlapping_chunk_count[i] will be used to store the number of chunks
        // that contributed to frame #i
        std::vector<std::vector<double>> overlapping_chunk_count(num_frames, std::vector<double>( num_classes, 0.0 ));

        // aggregated_mask[i] will be used to indicate whether
        // at least one non-NAN frame contributed to frame #i
        std::vector<std::vector<double>> aggregated_mask(num_frames, std::vector<double>( num_classes, 0.0 ));
        
        // for our use case, warm_up_window is num_frames_per_chunkx1
        double start = scores_frames.start;
        for( size_t i = 0; i < scores.size(); ++i )
        {
            size_t start_frame = frames.closest_frame( start );
            //std::cout<<"start_frame: "<<start_frame<<" with:"<<start<<std::endl;
            start += scores_frames.step; // python: chunk.start
            for( size_t j = 0; j < num_frames_per_chunk; ++j )
            {
                size_t _j = j + start_frame;
                for( size_t k = 0; k < num_classes; ++k )
                {
                    // score * mask * hamming_window * warm_up_window
                    aggregated_output[_j][k] += scores[i][j][k] * masks[i][j][k] * hamming_window[j][k];
                    overlapping_chunk_count[_j][k] += masks[i][j][k] * hamming_window[j][k];
                    if( masks[i][j][k] > aggregated_mask[_j][k] )
                    {
                        aggregated_mask[_j][k] = masks[i][j][k];
                    }
                }
            }
        }

#ifdef WRITE_DATA
        debugWrite3d( masks, "cpp_masks_in_aggregate" );
        debugWrite3d( scores, "cpp_scores_in_aggregate" );
        debugWrite2d( aggregated_output, "cpp_aggregated_output" );
        debugWrite2d( aggregated_mask, "cpp_aggregated_mask" );
        debugWrite2d( overlapping_chunk_count, "cpp_overlapping_chunk_count" );
#endif // WRITE_DATA

        post_frames.start = frames.start;
        post_frames.step = frames.step;
        post_frames.duration = frames.duration;
        post_frames.num_samples = num_samples;
        if( !skip_average )
        {
            for( size_t i = 0; i < aggregated_output.size(); ++i )
            {
                for( size_t j = 0; j < aggregated_output[0].size(); ++j )
                {
                    aggregated_output[i][j] /= std::max( overlapping_chunk_count[i][j], epsilon );
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
                if( abs( aggregated_mask[i][j] ) < std::numeric_limits<double>::epsilon() )
                {
                    aggregated_output[i][j] = missing;
                }
            }
        }

        return aggregated_output;

    }

    // np.partition( scores, -2, axis=-1)[:, :, -2, np.newaxis]
    static std::vector<std::vector<std::vector<double>>> preAggregateHook(
        const std::vector<std::vector<std::vector<double>>>& scores) 
    {
        std::vector<std::vector<std::vector<double>>> result;

        for (size_t i = 0; i < scores.size(); ++i) {
            std::vector<std::vector<double>> slice;
            for (size_t j = 0; j < scores[i].size(); ++j) {
                std::vector<double> row = scores[i][j];
                std::nth_element(row.begin(), row.end() - 2, row.end());
                std::vector<double> new_row = {row[row.size() - 2]};
                slice.push_back(new_row);
            }
            result.push_back(slice);
        }

        return result;
    }

};

struct DisSeg
{
    int i;
    int j;
    bool operator==( const DisSeg& other )
    {
        if( i == other.i && j == other.j )
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};

class SegmentModel : public OnnxModel 
{
private:
    double m_duration = 5.0;
    double m_step = 0.5;
    int m_batch_size = 32;
    int m_sample_rate = 16000;
    size_t m_num_samples = 0;
    bool m_runOnGPU = false;


public:
    SegmentModel(const std::string& model_path, bool runOnGPU)
        : OnnxModel(model_path, runOnGPU) 
        , m_runOnGPU( runOnGPU )
    {
    }


    // input: batch size x channel x samples count, for example, 32 x 1 x 80000
    // output: batch size x 293 x 4
    std::vector<std::vector<std::vector<float>>> infer( const std::vector<std::vector<float>>& waveform )
    {
        // Create a std::vector<float> with the same size as the tensor
        // TODO: do memory copy for better performance, Maybe use pointer only rather than stl container
        std::vector<float> audio( m_batch_size * waveform[0].size(), 0.0 );
        for( size_t i = 0; i < waveform.size(); ++i )
        {   
            for( size_t j = 0; j < waveform[0].size(); ++j )
            {   
                audio[i*waveform[0].size() + j] = waveform[i][j];
            }   
        }   

        if( m_runOnGPU )
        {
            // ??? stupid but workable sleep here
            // without it, output is wrong
            // if sleep 10 milliseconds, we got
            // Mismatched elements: 1 / 1096992 (9.12e-05%) if run 
            // python verifyEveryStepResult.py for before_aggregation
            // and final result exact same.
            // if sleep less than 10 milliseconds, got more mismatch
            usleep( 100000 );
        }

        // batch_size * num_channels (1 for mono) * num_samples
        //const int64_t batch_size = waveform.size();
        const int64_t batch_size = m_batch_size;
        const int64_t num_channels = 1;
        int64_t input_node_dims[3] = {batch_size, num_channels,
            static_cast<int64_t>(waveform[0].size())};
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
                memory_info_, const_cast<float*>(audio.data()), audio.size(),
                input_node_dims, 3);
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.emplace_back(std::move(input_ort));

        // TODO: Call Run with user provided buffer to avoid extra memory copy
        // see Run signature in 
        // https://onnxruntime.ai/docs/api/c/struct_ort_1_1_session.html
        auto ort_outputs = session_->Run(
                Ort::RunOptions{nullptr}, input_node_names_.data(), ort_inputs.data(),
                ort_inputs.size(), output_node_names_.data(), output_node_names_.size());

        const float* outputs = ort_outputs[0].GetTensorData<float>();
        auto outputs_shape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int len1 = waveform.size();
        int len2 = outputs_shape[1];
        int len3 = outputs_shape[2];
        std::cout<<"output shape:"<<len1<<"x"<<len2<<"x"<<len3<<std::endl;
        std::vector<std::vector<std::vector<float>>> res( len1, 
                std::vector<std::vector<float>>( len2, std::vector<float>( len3 )));
        // TODO: do memory copy for better performance
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

    // pyannote/audio/core/inference.py:202
    std::vector<std::vector<std::vector<float>>> slide(const std::vector<float>& waveform, 
            SlidingWindow& res_frames, bool& has_last_chunk )
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
        int idx = 0;
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
//#ifdef WRITE_DATA
//        std::string tt = "cpp_audio";
//        tt += std::to_string( idx++ );
//        debugWrite2d( chunks, tt.c_str());
//#endif // WRITE_DATA
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
            // Process last chunk if have, last chunk may not equal window_size
            // Make sure at least we have 1 element remaining
            if( i + 1 < num_samples )
            {
                has_last_chunk = true;
                // Starting and Ending iterators
                auto start = waveform.begin() + i;
                auto end = waveform.end();

                // To store the sliced vector, always window_size, for last chunk we pad with 0.0
                std::vector<float> chunk( window_size, 0.0 );

                // Copy vector using copy function()
                std::copy(start, end, chunk.begin());
                chunks.push_back( chunk ); 
            }
            auto tmp = infer( chunks );
            for( const auto& a : tmp )
            {
                outputs.push_back( a );
            }

        }

        // pyannote/audio/core/inference.py:343
        //         def __aggregate(
        // TODO: implement above method

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

    std::vector<float> crop( const std::vector<float>& waveform, std::pair<double, double> segment) 
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
        data.insert(data.begin(), pad_start, 0.0);
        data.insert(data.end(), pad_end, 0.0);

        return data;
    }

}; // SegmentModel

class Binarize {
public:
    Binarize(
        float onset = 0.5f,
        std::optional<float> offset = std::nullopt,
        float min_duration_on = 0.0f,
        float min_duration_off = 0.0f,
        float pad_onset = 0.0f,
        float pad_offset = 0.0f) 
        : onset(onset),
        offset(offset.value_or(onset)),
        min_duration_on(min_duration_on),
        min_duration_off(min_duration_off),
        pad_onset(pad_onset),
        pad_offset(pad_offset) 
    {
        // onset = Uniform(0.0, 1.0)
        // offset = Uniform(0.0, 1.0)
    }

    // pyannote/audio/utils/signal.py:254
    Annotation operator()(const std::vector<std::vector<double>>& scores, 
            SlidingWindow& frames) const 
    {
        int num_frames = scores.size();
        int num_classes = scores[0].size();  // Assuming consistent dimensions
        assert( num_classes == 1 );

        std::vector<double> timestamps(num_frames);
        for (int i = 0; i < num_frames; ++i) 
        {
            // Extract middle timestamp from slidingWindow (assuming start, end pairs)
            timestamps[i] = (frames[i].start + frames[i].end) / 2.0f;
        }

        Annotation active;

        for (int k = 0; k < num_classes; ++k) 
        {
            // label = k if scores.labels is None else scores.labels[k]
            int label = k;

            float start = timestamps[0];
            bool is_active = scores[0][k] > onset;

            for (int i = 1; i < num_frames; ++i) 
            {
                float t = timestamps[i];
                float y = scores[i][k];

                if (is_active) 
                {
                    if (y < offset) 
                    {
                        Segment region(start - pad_onset, t + pad_offset);
                        active.addSegment(start - pad_onset, t + pad_offset, label );
                        start = t;
                        is_active = false;
                    }
                } 
                else 
                {
                    if (y > onset) 
                    {
                        start = t;
                        is_active = true;
                    }
                }
            }

            if (is_active) 
            {
                active.addSegment(start - pad_onset, timestamps.back() + pad_offset, label );
            }
        }

        if (pad_offset > 0.0f || pad_onset > 0.0f || min_duration_off > 0.0f) {
            active.support(min_duration_off);
        }

        if (min_duration_on > 0.0f) 
        {
            active.removeShort( min_duration_on );
        }

        return active;
    }

private:
    float onset;
    float offset;
    float min_duration_on;
    float min_duration_off;
    float pad_onset;
    float pad_offset;
};


// class detectOverlappedSpeech{ // TODO: add this 

double self_frame_step = 0.017064846416382253;
double self_frame_duration = 0.017064846416382253;
double self_frame_start = 0.0;


// pyannote/core/feature.py
// +
// pyannote/core/segment.py
// mode='loose', fixed=None
template<typename T>
std::vector<std::vector<T>> crop_segment( const std::vector<std::vector<T>>& data,
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
    //SlidingWindow res( start, src.step, src.duration, n_samples );
    SlidingWindow res( start, src.step, src.duration, n_samples * src.duration * src.sample_rate );
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
            for( size_t j = 0; j < data[i].size(); ++j )
                tmp.push_back( data[i][j] );
            cropped_data.push_back( tmp );
        }
    }

    return cropped_data;
}

Annotation detectOverlappedSpeech( const std::string& waveFile, const std::string& segmentModel, bool runOnGPU )
{
    wav::WavReader wav_reader( waveFile );
    int num_channels = wav_reader.num_channels();
    int bits_per_sample = wav_reader.bits_per_sample();
    int sample_rate = wav_reader.sample_rate();
    const float* audio = wav_reader.data();
    int num_samples = wav_reader.num_samples();
    std::vector<float> input_wav{audio, audio + num_samples};

    // Print audio samples
    for( int i = 0; i < num_samples; ++i )
    {
        input_wav[i] = input_wav[i]*1.0f/32768.0;
    }

    //*************************************
    // 1. Segmentation stage
    //*************************************
    auto beg_seg = timeNow();
    std::cout<<"\n---segmentation ---"<<std::endl;
    SegmentModel mm( segmentModel, runOnGPU );
    std::vector<std::pair<double, double>> segments;
    SlidingWindow res_frames;
    bool has_last_chunk = false;
    auto segmentations = mm.slide( input_wav, res_frames, has_last_chunk );

#ifdef WRITE_DATA
    debugWrite3d( segmentations, "cpp_before_aggregation" );
#endif // WRITE_DATA

    // From float to double
    std::vector<std::vector<std::vector<double>>> tmp( segmentations.size(), 
            std::vector<std::vector<double>>( segmentations[0].size(), 
                std::vector<double>( segmentations[0][0].size())));
    for( int i = 0; i < segmentations.size(); ++i )
    {
        for( int j = 0; j < segmentations[0].size(); ++j )
        {
            for( int k = 0; k < segmentations[0][0].size(); ++k )
            {
                tmp[i][j][k] = static_cast<double>( segmentations[i][j][k] ); 
            }
        }
    }
    // Before aggregate, will change last dimension to 1
    // np.partition( scores, -2, axis=-1)[:, :, -2, np.newaxis]
    // pyannote/audio/pipelines/overlapped_speech_detection.py:132
    auto temp = PipelineHelper::preAggregateHook( tmp );

#ifdef WRITE_DATA
    debugWrite3d( temp, "cpp_after_hook" );
#endif // WRITE_DATA
    
    // TODO, dont know where this(0.017xxxx) from
    // this function is in the the slide function in python, but we tick it out to here
    SlidingWindow activations_frames;
    SlidingWindow pre_frame( self_frame_start, self_frame_step, self_frame_duration );
    auto activations = PipelineHelper::aggregate( temp, 
            res_frames, 
            pre_frame, 
            activations_frames, 
            true, 0.0, false );
#ifdef WRITE_DATA
    debugWrite2d( activations, "cpp_after_aggregate" );
#endif // WRITE_DATA

    // Those hard code number from config file which was used during traning stage
    // https://huggingface.co/pyannote/overlapped-speech-detection/blob/main/config.yaml
    float onset = 0.8104268538848918;
    float offset = 0.4806866463041527;
    float min_duration_off = 0.09791355693027545;
    float min_duration_on = 0.05537587440407595;
    Binarize binarize( onset, offset, min_duration_on, min_duration_off );

    /*
     * # remove padding that was added to last chunk
     * if has_last_chunk:
     *      aggregated.data = aggregated.crop(
     *          Segment(0.0, num_samples / sample_rate), mode="loose"
     *      )
     * */
    if( has_last_chunk )
    {
        Segment focus( 0.0, num_samples*1.0/sample_rate);
        SlidingWindow cropped_activations_frames;
        auto cropped_activations = crop_segment( activations, activations_frames, focus, 
                cropped_activations_frames );

#ifdef WRITE_DATA
    debugWrite2d( cropped_activations , "cpp_segmentations" );
#endif // WRITE_DATA

        return binarize( cropped_activations, cropped_activations_frames );
    }
    else
    {
#ifdef WRITE_DATA
    debugWrite2d( activations, "cpp_segmentations" );
#endif // WRITE_DATA

        return binarize( activations, activations_frames );
    }


}

void test()
{
    /*
    std::cout<<0.5<<" - "<<Helper::np_rint( 0.5 )<<std::endl;
    std::cout<<1.1<<" - "<<Helper::np_rint( 1.2 )<<std::endl;
    std::cout<<1.5<<" - "<<Helper::np_rint( 1.5 )<<std::endl;
    std::cout<<-1.5<<" - "<<Helper::np_rint( -1.5 )<<std::endl;
    std::cout<<-2.5<<" - "<<Helper::np_rint( -2.5 )<<std::endl;
    std::cout<<-3.5<<" - "<<Helper::np_rint( -3.5 )<<std::endl;
    std::cout<<2.5<<" - "<<Helper::np_rint( 2.5 )<<std::endl;
    std::cout<<3.5<<" - "<<Helper::np_rint( 3.5 )<<std::endl;
    std::cout<<3.6<<" - "<<Helper::np_rint( 3.6 )<<std::endl;
    */

    std::cout<<"Testing closest frame"<<std::endl;
    SlidingWindow sw( 0, 0.016875, 0.016875 );
    double start = 0.0;
    std::vector<std::pair<int, float>> frames;
    for( int i = 0; i < 10000; ++i )
    {
        auto cf = sw.closest_frame( start );
        frames.emplace_back( cf, start );
        start += 0.5;
    }

    // Read expected result from file
    std::vector<std::pair<int, float>> expected_frames;
    std::ifstream fd1("../src/test/closest_frame.txt"); //taking file as inputstream
    if( fd1 ) 
    {
        for( std::string line; getline( fd1, line ); )
        {
            std::string delimiter = ",";
            std::vector<std::string> v = Helper::split(line, delimiter);
            assert( v.size() == 2 );
            expected_frames.emplace_back( std::stoi( v[0] ), std::stof( v[1] ));
        }
    }
    assert( frames == expected_frames );
    std::cout<<"==> passed"<<std::endl;

}

std::stringstream toHHMMSS( float ms ) 
{

    std::stringstream os;
    int h = ms / (60 * 60);
    ms -= h * (60 * 60);

    int m = ms / (60);
    ms -= m * (60);

    int s = ms;
    ms -= s;
    int n = ms * 1000;

    os << std::setfill('0') << std::setw(2) << h << ':' << std::setw(2) << m
              << ':' << std::setw(2) << s << '.' << std::setw(3) << n;
    return os;
}

void printRunInfo( const char* model, const char* wavFile, bool runOnGPU )
{
    std::cout<<"----------------------------------------------------"<<std::endl;
    std::cout<<"Model file: "<<model<<std::endl;
    std::cout<<"Wav file: "<<wavFile<<std::endl;
    std::cout<<"Run inference on: "<<(runOnGPU?"GPU":"CPU")<<std::endl;
    std::cout<<"----------------------------------------------------"<<std::endl;
    std::cout<<std::endl;
}

int main(int argc, char* argv[]) 
{
    //test();
    //return 0;
    if( argc < 3 )
    {
        std::cout<<"program [segment model file] [wave file]"<<std::endl;
        return 0;
    }

    auto beg = timeNow();
    std::string segmentModel( argv[1] );
    std::string waveFile( argv[2] );
    bool runOnGPU = false;
    if( argc == 4 && strcasecmp( argv[3], "GPU" ) == 0 )
        runOnGPU = true;
    printRunInfo( argv[1], argv[2], runOnGPU );
    auto res = detectOverlappedSpeech( waveFile, segmentModel, runOnGPU );

    std::cout<<"\n----Summary----"<<std::endl;
    timeCost( beg, "Time cost" );
    std::cout<<"----------------------------------------------------"<<std::endl;
    auto diaRes = res.finalResult();
    for( const auto& dr : diaRes )
    {
        std::cout<<"[ "<<toHHMMSS( dr.start ).str()<<" --> "<<toHHMMSS(dr.end).str()<<" ]"<<std::endl;
    }
    std::cout<<"----------------------------------------------------"<<std::endl;
}
