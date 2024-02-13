#include <omp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

namespace KMEANS
{

class Point
{
private:
    int pointId, clusterId;
    int dimensions;
    vector<double> values;

    vector<double> lineToVec(string &line);

public:
    Point(int id, string line);

    int getDimensions() { return dimensions; }

    int getCluster() { return clusterId; }

    int getID() { return pointId; }

    void setCluster(int val) { clusterId = val; }

    double getVal(int pos) { return values[pos]; }
};

class Cluster
{
private:
    int clusterId;
    vector<double> centroid;
    vector<Point> points;

public:
    Cluster(int clusterId, Point centroid);

    void addPoint(Point p);

    bool removePoint(int pointId);

    void removeAllPoints() { points.clear(); }

    int getId() { return clusterId; }

    Point getPoint(int pos) { return points[pos]; }

    int getSize() { return points.size(); }

    double getCentroidByPos(int pos) { return centroid[pos]; }

    void setCentroidByPos(int pos, double val) { this->centroid[pos] = val; }
};

class KMeans
{
private:
    int K, iters, dimensions, total_points;
    vector<Cluster> clusters;
    string output_dir;

    void clearClusters();

    int getNearestClusterId(Point point);

public:
    KMeans(int K, int iterations, string output_dir);
    void run(vector<Point> &all_points);
};

} // namespace KMEANS
