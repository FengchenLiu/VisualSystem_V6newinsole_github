#include <iostream>
#include <string>
#include <vector>

#include "GraphItem.h"
#include "NormalModel.h"

using std::cout;
using std::cin;
using std::cerr;
using std::endl;

void test();

int main() {
    cout << "Hello, World!" << endl;

    // test our classes
    test();


    return 0;
}

void test(){
    cout << "Test function! ------------" << endl;
    cout << "Traid Area: " << endl;
    Triad center1, center2(10.2, 10.8, 11.6);

    cout << "center 1:" << endl;
    cout << "(" << center1.getX() << ", " << center1.getY() << ", " << center1.getZ() << ")" << endl;

    cout << "center 2:" << endl;
    cout << "(" << center2.getX() << ", " << center2.getY() << ", " << center2.getZ() << ")" << endl;

    // set cneter2
    center2.setX(50.1);
    center2.setY(50.2);
    center2.setZ(50.3);

    cout << "center 2:" << endl;
    cout << "(" << center2.getX() << ", " << center2.getY() << ", " << center2.getZ() << ")" << endl;

    std::vector<float> vecCenter1 = center1.cvtStdVec();

    cout << "std vector center1:" << endl;
    cout << "The size of standard vector 1: " << vecCenter1.size() << endl;
    cout << "(" << vecCenter1[0] << ", " << vecCenter1[1] << ", " << vecCenter1[2] << ")" << endl;


    cout << "Bounding Box Area" << endl;
    BoundingBox bbox1, bbox2(center1, 10, 10, 10),
    bbox3(20.1, 20.2, 20.3, 50, 50, 50);

    cout << endl << "bounding box 1: " << endl;
    bbox1.print();

    cout << endl << "bounding box 2: " << endl;
    bbox2.print();

    cout << endl << "bounding box 3: " << endl;
    bbox3.print();

    bbox3.setCenter(Triad(30.5, 30.6, 30.7));
    cout << endl << "center of the bounding box" << endl;
    bbox3.print();

    bbox3.setCenter(40.5, 40.6, 40.7);
    cout << endl << "center of the bounding box" << endl;
    bbox3.print();

    bbox3.setHeight(100.1);
    bbox3.setLength(100.2);
    bbox3.setWidth(100.3);

    cout << "changes in length, height, width:" << endl;
    bbox3.print();

    cout << endl << "Plane Area" << endl;
    Plane plane1, plane2(Triad(2.1, 2.2, 2.3), Triad(2.4, 2.5, 2.6),
                         BoundingBox(3.1, 3.2, 3.3, 3.4, 3.5, 3.6), 2);
    Plane plane3(10.1, 10.2, 10.3, 0.5, 0.6, 0.7,
                 11.1, 11.2, 11.3, 15.1, 15.2, 15.3, 3);

    cout << endl << "print info of plane1: " << endl;
    plane1.print();

    cout << endl << "print info of plane2: " << endl;
    plane2.print();

    cout << endl << "print info of plane3: " << endl;
    plane3.print();

    // getter test
    cout << "plane2 center getter" << endl;
    plane2.getCenter().print();

    cout << "plane2 normal getter" << endl;
    plane2.getNormal().print();

    cout << "plane2 boundingbox getter" << endl;
    plane2.getBoundingBox().print();

    // setter
    plane2.setCenter(Triad(4.1, 4.2, 4.3));
    plane2.setNormal(Triad(4.4, 4.5, 4.6));
    plane2.setBoundingBox(BoundingBox(4.1, 4.2, 4.3, 4.4, 4.5, 4.6));

    cout << "changed plane2 bounding box getter" << endl;
    plane2.print();

    /**************************************Stair Plane Test Area**************************************/
    // stair
    Plane sPlane1(Triad(0, 0, 0), Triad(0, 1, 0),
                  BoundingBox(-10, -10, 0, 10, 10, 10), 1);
    Plane sPlane2(Triad(0, 15, 25), Triad(0, 1, 0),
                  BoundingBox(-10, -10, 0, 10, 10, 10), 2);
    Plane sPlane3(Triad(0, 30, 50), Triad(0, 1, 0),
                  BoundingBox(-10, -10, 0, 10, 10, 10), 3);
    Plane nsPlane1(Triad(0, -20, -45), Triad(0, 1, 0),
                  BoundingBox(-10, -10, 0, 10, 10, 10), 4);
    Plane nsPlane2(Triad(0, 55, 90), Triad(0, 1, 0),
                  BoundingBox(-10, -10, 0, 10, 10, 10), 5);

    std::vector<Plane> prePlanes{sPlane2, sPlane1, sPlane3, nsPlane1, nsPlane2};

    Stair stair1;

    stair1.constructFromPlanes(prePlanes);
    cout << "number of planes of stair: " << stair1.getNumPlanes() << endl;
    //stair1.print();

    Plane esPlane1(Triad(0, 45, 75), Triad(0, 1, 0),
                   BoundingBox(-10, -10, 0, 10, 10, 10), 6);
    Plane esPlane2(Triad(0, -15, -25), Triad(0, 1, 0),
                   BoundingBox(-10, -10, 0, 10, 10, 10), 7);
    Plane ensPlane1(Triad(0, 100, -25), Triad(0, 1, 0),
                   BoundingBox(-10, -10, 0, 10, 10, 10), 8);

    std::vector<Plane> exPlanes{esPlane1, esPlane2, ensPlane1};

    stair1.mergeFromPlanes(exPlanes);
    cout << "size of remained vector: " << exPlanes.size() << endl;
    cout << "number of planes of merged stair: " << stair1.getNumPlanes() << endl;

    stair1.print();

    cout << "stair id: " << stair1.getId() << endl;
    stair1.setId(1);
    cout << "stair id: " << stair1.getId() << endl;

    Plane scPlane1(Triad(0, 0.5, 0), Triad(0, 1, 0),
                  BoundingBox(-10, -10, 30, 80, 80, 80), 1);
    Plane scPlane2(Triad(0, 14.8, 25), Triad(0, 1, 0),
                  BoundingBox(-10, -10, 20, 30, 40, 60), 2);

    std::vector<Plane> changedPlanes{scPlane1, scPlane2};

    bool changedResult = stair1.updateFromPlanes(changedPlanes);
    cout << "changed result from updated detected vectors: " << changedResult << endl;

    stair1.print();
    /**************************************Stair Plane Test Area**************************************/

    /**************************************Ramp Plane Test Area**************************************/
    Plane rPlane1(Triad(0, 0, 0), Triad(0, 0.95, 0),
                  BoundingBox(-10, -10, 0, 10, 10, 10), 1);
    Plane rsPlane2(Triad(0, 15, 25), Triad(0, 1, 0),
                  BoundingBox(-10, -10, 0, 10, 10, 10), 2);
    Plane rsPlane3(Triad(0, 30, 50), Triad(0, 1, 0),
                  BoundingBox(-10, -10, 0, 10, 10, 10), 3);

    std::vector<Plane> rpTestDetectedPlanes{rPlane1, rsPlane2, rsPlane3};

    cout << "size of rp test detected planes (phase 1): " << rpTestDetectedPlanes.size() << endl;
    Ramp ramp1;

    cout << "size of ramp 1 before extraction: " << ramp1.getNumPlanes() << endl;
    bool rampExtractionResult = ramp1.extractFromPlanes(rpTestDetectedPlanes, 1);
    if (rampExtractionResult){
        ramp1.setId(1);
    }
    ramp1.print();
    cout << "size of ramp 1 after extraction: " << ramp1.getNumPlanes() << endl;

    cout << "size of rp test detected planes (phase 2): " << rpTestDetectedPlanes.size() << endl;

    Stair stair2;
    cout << "size of stair 2 before construction: " << stair2.getNumPlanes() << endl;
    stair2.constructFromPlanes(rpTestDetectedPlanes);
    if (stair2.getNumPlanes() != 0) {
        stair2.setId(2);
    }
    stair2.print();
    cout << "size of stair 2 after construction: " << stair2.getNumPlanes() << endl;

    cout << "size of rp test detected planes (phase 3): " << rpTestDetectedPlanes.size() << endl;
    /**************************************Ramp Plane Test Area**************************************/

    /**************************************Level Ground Plane Test Area**************************************/
    Plane lgPlane1(Triad(0, 0, 0), Triad(0, 1, 0),
                   BoundingBox(-10, -10, 0, 10, 10, 10), 1);
    Plane lgPlane2(Triad(0, 15, 25), Triad(0, 1, 0),
                   BoundingBox(-10, -10, 0, 10, 10, 10), 2);
    Plane lgPlane3(Triad(0, 30, 50), Triad(0, 1, 0),
                   BoundingBox(-10, -10, 0, 10, 10, 10), 3);
    Plane lgPlane4(Triad(0, 45, 75), Triad(0, 1, 0),
                   BoundingBox(-10, -10, 0, 10, 10, 10), 4);
    Plane lgPlane5(Triad(0, 45, 200), Triad(0, 1, 0),
                   BoundingBox(-10, -10, 0, 1000, 10, 1000), 5);
    Plane lgPlane6(Triad(0, 45, 300), Triad(0, 0.97, 0),
                   BoundingBox(-10, -10, 0, 10, 10, 10), 6);

    std::vector<Plane> vecLevelGroundPlanes{lgPlane5, lgPlane2, lgPlane4, lgPlane1, lgPlane3, lgPlane6};

    cout << "size of level ground planes (phase 1): " << vecLevelGroundPlanes.size() << endl;
    Ramp lgRamp1;
    cout << "size of lgRamp1 before: " << lgRamp1.getNumPlanes() << endl;
    lgRamp1.extractFromPlanes(vecLevelGroundPlanes);
    cout << "size of lgRamp1 after: " << lgRamp1.getNumPlanes() << endl;
    cout << "size of level ground planes (phase 2): " << vecLevelGroundPlanes.size() << endl;

    Stair lgStair1;
    cout << "size of lgStair1 before: " << lgStair1.getNumPlanes() << endl;
    lgStair1.constructFromPlanes(vecLevelGroundPlanes);
    cout << "size of lgStair1 after: " << lgStair1.getNumPlanes() << endl;
    cout << "size of level ground planes (phase 3): " << vecLevelGroundPlanes.size() << endl;

    LevelGround lgLevelGround1;
    cout << "size of lgLevelGround1 before: " << lgLevelGround1.getNumPlanes() << endl;
    lgLevelGround1.extractFromPlanes(vecLevelGroundPlanes, 0);
    cout << "size of lgLevelGround1 after: " << lgLevelGround1.getNumPlanes() << endl;
    cout << "size of level ground planes (phase 4): " << vecLevelGroundPlanes.size() << endl;

    /**************************************Level Ground Plane Test Area**************************************/

    /**********************************Normal Dist Module Test Area**********************************/
    // Normal Distribution Test Area
    cout << "Normal Dist Module Test Area **********************************************" << endl;
    NormalDistribution normalDist(0, 1);
    cout << "prob: " << normalDist.calProbability(0) << endl;

    MixedNormalDistribution mnd(50, 87.5, 10, 20);

    cout << "std of normal 1: " << mnd.getNormal1().getStd() << endl;
    cout << "mean of normal 1: " << mnd.getNormal1().getMean() << endl;

    cout << "std of normal 2: " << mnd.getNormal2().getStd() << endl;
    cout << "mean of normal 2: " << mnd.getNormal2().getMean() << endl;

    mnd.calGreaterModel(80);
    std::pair<float, float> probabilities = mnd.calTwoProbabilities(80);

    cout << "prob1: " << probabilities.first << ", prob2: " << probabilities.second << endl;

    normalTwoProbabilities(probabilities.first, probabilities.second);

    cout << "normalized prob1: " << probabilities.first << ", normalized prob2: " << probabilities.second << endl;

    /**********************************Normal Dist Module Test Area**********************************/
}