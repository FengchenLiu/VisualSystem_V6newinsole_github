/*********************Transition Codes*********************/
/*                                                        */
/*                                                        */
/*********************Transition Codes*********************/
            /* auto currBack = vecPwcToeTriggerProjection.back();
            auto predBack = vecPredPosProjection.back();
            auto stepLengthBack = vecStepLength.back();
            auto posDistanceBack = vecDistances.back();

            auto currProjNormal = vecPwcToeProjNormal.back();
            auto predProjNormal = vecPredProjNormal.back();

            globalPredictionSlope = calSlopePlane(Eigen::Vector3d(predProjNormal[0], predProjNormal[1], predProjNormal[2])) / M_PI * 180;

            double disHeightAbs = calHeightDistancdArrayAbs(currBack, predBack);

            double disHeight = calHeightDistancdArray(currBack, predBack); */


            // LG -> SA -----------------------------------------------------------------------------------
            /* if (currBack[2] != 0 &&
                predBack[2] != 0 &&
                disHeightAbs > EFFECTIVE_HEIGHT_THRESHOLD && 
                disHeightAbs < 0.8 &&
                currLocomotionMode == 0 && 
                disHeight < 0 // &&
                //(calSlopePlane(Eigen::Vector3d(predProjNormal[0], predProjNormal[1], predProjNormal[2])) / M_PI * 180 < 7 ||
                // calSlopePlane(Eigen::Vector3d(predProjNormal[0], predProjNormal[1], predProjNormal[2])) / M_PI * 180 > 15)
                ) {
                
                std::cout << "pred plane id golbal: " << std::endl
                          << predPlaneIdGlobal << std::endl;

                if (surfaces.count(predPlaneIdGlobal)) {
                    std::cout << "pred plane id global: " << std::endl
                              << predPlaneIdGlobal << std::endl;

                    auto predPlaneCenter = surfaces[predPlaneIdGlobal] -> GetCenter();
                
                    std::cout << "Vwc" << std::endl
                              << currVwc << std::endl;

                    std::cout << "mode Vwc: " << std::endl
                              << sqrt(currVwc[0] * currVwc[0] + currVwc[1] * currVwc[1] + currVwc[2] * currVwc[2]) << std::endl;

                    std::cout << "dis pwc stair" << std::endl
                              << calDistancePwcStair(currBack, Triad(predPlaneCenter[0] * 1000, 
                                                        predPlaneCenter[1] * 1000, 
                                                        predPlaneCenter[2] * 1000)) << std::endl;

                    if (sqrt(currVwc[0] * currVwc[0] + currVwc[1] * currVwc[1] + currVwc[2] * currVwc[2]) < 0.5 && 
                        stepLengthBack > 0.5 &&
                        calDistancePwcStair(currBack, Triad(predPlaneCenter[0] * 1000, 
                                                        predPlaneCenter[1] * 1000, 
                                                        predPlaneCenter[2] * 1000)) > 0.75) {
                    
                    }
                    else {
                    //std::cout << "LG -> SA" << std::endl;
                    currLocomotionMode = 1;
                    }
                }
                else {
                    //std::cout << "LG -> SA" << std::endl;
                    currLocomotionMode = 1;
                }
            } */
            // LG -> SA -----------------------------------------------------------------------------------

            // SA -> LG -----------------------------------------------------------------------------------
            /* if (currBack[2] != 0 &&
                predBack[2] != 0 &&
                disHeightAbs < EFFECTIVE_HEIGHT_THRESHOLD_2 && 
                currLocomotionMode == 1 // &&
                // calSlopePlane(Eigen::Vector3d(predProjNormal[0], predProjNormal[1], predProjNormal[2])) / M_PI * 180 < 10
                ) {

                if (vecStairs.empty()) {
                    globalTransitionType = 2;
                    currLocomotionMode = 0;
                }
                else if (numSASteps >= vecStairs[0].getStairPlanes().size() + 1) {
                    globalTransitionType = 2;
                    currLocomotionMode = 0;
                }
                else {
                    globalTransitionType = 2;
                    currLocomotionMode = 0;
                }
            } */
            // SA -> LG -----------------------------------------------------------------------------------

            // LG -> SD -----------------------------------------------------------------------------------
            /* if (currBack[2] != 0 &&
                predBack[2] != 0 &&
                disHeightAbs > EFFECTIVE_HEIGHT_THRESHOLD && 
                currLocomotionMode == 0 && 
                disHeight > 0 
                && disHeightAbs < 0.6
                && vecStepLength.size() >= 2) {

                std::cout << "In LG -> SD if condition!" << std::endl;

                if (!vecStairs.empty() &&
                    !vecStairs[0].getStairPlanes().empty()) {
                    std::cout << "calDistancePwcStair" << std::endl
                              << calDistancePwcStair(currBack, vecStairs[0].getStairPlanes().back().getCenter()) << std::endl;
                }

                if ((!vecStairs.empty() &&
                    !vecStairs[0].getStairPlanes().empty() &&
                    stepLengthBack > 0.40 && 
                    calDistancePwcStair(currBack, vecStairs[0].getStairPlanes().back().getCenter()) > 0.55) ||
                    (!vecStairs.empty() &&
                    !vecStairs[0].getStairPlanes().empty() &&
                    calDistancePwcStair(currBack, vecStairs[0].getStairPlanes().back().getCenter()) > 1.8)
                    ) {
                }
                else {
                    std::cout << "LG -> SD" << std::endl;
                    std::cout << "1!!!!!!!!!" << std::endl;
                    currLocomotionMode = 2;
                }
            }
            else if (currBack[2] !=0 &&
                     currLocomotionMode == 0 &&
                     vecPwcToeTriggerProjection[vecPwcToeTriggerProjection.size() - 2][2] != 0 &&
                     abs((*(vecPwcToeTriggerProjection.end() - 2))[2] - currBack[2]) > EFFECTIVE_HEIGHT_THRESHOLD &&
                     (*(vecPwcToeTriggerProjection.end() - 2))[2] - currBack[2] > 0 &&
                     (*(vecPwcToeTriggerProjection.end() - 2))[2] - currBack[2] < 0.6){
                currLocomotionMode = 2;
                std::cout << "LG -> SD" << std::endl;
                std::cout << "2!!!!!!!!" << std::endl;
            } */
            // LG -> SD -----------------------------------------------------------------------------------

            // SD -> LG -----------------------------------------------------------------------------------
            /* if (currBack[2] != 0 &&
                predBack[2] != 0 &&
                disHeightAbs < EFFECTIVE_HEIGHT_THRESHOLD_2 && 
                currLocomotionMode == 2 //&&
                //!vecStairs.empty() //&&
                //numSDSteps >= vecStairs[0].getStairPlanes().size() + 1
                ) {
                std::cout << "SD -> LG" << std::endl;
                currLocomotionMode = 0;
            } */
            // SD -> LG -----------------------------------------------------------------------------------

            // LG -> RA -----------------------------------------------------------------------------------
            /* if (//predPlaneIdGlobal !=0 &&
                //!vecRamps.empty() && 
                currLocomotionMode == 0 &&
                currBack[2] != 0 &&
                predBack[2] != 0 &&
                disHeight < 0 &&
                disHeightAbs >= 0.05 &&
                disHeightAbs < 0.25 &&
                checPredictionIndex(vecRamps, predPlaneIdGlobal)
                //sqrt(currVwc[0] * currVwc[0] + currVwc[1] * currVwc[1] + currVwc[2] * currVwc[2]) > 0.5
                ) {
                
                std::cout << "LG -> RA Area *****************" << std::endl;
                currLocomotionMode = 3;
            } */
            // LG -> RA -----------------------------------------------------------------------------------

            // RA -> LG -----------------------------------------------------------------------------------
            /* if (//predPlaneIdGlobal !=0 &&
                !vecRamps.empty() &&
                currLocomotionMode == 3 &&
                currBack[2] != 0 &&
                predBack[2] != 0 &&
                disHeightAbs < 0.02) {

                std::cout << "RA -> LG Area *****************" << std::endl;
                currLocomotionMode = 0;
            } */
            // RA -> LG -----------------------------------------------------------------------------------

            // LG -> RD -----------------------------------------------------------------------------------
            /* if (predPlaneIdGlobal !=0 &&
                !vecRamps.empty() &&
                currLocomotionMode == 0 &&
                checPredictionIndex(vecRamps, predPlaneIdGlobal) &&
                stepLengthBack < 1 &&
                (currTimeStamp - RD_TimeStamp) > 2) {
                
                std::cout << "LG -> RD Area *****************1" << std::endl;
                RD_TimeStamp = currTimeStamp;
                currLocomotionMode = 4;
            }
            else if (predPlaneIdGlobal !=0 &&
                     currBack[2] != 0 &&
                     predBack[2] != 0 &&
                     disHeight > 0 &&
                     disHeightAbs >= 0.07 &&
                     disHeightAbs <= 0.2 &&
                     currLocomotionMode == 0 &&
                     stepLengthBack < 1 &&
                     (currTimeStamp - RD_TimeStamp) > 2) {
                
                std::cout << "LG -> RD Area *****************2" << std::endl;
                RD_TimeStamp = currTimeStamp;
                currLocomotionMode = 4;
            } */
            // LG -> RD -----------------------------------------------------------------------------------

            // RD -> LG -----------------------------------------------------------------------------------
            /* if (//predPlaneIdGlobal !=0 &&
                !vecRamps.empty() &&
                currLocomotionMode == 4 &&
                currBack[2] != 0 &&
                predBack[2] != 0 && 
                disHeightAbs < 0.04 &&
                (currTimeStamp - RD_TimeStamp) > 2
                //!checPredictionIndex(vecRamps, currPwcPlaneId) &&
                //stepLengthBack < 1
                ) {
                std::cout << "RD -> LG Area *****************" << std::endl;
                RD_TimeStamp = currTimeStamp;
                currLocomotionMode = 0;
            } */
            // RD -> LG -----------------------------------------------------------------------------------

            // SA -> RA -----------------------------------------------------------------------------------
            /* if (currLocomotionMode == 1 &&
                calSlopePlane(Eigen::Vector3d(predProjNormal[0], predProjNormal[1], predProjNormal[2])) / M_PI * 180 > 6 &&
                calSlopePlane(Eigen::Vector3d(predProjNormal[0], predProjNormal[1], predProjNormal[2])) / M_PI * 180 < 20 &&
                !vecRamps.empty()) {
                currLocomotionMode = 3;
                globalTransitionType = 1;
            } */
            // SA -> RA -----------------------------------------------------------------------------------

            // RA -> SA -----------------------------------------------------------------------------------
            /* if (currLocomotionMode == 3 &&
                (calSlopePlane(Eigen::Vector3d(predProjNormal[0], predProjNormal[1], predProjNormal[2])) / M_PI * 180 < 6) &&
                !vecStairs.empty()) {
                currLocomotionMode = 1;
                globalTransitionType = 3;
            } */
            // RA -> SA -----------------------------------------------------------------------------------
            // exit(1);

/*********************Transition Codes*********************/
/*                                                        */
/*                                                        */
/*********************Transition Codes*********************/
