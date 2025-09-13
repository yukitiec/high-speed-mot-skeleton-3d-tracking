#include "triangulation.h"

bool Triangulation::compareVectors(const std::vector<int>& a, const std::vector<int>& b)
{
    /**
    * @brief compare 2 data with first element. here first element is index of the left object
    */
    return a[0] < b[0];
}

void Triangulation::sortData(std::vector<std::vector<int>>& data)
{
    /**
    * @brief sort Data to arrange the data in index-left ascending order. like {{1,~},{2,~},{3,~}...}, data style is like {{idx_left,idx_right},....}
    * @param[in] data format is like {{idx_left,idx_right},....}
    */
    std::sort(data.begin(), data.end(), [this](const std::vector<int>& a, const std::vector<int>& b) {
        return compareVectors(a, b);
        });
}

void Triangulation::triangulation(std::vector<std::vector<std::vector<double>>>& data_left, std::vector<std::vector<std::vector<double>>>& data_right,
    std::vector<std::vector<int>>& matchingIndexes, std::vector<std::vector<std::vector<double>>>& data_3d)
{
    /**
    * @brief triangulate 3D points based on sequential data (seqData_left/right) and matching pairs from Matching.main()
    * @param[in] data_left/data_right shape is like {n_objects, sequence, (frame,label,left,top,width,height)}
    * @param[in] matchingIndexes shape is like {n_pairs, (index_left,index_right)}
    * @param[out] data_3d shape is like {num of objects, sequential, { frameIndex, X,Y,Z }}
    */

    //for all matching data
    int index_left;
    std::vector<double> left, right;
    std::vector<cv::Point2d> ps_left, ps_right;
    std::vector<cv::Point3d> ps_3d;
    double x_left, x_right, y_left, y_right;
    //std::cout << "0-0" << std::endl;
    for (std::vector<int>& matchIndex : matchingIndexes) //triangulate 3D points in the index-left ascending way
    {
        //calculate objects
        //left
        left = data_left[matchIndex[0]].back(); //{frameIndex, classLabel,left,top,width,height}
        x_left = left[2] + left[4] / 2.0;
        y_left = left[3] + left[5] / 2.0;
        ps_left.push_back(cv::Point2d(x_left, y_left));

        //right
        right = data_right[matchIndex[1]].back(); //{frameIndex, classLabel,left,top,width,height}
        x_right = right[2] + right[4] / 2.0;
        y_right = right[3] + right[5] / 2.0;
        ps_right.push_back(cv::Point2d(x_right, y_right));
    }
    //std::cout << "0-1" << std::endl;
    //triangulate points
    cal3D(ps_left, ps_right, 0, ps_3d);//0:cv::triangulatePoints, 1:stereo triangulation. 0 is better.
    //std::cout << "0-2" << std::endl;
    //save data in the data_3d
    double frame, label;
    int counter = 0;
    double X, Y, Z, X_prev, Y_prev, Z_prev, diff;
    for (std::vector<int>& matchIndex : matchingIndexes) //triangulate 3D points in the index-left ascending way
    {
        index_left = matchIndex[0];
        frame = data_left[index_left].back()[0];
        label = data_left[index_left].back()[1];
        X = ps_3d[counter].x;
        Y = ps_3d[counter].y;
        Z = ps_3d[counter].z;
        //std::cout << "0-3" << std::endl;
        //std::cout << "data_3d.size()=" << data_3d.size() << std::endl;
        if (!data_3d[index_left].empty()) {//not empty
            if (data_3d[index_left].back()[0] == 0.0)//not saved yet-> update with new data.
                data_3d[index_left][0] = std::vector<double>{ frame,label,X,Y,Z };//first element
            else {
                X_prev = data_3d[index_left].back()[2];
                Y_prev = data_3d[index_left].back()[3];
                Z_prev = data_3d[index_left].back()[4];
                diff = std::pow(((X - X_prev) * (X - X_prev) + (Y - Y_prev) * (Y - Y_prev) + (Z - Z_prev) * (Z - Z_prev)), 0.5);
                //if (diff < 1000.0)//1.0m/frame . ~150km/h
                data_3d[index_left].push_back({ frame,label,X,Y,Z });//{frame,label,x,y,z} in robot base frame.after second data.
            }
            //std::cout << "0-4" << std::endl;
        }
        counter++;
    }
}

void Triangulation::cal3D(std::vector<cv::Point2d>& pts_left, std::vector<cv::Point2d>& pts_right, int method_triangulate, std::vector<cv::Point3d>& results)
{

    //std::cout << "3" << std::endl;
    if (method_triangulate == 0) {//DLT (Direct Linear Translation)
        dlt(pts_left, pts_right, results);
    }
    else if (method_triangulate == 1) {//stereo triangulation
        stereo3D(pts_left, pts_right, results);
    }
}

void Triangulation::dlt(std::vector<cv::Point2d>& points_left, std::vector<cv::Point2d>& points_right, std::vector<cv::Point3d>& results)
{
    /**
    * @brief calculate 3D points with DLT method
    * @param[in] points_left, points_right {n_data,(xCenter,yCenter)}
    * @param[out] reuslts 3D points storage. shape is like (n_data, (x,y,z))
    */
    cv::Mat points_left_mat(points_left);
    cv::Mat undistorted_points_left_mat;
    cv::Mat points_right_mat(points_right);
    cv::Mat undistorted_points_right_mat;

    // Undistort the points
    cv::undistortPoints(points_left_mat, undistorted_points_left_mat, cameraMatrix_left, distCoeffs_left);
    cv::undistortPoints(points_right_mat, undistorted_points_right_mat, cameraMatrix_right, distCoeffs_right);

    // Reproject normalized coordinates to pixel coordinates
    cv::Mat normalized_points_left(undistorted_points_left_mat.rows, 1, CV_64FC2);
    cv::Mat normalized_points_right(undistorted_points_right_mat.rows, 1, CV_64FC2);

    for (int i = 0; i < undistorted_points_left_mat.rows; ++i) {
        double x, y;
        x = undistorted_points_left_mat.at<cv::Vec2d>(i, 0)[0];
        y = undistorted_points_left_mat.at<cv::Vec2d>(i, 0)[1];
        normalized_points_left.at<cv::Vec2d>(i, 0)[0] = cameraMatrix_left.at<double>(0, 0) * x + cameraMatrix_left.at<double>(0, 2);
        normalized_points_left.at<cv::Vec2d>(i, 0)[1] = cameraMatrix_left.at<double>(1, 1) * y + cameraMatrix_left.at<double>(1, 2);

        x = undistorted_points_right_mat.at<cv::Vec2d>(i, 0)[0];
        y = undistorted_points_right_mat.at<cv::Vec2d>(i, 0)[1];
        normalized_points_right.at<cv::Vec2d>(i, 0)[0] = cameraMatrix_right.at<double>(0, 0) * x + cameraMatrix_right.at<double>(0, 2);
        normalized_points_right.at<cv::Vec2d>(i, 0)[1] = cameraMatrix_right.at<double>(1, 1) * y + cameraMatrix_right.at<double>(1, 2);
    }

    // Output matrix for the 3D points
    cv::Mat triangulated_points_mat;

    // Triangulate points
    cv::triangulatePoints(projectMatrix_left, projectMatrix_right, normalized_points_left, normalized_points_right, triangulated_points_mat);
    //cv::triangulatePoints(projectMatrix_left, projectMatrix_right, undistorted_points_left_mat, undistorted_points_right_mat, triangulated_points_mat);

    // Convert homogeneous coordinates to 3D points
    triangulated_points_mat = triangulated_points_mat.t();
    cv::convertPointsFromHomogeneous(triangulated_points_mat.reshape(4), triangulated_points_mat);

    // Access triangulated 3D points
    results.clear();

    for (int i = 0; i < triangulated_points_mat.rows; i++) {
        cv::Point3d point;
        point.x = triangulated_points_mat.at<double>(i, 0);
        point.y = triangulated_points_mat.at<double>(i, 1);
        point.z = triangulated_points_mat.at<double>(i, 2);
        results.push_back(point);
    }

    // Convert from camera coordinate to robot base coordinateS
    for (auto& point : results) {
        double x = point.x;
        double y = point.y;
        double z = point.z;
        point.x = (transform_cam2base.at<double>(0, 0) * x + transform_cam2base.at<double>(0, 1) * y + transform_cam2base.at<double>(0, 2) * z + transform_cam2base.at<double>(0, 3)) / 1000.0;//[mm]->[m]
        point.y = (transform_cam2base.at<double>(1, 0) * x + transform_cam2base.at<double>(1, 1) * y + transform_cam2base.at<double>(1, 2) * z + transform_cam2base.at<double>(1, 3)) / 1000.0;//[mm]->[m]
        point.z = (transform_cam2base.at<double>(2, 0) * x + transform_cam2base.at<double>(2, 1) * y + transform_cam2base.at<double>(2, 2) * z + transform_cam2base.at<double>(2, 3)) / 1000.0;//[mm]->[m]
    }
}

void Triangulation::stereo3D(std::vector<cv::Point2d>& left, std::vector<cv::Point2d>& right, std::vector<cv::Point3d>& results)
{
    /**
    * @brief calculate 3D points with stereo method
    * @param[in] points_left, points_right {n_data,(xCenter,yCenter)}
    * @param[out] reuslts 3D points storage. shape is like (n_data, (x,y,z))
    */

    //undistort points
    cv::Mat points_left_mat(left);
    cv::Mat undistorted_points_left_mat;
    cv::Mat points_right_mat(right);
    cv::Mat undistorted_points_right_mat;

    // Undistort the points
    cv::undistortPoints(points_left_mat, undistorted_points_left_mat, cameraMatrix_left, distCoeffs_left);
    cv::undistortPoints(points_right_mat, undistorted_points_right_mat, cameraMatrix_right, distCoeffs_right);
    std::cout << "undistorted_points_left_mat=" << undistorted_points_left_mat << std::endl;

    // Reproject normalized coordinates to pixel coordinates
    //left
    cv::Mat normalized_points_left(undistorted_points_left_mat.rows, 2, CV_64F);
    double x, y;
    for (int i = 0; i < undistorted_points_left_mat.rows; ++i) {
        x = undistorted_points_left_mat.at<cv::Vec2d>(i, 0)[0];
        y = undistorted_points_left_mat.at<cv::Vec2d>(i, 0)[1];
        normalized_points_left.at<cv::Vec2d>(i, 0)[0] = cameraMatrix_left.at<double>(0, 0) * x + cameraMatrix_left.at<double>(0, 2);
        normalized_points_left.at<cv::Vec2d>(i, 0)[1] = cameraMatrix_left.at<double>(1, 1) * y + cameraMatrix_left.at<double>(1, 2);
    }
    //right
    cv::Mat normalized_points_right(undistorted_points_right_mat.rows, 2, CV_64F);
    for (int i = 0; i < undistorted_points_right_mat.rows; ++i) {
        x = undistorted_points_right_mat.at<cv::Vec2d>(i, 0)[0];
        y = undistorted_points_right_mat.at<cv::Vec2d>(i, 0)[1];
        normalized_points_right.at<cv::Vec2d>(i, 0)[0] = cameraMatrix_right.at<double>(0, 0) * x + cameraMatrix_right.at<double>(0, 2);
        normalized_points_right.at<cv::Vec2d>(i, 0)[1] = cameraMatrix_right.at<double>(1, 1) * y + cameraMatrix_right.at<double>(1, 2);
    }

    std::cout << "normalized_points_left_mat=" << normalized_points_left << std::endl;
    int size_left = normalized_points_left.rows;
    int size_right = normalized_points_right.rows;
    int size;
    if (size_left <= size_right) size = size_left;
    else size = size_right;

    cv::Point3d result;
    for (int i = 0; i < size; i++) {
        double xl = normalized_points_left.at<double>(i, 0); double xr = normalized_points_right.at<double>(i, 0);
        double yl = normalized_points_left.at<double>(i, 1); double yr = normalized_points_left.at<double>(i, 1);
        double disparity = xl - xr;
        double X = (double)(BASELINE / disparity) * (xl - oX_left - (fSkew / fY) * (yl - oY_left));
        double Y = (double)(BASELINE * (fX / fY) * (yl - oY_left) / disparity);
        double Z = (double)(fX * BASELINE / disparity);
        /* convert Camera coordinate to robot base coordinate */
        X = (transform_cam2base.at<double>(0, 0) * X + transform_cam2base.at<double>(0, 1) * Y + transform_cam2base.at<double>(0, 2) * Z + transform_cam2base.at<double>(0, 3)) / 1000.0;
        Y = (transform_cam2base.at<double>(1, 0) * X + transform_cam2base.at<double>(1, 1) * Y + transform_cam2base.at<double>(1, 2) * Z + transform_cam2base.at<double>(1, 3)) / 1000.0;
        Z = (transform_cam2base.at<double>(2, 0) * X + transform_cam2base.at<double>(2, 1) * Y + transform_cam2base.at<double>(2, 2) * Z + transform_cam2base.at<double>(2, 3)) / 1000.0;
        result.x = X; result.y = Y; result.z = Z;
        results.push_back(result);
    }
}

void Triangulation::makeDir(std::filesystem::path& dirPath) {
    /**
    * @brief make a directory.
    */

    try {
        if (std::filesystem::create_directory(dirPath)) {
            std::cout << "Directory created successfully: " << dirPath << std::endl;
        }
        else {
            std::cout << "Directory already exists or could not be created: " << dirPath << std::endl;
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    }
}