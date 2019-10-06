#include <ros/ros.h>
#include "optim_traj_gen/chomp_subroutine.h"
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Path.h>


namespace CHOMP{    
    class Wrapper{
        private: 
            // ros
            ros::NodeHandle nh;
            nav_msgs::Path current_path;
            string world_frame_id;

            ros::Publisher pub_path_cur_solution; // publisher for path 
            ros::Publisher pub_vis_problem; // corridor  
            ros::Publisher pub_marker_pnts_path; // pnts on the path 
            ros::Publisher pub_cur_control_point; // evaluated point at the current time (x_d(t) y_d(t)) 

            visualization_msgs::Marker pnts_on_path_marker;
            
            // map 
            DynamicEDTOctomap *edf_ptr; // Euclidean Distance Field 
            voxblox::EsdfServer *voxblox_server;  // voxblox server 
             
            // related parameters 
            double ground_rejection_height;            
            double dx; // obtain from octomap 
            double r_safe; // safe clearance (outside of r_safe, cost = 0) 
            OptimParam optim_param; // optimization parameters       
            OptimParam optim_param_default; // optimization parameters           
    
            int dim = 2; // we will plan on 2D plane only 
            // flags 
            bool is_map_load = false; // is map loaded             
            bool is_voxblox_pub = true; 
            bool is_problem_exist = false;
            bool is_solved = false;
            
            
            // chomp solver 
            Solver solver;
            // current problem 
            OptimProblem cur_problem;


             // optimization subroutine (private)
            void build_matrix(MatrixXd &M,VectorXd &h,MatrixXd &C, MatrixXd &d ,Corridor2D corridor,geometry_msgs::Point start,geometry_msgs::Point goal,OptimParam* param = NULL);
            VectorXd prepare_chomp(MatrixXd M,VectorXd h,MatrixXd C,MatrixXd d,Corridor2D corridor,geometry_msgs::Point start,geometry_msgs::Point goal,OptimParam* param = NULL );
            void solve_chomp(VectorXd x0);


        public:
            OptimResult recent_optim_result; // recent optimization result 

            int map_type; // 0 = octomap, 1 = voxblox 
            Wrapper(const ros::NodeHandle & );
            //ros
            void publish_routine();
            // map and edf 
            void load_map(octomap::OcTree* octree_ptr);
            void load_map(string file_name);            
            
            // utill function 
            void set_problem(OptimProblem new_problem);

            // result retreive 
            MatrixXd get_current_prediction_path();
            OptimParam get_default_optim_param();
            double get_ground_height() {return ground_rejection_height;};

            // solve            
            void optim_traj_gen(OptimProblem problem);

    };
}
