/**
 * @file chomp_base.h 
 * @author JBS (junbs95@gmail.com)
 * @brief This header contains basic class and struct for CHOMP  
 * @version 0.1
 * @date 2019-10-04
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#include <optim_traj_gen/chomp_utils.h>

using namespace Eigen;
using namespace std;

namespace CHOMP{

// for solving qp form 
struct QP_form{
    MatrixXd Q;
    MatrixXd H;
    MatrixXd A;
    MatrixXd b;
    MatrixXd Aeq;
    MatrixXd beq;  
    bool verbose;  
};

// Ax <= b (augmented)
struct LinearInequalityConstraint{
    MatrixXd A;
    MatrixXd b;
};

// this will be a box constraint 
struct BoxConstraint2D{
    double xl;
    double yl; 
    double xu; 
    double yu;
    // How will these be used? (TODO) 
    double t_start;
    double t_end;    
};

/**
 *  TYPEDEF  
 **/

typedef vector<int> BoxAllocSeq;  // {N1,N2,N3,,, Nm}
typedef vector<BoxConstraint2D> BoxConstraintSeq; // {B1,B2,...,Bm}
typedef pair<VectorXd,VectorXd> Path2D; // {(x1,x2,x3,...) , (y1,y2,y3,...)}  
typedef pair<VectorXd,Path2D> Traj2D; // {(t1,t2,...,tn) , (X1,X2,...)}

// corridor 
struct Corridor2D{    
    BoxConstraintSeq box_seq;
    BoxAllocSeq box_alloc_seq; 
    double height; 
    Corridor2D(){}; // default constructor 
    Corridor2D(BoxConstraintSeq box_seq,BoxAllocSeq box_alloc_seq,double height): 
    box_seq(box_seq),box_alloc_seq(box_alloc_seq),height(height) {};    
    vector<geometry_msgs::Point> get_corridor_intersect_points();
    visualization_msgs::MarkerArray get_corridor_markers(string world_frame_id);
};

struct chompTraj{

    bool is_solved = false;
    Traj2D trajectory; 
    double height; 
    
    // function overloading 
    geometry_msgs::Point evalute_point(double t); // x(t) y(t) z (constant)
    geometry_msgs::Point evalute_point(ros::Time t);
    
    geometry_msgs::Point evalute_vel(double t); // member x = xdot / member y = ydot 
    geometry_msgs::Point evalute_vel(ros::Time t); // member x = xdot / member y = ydot 

    geometry_msgs::Point evalute_accel(double t); // member x = xddot / member y = yddot 
    geometry_msgs::Point evalute_accel(ros::Time t); // member x = xddot / member y = yddot 
 
    geometry_msgs::Twist evalute_input(double t); // v(t) w(t)
    geometry_msgs::Twist evalute_input(ros::Time t);       
};

VectorXd solveqp(QP_form qp_prob,bool& is_ok);

/**
 * @brief Functor for shaping function from distance_raw
 * 
 */

class shaping_functor{
    private: 
        double r_safe;

    public:
        shaping_functor(double r_safe):r_safe(r_safe){};
        double operator()(double distance_raw){
        if (distance_raw <=0 )
            return (-distance_raw + 0.5*r_safe); 
        else if((0<distance_raw) and (distance_raw < r_safe) ){
            return 1/(2*r_safe)*pow(distance_raw - r_safe,2);                
        }else        
            return 0;
    };
};
/**
 * @brief gradient of shaping function   
 * 
 */
class shaping_functor_grad{
    private:
        double r_safe;
    public: 
        shaping_functor_grad(double r_safe):r_safe(r_safe){};
        double operator()(double distance_raw){
            if (distance_raw <=0 )
                return -1 ; 
            else if((0<distance_raw) and (distance_raw < r_safe) ){
                return 1/(r_safe)*(distance_raw - r_safe);                
            }else        
                return 0;
        }
};

/**
 * PARAMETERS
 */

// parameter to make a nonlinear cost (obstacle) 
struct CostParam{
    double r_safe;  // extent to which gradient will effct 
    double dx; // perturbation 
    double ground_height; // height for 2D problem formulation  
};

// parameter for optimization 
struct OptimParam{
    int max_iter;
    double termination_cond; // norm of innovation to be terminated  
    double descending_rate; // step rate
    double descending_rate_min; // step rate
    double descending_rate_max; // step rate        
    double weight_prior; // weight for prior
    double gamma; // discount
    int n_step; // discount
};

// information after optimization (record for history)
struct OptimInfo{
    vector<double> prior_cost_history;
    vector<double> nonlinear_cost_history;
    vector<double> total_cost_history;
};

// result to be obtained after optimization 
struct OptimResult{
    OptimInfo result_verbose;
    Eigen::VectorXd solution_raw; // raw solution from optimization = [x1 y1 x2 y2 x3 y3 ... ]'      
    void complete_solution(double t0,double tf,double height); // should be called to finish one trajectory. called after solution update 
    chompTraj chomp_traj; // processed result     
    double distance_min = 1; // minimum distance value (negative = containing obstacle point)

};

// components of one chomp problem 
struct OptimProblem{
    geometry_msgs::Point start;
    geometry_msgs::Point start_velocity;
    ros::Time t0; // raw ros_time 
    geometry_msgs::Point goal;
    ros::Time tf; // raw ros_time 
    Corridor2D corridor;
    VectorXd get_initial_guess();
    visualization_msgs::MarkerArray get_markers(string world_frame_id);
    visualization_msgs::Marker get_initial_guess_marker(string world_frame_id);
};

/**
 * @brief gradient containing prior and nonlinear together 
 * 
 */
struct OptimGrad{
    VectorXd prior_gard;
    VectorXd nonlinear_grad;
};


}