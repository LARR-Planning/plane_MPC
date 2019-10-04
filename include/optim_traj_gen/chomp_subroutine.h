#include <eigen3/Eigen/LU>
#include <qpOASES.hpp>
#include <numeric>
#include <optim_traj_gen/chomp_utils.h>


#define DIM_EXCEPTION 0; // dimenstion mismatch  

using namespace std;
using namespace Eigen;

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
 *  typedef 
 **/

typedef vector<int> BoxAllocSeq;  // {N1,N2,N3,,, Nm}
typedef vector<BoxConstraint2D> BoxConstraintSeq; // {B1,B2,...,Bm}

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
 * @brief This script solves the CHOMP programming : 1/2 x'Mx + hx + f(x)
 * This requires evaluation function for f(x) and grad_f(x) as a function pointer and other parameter
 */
namespace CHOMP{

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
    Eigen::VectorXd solution;
    double distance_min = 1; // minimum distance value (negative = containing obstacle point)
};

// components of one chomp problem 
struct OptimProblem{
    geometry_msgs::Point start;
    geometry_msgs::Point goal;
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

// solver class 
class Solver{

    private: 
        int map_type;     
        bool is_problem_set; // should be checked whether programming is set
        bool is_constrained; // is it constrained CHOMP ? 
        MatrixXd prior_inverse; 
        
        // in the prior cost 1/2xMx + hx 
        MatrixXd M;
        MatrixXd h;

        // in the linear constraints Cx <= d 
        MatrixXd C;
        MatrixXd d;
        // only one of the followings will be used for distnace query 
        DynamicEDTOctomap* edf_ptr = NULL; 
        voxblox::EsdfServer* esdf_ptr = NULL; 
        CostParam cost_param; // cost param


        // sub moudles 
        // cost (edf should be given first)
        Vector2d evaluate_costs(VectorXd x); // evaluate costs at x (not weigthed). {prior,nonlinear} if provided      
        Vector2d evaluate_costs2(VectorXd x); // evaluate costs at x  using voxblox batch evalution 
                        
        double cost_at_point(geometry_msgs::Point p); // c(x)
        double cost_obstacle(VectorXd x); 
        VectorXd grad_cost_obstacle(VectorXd x);        
        inline VectorXd grad_and_cost_obstacle2(VectorXd x,Vector2d &  costs,OptimGrad& grad);        
         
        // new moudle for project 
        LinearInequalityConstraint get_lin_ineq_constraint(BoxConstraintSeq box_seq,BoxAllocSeq alloc_seq);       

    public:
        //constructor     
        Solver();
        // setting optimization problem
        void set_problem(MatrixXd A,VectorXd b,DynamicEDTOctomap*,CostParam);
        void set_problem(MatrixXd A,VectorXd b,MatrixXd C,VectorXd d,DynamicEDTOctomap*,CostParam);
        void set_problem(MatrixXd A,VectorXd b,voxblox::EsdfServer*,CostParam);                      
        // solve and return the result
        OptimResult solve(VectorXd x0, OptimParam optimization_param); // run optimization routine          
        OptimResult solve2(VectorXd x0, OptimParam optimization_param); // run optimization routine using voxblox built-in function : batch evaluation                 

};

}