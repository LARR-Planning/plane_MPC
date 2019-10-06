#include <optim_traj_gen/chomp_base.h>


#define DIM_EXCEPTION 0; // dimenstion mismatch  

using namespace std;

/**
 * @brief This script solves the CHOMP programming : 1/2 x'Mx + hx + f(x)
 * This requires evaluation function for f(x) and grad_f(x) as a function pointer and other parameter
 */
namespace CHOMP{

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