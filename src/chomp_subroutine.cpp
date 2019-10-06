#include <optim_traj_gen/chomp_subroutine.h>

namespace CHOMP{

Solver::Solver(){};

// octomap mode / unconstrained  
void Solver::set_problem(MatrixXd A,VectorXd b,DynamicEDTOctomap* edf,CostParam cost_param){
    // compute inverse in advance for the future use in optimization loop
    prior_inverse = A.inverse();
    // set quadratic term
    this->M = A;
    this->h = b;
    this->edf_ptr = edf;
    this->is_constrained = false;
    if(this->edf_ptr!=NULL)
        is_problem_set = true;
    map_type = 0; // octomap         
    this->cost_param = cost_param;
}

// octomap mode / constrained 
void Solver::set_problem(MatrixXd A,VectorXd b,MatrixXd C,VectorXd d,DynamicEDTOctomap* edf,CostParam cost_param){
    // compute inverse in advance for the future use in optimization loop
    prior_inverse = A.inverse();
    // set quadratic term
    this->M = A;
    this->h = b;
    
    this->is_constrained = false;

    this->C = C;
    this->d = d;

    this->edf_ptr = edf;
    if(this->edf_ptr!=NULL)
        is_problem_set = true;
    map_type = 0; // octomap         
    this->cost_param = cost_param;
}



// voxblox mode 
void Solver::set_problem(MatrixXd A,VectorXd b,voxblox::EsdfServer* esdf_ptr,CostParam cost_param){
    // compute inverse in advance for the future use in optimization loop 
    prior_inverse = A.inverse();
    // set quadratic term
    this->M = A;
    this->h = b;
    this->esdf_ptr = esdf_ptr;
    if(this->esdf_ptr!=NULL)
        is_problem_set = true;
    map_type = 1; // voxblox        
    this->cost_param = cost_param;      
}


OptimResult Solver::solve(VectorXd x0, OptimParam optim_param){
    // history 
    OptimInfo optim_info;

    // parsing 
    double term_cond = optim_param.termination_cond;
    int max_iter = optim_param.max_iter;
    double weight_prior = optim_param.weight_prior;
    double learning_rate = optim_param.descending_rate;


    VectorXd x_prev = x0;
    VectorXd x;    


    // dimension of optimization variable 
    int N = x0.size();    
    double innovation = 1e+4;
    int iter = 0;
    ros::Time tic,toc; 
    double time_sum = 0;

    while (iter <= max_iter && innovation > term_cond){        
        // cost computation                 
        tic = ros::Time::now();    
        Vector2d costs = evaluate_costs(x_prev);
        double prior_cost = costs(0), nonlinear_cost = costs(1);
        
        // gradient computation                 
        VectorXd grad_nonlinear = grad_cost_obstacle(x_prev);
        VectorXd grad_cost; // VectorXd grad_cost = weight_prior * (A*x_prev + b) + grad_nonlinear;    // this is wrong 


        grad_cost =  weight_prior * (M*x_prev + h) + grad_nonlinear;                        
        
        
        toc = ros::Time::now();

        time_sum += (toc - tic).toSec();

        // record
        optim_info.nonlinear_cost_history.push_back(nonlinear_cost);
        optim_info.prior_cost_history.push_back(prior_cost);
        optim_info.total_cost_history.push_back(weight_prior*prior_cost + nonlinear_cost);
        VectorXd update; 


        if (this->is_constrained) // unconstrained 
            update = -learning_rate*(this->prior_inverse)*(grad_cost);
        else{ // constrained 
            
            // dual problem to consider linear inequality
            MatrixXd Q = this->C * this->prior_inverse * this->C.transpose()/(2*weight_prior);
            MatrixXd H = -(this->C * x_prev - this->d - 1/weight_prior*this->C * prior_inverse * grad_cost).transpose();
             
            int Nu = Q.cols(); // size of lagrangian multiplier 
            
            // enforce positivity of multiplier 
            MatrixXd Aineq(Nu,Nu); Aineq.setIdentity(); Aineq *= -1 ; MatrixXd bineq(Nu,1); bineq.setZero();
            // MatrixXd Aeq(1,Nu); Aeq.setZero(); MatrixXd beq(1,1); beq.setZero(); // dummy constraint 

            // construct qp for dual problem 
            QP_form qp_dual; qp_dual.verbose = false; 
        
            qp_dual.Q = Q; qp_dual.H = H;
            qp_dual.A = Aineq; qp_dual.b = bineq;
            // qp_dual.Aeq = Aeq; qp_dual.beq = beq; 

            // obtain multiplier in dual problem 
            bool is_ok;
            VectorXd u = solveqp(qp_dual,is_ok);

            update = -learning_rate*(this->prior_inverse)*(grad_cost + this->C.transpose()*u);
        }

        // update
        x = x_prev + update; 

        iter++;
        innovation = (x-x_prev).norm();  
        if((iter % 15 == 0) or (iter == 1)){
        // print 
        printf("[CHOMP] iter %d = obst_cost : %f / prior_cost %f / total_cost %f // innovation: %f\n",iter,
                nonlinear_cost,weight_prior*prior_cost,weight_prior*prior_cost + nonlinear_cost,innovation);
        
        printf("    grad : prior %f / obstacle %f / total %f / final update %f // learning rate: %f  /// eval_time = %f \n ",
        weight_prior * (M*x_prev + h).norm(),(grad_nonlinear).norm(),grad_cost.norm(),update.norm(),
        learning_rate,(time_sum /15.0));
        time_sum = 0;
        }
        // is it mature?
        if (iter > max_iter)
            std::cout<<"[CHOMP] reached maximum number of iteration."<<std::endl;
        x_prev = x;
    }

    std::cout<<"[CHOMP] optimization finished at "<<iter<<" iteration."<<std::endl;

    OptimResult optim_result;
    optim_result.result_verbose = optim_info;
    optim_result.solution_raw = x;
    return optim_result;
}


/**
 * @brief modified optimization solving routine  
 * @details momentum 
 * @param x0 : initial guess 
 * @param optim_param :
 * @return OptimResult 
 */
OptimResult Solver::solve2(VectorXd x0, OptimParam optim_param){
    // history 
    OptimInfo optim_info;

    // parsing 
    double term_cond = optim_param.termination_cond;
    int max_iter = optim_param.max_iter;
    double weight_prior = optim_param.weight_prior;
    double learning_rate = optim_param.descending_rate;


    VectorXd x_prev = x0; double cost_prev = 1e+4; double cost; double delta_cost;
    VectorXd x;    
    double dist_min = -1e+2;
    MatrixXd obst_grad_weight = MatrixXd::Identity(x_prev.size(),x_prev.size());
    // double  obst_grad_weight = 1; // dimension of optimization variable 
    int N = x0.size();    
    double innovation = 1e+4;
    int iter = 0;
    ros::Time tic,toc; 
    double time_sum = 0;
    while ((iter <= max_iter) && ((innovation > term_cond) or (dist_min < 0.)) ){        
        // cost computation              

        Vector2d costs;
        OptimGrad optim_grad;
        tic = ros::Time::now();
        VectorXd distance_set = grad_and_cost_obstacle2(x_prev,costs,optim_grad); // evaluate cost and grad simultanously 
        toc = ros::Time::now();
        time_sum += (toc - tic).toSec();

        double prior_cost = costs(0), nonlinear_cost = costs(1);
       // cout<<"distance_set"<<endl;
       // cout<<distance_set.transpose()<<endl;        
        // gradient computation                 
        VectorXd grad_cost = weight_prior * optim_grad.prior_gard + obst_grad_weight*optim_grad.nonlinear_grad;                
        
        cost = weight_prior*prior_cost + nonlinear_cost;
        delta_cost = abs(cost - cost_prev);

        
        // record
        optim_info.nonlinear_cost_history.push_back(nonlinear_cost);
        optim_info.prior_cost_history.push_back(prior_cost);
        optim_info.total_cost_history.push_back(weight_prior*prior_cost + nonlinear_cost);

        // update
        VectorXd update = -learning_rate*(this->prior_inverse)*(grad_cost);
        x = x_prev + update; 

        iter++;
        // innovation = (x-x_prev).norm();  // this innovation is gradient norm version
        innovation = delta_cost;  
        // adaptation of  step size by total cost                  
        // if (cost - cost_prev < 0){
        //     learning_rate = (1.2 * learning_rate > optim_param.descending_rate_max) ?  optim_param.descending_rate_max : (1.2*learning_rate);

        // }
        // else{
        //     learning_rate = (0.6*learning_rate < optim_param.descending_rate_min) ?  optim_param.descending_rate_min : (0.6*learning_rate) ;
        // }

        Index min_idx;
        dist_min = distance_set.minCoeff(&min_idx);
        //cout<<"distance min value"<<endl;
        //cout<< dist_min<<endl;
        // adaptation of obstacle weight by obstacle cost                  
        // if (dist_min < 0){
        //     obst_grad_weight = 1.5 * obst_grad_weight;
        // }
        // else
        //    obst_grad_weight = MatrixXd::Identity(x_prev.size(),x_prev.size()); 
        if((iter % 15 == 0) or (iter == 2)){
        // print 
        printf("[CHOMP] iter %d = obst_cost : %f / prior_cost %f / total_cost %f // cost_diff: %f / min_dist_val %f \n",iter,
                nonlinear_cost,weight_prior*prior_cost,weight_prior*prior_cost + nonlinear_cost,innovation,dist_min);

        printf("    grad : prior %f / obstacle %f / total %f / final update %f // learning rate: %f  /// eval_time = %f \n ",
        weight_prior*optim_grad.prior_gard.norm(),(obst_grad_weight*optim_grad.nonlinear_grad).norm(),grad_cost.norm(),update.norm(),
        learning_rate,(time_sum /15.0));
        time_sum = 0;
        }
		// is it mature?
        if (iter > max_iter)
            std::cout<<"[CHOMP] reached maximum number of iteration."<<std::endl;
        x_prev = x;

        cost_prev = cost;
	}

    std::cout<<"[CHOMP] optimization finished at "<<iter<<" iteration."<<std::endl;

    OptimResult optim_result;
    optim_result.result_verbose = optim_info;
    optim_result.solution_raw = x;
    optim_result.distance_min = dist_min;
    return optim_result;
}





// private modules  
Vector2d Solver::evaluate_costs(VectorXd x){
    Vector2d costs;
    double prior_term = (0.5*x.transpose()*M*x + h.transpose()*x)(0,0); // (0,0) is for conversion from matrix to a scalar
    double nonlinear_term = cost_obstacle(x); 
    costs(0) = prior_term;
    costs(1) = nonlinear_term;
    // cout<<"[in function evaluate_costs] evaluted cost for nonlinear term: "<<nonlinear_term<<endl;

    return costs;
} 

/**
 * @brief Use this function only in voxblox mode 
 * 
 * @param x 
 * @return Vector2d 
 */
Vector2d Solver::evaluate_costs2(VectorXd x){
    Vector2d costs;
    double prior_term = (0.5*x.transpose()*M*x + h.transpose()*x)(0,0); // (0,0) is for conversion from matrix to a scalar
    
    // replaced with batch evaluation 
    int H = x.size()/2; // this will be cols of Matrix3Xd matrix 
    MatrixXd xy_path = x; xy_path.resize(2,H);
    MatrixXd z_path = MatrixXd::Ones(1,H) * cost_param.ground_height;    
    
    Matrix3Xd batch_mat(3,H);
    batch_mat << xy_path , z_path; // final evaluation batch (3 x H) 
    Ref<const Matrix<double,3,-1,ColMajor>,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> batch_mat_ref(batch_mat);
    // cout<<"batch_mat_ref:"<<endl;
    // cout<<batch_mat_ref<<endl;
    VectorXd distance_set(H);
    VectorXi observation_set(H); 
    esdf_ptr->getEsdfMapPtr()->batchGetDistanceAtPosition(batch_mat_ref,distance_set,observation_set);
    // cout<<distance_set<<endl;
    costs(0) = prior_term;
    costs(1) = distance_set.sum();
    // cout<<"[in function evaluate_costs] evaluted cost for nonlinear term: "<<nonlinear_term<<endl;
    return costs;
} 




// evaluate cost at one point 
double Solver::cost_at_point(geometry_msgs::Point p){    
    try{
        if (!is_problem_set)
            throw 1;
        double distance_raw = 0;

        if (map_type == 0)
            distance_raw = edf_ptr->getDistance(octomap::point3d(p.x,p.y,p.z));
        else{
             Vector3d position(p.x,p.y,p.z);
             esdf_ptr->getEsdfMapPtr()->getDistanceAtPosition(position,&distance_raw);
        }
        // compute real cost from distance value 
        if (distance_raw <=0 )
           return (-distance_raw + 0.5*cost_param.r_safe); 
        else if((0<distance_raw) and (distance_raw < cost_param.r_safe) ){
            return 1/(2*cost_param.r_safe)*pow(distance_raw - cost_param.r_safe,2);                
        }else        
            return 0;

    }catch(exception e){
        std::cout<<"error in evaulating EDT value. Is edf completely loaded?"<<std::endl;
    }
}


// evalute cost of a path 
double Solver::cost_obstacle(VectorXd x){    
    // length(x) = dim x H
    double cost = 0;
    int H = x.size()/2;
    for(int i = 0;i<H-1;i++){
            geometry_msgs::Point p1;
            p1.x = x(2*i);
            p1.y = x(2*i+1);      
            p1.z = cost_param.ground_height;                          
            
            geometry_msgs::Point p2;
            p2.x = x(2*(i+1));
            p2.y = x(2*(i+1)+1);             
            p2.z = cost_param.ground_height;                          
            cost += (cost_at_point(p1) + cost_at_point(p2))/2 * sqrt(pow(p1.x - p2.x,2)+pow(p1.y - p2.y,2));        
        }

    return cost;
}
// evaluate gradient of cost of a path 
VectorXd Solver::grad_cost_obstacle(VectorXd x){
    // length(x) = dim x H
    int H = x.size()/2;
    double cost0 = cost_obstacle(x); // original cost 
    VectorXd grad(x.size());    
    for(int i = 0;i<H;i++){
        VectorXd pert_x(x.size()); pert_x.setZero(); pert_x(2*i) = cost_param.dx; grad(2*i) = (cost_obstacle(x+pert_x) - cost0)/cost_param.dx;
        VectorXd pert_y(x.size()); pert_y.setZero(); pert_y(2*i+1) = cost_param.dx; grad(2*i+1) = (cost_obstacle(x+pert_y) - cost0)/cost_param.dx;
    }
    return grad;
}


/**
 * @brief evalute cost gradient for prior and nonlinear 
 * 
 * @param x optim_var (2H x 1)
 * @param prior_and_obstacle_costs  
 * @param grad 
 * @return distance set. This is sometines needed as the obstacle detouring condition   
 */
inline VectorXd Solver::grad_and_cost_obstacle2(VectorXd x,Vector2d & prior_and_obstacle_costs,OptimGrad& grad){
    // length(x) = dim x H

    int H = x.size()/2; // this will be cols of Matrix3Xd matrix 
    MatrixXd xy_path = x; xy_path.resize(2,H);
    MatrixXd z_path = MatrixXd::Ones(1,H) * cost_param.ground_height;    
    
    Matrix3Xd batch_mat(3,H);
    batch_mat << xy_path , z_path; // final evaluation batch (3 x H) 
    Ref<const Matrix<double,3,-1,ColMajor>,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> batch_mat_ref(batch_mat);
    // cout<<"batch_mat_ref:"<<endl;
    // cout<<batch_mat_ref<<endl;
    VectorXd distance_set(H);
    VectorXi observation_set(H); 

    Matrix3Xd batch_grad(3,H);
    Ref<Matrix<double,3,-1,ColMajor>,0,Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> batch_grad_ref(batch_grad);
    esdf_ptr->getEsdfMapPtr()->batchGetDistanceAndGradientAtPosition(batch_mat_ref,distance_set,batch_grad_ref,observation_set);    

    // shaping function 
    shaping_functor shp_fun(cost_param.r_safe);
    shaping_functor_grad shp_fun_grad(cost_param.r_safe);

    // costs 
    prior_and_obstacle_costs(0) = (0.5*x.transpose()*M*x + h.transpose()*x)(0,0);   
    prior_and_obstacle_costs(1) = distance_set.unaryExpr(std::ref(shp_fun)).sum(); // TODO - should apply smoothing function
    
    // cout<<"cost from vox:"<<endl;
    // cout<<prior_and_obstacle_costs(1)<<endl;
    // cout<<"cost from FDM"<<endl;
    // cout<<cost_obstacle(x)<<endl;

    
    // grad 
    grad.prior_gard = (M*x + h);    
    VectorXd dc_dd = distance_set.unaryExpr(std::ref(shp_fun_grad)); 
//    cout<<batch_grad_ref<<endl;
    Matrix3Xd batch_grad_shp = batch_grad_ref*dc_dd.asDiagonal();     //chain rule
//    cout<<batch_grad_shp<<endl; 
    Matrix2Xd grad_xy = batch_grad_shp.block(0,0,2,H);
    grad.nonlinear_grad = (Map<VectorXd>(grad_xy.data(), grad_xy.cols()*grad_xy.rows()));
    // cout<<"grad from vox:"<<endl;
    // cout<<grad.nonlinear_grad<<endl;
    // cout<<"grad from FDM"<<endl;
    // cout<<grad_cost_obstacle(x)<<endl;
    
    return distance_set;
}

LinearInequalityConstraint Solver::get_lin_ineq_constraint(BoxConstraintSeq box_seq,BoxAllocSeq alloc_seq){
    
    int N_var_total = accumulate(alloc_seq.begin(),alloc_seq.end(),0);

    MatrixXd C(4*N_var_total,2*N_var_total); // 4 = xl / yl / xu / yu
    MatrixXd d(4*N_var_total,1);
    MatrixXd C_sub(4,2);

    // small matrix to be repeatedly inserted 
    C_sub << -1, 0,
              1, 0,
              0,-1,
              0, 1;

    int until_now = 0; // point index until now 
    int box_idx = 0; // from 0 to length(alloc_seq)-1
    

    // build the matrix to impose inequality constraints      
    for(auto it = alloc_seq.begin() ; it < alloc_seq.end(); it++,box_idx += 1){ // per block segment 
        for (int n = until_now ; n < until_now + *it ; n++){ // per points in this block 
            C.block(4*n,2*n,4,2) = C_sub;
            d.block(4*n,0,4,1) << -box_seq[box_idx].xl, 
                              box_seq[box_idx].xu,
                              -box_seq[box_idx].yl,
                              box_seq[box_idx].yu;
        }
        until_now += *it;
    }

    LinearInequalityConstraint ineq_const;
    ineq_const.A = C;
    ineq_const.b = d;
    return ineq_const;
};

VectorXd OptimProblem::get_initial_guess(){
        const int dim = 2;
         // Intiial guess generation 
        int N_pnt = accumulate(corridor.box_alloc_seq.begin(),corridor.box_alloc_seq.end(),0)-(corridor.box_seq.size()-1); // number of points to be optimized (dim * N = tot number of variable in optimization)
        int N = N_pnt * dim;
        int N_corridor = corridor.box_seq.size(); // number of corridors 

        
        VectorXd n_regress(N_corridor+1); // regression input

        n_regress(0) = 1;
        int corridor_alloc_sum = 1;
        for(int i = 0; i < N_corridor ; i++ ){           
            corridor_alloc_sum += (corridor.box_alloc_seq[i]-1);
            n_regress(i+1) = corridor_alloc_sum;             
        }
       
        // decompose point vector / prepare regression target 
        vector<geometry_msgs::Point> inter_pnts = corridor.get_corridor_intersect_points();
        VectorXd xs_regress(N_corridor + 1),ys_regress(N_corridor + 1);
        xs_regress(0) = start.x; ys_regress(0) = start.y; //insert start point

        int i = 1; 
        for (auto it = inter_pnts.begin() ; it < inter_pnts.end() ; it++, i++ ){
            xs_regress(i) = it->x; ys_regress(i) = it->y;
        }
        xs_regress(N_corridor) = goal.x;  
        ys_regress(N_corridor) = goal.y;  
        
        /** DEBUG 
        cout<<"n_regress: "<<endl;
        cout << n_regress <<endl;      
         
        cout<<"x_regress: "<<endl;
        cout << xs_regress <<endl;      
         
        cout<<"y_regress: "<<endl;
        cout << ys_regress <<endl;      
        **/

        VectorXi ns = VectorXi::LinSpaced(N_pnt,1,N_pnt);   // index starts from 1 
        VectorXd x0(N); // initial guess of trajectory (augmented)

        for (int n=0;n<N_pnt;n++){
            x0(dim*n) = interpolate(n_regress,xs_regress,ns(n),false);  
            x0(dim*n+1) = interpolate(n_regress,ys_regress,ns(n),false);
        }

    return x0;
}

visualization_msgs::Marker OptimProblem::get_initial_guess_marker(string world_frame_id){

    const int dim = 2;    
    visualization_msgs::Marker knots_marker;
    knots_marker.header.frame_id = world_frame_id;
    knots_marker.id = 0;
    knots_marker.type = visualization_msgs::Marker::SPHERE_LIST;    
    knots_marker.color.r = 10.0/255.0;
    knots_marker.color.g = 50.0/255.0;
    knots_marker.color.b = 1.0;
    knots_marker.color.a = 0.3;
    knots_marker.pose.orientation.w = 1.0;
    double scale = 0.08; 
    knots_marker.scale.x = scale;
    knots_marker.scale.y = scale;
    knots_marker.scale.z = scale;       
    
    VectorXd x0 = get_initial_guess();
    int N_pnt = x0.size()/dim;
    for (int n = 0; n<N_pnt ; n++){       
        geometry_msgs::Point pnt; 
        pnt.x = x0(2*n);
        pnt.y = x0(2*n+1);
        pnt.z = corridor.height;
        knots_marker.points.push_back(pnt);    
    }
    return knots_marker;    
}

visualization_msgs::MarkerArray OptimProblem::get_markers(string world_frame_id){
    
    // prepare markers for start, goal and coridor 
    visualization_msgs::MarkerArray markers = corridor.get_corridor_markers(world_frame_id); 

    double height = corridor.height; 
    
    visualization_msgs::Marker start_marker; 
    visualization_msgs::Marker goal_marker; 
    
    float scale = 0.16;
    goal_marker.action = 0;
    goal_marker.type = visualization_msgs::Marker::SPHERE;
    goal_marker.header.frame_id = world_frame_id;
    goal_marker.pose.position = goal;
    goal_marker.pose.position.z = height; // hegith is arbitrarily
    goal_marker.scale.x = scale;
    goal_marker.scale.y = scale;
    goal_marker.scale.z = scale;
    goal_marker.color.r = 1; // current gaol 
    goal_marker.color.a = 0.8;
    goal_marker.id = corridor.box_seq.size();
    
    start_marker = goal_marker;
    start_marker.color.r = 0;
    start_marker.id = corridor.box_seq.size()+1;
    start_marker.pose.position = start;
    start_marker.pose.position.z = height;

    markers.markers.push_back(start_marker);
    markers.markers.push_back(goal_marker);    
    markers.markers.push_back(get_initial_guess_marker(world_frame_id));

    return markers;
}
}