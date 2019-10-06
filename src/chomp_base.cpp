#include <optim_traj_gen/chomp_base.h>

using namespace CHOMP;

// designed only for solving dual problem 
VectorXd CHOMP::solveqp(QP_form qp_prob,bool& is_ok){
    is_ok = true;
    MatrixXd Q = qp_prob.Q;
    MatrixXd H = qp_prob.H;
    MatrixXd Aineq = qp_prob.A;
    MatrixXd bineq = qp_prob.b;
    MatrixXd Aeq = qp_prob.Aeq;
    MatrixXd beq = qp_prob.beq;
    if(qp_prob.verbose){
        cout<<"Q: "<<endl;
        cout<<Q<<endl;
        
        cout<<"H: "<<endl;
        cout<<H<<endl;
            
    }
    USING_NAMESPACE_QPOASES;

    int N_var = Q.rows();
    int N_ineq_const = Aineq.rows();
    int N_eq_const = Aeq.rows();    
    int N_const = N_ineq_const + N_eq_const;

    real_t H_qp[N_var*N_var];
    real_t g[N_var];
    real_t lb[N_var];

    // cost
    for (int i = 0;i<N_var;i++){
        g[i] = H(0,i);            
        for(int j = 0;j<N_var;j++)
            H_qp[j*N_var+i] = 2*Q(i,j);
    }

   // positivity  
    for (int i = 0; i<N_var;i++)
        lb[i] = 0;


    int_t nWSR = 2000;
    
	QProblem qp_obj(N_var,0,HST_SEMIDEF); // here the second argument was the number of rows in Ax <= b
    //std::cout<<"hessian type: "<<qp_obj.getHessianType()<<endl;
    Options options;
	options.printLevel = PL_LOW;
	qp_obj.setOptions(options);
	qp_obj.init(H_qp,g,NULL,lb,NULL,NULL,NULL,nWSR);
    if(qp_obj.isInfeasible()){
        cout<<"[QP solver] warning: problem is infeasible. "<<endl;
        is_ok = false;
    }
    real_t xOpt[N_var];
    qp_obj.getPrimalSolution(xOpt);

    if(not qp_obj.isSolved()){
        cout<<"[QP solver] quadratic programming has not been solved "<<endl;
        is_ok = false;
    }
    
    VectorXd sol(N_var);

    for(int n = 0; n<N_var;n++)
        sol(n) = xOpt[n];

    // cout << "solution" <<endl;
    // cout << sol <<endl;
    return sol;        
}

vector<geometry_msgs::Point> Corridor2D::get_corridor_intersect_points(){
    int N_box = box_seq.size();    
    vector<geometry_msgs::Point> intersect_pnts(N_box-1);  // the number of total intersection points 
    for ( int box_idx = 0; box_idx < N_box-1 ; box_idx ++){
        geometry_msgs::Point cur_intersect;
        cur_intersect.x = (max(box_seq[box_idx].xl,box_seq[box_idx+1].xl) + min(box_seq[box_idx].xu,box_seq[box_idx+1].xu))/2;   
        cur_intersect.y = (max(box_seq[box_idx].yl,box_seq[box_idx+1].yl) + min(box_seq[box_idx].yu,box_seq[box_idx+1].yu))/2;   
        intersect_pnts[box_idx] = cur_intersect; 
    }        
    return intersect_pnts;
 }

visualization_msgs::MarkerArray Corridor2D::get_corridor_markers(string world_frame_id){

    visualization_msgs::MarkerArray safe_corridor_marker;

    visualization_msgs::Marker safe_corridor_marker_single_base;    
    safe_corridor_marker_single_base.header.frame_id = "world";
    safe_corridor_marker_single_base.ns = "sf_corridor";
    safe_corridor_marker_single_base.type = visualization_msgs::Marker::CUBE;
    safe_corridor_marker_single_base.action = 0;
    safe_corridor_marker_single_base.color.a = 0.5;
    safe_corridor_marker_single_base.color.r = 170.0/255.0;
    safe_corridor_marker_single_base.color.g = 1.0;
    safe_corridor_marker_single_base.color.b = 1.0;    
    const double box_dim_height = 0.5; // z scale of box

    // construct marker one by one  
    int N_box = box_seq.size();    

    for (int box_idx = 0; box_idx < N_box ; box_idx ++ ){
        double x_center,y_center; 
        x_center = (box_seq[box_idx].xl + box_seq[box_idx].xu)/2;
        y_center = (box_seq[box_idx].yl + box_seq[box_idx].yu)/2;

        // pose     
        safe_corridor_marker_single_base.pose.position.x = x_center;                     
        safe_corridor_marker_single_base.pose.position.y = y_center;                     
        safe_corridor_marker_single_base.pose.position.z = height;                     
        safe_corridor_marker_single_base.pose.orientation.x = 0;
        safe_corridor_marker_single_base.pose.orientation.y = 0;
        safe_corridor_marker_single_base.pose.orientation.z = 0;
        safe_corridor_marker_single_base.pose.orientation.w = 1;

       // scale 
        safe_corridor_marker_single_base.scale.x = abs(box_seq[box_idx].xl - box_seq[box_idx].xu);
        safe_corridor_marker_single_base.scale.y = abs(box_seq[box_idx].yl - box_seq[box_idx].yu);
        safe_corridor_marker_single_base.scale.z = box_dim_height;
        
        // index
        safe_corridor_marker_single_base.id = box_idx; 

        // push back 
        safe_corridor_marker.markers.push_back(safe_corridor_marker_single_base);
    }

    return safe_corridor_marker;
}

/**
 * @brief fill chomp trajectory struct with necessary information 
 * @param t0 : value = t0_raw - t_ref (for numerical stability)
 * @param tf : value = tf_raw - t_ref (for numerical stability)
 */
void OptimResult::complete_solution(double t0,double tf,double height){

    int N_pnt = solution_raw.size()/2; // number of points  
    VectorXd ts = VectorXd::LinSpaced(N_pnt,t0,tf);
    
    chomp_traj.is_solved = true;     
    // fill time 
    chomp_traj.trajectory.first = ts;
    chomp_traj.trajectory.second.first = VectorXd(N_pnt);
    chomp_traj.trajectory.second.second= VectorXd(N_pnt);    
    chomp_traj.height = height;


    // fill xy  
    for(int n = 0; n < N_pnt ; n++){
        chomp_traj.trajectory.second.first(n) = solution_raw(2*n); // x
        chomp_traj.trajectory.second.second(n) = solution_raw(2*n+1); // y
    }
}
/**
 * @brief evalute point given t 
 * 
 * @param t : value = t0_raw - t_ref (for numerical stability) 
 * @return geometry_msgs::PointStamped 
 */
geometry_msgs::Point chompTraj::evalute_point(double t){

    geometry_msgs::Point eval_pnt; 
    
    if (is_solved == true){
        eval_pnt.x = interpolate(trajectory.first,trajectory.second.first,t,false);
        eval_pnt.y = interpolate(trajectory.first,trajectory.second.second,t,false);
        eval_pnt.z = height;
    }else{
        ROS_WARN("[CHOMP] trajectory has not been initialized. evaluation of point is impossible");
    }
    return eval_pnt; 
};

geometry_msgs::Point chompTraj::evalute_point(ros::Time t){
    return evalute_point(t.toSec());    
};

/**
 * @brief evalute velocity given t 
 * 
 * @param t 
 * @return geometry_msgs::Point (no other option.. to represent velocity)
 */

geometry_msgs::Point chompTraj::evalute_vel(double t){

    double dt = trajectory.first(2) - trajectory.first(1);
    double xdot = (evalute_point(t+dt).x - evalute_point(t).x)/dt;
    double ydot = (evalute_point(t+dt).y - evalute_point(t).y)/dt;

    geometry_msgs::Point vel; 
    vel.x = xdot; 
    vel.y = ydot;     
    vel.z = 0.0; 
    return vel; 
}

geometry_msgs::Point chompTraj::evalute_vel(ros::Time t){
    
    return evalute_vel(t.toSec()); 
}


geometry_msgs::Point chompTraj::evalute_accel(double t){

    double dt = trajectory.first(2) - trajectory.first(1);
    double xddot = (evalute_vel(t+dt).x - evalute_vel(t).x)/dt;
    double yddot = (evalute_vel(t+dt).y - evalute_vel(t).y)/dt;
    
    geometry_msgs::Point accel; 
    accel.x = xddot; 
    accel.y = yddot;     
    accel.z = 0.0;
    
    return accel; 
}



/**
 * @brief return (v,w) based on differential flatness
 * 
 * @param t evaluation time 
 * @return geometry_msgs::Twist (v,w) 
 */
geometry_msgs::Twist chompTraj::evalute_input(double t){

    double xdot = evalute_vel(t).x;
    double ydot = evalute_vel(t).y;

    double xddot = evalute_accel(t).x;
    double yddot = evalute_accel(t).y;
     
    geometry_msgs::Twist input; 

    // linear input 
    input.linear.x = sqrt(pow(xdot,2) + pow(ydot,2));    
    
    // printf("xddot / yddot : [%f,%f]\n",xddot,yddot);

    // angular input 
    if (input.linear.x > 1e-4)
        input.angular.z = (yddot * xdot - xddot * ydot)/(pow(xdot,2) + pow(ydot,2));
    else{    
        input.angular.z = 0.0 ;    
        ROS_WARN("[CHOMP] linear velocity is almost zero. angular velocity is set as zero. Does the car reached goal?");
    }
    
    return input; 
};

geometry_msgs::Twist chompTraj::evalute_input(ros::Time t){
    return evalute_input(t.toSec());
};




