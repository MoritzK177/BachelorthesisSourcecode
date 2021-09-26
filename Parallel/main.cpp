#include<iostream>
#include<limits>
#include <array>
#include <algorithm>
#include <cmath>
#include "settings.h"
#include "HeapAndStructVector.h"
#include <cassert>
#include <vector>
#include <string>
#include <utility>
#include <omp.h>
#include <functional>
#include <chrono>
#include "numeric"


/* legend of the statuses:
 * 0 = FAR
 * 1 = BAND_NEW
 * 2 = BAND_OLD
 * 3 = KNOWN_FIX
 * 4 = KNOWN_NEW
 * 5 = KNOWN_OLD */

bool in_barrier( const int x,const int y,const int z)
{
    const double temp_x= x*settings::h - 0.5;
    const double temp_y= y*settings::h - 0.5;
    const double temp_z= z*settings::h - 0.5;

    const double w = 1.0/24;
    const double big_r = std::sqrt(std::pow(temp_x,2)+ std::pow(temp_y,2)+ std::pow(temp_z,2));
    const double small_r = std::sqrt(std::pow(temp_x,2)+ std::pow(temp_y,2));
    const bool in_barrier_one = (0.15 < big_r && big_r< 0.15+w) && !(small_r<0.05 && temp_z<0);
    const bool in_barrier_two = (0.25 < big_r && big_r< 0.25+w) && !(small_r<0.1 && temp_z>0);
    const bool in_barrier_three = (0.35 < big_r && big_r< 0.35+w) && !(small_r<0.10 && temp_z<0);
    const bool in_barrier_four = (0.45 < big_r && big_r< 0.45+w) && !(small_r<0.1 && temp_z>0);
    return in_barrier_one||in_barrier_two||in_barrier_four||in_barrier_three;
}
//speed and mask functions
double speed_funct(const int number, const int x,const int y,const int z)
{
    switch (number) {
        case 1 :    return 1.0;
        case 2 :    return 1+ 0.5*std::sin(20*M_PI*settings::h*x)*std::sin(20*M_PI*settings::h*y)*std::sin(20*M_PI*settings::h*z);
        case 3 :    return (1- 0.99*std::sin(2*M_PI*settings::h*x)*std::sin(2*M_PI*settings::h*y)*std::sin(2*M_PI*settings::h*z));
        case 4 :    return 1*(pow(std::sin(x*settings::h),2)+pow(std::cos(y*settings::h),2)+0.1);
        case 5 :    if(in_barrier(x,y,z)) return 0;
                    else return 1;
        case 6 :    return 0.001*(pow(std::sin(x*settings::h),2)+pow(std::cos(y*settings::h),2)+0.1);

        default :   std::cout<<"UNDEFINED FUNCTION; RETURNING ZERO !"<<std::endl;
            return 0;
    }
}
bool in_mask(const int number,const int x,const int y ,const int z)
{
    switch (number) {
        case 1 :    return (pow(x*settings::h -0.5,2)+ pow(y*settings::h -0.5,2)+pow(z*settings::h -0.5,2))<=1.0/16;
        case 2 :    return x==0&&y==0&&z==0;
        case 3 :    return (pow(x*settings::h -0.25,2)+ pow(y*settings::h -0.25,2)+pow(z*settings::h -0.25,2))<=1.0/256||x*settings::h<=0.875&&x*settings::h >=0.625&& y*settings::h<=0.875&&y*settings::h >=0.625&& z*settings::h<=0.875&&z*settings::h >=0.625;
        case 4 :    return x== settings::x_global_grid_size/2&&y== settings::y_global_grid_size/2&&z== settings::z_global_grid_size/2;
        default :   std::cout<<"UNDEFINED MASK; RETURNING FALSE !"<<std::endl;
            return false;
    }
}

//Helper function to return the respective array indices
int global_arr_index(const int x, const int y, const int z)
{
    return x+y*settings::x_global_grid_size+z*settings::x_global_grid_size*settings::y_global_grid_size;
}

int local_arr_index(const int x, const int y, const int z)
{
    return x+y*settings::x_local_grid_size+z*settings::x_local_grid_size*settings::y_local_grid_size;
}
int process_index(const int x, const int y, const int z)
{
    return x+y*settings::x_num_processes+z*settings::x_num_processes*settings::y_num_processes;
}
bool at_subdomain_border(const int x,const int y,const int z){
    const bool at_lower_borders = x==0 || y==0 || z==0 || x==1 || y==1||z==1;
    const bool at_upper_borders = x==(settings::x_local_grid_size-1) || y==(settings::y_local_grid_size-1) || z==(settings::z_local_grid_size-1) || x==(settings::x_local_grid_size-2) || y==(settings::y_local_grid_size-2) || z==(settings::z_local_grid_size-2);
    return(at_lower_borders||at_upper_borders);
}

bool is_in_neighbor(const int node_x, const int node_y, const int node_z, const int proc_x_diff, const int proc_y_diff, const int proc_z_diff){
    //process cant be its own neighbor
    assert(proc_x_diff !=0 ||proc_y_diff !=0 ||proc_z_diff !=0);
    assert(std::abs(proc_x_diff)<= 1 && std::abs(proc_y_diff)<=1 &&std::abs(proc_z_diff)<=1);

    const int x_max = settings::x_local_grid_size-1;
    const int y_max = settings::y_local_grid_size-1;
    const int z_max = settings::z_local_grid_size-1;

    bool res_x{true};
    if(proc_x_diff == 1){
        res_x = (node_x == 0||node_x == 1);
    }
    else if(proc_x_diff == -1){
        res_x = (node_x == x_max||node_x == x_max-1);
    }

    bool res_y{true};
    if(proc_y_diff == 1){
        res_y = (node_y == 0||node_y == 1);
    }
    else if(proc_y_diff == -1){
        res_y = (node_y == y_max||node_y == y_max-1);
    }

    bool res_z{true};
    if(proc_z_diff == 1){
        res_z = (node_z == 0||node_z == 1);
    }
    else if(proc_z_diff == -1){
        res_z = (node_z == z_max||node_z ==z_max-1);
    }

    return (res_x && res_y && res_z);
}

void get_node_neighbors2( const int x, const int y, const int z, std::vector<int> &res)
{
/* updates the vector of the coordinates of the neighboring
 * nodes
 * in the format<x_1,y_1,z_1,x_2,....
 * */
    //assumes the vector is empty
    assert(res.size()==0);

    const int x_max{settings::x_local_grid_size-1};
    const int y_max{settings::y_local_grid_size-1};
    const int z_max{settings::z_local_grid_size-1};

    if(x>0)
    {
        res.push_back(x-1);
        res.push_back(y);
        res.push_back(z);
    }
    if(x<x_max)
    {
        res.push_back(x+1);
        res.push_back(y);
        res.push_back(z);
    }
    if(y>0)
    {
        res.push_back(x);
        res.push_back(y-1);
        res.push_back(z);
    }
    if(y<y_max)
    {
        res.push_back(x);
        res.push_back(y+1);
        res.push_back(z);
    }
    if(z>0)
    {
        res.push_back(x);
        res.push_back(y);
        res.push_back(z-1);
    }
    if(z<z_max)
    {
        res.push_back(x);
        res.push_back(y);
        res.push_back(z+1);
    }
}
std::vector<int> get_process_neighbors(const int x, const int y, const int z)
{
    /* updates the vector of the coordinates of the neighboring
 * processes
 * in the format<x_1,y_1,z_1,x_2,....
 * */
    std::vector<int> res={};
    res.reserve(78);
    const int x_max{settings::x_num_processes-1};
    const int y_max{settings::y_num_processes-1};
    const int z_max{settings::z_num_processes-1};

    for(int i = -1; i<2 ; ++i){
        for(int j = -1; j<2 ; ++j){
            for(int k = -1; k<2 ; ++k){
                //process cant be its own neighbor
                if(i !=0 || j!=0 || k!=0){
                    if( x+i >=0 && x+i <= x_max && y+j >=0 && y+j <= y_max && z+k >=0 && z+k <= z_max){
                        res.push_back(x+i);
                        res.push_back(y+j);
                        res.push_back(z+k);
                    }
                }
            }
        }
    }
    return res;
}

int get_index(const int x, const int y, const int z, const int neighbor_x, const int neighbor_y, const int neighbor_z){
    //Assumes the input is actually a neighbor
    int res{-1};
    //get neighboring proccesses:
    std::vector<int> neighbors = get_process_neighbors(x, y, z);
    for (int i = 0; i < neighbors.size(); i += 3) {
        //TODO Can maybe be parallelized more
        if(neighbor_x == neighbors[i] && neighbor_y == neighbors[i + 1] && neighbor_z == neighbors[i + 2]){
            res = i/3;
        }
    }
    //should work if actually neighbor
    assert(res >= 0);
    return res;

}
void initialize_subdomain(const int function_number, const int mask_number,SubdomainData &subdomain)
{
    //some bools to check which point is outside of the global domain

    const bool at_x_lower_border{subdomain.x_offset == 0};
    const bool at_x_upper_border{subdomain.x_offset == settings::x_global_grid_size-(settings::x_local_grid_size-2)};
    const bool at_y_lower_border{subdomain.y_offset == 0};
    const bool at_y_upper_border{subdomain.y_offset == settings::y_global_grid_size-(settings::y_local_grid_size-2)};
    const bool at_z_lower_border{subdomain.z_offset == 0};
    const bool at_z_upper_border{subdomain.z_offset == settings::z_global_grid_size-(settings::z_local_grid_size-2)};

    //loops to initialize the points outside of the original domain and the weights and speeds of all points
    for(int x=0; x<settings::x_local_grid_size; ++x){
        for(int y=0; y<settings::y_local_grid_size; ++y){
            for(int z=0; z<settings::z_local_grid_size; ++z){
                //set every weight to infinity and status to far
                subdomain.weight_array[local_arr_index(x,y,z)] = std::numeric_limits<double>::infinity();
                subdomain.status_array[local_arr_index(x,y,z)]='0';

                //if a point is outside of the original domain, it gets the respective status maybe better to give known_fix 3 instead of 6
                if(x==0 && at_x_lower_border){
                    subdomain.speed_array[local_arr_index(x,y,z)]=0;
                    continue;
                }
                if(y==0 && at_y_lower_border){
                    subdomain.speed_array[local_arr_index(x,y,z)]=0;
                    continue;
                }
                if(z==0 && at_z_lower_border){
                    subdomain.speed_array[local_arr_index(x,y,z)]=0;
                    continue;
                }
                if(x==settings::x_local_grid_size-1 && at_x_upper_border){
                    subdomain.speed_array[local_arr_index(x,y,z)]=0;
                    continue;
                }
                if(y==settings::y_local_grid_size-1 && at_y_upper_border){
                    subdomain.speed_array[local_arr_index(x,y,z)]=0;
                    continue;
                }
                if(z==settings::z_local_grid_size-1 && at_z_upper_border){
                    subdomain.speed_array[local_arr_index(x,y,z)]=0;
                    continue;
                }
                //the rest of the points have a well defined value in the speed and mask function
                subdomain.speed_array[local_arr_index(x,y,z)]= speed_funct(function_number, x + subdomain.x_offset-1, y + subdomain.y_offset-1, z+subdomain.z_offset-1);
                if(in_mask(mask_number, x+subdomain.x_offset-1, y+subdomain.y_offset-1, z+subdomain.z_offset-1)){
                    subdomain.weight_array[local_arr_index(x,y,z)]= 0;
                    subdomain.status_array[local_arr_index(x,y,z)] ='3';
                }
            }
        }
    }

}

double solve_eikonal_quadratic_3d(SubdomainData &subdomain, const int x, const int y, const int z){
    //TODO maybe improve or  check compatability of old version
    double temp_res = std::numeric_limits<double>::infinity();
    const double speed= subdomain.speed_array[local_arr_index(x,y,z)];
    assert(speed >=0);
    //If the speed is zero(outside of original domain) we should instantly return infinity
    if(speed == 0){
        return temp_res;
    }
    const int node_index = local_arr_index(x,y,z);
    double min_res_array[3];
    double h_array[3];

    //Check the x direction to set ψ_1 and h_array[0]
    int d{0};
    if(x > 0){
        int curr_index = local_arr_index(x-1,y,z);
        if((subdomain.status_array[curr_index]=='3' ||subdomain.status_array[curr_index]=='4' ||subdomain.status_array[curr_index]=='5')&& subdomain.weight_array[curr_index]< subdomain.weight_array[node_index]){
            d=-1;
        }
    }
    if(x<settings::x_local_grid_size-1){
        int curr_index = local_arr_index(x+1,y,z);
        if((subdomain.status_array[curr_index]=='3' ||subdomain.status_array[curr_index]=='4' ||subdomain.status_array[curr_index]=='5')&& subdomain.weight_array[curr_index]< subdomain.weight_array[node_index]){
            if(d==0){ // ||subdomain.weight_array[curr_index]< subdomain.weight_array[local_arr_index(x-1,y,z)]){
                d=1;
            }
            else if(subdomain.weight_array[curr_index]< subdomain.weight_array[local_arr_index(x-1,y,z)]){
                d=1;
            }
        }
    }
    if(d!=0){
        min_res_array[0]=subdomain.weight_array[local_arr_index(x+d,y,z)];
        h_array[0]=pow(settings::h,-1);
    }
    else{
        min_res_array[0]=0;
        h_array[0]=0;
    }
    //Check the y direction to set ψ_2 and h_array[1]
    d=0;
    if(y > 0){
        int curr_index = local_arr_index(x,y-1,z);
        if((subdomain.status_array[curr_index]=='3' ||subdomain.status_array[curr_index]=='4' ||subdomain.status_array[curr_index]=='5')&& subdomain.weight_array[curr_index]< subdomain.weight_array[node_index]){
            d=-1;
        }
    }
    if(y<settings::y_local_grid_size-1){
        int curr_index = local_arr_index(x,y+1,z);
        if((subdomain.status_array[curr_index]=='3' ||subdomain.status_array[curr_index]=='4' ||subdomain.status_array[curr_index]=='5')&& subdomain.weight_array[curr_index]< subdomain.weight_array[node_index]){
            if(d==0){// ||subdomain.weight_array[curr_index]< subdomain.weight_array[local_arr_index(x,y-1,z)]){
                d=1;
            }
            else if(subdomain.weight_array[curr_index]< subdomain.weight_array[local_arr_index(x,y-1,z)]){
                d=1;
            }
        }
    }
    if(d!=0){
        min_res_array[1]=subdomain.weight_array[local_arr_index(x,y+d,z)];
        h_array[1]=pow(settings::h,-1);
    }
    else{
        min_res_array[1]=0;
        h_array[1]=0;
    }
    //Check the z direction to set ψ_3 and h_array[2]
    d=0;
    if(z > 0){
        int curr_index = local_arr_index(x,y,z-1);
        if((subdomain.status_array[curr_index]=='3' ||subdomain.status_array[curr_index]=='4' ||subdomain.status_array[curr_index]=='5')&& subdomain.weight_array[curr_index]< subdomain.weight_array[node_index]){
            d=-1;
        }
    }
    if(z<settings::z_local_grid_size-1){
        int curr_index = local_arr_index(x,y,z+1);
        if((subdomain.status_array[curr_index]=='3' ||subdomain.status_array[curr_index]=='4' ||subdomain.status_array[curr_index]=='5')&& subdomain.weight_array[curr_index]< subdomain.weight_array[node_index]){
            if(d==0){// ||subdomain.weight_array[curr_index]< subdomain.weight_array[local_arr_index(x,y,z-1)]){
                d=1;
            }
            else if(subdomain.weight_array[curr_index]< subdomain.weight_array[local_arr_index(x,y,z-1)]){
                d=1;
            }
        }
    }
    if(d!=0){
        min_res_array[2]=subdomain.weight_array[local_arr_index(x,y,z+d)];
        h_array[2]=pow(settings::h,-1);
    }
    else{
        min_res_array[2]=0;
        h_array[2]=0;
    }
    int num_dir{0};
    for(int i=0; i<3;++i){
        if(h_array[i] >0){
            ++num_dir;
        }
    }

    while(num_dir!=0){
        double a = pow(h_array[0],2)+pow(h_array[1],2)+pow(h_array[2],2);
        double b = -2*(pow(h_array[0],2)*min_res_array[0]+pow(h_array[1],2)*min_res_array[1]+pow(h_array[2],2)*min_res_array[2]);
        //double speed=speed_funct(x,y,z);
        double c = pow(h_array[0] * min_res_array[0], 2) + pow(h_array[1] * min_res_array[1], 2) +pow(h_array[2] * min_res_array[2], 2) - pow(speed, -2);

        if((pow(b,2)-4*a*c)>=0){
            double psi_t = (-1*b+std::sqrt(pow(b,2)-4*a*c))/(2*a);
            if(min_res_array[0]< psi_t && min_res_array[1]<psi_t &&min_res_array[2]<psi_t){
                temp_res=std::min(psi_t, temp_res);
            }
        }
        //is always an integer(for the warning)
        int index_to_del = std::distance(min_res_array, std::max_element(min_res_array, min_res_array + 3));
        min_res_array[index_to_del]=0;
        h_array[index_to_del]=0;
        num_dir -=1;
    }
    return temp_res;

}

void update_neighbors2(SubdomainData &subdomain, const int x, const int y, const int z){

    subdomain.node_neighbors.clear();
    get_node_neighbors2(x,y,z, subdomain.node_neighbors);
    const int curr_node_index = local_arr_index(x, y, z);
    for (auto i = 0; i < subdomain.node_neighbors.size(); i += 3){
        const int neighbor_index = local_arr_index(subdomain.node_neighbors[i], subdomain.node_neighbors[i+1], subdomain.node_neighbors[i+2]);
        //WeightedPoint neighbor{neighbors[i], neighbors[i+1], neighbors[i+2], subdomain.weight_array[neighbor_index]};

        //we cant update fixed points (known fix, ghost points are handled in solve_quadratic
        if(subdomain.status_array[neighbor_index]!='3' && subdomain.weight_array[curr_node_index]< subdomain.weight_array[neighbor_index]){
            const double temp = solve_eikonal_quadratic_3d(subdomain, subdomain.node_neighbors[i], subdomain.node_neighbors[i + 1], subdomain.node_neighbors[i + 2]);
            //if the solver calculates a smaller value we update the point to BAND_NEW
            if (temp < subdomain.weight_array[neighbor_index]) {
                subdomain.weight_array[neighbor_index] = temp;
                subdomain.status_array[neighbor_index] = '1';

                //int check = subdomain.h.get_heap_index(neighbors[i], neighbors[i + 1], neighbors[i + 2]);

                if (subdomain.h.get_heap_index(subdomain.node_neighbors[i], subdomain.node_neighbors[i + 1], subdomain.node_neighbors[i + 2]) == -1) {
                    subdomain.h.insertKey(WeightedPoint{subdomain.node_neighbors[i], subdomain.node_neighbors[i + 1], subdomain.node_neighbors[i + 2], temp});
                }

                else {
                    subdomain.h.decreaseKey(subdomain.h.get_heap_index(subdomain.node_neighbors[i], subdomain.node_neighbors[i + 1], subdomain.node_neighbors[i + 2]), temp);
                }

            }
        }

    }

}

void initialize_heap(SubdomainData &subdomain){
    for(int x=0 ; x<settings::x_local_grid_size; ++x) {
        for (int y = 0; y < settings::y_local_grid_size; ++y) {
            for (int z = 0; z < settings::z_local_grid_size; ++z) {
                if (subdomain.status_array[local_arr_index(x, y, z)] == '3') {
                    update_neighbors2(subdomain, x, y, z);
                }
            }
        }
    }

}

void collect_overlapping_data2(SubdomainData &subdomain, std::vector<std::array<std::vector<ExchangeData>,26>> &exchange_vector){
    subdomain.count_new=0;
    //recover the domains' indices:
    const int x_index = subdomain.x_offset/(settings::x_local_grid_size -2);
    const int y_index = subdomain.y_offset/(settings::y_local_grid_size -2);
    const int z_index = subdomain.z_offset/(settings::z_local_grid_size -2);
    //get neighboring proccesses:
    for(int x=0; x<settings::x_local_grid_size; ++x) {
        for (int y = 0; y < settings::y_local_grid_size; ++y) {
            for (int z = 0; z < settings::z_local_grid_size; ++z) {

                const char curr_node_status{subdomain.status_array[local_arr_index(x,y,z)]};
                if(at_subdomain_border(x,y,z) && (curr_node_status == '1' || curr_node_status == '4')) {
                    //++subdomain.count_new;
                    bool increase_count_new = false;

                    for (std::size_t i = 0; i < subdomain.process_neighbors.size(); i += 3) {
                        const int neighbor_index = process_index( subdomain.process_neighbors[i], subdomain.process_neighbors[i + 1], subdomain.process_neighbors[i + 2]);

                        if (is_in_neighbor(x, y, z, x_index - subdomain.process_neighbors[i], y_index - subdomain.process_neighbors[i + 1],z_index - subdomain.process_neighbors[i + 2])) {
                            if(!increase_count_new){
                                increase_count_new = true; //only if our border point is in a neighboring process we should increase count_new
                            }
                            int process_index_in_neighbor = subdomain.index_in_neighbor[i/3];
                            //give the GLOBAL coordinates as ExchangeData
                            ExchangeData data{x+subdomain.x_offset-1, y+subdomain.y_offset-1, z+subdomain.z_offset-1, subdomain.weight_array[local_arr_index(x, y, z)]};
                            exchange_vector[neighbor_index][process_index_in_neighbor].push_back(data);
                        }
                    }

                    if(increase_count_new){ //We should only increase the count and update the status if the point is in a neighbouring domain
                        ++subdomain.count_new;
                        if (curr_node_status == '1') {
                            subdomain.status_array[local_arr_index(x, y, z)] = '2';
                        }
                        else {
                            assert(curr_node_status == '4');
                            subdomain.status_array[local_arr_index(x, y, z)] = '5';
                        }
                    }

                }
            }
        }
    }
}

void integrate_overlapping_data(SubdomainData &subdomain, double bound_band, std::array<std::vector<ExchangeData>,26> &exchange_vector) {
    //TODO can be further parallelized, look at suggestions
    for(int i=0; i<exchange_vector.size();++i){
        for(int j=0; j < exchange_vector[i].size();++j){
            ExchangeData temp_data = exchange_vector[i][j];
            //transform the GLOBAL coordinates into local ones
            const int temp_x = temp_data.x-subdomain.x_offset+1;
            const int temp_y = temp_data.y-subdomain.y_offset+1;
            const int temp_z = temp_data.z-subdomain.z_offset+1;

            const int temp_index = local_arr_index(temp_x, temp_y, temp_z);
            const double temp_val =  temp_data.value;
            const double comp = subdomain.weight_array[temp_index];
            const char status = subdomain.status_array[temp_index];

            const int index = subdomain.h.get_heap_index(temp_x, temp_y, temp_z);

            if(temp_val<subdomain.weight_array[temp_index]) {
                subdomain.weight_array[temp_index] = temp_val;

                if (subdomain.weight_array[temp_index] > bound_band) {
                    subdomain.status_array[temp_index] = '2';
                }
                else {
                    subdomain.status_array[temp_index] = '5';
                }
                const int heap_index = subdomain.h.get_heap_index(temp_x, temp_y, temp_z);
                if(heap_index==-1){
                    subdomain.h.insertKey(WeightedPoint{temp_x, temp_y, temp_z, temp_val});
                }
                else {
                    subdomain.h.decreaseKey(heap_index, temp_val);
                }
            }

        }
    }
}
void march_narrow_band(SubdomainData &subdomain, const double bound_band) {
    while(true){
        if(subdomain.h.get_size()==0){
            break;
        }
        const WeightedPoint curr_point = subdomain.h.getMin();
        //int check24 = subdomain.h.get_heap_index(1,8,20);
        const int curr_index= local_arr_index(curr_point.m_x, curr_point.m_y, curr_point.m_z);
        const double value = curr_point.weight;

        if(value> bound_band){
            break;
        }
        if(subdomain.status_array[curr_index]!= '5'){
            subdomain.status_array[curr_index] = '4';
        }

        subdomain.h.extractMin();
        subdomain.weight_array[curr_index]= value;
        update_neighbors2(subdomain, curr_point.m_x, curr_point.m_y, curr_point.m_z);
    }
}
void id(const int function_number, const int mask_number)
{
    std::cout<<"The function f(x,y,z) = ";
    switch (function_number) {
        case 1 :    std::cout<< "1.0" <<std::endl;
            break;
        case 2 :    std::cout<< "1+ 0.5*std::sin(20*PI*h*x)*std::sin(20*PI*h*y)*std::sin(20*PI*h*z)" <<std::endl;
            break;
        case 3 :    std::cout<< "(1- 0.99*std::sin(2*PI*h*x)*std::sin(2*PI*h*y)*std::sin(2*PI*h*z))" <<std::endl;
            break;
        case 4 :    std::cout<< "0.001*(pow(std::sin(x*h),2)+pow(std::cos(y*h),2)+0.1)"<<std::endl;
            break;
        case 5 :    std::cout<< "Spheric barriers, speed in barriers 0, else 1"<<std::endl;
            break;
        default :   std::cout<<"UNDEFINED FUNCTION!"<<std::endl;
            break;
    }
    std::cout<<"The mask: ";
    switch (mask_number) {
        case 1 :    std::cout<< "Ball in the center, radius 1/4" <<std::endl;
            break;
        case 2 :    std::cout<< "Origin" <<std::endl;
            break;
        case 3 :    std::cout<< "Ball at 0.25/0.25/0.25, radius 1/16 and Cube at 0.75/0.75/0.75 with diameter 1/8" <<std::endl;
            break;
        case 4 :    std::cout<< "Point in the middle" <<std::endl;
            break;
        default :   std::cout<<"UNDEFINED MASK!"<<std::endl;
            break;
    }
}

int main() {
    int function_number;
    int mask_number;
    id(function_number,mask_number);
    double width_band{std::numeric_limits<double>::infinity()};
    //Detailed description of the stride sizes is in the sources
    double stride{2* settings::h};
    double min_val_global = width_band;
    double min_array[settings::total_num_processes];
    int count_array[settings::total_num_processes];
    int count_global = 0;
    bool flag = true;
    double bound_band = width_band;
    //const int function_number{1};
    //const int mask_number{1};

    //all the largest elements there get allocated with new
    SubdomainData subdomain_array[settings::total_num_processes];
    std::vector<std::array<std::vector<ExchangeData>,26>> exchange_vector;
    exchange_vector.resize(settings::total_num_processes);
    for(int i=0;i<settings::total_num_processes;++i){
        for(int j =0 ; j<26;++j){
            exchange_vector[i][j].reserve(4*std::max({settings::x_local_grid_size,settings::y_local_grid_size,settings::z_local_grid_size}));
        }
    }
    //experimental adaptation of the stride size with the median of the function values:
    /*
    std::vector<double> FunctionValueVector;
    for (int x = 0; x < settings::x_global_grid_size; ++x) {
        for (int y = 0; y < settings::y_global_grid_size; ++y) {
            for (int z = 0; z < settings::x_global_grid_size; ++z) {
                FunctionValueVector.push_back(speed_funct(function_number,x,y,z));
            }
        }
    }
    std::sort(FunctionValueVector.begin(),FunctionValueVector.end());
    double median=(FunctionValueVector[settings::total_global_grid_size/2.0]+FunctionValueVector[settings::total_global_grid_size/2.0 -1])/2;
    //std::cout<<"Median:"<<median<<std::endl;
    stride=2*std::pow(median,-2)*settings::h;*/

    //setting domain offsets:
    for (int x = 0; x < settings::x_num_processes; ++x) {
        for (int y = 0; y < settings::y_num_processes; ++y) {
            for (int z = 0; z < settings::z_num_processes; ++z) {
                int i = process_index(x, y, z);
                subdomain_array[i].x_offset = x * (settings::x_local_grid_size - 2);
                subdomain_array[i].y_offset = y * (settings::y_local_grid_size - 2);
                subdomain_array[i].z_offset = z * (settings::z_local_grid_size - 2);
                subdomain_array[i].node_neighbors.resize(18);
                subdomain_array[i].process_neighbors = get_process_neighbors(x,y,z);
                for (std::size_t j = 0; j < subdomain_array[i].process_neighbors.size(); j += 3) {
                    const int neighbor_x = subdomain_array[i].process_neighbors[j];
                    const int neighbor_y = subdomain_array[i].process_neighbors[j+1];
                    const int neighbor_z = subdomain_array[i].process_neighbors[j+2];
                    subdomain_array[i].index_in_neighbor.push_back(get_index(neighbor_x, neighbor_y, neighbor_z,x,y,z));
                }
                assert(subdomain_array[i].process_neighbors.size() == subdomain_array[i].index_in_neighbor.size()*3);
            }
        }
    }
    //INITIALIZE_INTERFACE
    auto startTime = std::chrono::system_clock::now();
    double startTimeMarch = 0;
    omp_set_num_threads(48);
//master thread creates the tasks for the worker threads
#pragma omp parallel default(none) shared(function_number, mask_number,subdomain_array, stride, width_band, exchange_vector, min_val_global, count_global, count_array, min_array, flag, bound_band)// startTimeMarch) //num_threads(4)
#pragma omp master
    {
        //INITIALIZE HEAP AND SUBDOMAINS
        for (int i = 0; i < settings::total_num_processes; ++i) {
#pragma omp task
            {
                initialize_subdomain(function_number, mask_number, subdomain_array[i]);
                initialize_heap(subdomain_array[i]);
            }
        }
#pragma omp taskwait
        while (flag) {
            for (int i = 0; i < settings::total_num_processes; ++i) {
                count_array[i] = subdomain_array[i].count_new;
                int size = subdomain_array[i].h.get_size();
                if (subdomain_array[i].h.get_size() > 0) {
                    //std::cout<<"Process: "<< i <<" minimum: "<< subdomain_array[i].h.getMin().weight<<std::endl;
                    min_array[i] = subdomain_array[i].h.getMin().weight;
                } else min_array[i] = width_band;
            }
#pragma omp parallel for reduction(min:min_val_global)
            for (int i = 0; i < settings::total_num_processes; ++i) {
                min_val_global = std::min(min_val_global, min_array[i]);
            }
            //int count_global =0;
#pragma omp parallel for reduction(max:count_global)
            for (int i = 0; i < settings::total_num_processes; ++i) {
                count_global = std::max(count_global, count_array[i]);
            }

            if ((min_val_global >= width_band) && (count_global == 0)) {
                flag = false;
            }
            bound_band = std::min(min_val_global + stride, width_band);
            count_global = 0;
            min_val_global = width_band;
            //march band
            for (int i = 0; i <settings::total_num_processes; ++i) {
#pragma omp task
                {
                    march_narrow_band(subdomain_array[i], bound_band);
                }
            }
#pragma omp taskwait

            for(int i=0;i<settings::total_num_processes;++i){
                for(int j =0 ; j<26;++j){
                    exchange_vector[i][j].clear();
                }
            }
            //collect and transfer data
            for (int i = 0; i < settings::total_num_processes; ++i) {
#pragma omp task
                collect_overlapping_data2(subdomain_array[i], std::ref(exchange_vector));
            }
#pragma omp taskwait


            //integrate data
            for (int i = 0; i < settings::total_num_processes; ++i) {
#pragma omp task
                {
                    integrate_overlapping_data(subdomain_array[i], bound_band, std::ref(exchange_vector[i]));
                    //exchange_vector[i].clear();
                }
            }
#pragma omp taskwait
            //march band
            for (int i = 0; i < settings::total_num_processes; ++i) {
#pragma omp task
                march_narrow_band(subdomain_array[i], bound_band);
            }
#pragma omp taskwait
        }
    }

    auto endTime = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = endTime - startTime;
    //std::cout << "STRIDE: " << stride << std::endl;
    //std::cout << "MARCH: " << startTimeMarch << std::endl;
    //std::cout << "TOTAL: " << elapsed_seconds.count() << std::endl;
    double *weight_array{new double[settings::total_global_grid_size]{}};
    for (int x = 0; x < settings::x_global_grid_size; ++x) {
        for (int y = 0; y < settings::y_global_grid_size; ++y) {
            for (int z = 0; z < settings::z_global_grid_size; ++z) {

                int process_xid = x / (settings::x_local_grid_size - 2);
                int process_yid = y / (settings::y_local_grid_size - 2);
                int process_zid = z / (settings::z_local_grid_size - 2);
                int x_local_coord = (x % (settings::x_local_grid_size - 2)) + 1;
                int y_local_coord = (y % (settings::y_local_grid_size - 2)) + 1;
                int z_local_coord = (z % (settings::z_local_grid_size - 2)) + 1;
                weight_array[global_arr_index(x, y, z)] = subdomain_array[process_index(process_xid, process_yid,
                                                                                        process_zid)].weight_array[local_arr_index(
                        x_local_coord, y_local_coord, z_local_coord)];
            }

        }
    }

    int count{0};
    for (int i = 0; i < settings::total_num_processes; ++i) {
        for (int x = 1; x < settings::x_local_grid_size - 1; ++x) {
            for (int y = 1; y < settings::y_local_grid_size - 1; ++y) {
                for (int z = 1; z < settings::z_local_grid_size - 1; ++z) {
                    int j = local_arr_index(x, y, z);
                    double c = subdomain_array[i].weight_array[j];
                    if (subdomain_array[i].weight_array[j] < 100000) {
                        ++count;
                        //std::cout << "STATUS: " << subdomain_array[i].status_array[j] << std::endl;
                        //std::cout << "WEIGHT: " << subdomain_array[i].weight_array[j] << std::endl;
                        //std::cout << "SPEED: " << subdomain_array[i].speed_array[j] << "\n" << std::endl;
                    }
                }
            }
        }
    }

    //Comment out the following lines for output, python code will transform it into a .vts file

    /*std::ofstream myfile;
    myfile.open("output.txt");
    myfile << "Dimension information\n" << settings::x_global_grid_size << "\n" << settings::y_global_grid_size << "\n"
           << settings::z_global_grid_size << "\n";
    myfile << "Mask information\n";
    for (int x = 0; x < settings::x_global_grid_size; ++x) {
        for (int y = 0; y < settings::y_global_grid_size; ++y) {
            for (int z = 0; z < settings::z_global_grid_size; ++z) {
                myfile << in_mask(x, y, z) << "\n";
            }
        }
    }

    myfile << "Result information\n";
    for (int i = 0; i < settings::total_global_grid_size; ++i) {
        myfile << weight_array[i] << "\n";
    }
    myfile.close();*/

    std::cout<<"Runtime: "<< elapsed_seconds.count()<<std::endl;
    return 0;
}
