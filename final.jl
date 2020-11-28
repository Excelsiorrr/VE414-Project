using Distributions
using CSV
using LinearAlgebra
using Random
const SIG = 5
const original_data = CSV.read("/Users/ruiming_lu/Documents/Undergraduate Studies/VE414/project/data_proj_414.csv")
const index = 0
original_data[:X] = [convert(Int64, floor((1+original_data[i,:X])*SIG)) for i in 1:1:size(original_data)[1]]
original_data[:Y] = [convert(Int64, floor((1+original_data[i,:Y])*SIG)) for i in 1:1:size(original_data)[1]]




const data_empty = original_data[original_data[:Close].==0,:][:,2:3];
const data_close = original_data[original_data[:Close].>0,:][:,[2:3;8:9]];
const data_far_c = data_close[data_close[:X].>180,:];
data_far_c = data_far_c[data_far_c[:X].<300,:];
data_far_c = data_far_c[data_far_c[:Y].>250,:];
data_far_c = data_far_c[data_far_c[:Y].<300,:];

const grid = -ones(108*SIG,108*SIG);
const occupied = zeros(108*SIG,108*SIG);
const frequency = zeros(108*SIG,108*SIG);
const fruit_at= zeros(108*108*SIG*SIG,2);
const forest = zeros(108*SIG,108*SIG);
#how many grids in circle1
function circle_1()
    index = 0
    for i in (-SIG):(SIG)
        for j in (-SIG):(SIG)
            if (-i)^2+(-j)^2 <= SIG^2
                index = index + 1
            end
        end
    end
    return index
end

const index_1 = circle_1()
#return an array of grids r=1
function close_grid(x,y)
    A = Array{Int64}(undef,index_1,3)
    index = 0
    for i in (x-SIG):(x+SIG)
        for j in (y-SIG):(y+SIG)
            if (x-i)^2+(y-j)^2 <= SIG^2 && i>0 && j>0
                index = index + 1
                A[index,1] = i
                A[index,2] = j
                A[index,3] = Int64(frequency[i,j])
            end
        end
    end
    return A[1:index,1:3]
end

function sum_close_grid(x,y)
    arry = close_grid(x,y)
    deviation = 0
    for i in 1:size(arry)[1]
        if grid[arry[i,1],arry[i,2]]!=-1
            deviation = deviation+grid[arry[i,1],arry[i,2]]
        end
    end
    return deviation
end

function sample_vari()
    arry = zeros(size(data_close)[1])
    summm = 0.0
    for i in 1:size(data_close)[1]
        t1 = sum_close_grid(data_close[i,1],data_close[i,2])
        t2 = data_close[i,3]
        arry[i] = t1-t2
    end
    mu = sum(arry)/size(data_close)[1]
    for i in 1:size(data_close)[1]
        summm = (mu - arry[i])^2 + summm
    end
    return summm/size(data_close)[1]
end

function sample_devi()
    summ = 0.0
    for i in 1:size(data_close)[1]
        t1 = sum_close_grid(data_close[i,1],data_close[i,2])
        t2 = data_close[i,3]
        summ = t1-t2 + summ
    end
    return summ
end

function sample_devi_square()
    summ = 0.0
    for i in 1:size(data_close)[1]
        t1 = sum_close_grid(data_close[i,1],data_close[i,2])
        t2 = data_close[i,3]
        summ = (t1-t2)^2 + summ
    end
    return summ
end

function fruit()
    #0 for definitely empty grids
    for i in 1:size(data_empty)[1]
        center_x = data_empty[i,1]
        center_y = data_empty[i,2]
        arry = close_grid(center_x,center_y)
        for j in 1:size(arry)[1]
            grid[arry[j,1],arry[j,2]] = 0
        end
    end
    #loading frequency for each small grid
    for i in 1:size(data_close)[1]
        arry = close_grid(data_close[i,1],data_close[i,2])
        for j in 1:size(arry)[1]
            if grid[arry[j,1],arry[j,2]] == -1
                frequency[arry[j,1],arry[j,2]] = frequency[arry[j,1],arry[j,2]]+1
            end
        end
    end
    for i in 1:size(data_close)[1]
        ideal_fruit = data_close[i,3]
        prev_fruit = Int64(sum_close_grid(data_close[i,1],data_close[i,2]))
        if ideal_fruit < prev_fruit
            continue
        end
        close_arry = close_grid(data_close[i,1],data_close[i,2])
        close_arry = close_arry[sortperm(close_arry[:, 3]), :]
        #remove all zeros
        if findlast(isequal(0),close_arry[:,3]) != nothing
            close_arry = close_arry[(findlast(isequal(0),close_arry[:,3])+1):size(close_arry)[1],:]
        end

        if size(close_arry)[1] == 0
            continue
        end
        ii = size(close_arry)[1]
        while ii > 0
            if occupied[close_arry[ii,1],close_arry[ii,2]] == 1
                close_arry = close_arry[1:end .!= ii,:]
            end
            ii = ii - 1
        end
        if size(close_arry)[1] == 0
            continue
        end
        client = minimum([ideal_fruit-prev_fruit,size(close_arry)[1]])
        for j in 1:client
            grid[close_arry[j,1],close_arry[j,2]] = 1
        end
        close_arry = close_grid(data_close[i,1],data_close[i,2])
        for j in 1:size(close_arry)[1]
            occupied[close_arry[j,1],close_arry[j,2]] = 1
        end
    end
    println("Iteration:  ",0)
    println("TSS: ",sample_devi_square())
    println("SUM: ",sample_devi())
    println("VAR: ",sample_vari())
    return sample_devi_square()
end

function convolution()
    fruit_at = fruit_coordinate(grid)
    for i in 1:size(fruit_at)[1]
        #println("[",i,"/",size(fruit_at)[1],"]")
        o_x = Int64(fruit_at[i,1])
        o_y = Int64(fruit_at[i,2])
        delta = Int64(ceil(0.2*SIG))
        dir_xy = [o_x o_y+delta Inf;
                  o_x-delta o_y Inf;
                  o_x+delta o_y Inf;
                  o_x o_y-delta Inf;]
                  #o_x-delta o_y+delta Inf;
                  #o_x+delta o_y+delta Inf;
                  #o_x-delta o_y-delta Inf;
                 # o_x+delta o_y-delta Inf;
        o_devi = sample_devi_square()
        for j in 1:size(dir_xy)[1]
            n_x = Int64(dir_xy[j,1])
            n_y = Int64(dir_xy[j,2])
            if grid[n_x,n_y] == -1
                grid[o_x,o_y] = -1
                grid[n_x,n_y] = 1
                n_devi = sample_devi_square()
                if abs(n_devi) < abs(o_devi)
                    dir_xy[j,3] = n_devi
                end
                grid[o_x,o_y] = 1
                grid[n_x,n_y] = -1
            end
        end
        dir_xy = dir_xy[sortperm(dir_xy[:, 3]), :]
        if dir_xy[1,3] < Inf
            grid[o_x,o_y] = -1
            grid[Int64(dir_xy[1,1]),Int64(dir_xy[1,2])] = 1
            fruit_at[i,1] = dir_xy[1,1]
            fruit_at[i,2] = dir_xy[1,2]
        end
    end
    println("TSS: ",sample_devi_square())
    println("SUM: ",sample_devi())
    println("VAR: ",sample_vari())
    return sample_devi_square()
end

function optimization()
    iter = 2
    prev = 0.0
    curr = 0.0
    t = false
    for i in 1:iter
        if i == 1
            prev = fruit()
            println("Iteration: ",i)
            curr = convolution()
        else
            prev = sample_devi_square()
            println("Iteration: ",i)
            curr = convolution()
        end
        println("Current optimization rate: ",abs(prev-curr)/prev)

        if abs(prev-curr)/prev < 0.01*0.5
            println("Optimization Complete with ",0.5,"%")
            return;
        end
    end
    return 80;
    #println("Warning!!! Not optimaized after ",iter," iterations!")
end

function far_deviation()
    n = size(data_far_c)[1];
    devia = zeros(n,1);
    for i in 1:n
        x = data_far_c[i,1];
        y = data_far_c[i,2];
        devia[i] = fruit_nearby(fruit_coordinate(grid),x,y) - data_far_c[i,4];
    end
    return devia;
end


function fruit_coordinate(ggrid)
    indexx = 0
    for i in 1:108*SIG
        for j in 1:108*SIG
            if ggrid[i,j] == 1.0
                indexx = indexx + 1;
            end
        end
    end
    fruit_cor = zeros(indexx,2)
    indexx = 1
    for i in 1:108*SIG
        for j in 1:108*SIG
            if ggrid[i,j] == 1.0
                fruit_cor[indexx,1] = i;
                fruit_cor[indexx,2] = j;
                indexx = indexx + 1;
            end
        end
    end
    return fruit_cor
end

function fruit_nearby(fruit,x,y)
    n = size(fruit)[1];
    num = 0;
    for i in 1:n
        if sqrt((fruit[i,1]-x)^2+(fruit[i,2]-y)^2) <= 3*SIG
            num = num + 1;
        end
    end
    return num;
end

function add_far()
    fruit_at = fruit_coordinate(grid);
    tt = far_deviation();
    mu = mean(tt);
    varr = var(tt;dims=1)[1];
    return mu,varr;
end

function k_means(X,n_clusters,r)
    n = size(X)[1];
    X_mean = mean(X,dims=1);
    prev_mu = zeros(n_clusters,2);
    rng = MersenneTwister(50*n_clusters+r);
    rand_arry = shuffle(rng, Vector(1:n));
    ii = 1;
    for i in 1:n_clusters
        #prev_mu[i,1] = (120)/(n_clusters-1)*(i-1)+180;
        #prev_mu[i,2] = 300;
        if i == 1
            prev_mu[i,1] = X[rand_arry[ii],1];
            prev_mu[i,2] = X[rand_arry[ii],2];
            ii = ii + 1;
        else
            v = false
            while !v
                #println(ii)
                v = true;
                for j in 1:i
                    if abs(prev_mu[j,1]-X[rand_arry[ii],1])<2 && abs(prev_mu[j,2]-X[rand_arry[ii],2])<2
                        v=false;
                        break;
                    end
                    # if fruit_nearby(X,X[rand_arry[ii],1],X[rand_arry[ii],2]) < 20
                    #     v = false;
                    #     break;
                    # end

                end
                ii = ii + 1;
            end
            prev_mu[i,1] = X[rand_arry[ii],1];
            prev_mu[i,2] = X[rand_arry[ii],2];
        end
    end
    deviation = Inf
    labels = zeros(n,1);
    prev_indexx = zeros(n_clusters,1);
    prev_labels = zeros(n,1);
    valid = true;
    while deviation > 1
        #println(prev_mu)
        #update
        labels = zeros(n,1);
        temp = zeros(n_clusters,1);
        for i in 1:n
            for j in 1:n_clusters
                temp[j] = sqrt((X[i,1]-prev_mu[j,1])^2+(X[i,2]-prev_mu[j,2])^2);
            end
            labels[i] =findmin(temp)[2][1];
        end
        new_mu = zeros(n_clusters,2);
        indexx = zeros(n_clusters,1);
        for j in 1:n_clusters
            for i in 1:n
                if labels[i] == j
                    new_mu[j,1] = new_mu[j,1] + X[i,1];
                    new_mu[j,2] = new_mu[j,2] + X[i,2];
                    indexx[j] = indexx[j] + 1;
                end
            end
        end
        if count(i->i<2,indexx) !=0
            valid = false;
            #println(valid);
            break;
        end
        deviation = 0.0
        for i in 1:n_clusters
            new_mu[i,:] = new_mu[i,:]/indexx[i];
            deviation = deviation + sqrt((new_mu[i,1]-prev_mu[i,1])^2+(new_mu[i,2]-prev_mu[i,2])^2)
        end
        prev_mu = new_mu;
        prev_indexx = indexx;
        prev_labels = labels;
        #println(new_mu)
    end
    return prev_mu,prev_labels,prev_indexx,valid;
    #cluster = Array{Array}(undef,n_clusters,1)
end

function initialize_clusters(X,n_clusters,r)
    pi_k = zeros(n_clusters,1);
    mu_k = zeros(n_clusters,2);
    cov_k = Array{Array}(undef,n_clusters,1);
    gamma_nk =  Array{Array}(undef,n_clusters,1);
    totals = Array{Array}(undef,n_clusters,1);
    #X_mean = mean(X,dims=1)
    valid = false;
    mu_k,labels,indexx,valid = k_means(X,n_clusters,r);
    while !valid
        #println(r);
        r = r+1;
        mu_k,labels,indexx,valid = k_means(X,n_clusters,r);
        if r>10000
            return;
        end
    end
    for i in 1:n_clusters
        pi_k[i] = 1.0/n_clusters;
        #mu_k[i,1] = 2*X_mean[1]*i/(n_clusters+1);  #i*108*SIG/(n_clusters+1);
        #mu_k[i,2] = 2*X_mean[2]*i/(n_clusters+1);  #i*108*SIG/(n_clusters+1);
    end
    x0 = Array{Array}(undef,n_clusters,1)
    for i in 1:n_clusters
        cov_k[i] = [1.0 0.0;0.0 1.0];
    end
    iter = 0.0;
    return pi_k, mu_k, cov_k, gamma_nk,totals,labels,indexx,iter,r;
end

function gaussian(mu,cov,x)
    eigenn = eigen(cov).values;
    dett = 0.0
    if eigenn[1]==0
        dett = eigenn[2];
    elseif eigenn[2]==0
        dett = eigenn[1];
    else
        dett = eigenn[1]*eigenn[2];
    end
    t = 1/(2*pi*sqrt(abs(dett)))*exp(-0.5*transpose(x-mu)*pinv(cov)*(x-mu));
    return t;
end

function expectation_step(X,clusters,n_clusters)
    totals = zeros(size(X)[1],1);
    for i in 1:n_clusters
        pi_k = clusters[1][i];
        mu_k = clusters[2][i,:];
        cov_k = clusters[3][i,:][1];
        gamma_nk = zeros(size(X)[1],1);
        for j in 1:size(X)[1]
            #println(j," ",gaussian(mu_k,cov_k,X[j,:]))
            gamma_nk[j] = pi_k*gaussian(mu_k,cov_k,X[j,:]);#pdf(dis,X[j,:]);
        end
        for j in 1:size(X)[1]
            totals[j] = totals[j]+gamma_nk[j];
        end
        clusters[5][i] = totals;
        clusters[4][i] = gamma_nk;
    end

    for i in 1:n_clusters
        for j in 1:size(X)[1]
            if (clusters[5][i][j]==0)
                clusters[4][i][j] = 0;
            else
                clusters[4][i][j] = clusters[4][i][j]/clusters[5][i][j];
            end
        end
    end
    return clusters;
end

function maximization_step(X,clusters,n_clusters)
    n = size(X)[1]
    for i in 1:n_clusters
        gamma_nk = clusters[4][i];
        cov_k = zeros(size(X)[2],size(X)[2]);
        mu_k = zeros(2,1);
        n_k = sum(gamma_nk);
        pi_k = Float64(n_k/n);
        for j in 1:n
            if n_k > 0
                mu_k[1] = mu_k[1] + X[j,1]*gamma_nk[j]/n_k;
                mu_k[2] = mu_k[2] + X[j,2]*gamma_nk[j]/n_k;
            end
        end

        diff = zeros(n,2);
        for j in 1:n
            diff[j,1] = X[j,1] - mu_k[1];
            diff[j,2] = X[j,2] - mu_k[2];
        end

        for j in 1:n
            cov_k = cov_k + gamma_nk[j]*([diff[j,1]*diff[j,1] diff[j,1]*diff[j,2];diff[j,2]*diff[j,1] diff[j,2]*diff[j,2]]);
        end
        cov_k = cov_k/n_k;
        clusters[1][i] = pi_k;
        clusters[2][i,:] = mu_k;
        clusters[3][i] = cov_k;
    end
    return clusters;
end

function get_likelihood(X,clusters,n_clusters)
    summ = 0;
    for j in 1:n_clusters
        for i in 1:size(X)[1]
            if (clusters[5][j][i] > 0)
                summ = summ + log(clusters[5][j][i]);
            end
        end
    end
    return summ;
end

function train_gmm(X,n_clusters,r)
    prev_clusters = initialize_clusters(X,n_clusters,r);
    clusters = initialize_clusters(X,n_clusters,r);
    n_epochs = 200;
    likelihoods = zeros(n_epochs,1);
    firstt = true;
    for i in 1:n_epochs
        try
            #print(i)
            clusters = expectation_step(X,clusters,n_clusters);
        catch
            println("  LIEKAI")
            prev_clusters = (prev_clusters[1],prev_clusters[2],prev_clusters[3],prev_clusters[4],prev_clusters[5],prev_clusters[6],prev_clusters[7],i-1,prev_clusters[9]);
            return prev_clusters,false;
        end
        clusters = maximization_step(X,clusters,n_clusters);
        likelihood = get_likelihood(X,clusters,n_clusters);
        likelihoods[i] = likelihood;
        if i>=2
            println("Epochs: ",i,"  ",abs((likelihood-likelihoods[i-1])/likelihoods[i-1]));
            if abs((likelihood-likelihoods[i-1])/likelihoods[i-1]) < 0.01*5
                # if firstt
                #     firstt = false;
                # else
                clusters = (clusters[1],clusters[2],clusters[3],clusters[4],clusters[5],clusters[6],clusters[7],i,clusters[9]);
                println("NIUPI")
                return clusters,true;
                #end
            else
                firstt = true;
            end
        end
        prev_clusters = clusters;
        #println(n_clusters," clusters in ","Epoch: ",i+1," Likelihood: ",likelihood);
    end
    println("XUE MA LIEKAI")
end

function train_gmm_ex(X,n_clusters,r)
    if_found = false
    while true
        (clusters,if_found) = train_gmm(X,n_clusters,r);
        r = r +20;
        if if_found
            return clusters;
        end
    end
end

function BIC(X,clusters,n_clusters)
    #pi_k, mu_k, cov_k, gamma_nk,totals,labels,indexx;
    N = size(X)[1];
    d = size(X)[2];
    m = n_clusters;
    summ = 0.0;
    for i in 1:N
        #println(i)
        mu = clusters[2][Int64(clusters[6][i]),:];
        summ = summ + sqrt((mu[1]-X[i,1])^2+(mu[2]-X[i,2])^2);
    end
    cl_var = summ*(1.0/(N-m)/d);
    const_term = 0.5*m*log(N)*(d+1);
    bic_arry = zeros(n_clusters,1);
    for i in 1:m
        bic_arry[i] = clusters[7][i]*log(clusters[7][i])-clusters[7][i]*log(N)-((clusters[7][i]*d)/2)*log(2*pi*cl_var)-((clusters[7][i]-1)*d/2);
    end
    return sum(bic_arry)-const_term;
end

function Elbow(X,clusters,n_clusters)
    #pi_k, mu_k, cov_k, gamma_nk,
    #totals,labels,indexx;
    n = size(X)[1];
    wss = zeros(n_clusters,1);
    for i in 1:n
        t = Int64(clusters[6][i]);
        mu = clusters[2][t,:]
        wss[t] = wss[t] + sqrt((mu[1]-X[i,1])^2+(mu[2]-X[i,2])^2);
    end
    return sum(wss);
end


@time begin
    trees = optimization();
    clusters=train_gmm_ex(fruit_coordinate(grid),trees,1);
end
