%% Sketching algorithm for finding the closest point on a convex hull
% written by: Roozbeh Yousefzadeh
% last updated August 2021

%% read the dataset
[trainD,testD] = loadData("mnist",0,0,0);
% trainD.X = double(trainD.X);  testD.X = double(testD.X);
%%
tol = 1e-8; % tolerance for residual
tolx = 1e-6; % tolerance for dismissing variables with small values
maxit = 1; % this can be large, but it is not necessary as we currently solve the subspace problem exactly
eta = 5;  % number of pieces for sketching
q = testD.X(1,:); % query point - we can loop over all testing samples

xopt = sketch(eta,trainD,q,maxit,tol,tolx);

%% Sketching function
function [xopt] = sketch(eta,trainD,q,maxit,tol,tolx)
    ssm = 1;
    printlevel = 0;

    ntrain = size(trainD.X);
    % ntrain = 10000;

    xopt = 1;
    tic
    for iter_sk = 1:eta
        fprintf('----sketching iter %d \n',iter_sk);
        D = trainD.X(1:ntrain/eta*iter_sk,:);
        [n,d] = size(D);
        c = -D*q';
        l = zeros(n,1);
        u = ones(n,1);

        x0 = zeros(n,1);
        x0(1:numel(xopt)) = xopt;
        [xopt,~,~,~,res,~,stat] = bqp_pg_2(c,D,l,u,x0,maxit,ssm,printlevel,tol,tolx,q);
    end
    toc
    fprintf('distance to query is %.5f\n',norm(xopt'*D-q));
end

%% QP function
function [x,zl,zu,obj,res,iter,stat] = bqp_pg_2(c,D,l,u,x0,maxit,ssm,printlevel,tol,tolx,q)
n = size(c,1);
iter = 0;
x = projection(x0,l,u);
stat = 1;
res = 10*tol;
obj = 0;
zl = []; zu = zl;
%% Functions
qf = @(z) 0.5*z'*D*(D'*z) + c'*z;
grad = @(z) (D*(D'*z)+c);
% g = @(z) (D*D'*z+c)./sum(D*D'*z+c)-1/n; % normalizing the gradient - not helpful
% fprintf('     iter  ,   obj  ,   residual\n');
%% Main loop
while iter <= maxit
%     [k,res,zl,zu] = kkt(x,D,c,l,u);
%     obj = qf(x);
%     solution = [iter,obj,res];
    x_prev = x;
    %% Print
    if printlevel == 1
        fprintf('%7.0f     %7.2f    %7.2f \n' , solution);
    end
%     %% KKT condition
%     if k==1
%         stat=0;
%         fprintf('\nKKT satisfied\n\n');
%         break
%     end
%     %% Residual tolerance
%     if res<tol
%         stat=0;
%         fprintf('\nresidual satisfied\n\n');
%         %display(res);
%         break
%     end
    %% Cauchy step
    t_bar = inf(size(x));
    gg = grad(x);
    t1 = (x-u)./gg;
    t2 = (x-l)./gg;
    for i = 1:n
        if gg(i)<0 && u(i)<inf
            t_bar(i) = t1(i);
        elseif gg(i)>0 && u(i)>-inf
            t_bar(i) = t2(i);
        else
            t_bar(i) = inf;
        end
    end
    t_sort = unique(t_bar(t_bar~=0));
    p = -gg;
    t1 = 0;
    xc = x;
    for i = 1:size(t_sort)
        t2 = t_sort(i);
        p = gp(p,t1,t_bar);
        t = tfinder(t1,t2,p,c,D,xc);
        xc = xc + t*p;
        if t>=t1 && t<t2
            break
        end
        t1=t2;
    end
    xc = projection(xc,l,u);
%     toc
    %% Aproximate solution to accelerate
    if ssm==1
        xc(xc<tolx) = 0;
        xs = xc;
        sl = xc-l; %slack l
        active_set = ~sl';
        inactive_set = 1:n;
        inactive_set(active_set) = [];
        Di = D(inactive_set,:);
        xi = xc(inactive_set);
        
        ni = numel(inactive_set);
%         fprintf('%d inactive constraints out of %d total\n',ni,n);
        lb = zeros(ni,1);  ub = ones(ni,1);
        options = optimoptions('lsqlin','Algorithm','interior-point','Display','none',...
            'OptimalityTolerance',1e-8);
        xi = lsqlin(Di',q',[],[],ones(1,ni),1,lb,ub,xi,options);

        xs(inactive_set) = xi;
        xs(xs<tolx) = 0;
        x = projection(xs,l,u);
        fprintf('%d inactive constraints out of %d total\n',sum(xs>0),n);
    else
        x = xc;
    end
%     toc
    %% change in solution
    if norm(x-x_prev) < tol && iter > 1
        stat = 0;
        fprintf('no improvement in optimal solution\n');
        break
    end
    %% maximum iteration
    iter = iter+1;
    if iter==maxit+1
        fprintf('max iteration reached\n\n');
    end
end
end

%% Projection Operator
function P = projection(x,l,u)
    P = max(min(x,u),l);
end

%% Gradient Projection Operator
function p = gp(p,t1,t_bar)
    p(t_bar<=t1)=0;
end

%% tfinder Operator
function t = tfinder(t1,t2,p,c,D,x)
    f1=c'*p+x'*D*(D'*p);
    f2=p'*D*(D'*p);
    if f1>0
        t=t1;
        %% ('case1');
    elseif -f1/f2>=0 && -f1/f2<t2-t1
        t=t1-f1/f2;
        %% ('case2');
    else
    t = t2;
    %% ('case3');
    end
end

%% KKT and residual
function [v,res,yl,ye] = kkt(x,D,c,l,u)
    yl=(D*D'*x+c); %lagrange l
    tolx = 1e-8;
    active = find(x<tolx);
    inactive = x>tolx;
    ye = mean(yl(inactive));
    se = sum(x) - 1;
    ye(se~=0)=0;
    sl=x-l; %slack l
    yl(sl~=0)=0;
    yl(active) = yl(active) - ye;
    v=0;
    sl(sl==inf)=1e5;
    %%          feasibility                         complimentary                 stationary                    duality
    res = norm(min(sl,0))+norm(sum(x)-1)  +  norm(sl.*yl)+norm(se.*ye)  +  norm(D*D'*x+c-yl-ye)  +  norm(min(yl,0))+norm(min(ye,0));
    if res==0
        v = 1; % KKT satistied
    end
end