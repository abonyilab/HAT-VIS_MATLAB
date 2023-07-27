clear all
close all

%%
%Add Houses...
% Load data 
[EC1num,EC1txt]=xlsread('hat_vis_electric_car_qfd_example.xlsx', 'Munka1'); % Electric car
% [EC1num,EC1txt]=xlsread('hat_vis_electric_car_qfd_example.xlsx', 'Munka4'); % Electric car

% [EC1num,EC1txt]=xlsread('hat_vis_electric_car_qfd_example.xlsx', 'Munka2'); % Electric car

%Dual
% [EC1num,EC1txt]=xlsread('hat_vis_electric_car_qfd_example.xlsx', 'Munka3'); % Electric car

%% Houses one by one
% 
%Munka1, Munka4
id{1}=EC1txt(2:9,2)% ranks in unsorted list - id of the requirement
w{1}=EC1num(3:10,1) % weights (importance)
A{1}=EC1num(3:10,2:end)
id2=EC1txt(1, 4:end) ;
% Weighted matrix
A{1} = A{1}.*w{1};


% %Munka2
% id{1}=EC1txt(3:10,1) % ranks in unsorted list - id of the requirement
% w{1}=EC1num(1:8,3) % weights (importance)
% A{1}=EC1num(1:8,4:end)
% id2=EC1txt(2,2:end);
% % Weighted matrix
% A{1} = A{1}.*w{1};



% %%Dual
% id{1}=EC1txt(1,2:9) % ranks in unsorted list - id of the requirement
% w{1}=EC1num(1,3:end) % weights (importance)
% A{1}=EC1num(2:12,3:end)
% id2=EC1txt(3:13,1);
% % Weighted matrix
% A{1} = A{1}.*w{1};

%% Name of the edges
edgenames ={'ES1','ES2','ES3','ES4','ES5','ES6','ES7','ES8','ES9','ES10','ES11'}
nodenames = {'CR1','CR2','CR3','CR4','CR5','CR6','CR7','CR8'}

edge_name_list = cell(size(id2, 2), 2);
for i = 1:size(id2, 2)
    edge_name_list{i, 1} = edgenames(i);
    edge_name_list{i, 2} = id2(i);
end

%% conv. and order
for i=1:length(A)       % size(A, 2)
    zero=isnan(A{i});	% where in A{i} is there any not-a-number
    A{i}(zero)=0       % the marked element of A{i} will be replaced with zero

    for row=1:size(A{i},1)
        for column=1:size(A{i},2)
            if isnan(A{i}(row, column))
                A{i}(row, column)=0;
            end
        end
    end

    A{i}(isnan(A{i}))=0;

    [dum,indexs]=sort(id{i});
    A{i}=A{i}(indexs,:);
end   
%%
Nn = cell(length(A), 1);
Mm = cell(length(A), 1);
for i=1:length(A)
    Nn{i}= size(A{i},1);
    Mm{i}= size(A{i},2)
end
%% Sort the nodes
for l=1:length(A) 
    N=Nn{l};
    M=Mm{l};
    x=[1:N]';
    y=[1:M]'; 
    b=zeros(N,1);  
    for it=1:10
    for i=1:N
        bn=0;
        bd=0;
        for m=1:M 
            for n=1:N
                bn= bn+x(n)*A{l}(n,m)*A{l}(i,m);
                bd= bd+ A{l}(n,m)*A{l}(i,m);
            end
        end
        b(i)=bn/bd;
    end    
    [dum,x]=sort(b);   
    end
end
%% Sort the edges 
for l=1:length(A) 
    N=Nn{l};
    M=Mm{l};
    x=[1:N]';
    y=[1:M]'; 
    d=zeros(M,1); 
    for it=1:10
    for j=1:M
        dn=0;
        dd=0;
        for n=1:N
            for m=1:M
                dn= dn+y(m)*A{l}(n,m)*A{l}(n,j);
                dd= dd+ A{l}(n,m)*A{l}(n,j);
            end
        end
        d(j)=dn/dd;
    end    
    [dum,y]=sort(d);   
    end 
    % subplot(1, length(A), l);
    figure()
    spy(A{l}(x, y));
    title(['House of Noises ', num2str(l)]);
    hold on
end


%% MDS-based visualizaton of the objects 
D=cell(length(A), 1);
Y=cell(length(A), 1);
simO = cell(length(A), 1);
simOT = cell(length(A), 1);
for l=1:length(A) 
    N=Nn{l};
    M=Mm{l};
    simO{l}=[];
    for k=1:N
        simO{l}=[simO{l} sum(min(A{l},repmat(A{l}(k,:),N,1)),2)./sum(max(A{l},repmat(A{l}(k,:),N,1)),2)];
    end    
    
    %transifive similarity  
    simOT{l}=simO{l};
    for it=1:N
        for n=1:N
            simOT{l}=max(simOT{l},repmat(simOT{l}(n,:),N,1).*repmat(simOT{l}(:,n),1,N));
        end
    end    
    D{l}=1-simOT{l};
    Y{l} = cmdscale(D{l},2);
end

%% plot the edges as bounding ellipsoids or convex hullls



MarkerVec = {'r-', 'g-', 'b-', 'k-', 'c-', 'm-', ...
            'r-.', 'g-.', 'b-.', 'k-.', 'c-.', 'm-.', ...
            'r:', 'g:', 'b:', 'k:', 'c:', 'm:', ...
            'r--', 'g--', 'b--', 'k--', 'c--', 'm--', ...
            'ro-', 'go-', 'bo-', 'ko-', 'co-', 'mo-', ...
            'ro-.', 'go-.', 'bo-.', 'ko-.', 'co-.', 'mo-.', ...
            'ro:', 'go:', 'bo:', 'ko:', 'co:', 'mo:', ...
            'ro--', 'go--', 'bo--', 'ko--', 'co--', 'mo--', ...
             'r+-', 'g+-', 'b+-', 'k+-', 'c+-', 'm+-', ...
            'r+-.', 'g+-.', 'b+-.', 'k+-.', 'c+-.', 'm+-.', ...
            'r+:', 'g+:', 'b+:', 'k+:', 'c+:', 'm+:', ...
            'r+--', 'g+--', 'b+--', 'k+--', 'c+--', 'm+--', ...
        };
linedex={'-','--','-.',':'};
all_marks = {'o','+','*','.','x','s','d','^','v','>','<','p','h'};



%%
for l=1:length(A)
    N=Nn{l};
    M=Mm{l};
    figure()
    % subplot(1, length(A), l);
    plot(Y{l}(:,1),Y{l}(:,2),'.')
    hold on
    text(Y{l}(:,1),Y{l}(:,2),num2str([1:N]'),'FontSize',14)
end

%%
% hold on 
% P=cell(length(A), 1);
% G=cell(length(A), 1);
% T=cell(length(A), 1);
% p=cell(length(A), 1);
% for l=1:length(A)
%     figure()
%     N=Nn{l};
%     M=Mm{l};
%     for m=1:M
%         P{l}=Y{l}(find(A{l}(:,m)>0),:);
%     
%         if size(unique(P{l}, 'rows'), 1) < 3
%             k = 1:size(unique(P{l}, 'rows'), 1);
%             av = P{l}(k, :);
%         else
%     
%         [k,av] = convhull(P{l});
%         end
%         Marker=MarkerVec{mod(m,length(MarkerVec))+1};
%         fill(P{l}(k,1),P{l}(k,2),Marker(1),'FaceAlpha',0.25)
%         hold on
%         plot(P{l}(k,1),P{l}(k,2),Marker,'LineWidth',2)
%         text(P{l}(k,1),P{l}(k,2),num2str(P{l}(k)),'FontSize',14);
%         hold on 
%        if max(std(P{l}(k,:)))>1e-3
%         [eA , eCent] = MinVolEllipse(P{l}(k,:)', 0.01);
%        else 
%          eA=eye(2)*1e3; %minimum size ... 
%          eCent=mean(P{l}(k,:))';
%        end 
%        if size(P{l}(k,:),1)==1
%             eA=eye(2)*1e3;
%             eCent=(P{l}(k,:))';
%        end    
%         Ellipse_plot(eA*0.9,eCent,Marker);
%         hold on 
%     end
%     axis off
% 
% 
% % MST of the whole network 
% hold on 
% 
% % for l=1:length(A)
% G{l} = graph((simO{l}>0).*(1-simO{l}));
% T{l} = minspantree(G{l});
% p{l} = plot(G{l},'XData',Y{l}(:,1),'YData',Y{l}(:,2),...
%         'EdgeLabel',1-G{l}.Edges.Weight, 'EdgeColor','b','EdgeFontSize',14,'NodeLabel',[1:N],'NodeFontSize',14);
% [T{l},pred] = minspantree(G{l});
% highlight(p{l},T{l})
% end    
% % print('HyperGraph','-f2','-dpng')


%% Plot the edges as MST-s

dy=cell(length(A), 1);
dx=cell(length(A), 1);
Cr=cell(length(A), 1);
As=cell(length(A), 1);

Y_copy=round(Y{l}(:,:),4);

for i = 1:size(Y_copy, 1)
    for j = i+1:size(Y_copy, 1)
        if abs(Y_copy(i, :) - Y_copy(j, :)) < 0.1
        end
     end
end


%%

for l=1:length(A)
    figure()
    N=Nn{l};
    M=Mm{l};
    dy{l}=range(Y{l}(:,2))/100000/M;
    dx{l}=range(Y{l}(:,1))/10/M;
    
    As{l}=(simO{l}>0).*(1-simO{l}); %this is the adjacency matrix of the weighted similarity network (whole) 
    
    hold on 
    layoutOptions = struct('Iterations', 100, 'UseGravity', true);
    ures=[];
    Cr=rand(M,3);
    for m=1:M 
        Marker=MarkerVec{mod(m,length(MarkerVec))+1};
        lineStyle = linedex{mod(m, 4) + 1};
        selnodes=find(A{l}(:,m)>0); %nodes involved in the m-th edge 
        P{l}=Y{l}(selnodes,:); %the position of these nodes
        G{l} = graph(As{l}(selnodes,selnodes));
        T{l} = minspantree(G{l});

        p{m} = plot(T{l},'XData',P{l}(:,1),'YData',P{l}(:,2)+dy{l}*m,...
         'EdgeColor',Cr(m,:),'EdgeFontSize',10,'NodeColor',Cr(m,:),... %Marker(1)
         'LineWidth',5,... %'EdgeLabel',1-T.Edges.Weight,
         'MarkerSize',15,'Marker',all_marks{mod(m,13)+1},'LineStyle', lineStyle, 'Displayname', edgenames{m});%Marker(2:end)); %,
        set(p{m},'EdgeAlpha',1);

        text(Y_copy(:,1),Y_copy(:,2),nodenames([1:N]'),'FontSize',20,'HorizontalAlignment','left','VerticalAlignment','top');
        p{m}.NodeLabel={};
        %highlight(p,T)
        
        legend show
    
        K{l}=Y{l}(find(A{l}(:,m)>0),:);

        if size(unique(K{l}, 'rows'), 1) < 3
            k = 1:size(unique(K{l}, 'rows'), 1);
            av = K{l}(k, :);
        else

         end
          
    end
    
end    
axis off



print('HyperGraph2','-f3','-dpng')


%% Functions needed to plot the bounding ellopsoids 

function [A , c] = MinVolEllipse(P, tolerance)
% [A , c] = MinVolEllipse(P, tolerance)
% Finds the minimum volume enclsing ellipsoid (MVEE) of a set of data
% points stored in matrix P. The following optimization problem is solved: 
%
% minimize       log(det(A))
% subject to     (P_i - c)' * A * (P_i - c) <= 1
%                
% in variables A and c, where P_i is the i-th column of the matrix P. 
% The solver is based on Khachiyan Algorithm, and the final solution 
% is different from the optimal value by the pre-spesified amount of 'tolerance'.
%
% inputs:
%---------
% P : (d x N) dimnesional matrix containing N points in R^d.
% tolerance : error in the solution with respect to the optimal value.
%
% outputs:
%---------
% A : (d x d) matrix of the ellipse equation in the 'center form': 
% (x-c)' * A * (x-c) = 1 
% c : 'd' dimensional vector as the center of the ellipse. 
% 
% example:
% --------
%      P = rand(5,100);
%      [A, c] = MinVolEllipse(P, .01)
%
%      To reduce the computation time, work with the boundary points only:
%      
%      K = convhulln(P');  
%      K = unique(K(:));  
%      Q = P(:,K);
%      [A, c] = MinVolEllipse(Q, .01)
%
%
% Nima Moshtagh (nima@seas.upenn.edu)
% University of Pennsylvania
%
% December 2005
% UPDATE: Jan 2009
%%%%%%%%%%%%%%%%%%%%% Solving the Dual problem%%%%%%%%%%%%%%%%%%%%%%%%%%%5
% ---------------------------------
% data points 
% -----------------------------------
[d N] = size(P);
Q = zeros(d+1,N);
Q(1:d,:) = P(1:d,1:N);
Q(d+1,:) = ones(1,N);
% initializations
% -----------------------------------
count = 1;
err = 1;
u = (1/N) * ones(N,1);          % 1st iteration
% Khachiyan Algorithm
% -----------------------------------
while err > tolerance,
    X = Q * diag(u) * Q';       % X = \sum_i ( u_i * q_i * q_i')  is a (d+1)x(d+1) matrix
    M = diag(Q' * inv(X) * Q);  % M the diagonal vector of an NxN matrix
    [maximum j] = max(M);
    step_size = (maximum - d -1)/((d+1)*(maximum-1));
    new_u = (1 - step_size)*u ;
    new_u(j) = new_u(j) + step_size;
    count = count + 1;
    err = norm(new_u - u);
    u = new_u;
end
%%%%%%%%%%%%%%%%%%% Computing the Ellipse parameters%%%%%%%%%%%%%%%%%%%%%%
% Finds the ellipse equation in the 'center form': 
% (x-c)' * A * (x-c) = 1
% It computes a dxd matrix 'A' and a d dimensional vector 'c' as the center
% of the ellipse. 
U = diag(u);
% the A matrix for the ellipse
% --------------------------------------------
A = (1/d) * inv(P * U * P' - (P * u)*(P*u)' );
% center of the ellipse 
% --------------------------------------------
c = P * u;

end

function Ellipse_plot(A, C, Marker)
%
%  Ellipse_Plot(A,C,N) plots a 2D ellipse or a 3D ellipsoid 
%  represented in the "center" form:  
%               
%                   (x-C)' A (x-C) <= 1
%
%  A and C could be the outputs of the function: "MinVolEllipse.m",
%  which computes the minimum volume enclosing ellipsoid containing a 
%  set of points in space. 
% 
%  Inputs: 
%  A: a 2x2 or 3x3 matrix.
%  C: a 2D or a 3D vector which represents the center of the ellipsoid.
%  N: the number of grid points for plotting the ellipse; Default: N = 20. 
%
%  Example:
%  
%       P = rand(3,100);
%       t = 0.001;
%       [A , C] = MinVolEllipse(P, t)
%       figure
%       plot3(P(1,:),P(2,:),P(3,:),'*')
%       hold on
%       Ellipse_plot(A,C)
%  
%
%  Nima Moshtagh
%  nima@seas.upenn.edu
%  University of Pennsylvania
%  Feb 1, 2007
%  Updated: Feb 3, 2007
%%%%%%%%%%%  start  %%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 20; % Default value for grid
% See if the user wants a different value for N.
%----------------------------------------------
% check the dimension of the inputs: 2D or 3D
%--------------------------------------------
if length(C) == 3,
    Type = '3D';
elseif length(C) == 2,
    Type = '2D';
else
    display('Cannot plot an ellipse with more than 3 dimensions!!');
    return
end
% "singular value decomposition" to extract the orientation and the
% axes of the ellipsoid
[U D V] = svd(A);
if strcmp(Type, '2D'),
    % get the major and minor axes
    %------------------------------------
    a = 1/sqrt(D(1,1));
    b = 1/sqrt(D(2,2));
    theta = [0:1/N:2*pi+1/N];
    % Parametric equation of the ellipse
    %----------------------------------------
    state(1,:) = a*cos(theta); 
    state(2,:) = b*sin(theta);
    % Coordinate transform 
    %----------------------------------------
    X = V * state;
    X(1,:) = X(1,:) + C(1);
    X(2,:) = X(2,:) + C(2);
    
elseif strcmp(Type,'3D'),
    % generate the ellipsoid at (0,0,0)
    %----------------------------------
    a = 1/sqrt(D(1,1));
    b = 1/sqrt(D(2,2));
    c = 1/sqrt(D(3,3));
    [X,Y,Z] = ellipsoid(0,0,0,a,b,c,N);
    
    %  rotate and center the ellipsoid to the actual center point
    %------------------------------------------------------------
    XX = zeros(N+1,N+1);
    YY = zeros(N+1,N+1);
    ZZ = zeros(N+1,N+1);
    for k = 1:length(X),
        for j = 1:length(X),
            point = [X(k,j) Y(k,j) Z(k,j)]';
            P = V * point;
            XX(k,j) = P(1)+C(1);
            YY(k,j) = P(2)+C(2);
            ZZ(k,j) = P(3)+C(3);
        end
    end
end
% Plot the ellipse
%----------------------------------------
if strcmp(Type,'2D'),
    plot(X(1,:),X(2,:),Marker,'LineWidth',2);
    hold on;
    %plot(C(1),C(2),'r*');
    axis equal, grid
else
    mesh(XX,YY,ZZ);
    axis equal
    hidden off
end

end
            
