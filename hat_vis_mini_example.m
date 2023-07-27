clear all 
close all
%%

IM = [1 1 0;                % Incidence Matrix
      1 1 0;
      1 0 1;
      0 1 1;
      0 0 1];
HG1 = Hypergraph('IM',IM)    % Hypergraph object
%%
HG1.plot();                  % Plotting command 
title('Incidence Matrix v1'); xlabel('Hyperedges'); ylabel('Vertices');
set(gca,'XTick',[]); set(gca,'YTick',[])

%%
HAT.plotIncidenceMatrix(HG1, '*', {'r', 'b', 'g'}); % Plotting command 2
title('Incidence Matrix v2'); xlabel('Hyperedges'); ylabel('Vertices')
set(gca,'XTick',[]); set(gca,'YTick',[]);
%%
v = 10;     % Number of vertices
e = 20;     % Number of hyperedges
k = 4;      % Size of hyperedges

HG2 = HAT.uniformErdosRenyi(v, e, k);

HG2.plot();
title('Incidence Matrix'); xlabel('Hyperedges'); ylabel('Vertices')
set(gca,'XTick',[]); set(gca,'YTick',[]);

%%
A = HG1.adjTensor

D = HG1.degreeTensor

L1 = D - A;                 % Method 1
L2 = HG1.laplacianTensor    % Method 2

%%
C = HG1.cliqueGraph;
plot(graph(C)); title('Clique Expansion')

%%
S = HG1.starGraph;
X = [1 1 1 1 1 0 0 0]';
Y = [0 1 2 3 4 0.5 2 3.5]';
p = plot(graph(S)); title('Star Expansion')
p.XData = X; p.YData = Y;

%%
edgenames ={'e1', 'e2', 'e3'}
nodenames = {'v1, v2','v1, v2','v3','v4','v5'}

[N,M]=size(IM)
%% Sort the nodes

x=[1:N]';
y=[1:M]'; 
b=zeros(N,1);    

for it=1:10
for i=1:N
    bn=0;
    bd=0;
    for m=1:M 
        for n=1:N
            bn= bn+x(n)*IM(n,m)*IM(i,m);
            bd= bd+ IM(n,m)*IM(i,m);
        end
    end
    b(i)=bn/bd;
end    
[dum,x]=sort(b);   
end

%% Sort the edges

d=zeros(M,1);    
for it=1:10
for j=1:M
    dn=0;
    dd=0;
    for n=1:N 
        for m=1:M
            dn= dn+y(m)*IM(n,m)*IM(n,j);
            dd= dd+ IM(n,m)*IM(n,j);
        end
    end
    d(j)=dn/dd;
end    
[dum,y]=sort(d);   
end
figure(2) 
spy(IM(x,y))

%% MDS-based visualizaton of the objects

simO=[];
for k=1:N
    simO=[simO sum(min(IM,repmat(IM(k,:),N,1)),2)./sum(max(IM,repmat(IM(k,:),N,1)),2)];
end    


%transitive similarity  
simOT=simO;
for it=1:N
    for n=1:N
        simOT=max(simOT,repmat(simOT(n,:),N,1).*repmat(simOT(:,n),1,N));
    end
end    
D=1-simO;
Y = cmdscale(D,2);


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



figure(2)
clf
plot(Y(:,1),Y(:,2),'.')%, 'MarkerSize',35)
hold on
text(Y(:,1),Y(:,2),num2str([1:N]'),'FontSize',20)


%% MST of the whole network


hold on 
G = graph((simO>0).*(1-simO));
T = minspantree(G);
p = plot(G,'XData',Y(:,1),'YData',Y(:,2),...
     'EdgeColor','b','NodeLabel',[1:N],'EdgeLabel',1-G.Edges.Weight);
highlight(p,T)

%print('HyperGraph','-f2','-dpng')


%% Plot the edges as MST-s


figure(3)
clf
%plot(Y(:,1),Y(:,2),'.')
hold on
%text(Y(:,1),Y(:,2),num2str([1:N]'))
dy=range(Y(:,2))/100000/M;
dx=range(Y(:,1))/10/M;

Y_copy=round(Y(:,:),4);



As=(simO>0).*(1-simO); %this is the adjacency matrix of the weighted similarity network (whole) 


hold on 
for m=1:M
    Cr=rand(M,3);   
    Marker=MarkerVec{mod(m,length(MarkerVec))+1};
    selnodes=find(IM(:,m)>0); %nodes involved in the m-th edge 
    P=Y(selnodes,:); %the position of these nodes
    G = graph(As(selnodes,selnodes));
    T = minspantree(G);
    p = plot(T,'XData',P(:,1),'YData',P(:,2)+dy*m,...
     'EdgeColor',Cr(m,:),'NodeColor',Cr(m,:),... %Marker(1)
     'NodeLabel',selnodes,'LineWidth',5,... %'EdgeLabel',1-T.Edges.Weight,
     'MarkerSize',5,'Marker',all_marks{mod(m,13)+1},'LineStyle', Marker(2:end),'Displayname', edgenames{m});%); %,
    set(p,'EdgeAlpha',1);
    %hold on 
    %highlight(p,T)
    hold on 
   
    text(Y_copy(:,1),Y_copy(:,2),nodenames([1:N]'),'FontSize',20,'HorizontalAlignment','left','VerticalAlignment','top');
    p.NodeLabel={};

    % legend show
    
end
legend show

axis off
