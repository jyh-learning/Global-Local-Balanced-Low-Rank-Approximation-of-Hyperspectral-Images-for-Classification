% This is the code for Global and Local Adversarial Low-rank Representation
% of Hyperspectral Images for Classification. This function solves the
% model min_{Y1, Y2, E} |Y1|_norm1 - \alpha |Y1|_norm2 + \beta

function [Y1,E1,Y2,E2,obj] = GLALRR(X,opts)
[m,n,p]=size(X);
% I=eye(m*n);

%%

idxx=opts.idxx;
idxy=opts.idxy;
idxz=opts.idxz;
lambda=opts.lambda;
% lambda2=opts.lambda2;
% % alpha=opts.alpha;

alpha=opts.alpha;
beta=opts.beta;


w1=1/(1+beta);
w2=1-w1;



tol = 1e-8; 
max_iter = 500;
rho = 1.1;
mu = 1e-4;
max_mu = 1e10;




dim = size(X);
L = zeros(dim);
Y1 = L;
Y2 =L;
E1 = L;
E2=L;

Lambda1=L;
% Lambda21=L;
% Lambda22=L;
% Lambda31=L;
% Lambda32=L;
Lambda2=L;
% Lambda22=L;
% Lambda3=L;
Lambda3=L;

iter = 0;
obj=[];
for iter = 1 : max_iter
    
    % update Y1
    Temp=0.5*(X+Y2-E1+(Lambda1-Lambda2)/mu);
    for i=1:length(idxx)-1
        for j=1:length(idxy)-1
             Y1(idxx(i):idxx(i+1)-1,idxy(j):idxy(j+1)-1,:)=t2m_rpca(Temp(idxx(i):idxx(i+1)-1,idxy(j):idxy(j+1)-1,:),w1/(mu*2));
        end
    end
    
    Lvector=reshape(Y1,m*n,p);
    Tvector=reshape(Temp,m*n,p);
    for iii=1:5
%         Lvector=(S1-Lambda21/mu+beta*Grd_sub/mu);
        Grd_sub =cal_subgradient_nuclear(Lvector);
        Tvector=Tvector+(alpha/(1+beta))*Grd_sub/(mu);
%         Lvector=(Lvector+beta*Grd_sub/mu);
    end
    Temp=reshape(Tvector,m,n,p);
    %%%%%%%%%%%%%%%%%%%
    for i=1:length(idxx)-1
        for j=1:length(idxy)-1
%             S1(idxx(i):idxx(i+1)-1,idxy(j):idxy(j+1)-1,:)=...
%                 prox_tnn(Temp(idxx(i):idxx(i+1)-1,idxy(j):idxy(j+1)-1,:),w1/(mu*2));
             Y1(idxx(i):idxx(i+1)-1,idxy(j):idxy(j+1)-1,:)=t2m_rpca(Temp(idxx(i):idxx(i+1)-1,idxy(j):idxy(j+1)-1,:),w1/(mu*2));
        end
    end
    
    
    % update Y2
    Temp=(Y1+(Lambda2)/mu);
    for k=1:length(idxz)-1
        Y2(:,:,idxz(k):idxz(k+1)-1)=...
            prox_nuclear(Temp(:,:,idxz(k):idxz(k+1)-1),w2/(mu));      
    end
    
    
    

   
    
    % update E1
   Temp=0.5*(X+E2-Y1+(Lambda1-Lambda3)/mu);
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   for i=1:length(idxx)-1
        for j=1:length(idxy)-1
            E1(idxx(i):idxx(i+1)-1,idxy(j):idxy(j+1)-1,:)=...
                prox_l1(Temp(idxx(i):idxx(i+1)-1,idxy(j):idxy(j+1)-1,:),((w1*lambda)/(1+beta))/(mu*2));
        end
   end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
    % update E2
    Temp=(E1+(Lambda3)/mu);
    for k=1:length(idxz)-1
        E2(:,:,idxz(k):idxz(k+1)-1)=...
            prox_l1(Temp(:,:,idxz(k):idxz(k+1)-1),((w2*lambda)/(1+beta))/(mu*1));      
    end
    
    
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
%     % update L
%     
%     L=(S1+S2-Lambda21/mu-Lambda22/mu)/2;
%     Lvector=reshape(L,m*n,p);
%     
%     for iii=1:5
% %         Lvector=(S1-Lambda21/mu+beta*Grd_sub/mu);
%         Grd_sub =cal_subgradient_nuclear(Lvector);
%         Lvector=(Lvector+beta*Grd_sub/mu);
%     end
%     L=reshape(Lvector,m,n,p);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%     % update E
%     E=(V1+V2-Lambda31/mu-Lambda32/mu)/2;

    
  
    
    d1=(X-Y1-E1);
%     d12=(X-S2-V2);
    d2=(Y1-Y2);
    d3=(E1-E2);
    
    chg = max([ max(abs(d1(:))),max(abs(d2(:))),...
        max(abs(d3(:))) ]);
    obj(iter)=chg;
    
    if chg < tol
        break;
    end 
    
    Lambda1=Lambda1+mu*d1;
%     Lambda12=Lambda12+mu*d12;
    Lambda2=Lambda2+mu*d2;
    Lambda3=Lambda3+mu*d3;

%     Lambda21=Lambda21+mu*d21;
%     Lambda22=Lambda22+mu*d22;
%     Lambda31=Lambda31+mu*d31;
%     Lambda32=Lambda32+mu*d32;

    
    mu = min(rho*mu,max_mu);    
end

% obj = tnnL+lambda*norm(S(:),1);
% err = norm(dY(:));
