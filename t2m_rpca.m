function outputTensor=t2m_rpca(inputTensor,tau)

TempTensorB=inputTensor;
[mTemp,nTemp,pTemp]=size(TempTensorB);
TempMB=reshape(TempTensorB,mTemp*nTemp,pTemp);
            % SVD 
% Stemp=prox_nuclear(TempMB,1/(mu*2));
Stemp=prox_nuclear(TempMB,tau);
       

            TempT=reshape(Stemp,mTemp,nTemp,pTemp);
            outputTensor=TempT;

