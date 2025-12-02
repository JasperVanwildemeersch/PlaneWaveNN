% this script processes the NN estimate per realization, and givesb the NN
% estimate over several realizations
clear; close all;
fAll = [2,4];
lenF = length(fAll);
G = 6.67430*10^-11;
rho = 2800;
IVolEst = [];
ISurfEst = [];
ITotEst = [];
for i = 1:1:lenF
    fName = ['NewFreq',num2str(fAll(i)),'Hz.mat'];
    load(fName);
    sFASDX = 1/sqrt(mean(abs(uCavASDX).^2));
    sFASDY = 1/sqrt(mean(abs(uCavASDY).^2));
    sFASDZ = 1/sqrt(mean(abs(uCavASDZ).^2));
    
    IVolEst(i,1) = G*rho*sFASDX*sqrt(mean(abs(IVolTotAll(:,1)).^2));
    IVolEst(i,2) = G*rho*sFASDY*sqrt(mean(abs(IVolTotAll(:,2)).^2));
    IVolEst(i,3) = G*rho*sFASDZ*sqrt(mean(abs(IVolTotAll(:,3)).^2));
    
    ISurfEst(i,1) = G*rho*sFASDX*sqrt(mean(abs(ISurfTotAll(:,1)).^2));
    ISurfEst(i,2) = G*rho*sFASDY*sqrt(mean(abs(ISurfTotAll(:,2)).^2));
    ISurfEst(i,3) = G*rho*sFASDZ*sqrt(mean(abs(ISurfTotAll(:,3)).^2));
    
    ITotEst(i,1) = G*rho*sFASDX*sqrt(mean(abs(IVolTotAll(:,1) - ISurfTotAll(:,1)).^2));
    ITotEst(i,2) = G*rho*sFASDY*sqrt(mean(abs(IVolTotAll(:,2) - ISurfTotAll(:,2)).^2));
    ITotEst(i,3) = G*rho*sFASDZ*sqrt(mean(abs(IVolTotAll(:,3) - ISurfTotAll(:,3)).^2));
    
    disp('Stop');
end

figure(1);
subplot(1,3,1)
hold on;
plot(fAll,IVolEst(:,1),'b','LineWidth',2);
plot(fAll,ISurfEst(:,1),'ro');
plot(fAll,ITotEst(:,1),'mo');
plot(fAll,G*rho*4*pi/3*ones(length(fAll),1),'g','LineWidth',2);
hold off;

subplot(1,3,2)
hold on;
plot(fAll,IVolEst(:,2),'b','LineWidth',2);
plot(fAll,ISurfEst(:,2),'ro');
plot(fAll,ITotEst(:,2),'mo');
plot(fAll,G*rho*4*pi/3*ones(length(fAll),1),'g','LineWidth',2);
hold off;

subplot(1,3,3)
hold on;
plot(fAll,IVolEst(:,3),'b','LineWidth',2);
plot(fAll,ISurfEst(:,3),'ro');
plot(fAll,ITotEst(:,3),'mo');
plot(fAll,G*rho*4*pi/3*ones(length(fAll),1),'g','LineWidth',2);
hold off;