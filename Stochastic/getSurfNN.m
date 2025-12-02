function [ISurf] = getSurfNN(ampOut,x,y,z,zCav,dsUnitVec)
% this function was written to get the NN due to the outer surface
rVec = [x,y,(z-zCav)];
rDist = rVec.^2;
rDist = sum(rDist,2);
rDist = sqrt(rDist);

rDist3 = rDist.^3;
% dS = dx*dy;
% dsVec = dS*dsUnitVec;

Ids = sum(ampOut.*dsUnitVec,2);

Ids = Ids./rDist3;

I1X = Ids.*rVec(:,1);
I1Y = Ids.*rVec(:,2);
I1Z = Ids.*rVec(:,3);

%I1X = sum(I1X);I1Y = sum(I1Y);I1Z = sum(I1Z); 
ISurf = [I1X,I1Y,I1Z];
end