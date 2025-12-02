% this script estimates the Newtonian acceleration from a stochastic
% underground seismic field produced from a superposition of plane P and S waves

clear; close all;
%% simulation parameters
simCase = "halfSpace"; % or "fullSpace" a string
% some parameters that remain unchanged
vP = 4000; % P wave velocity in m/s
vS = 3000; % S -wave velocity in m/s
rho = 2800; % same as the lower limit paper in kg/m^3
nRea = 100; % total number of realizations, about 100, can test with just 1
Nwaves = 100; % number of plane waves per realization
G = 6.67430*10^-11; % gravitational constant

fAll = 4; % a single frequency value, units in Hz, range [2,8]
R = 5000; % maximum radius of integration in meters
% the way you set up the problem depends whether it is full-space or
% half-space
if(simCase=="halfSpace")
    nR = 300; % number of points along radial direction for generation the Gauss-Legendre 
    % quadrature points
    % theta is the angle measured clockwise from the vertical
    nTheta = 100; % number of points along theta
    % phi is measured in the horizontal plane
    nPhi = 500; % number of points along phi
    rCav = 20; % radius of cavern
    % loaction of cavity is at (0,0,-250)
    zCav = -250; % depth of cavern, change it to 0 for full space
    xCav = 0; yCav = 0;
    % bounds along radial directions, finer sampling near the cavity
    % note that GL points are generating finely between r = 210 to 290 m
    % this is for half space
    bounds = [0, -zCav - 2*rCav, -zCav + 2*rCav,R];
    
    pointsPerRegion = [30,30,240]; % for half space test
    % note that 30+30+240 must be equal nR, a limitation, fix later
    rVec = [];
    wR = [];

    % for half space
    for i = 1:3
        [r_sub, w_sub] = lgwt(pointsPerRegion(i), bounds(i), bounds(i+1));
        rVec = [rVec; r_sub];
        wR = [wR; w_sub];
    end
    
    % -- Gauss–Legendre in theta [pi/2,pi], for half space
    theta1 = pi/2; theta2 = pi;
    [theta, wTheta] = lgwt(nTheta, theta1, theta2);

    % -- Trapezoidal in [0, 2pi), phi remains unchanged irespective of
    % half space or ful space
    phi = linspace(0, 2*pi, nPhi+1); phi(end) = [];
    wPhi = 2*pi / nPhi;

elseif(simCase=="fullSpace")
    nR = 300; % this remains same irrespective of half or full sphere
    nTheta = 200; % must be doubled because the integration happens from 0-pi
    nPhi = 500; % remains unchanged
    rCav = 20; % remains unchanged
    zCav = 0; % cavity at center and matter all around it
    xCav = 0; yCav = 0; % same as half-space case
    bounds = [0,rCav,R]; % because now the cavity is at the center of the sphere
    pointsPerRegion = [10, 290];  % for full space test, 10+290=nR
    rVec = [];
    wR = [];
    %for full space
    for i = 1:2
        [r_sub, w_sub] = lgwt(pointsPerRegion(i), bounds(i), bounds(i+1));
        rVec = [rVec; r_sub];
        wR = [wR; w_sub];
    end
    theta1 = 0; theta2 = pi;
    % for full space, limits will be from [0,pi]
    [theta, wTheta] = lgwt(nTheta,theta1,theta2);
    phi = linspace(0, 2*pi, nPhi+1); phi(end) = [];
    wPhi = 2*pi / nPhi;
end
% for the above there is an efficient way to determine the parameters
% do later

%% main program
% -- Build flat theta–phi grid
[TH, PH] = meshgrid(theta, phi);      % size Nphi × Ntheta
thetaFlat = TH(:);                   % (Nphi*Ntheta) × 1
phiFlat = PH(:);

[thetaPhi, weights] = createSurfPoints(max(rVec), nTheta, nPhi, theta1, theta2);
sThetaCPhi = sin(thetaPhi(:,1)).*cos(thetaPhi(:,2));
sThetaSPhi = sin(thetaPhi(:,1)).*sin(thetaPhi(:,2));
cTheta = cos(thetaPhi(:,1));
dsUnitVec = [sThetaCPhi,sThetaSPhi,cTheta];

% -- Precompute angular components
sinTheta = sin(thetaFlat);
cosTheta = cos(thetaFlat);
cosPhi   = cos(phiFlat);
sinPhi   = sin(phiFlat);

% -- Repeat Gauss–Legendre weights without interpolation
wtTheta = repmat(wTheta(:)', nPhi, 1);
wtTheta = wtTheta(:);  % Flattened to match thetaFlat

% -- Angular weights (excluding r^2, introduced later)
angularWeights = sinTheta .* wtTheta * wPhi;  % size (Nphi*Ntheta) × 1

xyzCav = [];
IVolTotAll = [];
% predefined matrix to store the displacement at expected displacement at
% the center of the cavity
uCavASDX = zeros(nRea,length(fAll));
uCavASDY = zeros(nRea,length(fAll));
uCavASDZ = zeros(nRea,length(fAll));

for reaNo = 1:1:nRea
    disp(['Performing realization = ',num2str(reaNo)]);
    % Random directions over unit sphere
    thetaWP = acos(2*rand(Nwaves,1) - 1);     % [0, pi]
    phiWP = 2*pi*rand(Nwaves,1);            % [0, 2pi]
    phaseP = 2*pi*rand(Nwaves,1);
    
    thetaWS = acos(2*rand(Nwaves,1) - 1);     % [0, pi]
    phiWS = 2*pi*rand(Nwaves,1);            % [0, 2pi]
    phaseS = 2*pi*rand(Nwaves,1);
    
    % basically the polarization angle of the wave
    mixAngle = 2*pi*rand(Nwaves,1);
    for fNo = 1:1:length(fAll)
        IVolTot = [0,0,0];
        f = fAll(fNo);
        % evaluate the synthetic field at the center of the cavity once
        % this is meant to store the asd per realization, and compute the scale factor later
        [uPCav, uSCav] = synthStochFieldInde([xCav,yCav,zCav],Nwaves,f,vP,vS,thetaWP,phiWP,phaseP,...
            thetaWS,phiWS,phaseS,mixAngle);
        uCavTot = uPCav+uSCav;
        uCavASDX(reaNo,fNo) = abs(uCavTot(1,1));
        uCavASDY(reaNo,fNo) = abs(uCavTot(1,2));
        uCavASDZ(reaNo,fNo) = abs(uCavTot(1,3));
        
        for i = 1:nR
            r = rVec(i);
            wr = wR(i);
            
            % Cartesian coordinates of current shell
            x = r * sinTheta .* cosPhi;
            y = r * sinTheta .* sinPhi;
            z = r * cosTheta;
            
            % Mask out cavity
            insideCavity = (x.^2 + y.^2 + (z - zCav).^2) < rCav^2;
            %         xyzCav = [xyzCav;[x(insideCavity,1),y(insideCavity,1),z(insideCavity,1)]];
            % Exclude points inside cavity
            x(insideCavity) = [];
            y(insideCavity) = [];
            z(insideCavity) = [];
            
            %create the stochastic field on [x,y,z]
            [uP, uS] = synthStochFieldInde([x,y,z], Nwaves, f, vP, vS,thetaWP,phiWP,phaseP,...
                thetaWS,phiWS,phaseS,mixAngle);
            uTot = uP+uS;
            
            w = angularWeights(~insideCavity);
            
            % Evaluate function
            IVol = getVolNN(uTot,[x, y, z],zCav);  % returns vector same size as x
            
            % Accumulate contribution from this r shell
            
            IVolTot(1,1) = IVolTot(1,1) + sum(IVol(:,1) .* r^2 .* w) * wr;
            IVolTot(1,2) = IVolTot(1,2) + sum(IVol(:,2) .* r^2 .* w) * wr;
            IVolTot(1,3) = IVolTot(1,3) + sum(IVol(:,3) .* r^2 .* w) * wr;
            
            %disp('Stop');
            if(i==nR)
                % evaluate the surface integral for the outer boundary
                [ISurf] = getSurfNN(uTot,x,y,z,zCav,dsUnitVec);
    
                % Perform the integral
                ISurfTot(1,1) = sum(ISurf(:,1).*weights(:,1));
                ISurfTot(1,2) = sum(ISurf(:,2).*weights(:,1));
                ISurfTot(1,3) = sum(ISurf(:,3).*weights(:,1));
            end
        end
        IVolTotAll(reaNo,:) = [(IVolTot(1,1)),(IVolTot(1,2)),(IVolTot(1,3))];
        ISurfTotAll(reaNo,:) = [(ISurfTot(1,1)),(ISurfTot(1,2)),(ISurfTot(1,3))];
        %k = 2*pi*f/vP;
        %kr1 = k*r1; kr2 = k*r2;
        
        %volNNTheo(fNo,1) = (cos(kr2)/(kr2)^2-sin(kr2)/(kr2)^3)-(cos(kr1)/(kr1)^2-sin(kr1)/(kr1)^3);
        
        %     figure(1);
        %     hold on;
        %     plot(f,G*rho*abs(IVolTot(1,1)),'b*');
        %     hold off;
    end
end
figure(1);
subplot(1,3,1);
hold on;
plot(1:nRea,G*rho*abs(IVolTotAll(:,1)),'bo');
plot(1:nRea,G*rho*abs(ISurfTotAll(:,1)),'ro');
hold off;

subplot(1,3,2);
hold on;
plot(1:nRea,G*rho*abs(IVolTotAll(:,2)),'bo');
plot(1:nRea,G*rho*abs(ISurfTotAll(:,2)),'ro');
hold off;

subplot(1,3,3);
hold on;
plot(1:nRea,G*rho*abs(IVolTotAll(:,3)),'bo');
plot(1:nRea,G*rho*abs(ISurfTotAll(:,3)),'ro');
hold off;

% save these per frequency
fPathSave = 'C:\Users\soume\Dropbox\EinsteinTelescopeSurvey\PlaneWaveNN\RadialIntegration\Stochastic\SaveSim\';
fNameSave = ['NewFreq',num2str(f),'Hz.mat'];
save([fPathSave,fNameSave],'uCavASDX','uCavASDY','uCavASDZ','IVolTotAll','ISurfTotAll','nRea','f');
