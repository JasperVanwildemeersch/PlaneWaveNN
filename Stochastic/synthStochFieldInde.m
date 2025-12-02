function [uP, uS] = synthStochFieldInde(xyz, Nwaves, f, vP, vS,thetaP,phiP,phaseP,...
    thetaS,phiS,phaseS,mixAngle)
% xyz: [Npoints x 3] array of positions (x, y, z)
% Nwaves: number of plane waves (per type)
% f: frequency in Hz
% vP, vS: P and S wave speeds
% Returns:
%   uP, uS: [Npoints x 3] complex displacement field vectors from P and S waves

% Constants
omega = 2*pi*f;
kP = omega / vP;
kS = omega / vS;
Npts = size(xyz,1);

%% -------- P-Waves -------- %%
%thetaP = acos(2*rand(Nwaves,1) - 1);
%phiP   = 2*pi*rand(Nwaves,1);
khatP  = [sin(thetaP).*cos(phiP), sin(thetaP).*sin(phiP), cos(thetaP)];
%phaseP = 2*pi*rand(Nwaves,1);

uP = zeros(Npts, 3);
for i = 1:Nwaves
    kvec = kP * khatP(i,:);
    arg  = xyz * kvec.' + phaseP(i);
    amp  = exp(-1i * arg);
    uP   = uP + amp .* khatP(i,:);  % longitudinal displacement
end
uP = uP / sqrt(Nwaves);

%% -------- S-Waves (SH + SV) -------- %%
%thetaS = acos(2*rand(Nwaves,1) - 1);
%phiS   = 2*pi*rand(Nwaves,1);
khatS  = [sin(thetaS).*cos(phiS), sin(thetaS).*sin(phiS), cos(thetaS)];
%phaseS = 2*pi*rand(Nwaves,1);
%mixAngle = 2*pi*rand(Nwaves,1);  % random mixture between SH and SV

uS = zeros(Npts, 3);
for i = 1:Nwaves
    n = khatS(i,:);
    % Basis vectors perpendicular to n
    if abs(n(3)) < 0.99
        eSH = cross(n, [0,0,1]);
    else
        eSH = cross(n, [0,1,0]);
    end
    eSH = eSH / norm(eSH);
    eSV = cross(eSH, n);  % also unit vector

    % Random linear combo
    pol = cos(mixAngle(i)) * eSH + sin(mixAngle(i)) * eSV;

    kvec = kS * n;
    arg  = xyz * kvec.' + phaseS(i);
    amp  = exp(-1i * arg);
    uS   = uS + amp .* pol;
end
uS = uS / sqrt(Nwaves);

end