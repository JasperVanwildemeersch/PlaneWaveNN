function [thetaPhi, weights] = createSurfPoints(R, Ntheta, Nphi, theta1, theta2)
    % Generate flattened list of [theta, phi] points and weights for
    % integrating over the surface of a hemisphere of radius R
    % theta in [pi/2, pi], phi in [0, 2pi)

    % Gauss–Legendre nodes and weights in in [pi/2, pi]
    [theta, wtheta] = lgwt(Ntheta, theta1, theta2);  % both are Ntheta x 1

    % Uniform quadrature for [0, 2pi)
    phi = linspace(0, 2*pi, Nphi + 1); phi(end) = [];
    wphi = (2*pi) / Nphi;  % scalar

    % Create all (theta, phi) combinations
    [THETA, PHI] = meshgrid(theta, phi);         % Nphi × Ntheta
    thetaPhi = [THETA(:), PHI(:)];               % (Nphi*Ntheta) × 2

    % Evaluate weights: R² sin(?) w? w? at each grid point
    [WTHETA, ~] = meshgrid(wtheta, phi);         % Nphi × Ntheta
    [SIN_THETA, ~] = meshgrid(sin(theta), phi);  % Nphi × Ntheta

    weights = R^2 * wphi * (SIN_THETA .* WTHETA);  % Nphi × Ntheta
    weights = weights(:);                         % (Nphi*Ntheta) × 1
end