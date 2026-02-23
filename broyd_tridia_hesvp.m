function y = broyd_tridia_hesvp(x, v)
    p = 7/3;
    n = length(x);
    y = zeros(n,1);

    % pad boundaries
    xpad = [0; x; 0];
    vpad = [0; v; 0];

    for i = 1:n
        xi = xpad(i+1);

        % residual r_i
        ri = (3 - 2*xi)*xi - xpad(i) - xpad(i+2) + 1;

        ari = abs(ri);
        if ari < 1e-14
            % not C^2 at ri=0 for p=7/3; safeguard (rare)
            continue;
        end

        % grad r_i coefficients on (i-1, i, i+1)
        a = -1;
        b = 3 - 4*xi;
        c = -1;

        % s = (grad r_i)^T v
        s = a*vpad(i) + b*vpad(i+1) + c*vpad(i+2);

        % phi'(ri), phi''(ri) for |ri|^p
        phi1 = p * (ari^(p-2)) * ri;
        phi2 = p * (p-1) * (ari^(p-2));

        % phi'' * grad r_i * s
        if i-1 >= 1
            y(i-1) = y(i-1) + phi2 * a * s;
        end
        y(i) = y(i) + phi2 * b * s;
        if i+1 <= n
            y(i+1) = y(i+1) + phi2 * c * s;
        end

        % phi'(ri) * Hess(r_i) * v, where Hess(r_i) = -4 e_i e_i^T
        y(i) = y(i) + phi1 * (-4 * v(i));
    end
end