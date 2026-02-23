function y = penalty_1_hesvp(x, v)
    c = 100000;

    % last residual
    fn1 = (x.'*x) - 1/4;

    % hessian is: h = (2/c + 4*fn1) I + 8 * x'
    a = (2/c) + 4*fn1;

    y = a*v + 8*x*(x.'*v);
end