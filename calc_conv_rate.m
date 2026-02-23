function p = calc_conv_rate(xseq)
    
    cols = size(xseq, 2);
    if cols < 4
        p = NaN;
        return;
    end

    x_star = xseq(:, end);

    errors = vecnorm(xseq(:, 1:end-1) - x_star);

    valid_indices = find(errors > 1e-12);

    if length(valid_indices) < 3
        p = NaN;
        return;
    end

    W = min(12, length(valid_indices));
    idxW = valid_indices(end-W+1:end);

    e = errors(idxW);

    dec_triples = (e(1:end-2) > e(2:end-1)) & (e(2:end-1) > e(3:end));
    I = find(dec_triples);

    if isempty(I)
        p = NaN;
        return;
    end

    e_prev = e(I);
    e_curr = e(I+1);
    e_next = e(I+2);

    log_bot = log(e_curr ./ e_prev);
    log_top = log(e_next ./ e_curr);

    good = abs(log_bot) >= 1e-8;
    if ~any(good)
        p = NaN;
        return;
    end

    p_local = log_top(good) ./ log_bot(good);

    p = median(p_local);
end