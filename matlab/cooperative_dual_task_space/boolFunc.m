function bv = boolFunc(dq, q, q_min, q_max)

if q(1) > q_min(1) && q(1) < q_max(1) || q(2) > q_min(2) && q(2) < q_max(2) || q(3) > q_min(3) && q(3) < q_max(3) || q(4) > q_min(4) && q(4) < q_max(4) || q(5) > q_min(5) && q(5) < q_max(5) || ...
        q(6) > q_min(6) && q(6) < q_max(6) || q(7) > q_min(7) && q(7) < q_max(7) || q(8) > q_min(8) && q(8) < q_max(8) || q(9) > q_min(9) && q(9) < q_max(9) || ...
        q(10) > q_min(10) && q(10) < q_max(10) || q(11) > q_min(11) && q(11) < q_max(11) || q(12) > q_min(12) && q(12) < q_max(12) || q(13) > q_min(13) && q(13) < q_max(13) || ...
        q(14) > q_min(14) && q(14) < q_max(14)
    bv = true;
    
elseif q(1) <= q_min(1) && dq(1) >= 0 || q(2) <= q_min(2) && dq(2) >= 0 || q(3) <= q_min(3) && dq(3) >= 0 || q(4) <= q_min(4) && dq(4) >= 0 || q(5) <= q_min(5) && dq(5) >= 0 || ...
       q(6) <= q_min(6) && dq(6) >= 0 || q(7) <= q_min(7) && dq(7) >= 0 || q(8) <= q_min(8) && dq(8) >= 0 || q(9) <= q_min(9) && dq(9) >= 0 || q(10) <= q_min(10) && dq(10) >= 0 || ...
       q(11) <= q_min(11) && dq(11) >= 0 || q(12) <= q_min(12) && dq(12) >= 0 || q(13) <= q_min(13) && dq(13) >= 0 || q(14) <= q_min(14) && dq(14) >= 0
    bv = true;
    
elseif q(1) <= q_min(1) && dq(1) < 0 || q(2) <= q_min(2) && dq(2) < 0 || q(3) <= q_min(3) && dq(3) < 0 || q(4) <= q_min(4) && dq(4) < 0 || q(5) <= q_min(5) && dq(5) < 0 || ...
       q(6) <= q_min(6) && dq(6) < 0 || q(7) <= q_min(7) && dq(7) < 0 || q(8) <= q_min(8) && dq(8) < 0 || q(9) <= q_min(9) && dq(9) < 0 || q(10) <= q_min(10) && dq(10) < 0 || ...
       q(11) <= q_min(11) && dq(11) < 0 || q(12) <= q_min(12) && dq(12) < 0 || q(13) <= q_min(13) && dq(13) < 0 || q(14) <= q_min(14) && dq(14) < 0
    bv = false;
    
elseif q(1) >= q_max(1) && dq(1) <= 0 || q(2) >= q_max(2) && dq(2) <= 0 || q(3) >= q_max(3) && dq(3) <= 0 || q(4) >= q_max(4) && dq(4) <= 0 || q(5) >= q_max(5) && dq(5) <= 0 || ...
       q(6) >= q_max(6) && dq(6) <= 0 || q(7) >= q_max(7) && dq(7) <= 0 || q(8) >= q_max(8) && dq(8) <= 0 || q(9) >= q_max(9) && dq(9) <= 0 || q(10) >= q_max(10) && dq(10) <= 0 || ...
       q(11) >= q_max(11) && dq(11) <= 0 || q(12) >= q_max(12) && dq(12) <= 0 || q(13) >= q_max(13) && dq(13) <= 0 || q(14) >= q_max(14) && dq(14) <= 0
    bv = true;
    
else
    
    bv = false;
    
end

end