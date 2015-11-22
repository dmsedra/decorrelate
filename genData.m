%generate X = dxn ~ N(0,v)
function X = genData(d,n)
    v = 2*rand(d,1); %generate variances
    
    X = zeros(d,n);
    for i = 1:d
        X(i,:) = normrnd(0,v(i),1,n);
    end
end