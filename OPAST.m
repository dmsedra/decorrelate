%takes in X nxd where each example is length n
%and there are d of them
%0<r<1 is momentum term
function  OPAST(alpha,r,X)
    n = size(X,1);
    d = size(X,2);
    
    %double check 0 mean
    X = X-repmat(mean(X,1),n,1);


    T = 1;
    W = [eye(r); zeros(n-r,r)];
    Z = eye(r);
    for j = 1:T
        for i=1:d
            x = X(:,i);
            y = W'*x;
            q = 1/alpha*Z*y;
            gamma = 1/(1+y'*q);
            tau = (1/norm(q)^2)*(1/sqrt(1+gamma^2*norm(q)^2*(norm(x)^2-norm(y)^2))-1);

            p = W*(tau*q-gamma*(1+tau*norm(q)^2)*y) + gamma*(1+tau*norm(q)^2)*x;

            Z = 1/alpha*Z - gamma*(q*q');
            W = W + (p*q');
        end
    end
    
    test(W,Z,X);
end

function test(W,Z,X) 
    Cxx = X*X'
    %inspect returned eigen vectors
    [Q,~] = eig(Z);
    T1 = W*Q';
    [T2,~] = eig(Cxx);
    
    W
    Z
    
    diag(diag(Z).^-0.5)
    %check whitening
    Uwhite = W'
    Cxx
    diag(Cxx)
    out = Uwhite*Cxx*Uwhite'
    diag(out)
    
end