%takes in X nxd where each example is length n
%and there are d of them
%0<r<1 is momentum term
%T number of passes over entire dataset
function GOPAST(alpha,r,X)
    n = size(X,1);
    d = size(X,2);
    
    %double check 0 mean
    X = X-repmat(mean(X,2),1,d);


    T = 1;%number of passes over the data
    W = [eye(r); zeros(n-r,r)];
    Z = eye(r);
    for j = 1:T%pass over data
        for i=1:d%pass over each example
            x = X(:,i);
            y = W'*x;
            q = 1/alpha*Z*y;
            gamma = 1/(1+y'*q);
            tau = (1/norm(q)^2)*(1/sqrt(1+gamma^2*norm(q)^2*(norm(x)^2-norm(y)^2))-1);

            p = W*(tau*q-gamma*(1+tau*norm(q)^2)*y) + gamma*(1+tau*norm(q)^2)*x;

            Z = 1/alpha*Z - gamma*(q*q');
            W = W + (p*q');
           
        end
        test(W,Z,X);
        
        %now update by GOPAST
        r = size(Z,1);
        for k=1:r*(r-1)/2
            [~,p] = max(abs(Z(:)));
            [l,m] = ind2sub(size(Z),p);%row col of largest element
            g = [Z(l,l)-Z(m,m);2*real(Z(l,m));2*imag(Z(l,m))];
            v = g/norm(g)*sign(g(1));

            c = sqrt((v(1)+1)/2);
            s = (v(2) + j*v(3))/2*c;

            Zcopy = Z;
            Zcopy(:,l) = c*Z(:,l) + conj(s)*Z(:,m);
            Zcopy(:,m) = c*Z(:,m) - s*Z(:,l);
            Zcopy(l,:) = c*Z(l,:) + s*Z(m,:);
            Zcopy(m,:) = c*Z(m,:) - conj(s)*Z(l,:);
            Z = Zcopy;

            Wcopy = c*W(:,l) + conj(s)*W(:,m);
            W(:,m) = c*W(:,m) - s*W(:,l);
            W(:,l) = Wcopy;
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
    Uwhite = W';
    Cxx
    diag(Cxx)
    out = Uwhite*Cxx*Uwhite'
    diag(out)
    diag(out)./diag(Cxx)
end